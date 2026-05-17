"""Phase 6 -- Generalisierung ueber Geometrien.

Trainiert eine Policy auf dem kompletten Trainings-Pool mit Domain Randomization
(Skalierung, Rotation). Nutzt 8 parallele Worker (SubprocVecEnv) und ein
etwas groesseres neuronales Netz, um die Kapazitaet fuer verschiedene
Geometrien bereitzustellen.

Ein Eval-Callback testet das Modell alle 100k Steps auf dem ungesehenen
Test-Pool (mode="test") und speichert das beste Modell, sobald die harten
Erfolgskriterien aus `cfg.success` erreicht werden.
"""

import os
import time
import json
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
    from stable_baselines3.common.torch_layers import NatureCNN
except ImportError:
    print("FEHLER: stable-baselines3 oder torch fehlen!")
    print("Bitte installieren: pip install stable-baselines3[extra] torch tensorboard")
    import sys
    sys.exit(1)

# Um relative Imports zu erlauben
try:
    from .config import RLConfig, default_config
    from .env import PlasmaCutterEnv
    from .baseline import list_geometries, split_train_test
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.rl.config import RLConfig, default_config
    from plasma_cutter.rl.env import PlasmaCutterEnv
    from plasma_cutter.rl.baseline import list_geometries, split_train_test


class InfoCallback(BaseCallback):
    """Loggt Episoden-Infos in Tensorboard waehrend des Trainings."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                infos = self.locals.get("infos", [])
                if i < len(infos):
                    info = infos[i]
                    self.logger.record("rollout/ep_coverage", info.get("coverage", 0.0))
                    self.logger.record("rollout/ep_time", info.get("total_time", 0.0))
                    self.logger.record("rollout/ep_pierces", info.get("n_pierces", 0))
                    self.logger.record("rollout/is_feasible", float(info.get("is_feasible", True)))
        return True


class EvalCallback(BaseCallback):
    """Evaluiert das Modell deterministisch auf dem Test-Set.
    
    Vergleicht die Leistung gegen die Heuristik-Baseline und speichert
    das beste Modell ab, wenn die Erfolgskriterien aus `cfg.success`
    erfuellt sind.
    """
    def __init__(self, cfg: RLConfig, eval_freq: int, save_dir: str, verbose=1):
        super().__init__(verbose)
        self.cfg = cfg
        self.eval_freq = eval_freq
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_success_rate = -1.0
        
        # Test-Pool Geometrien ermitteln
        all_files = list_geometries(self.cfg)
        _, self.test_files = split_train_test(all_files, self.cfg)
        
        # Baseline-JSON laden fuer Zeitvergleiche.
        # Hart fehlschlagen, wenn die Datei fehlt: ein stiller Fallback
        # auf "time_ok=True" wuerde im EvalCallback Erfolg vortaeuschen
        # und ein nicht erreichtes Ziel als erreicht melden.
        baseline_file = Path(__file__).resolve().parent / "results" / f"baseline_v{self.cfg.version}.json"
        if not baseline_file.exists():
            raise FileNotFoundError(
                f"Baseline-Datei {baseline_file} nicht gefunden. "
                f"Phase 1 (`python -m plasma_cutter.rl.baseline`) muss vor "
                f"dem Generalisierungstraining mit derselben cfg.version="
                f"'{self.cfg.version}' gelaufen sein -- sonst ist das "
                f"Erfolgskriterium nicht ueberpruefbar."
            )
        with open(baseline_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Wir mappen geometry_file -> baseline_total_time
        self.baseline_data = {
            entry["geometry_file"]: entry["total_time"]
            for entry in data.get("test", [])
        }
        if not self.baseline_data:
            raise RuntimeError(
                f"Baseline-Datei {baseline_file} enthaelt keine Test-Eintraege. "
                f"Pruefe Phase 1 -- der Test-Split war beim Baseline-Lauf leer."
            )
            
        # Wir brauchen eine eigene Env, um sequentiell zu evaluieren
        # Diese Env nutzt keine Randomisierung, damit die Evaluierung determ. bleibt
        self.eval_env = PlasmaCutterEnv(self.cfg, mode="test")
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True
        
    def _run_evaluation(self):
        if self.verbose > 0:
            print(f"\n[{self.num_timesteps}] Starte Evaluation auf {len(self.test_files)} Test-Geometrien...")
            
        success_count = 0
        total_test_cov = 0.0
        
        # Wir testen jede Test-Geometrie einmal mit fixem Seed
        for test_file in self.test_files:
            obs, info = self.eval_env.reset(seed=42, options={"geometry_file": test_file.name})
            done = False
            
            while not done:
                # Deterministische Aktion (argmax)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
            final_cov = info["coverage"]
            final_time = info["total_time"]
            total_test_cov += final_cov
            
            # Erfolgskriterien pruefen
            cov_ok = final_cov >= self.cfg.success.coverage_target
            
            baseline_time = self.baseline_data.get(test_file.name)
            if baseline_time is None or baseline_time <= 0:
                raise RuntimeError(
                    f"Keine gueltige Baseline-Zeit fuer Test-Geometrie "
                    f"'{test_file.name}'. Train/Test-Split hat sich seit "
                    f"dem Baseline-Lauf veraendert -- Phase 1 neu laufen lassen."
                )
            time_ratio = final_time / baseline_time
            time_ok = time_ratio <= self.cfg.success.time_ratio_vs_baseline
                
            is_feasible = info.get("is_feasible", True)
            
            if cov_ok and time_ok and is_feasible:
                success_count += 1
                
        success_rate = success_count / len(self.test_files)
        mean_cov = total_test_cov / len(self.test_files)
        
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_coverage", mean_cov)
        
        if self.verbose > 0:
            print(f"Eval Success Rate: {success_rate*100:.1f}% (Ziel: {self.cfg.success.min_success_rate*100:.1f}%)")
            print(f"Eval Mean Coverage: {mean_cov*100:.1f}%")
            
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            if self.verbose > 0:
                print(f"Neue Best-Success-Rate: {success_rate*100:.1f}%. Speichere bestes Modell...")
            self.model.save(str(self.save_dir / "best_model"))
            
        # Wenn wir das Erfolgskriterium geknackt haben, speichern wir einen expliziten Milestone
        if success_rate >= self.cfg.success.min_success_rate:
            if self.verbose > 0:
                print("ERFOLGSKRITERIUM ERFUELLT! Modell wird als Milestone gespeichert.")
            self.model.save(str(self.save_dir / f"milestone_success_{self.num_timesteps}"))
            
    def _on_training_end(self) -> None:
        self.eval_env.close()


def make_env_fn(cfg: RLConfig, seed: int):
    """Factory-Funktion fuer SubprocVecEnv"""
    def _init():
        # Wir verwenden den cfg seed als Basis plus den worker index
        # Mode ist 'train', also greift er auf den Train-Pool zu
        env = PlasmaCutterEnv(cfg, mode="train")
        # Hier koennten wir env = Monitor(env) wrappen, aber VecMonitor
        # auf der auesseren VecEnv uebernimmt das Logging eleganter.
        # Wichtig: gym erwartet eigentlich, dass der Seed per reset()
        # gesetzt wird, was SB3 intern macht.
        return env
    return _init


def train_phase6():
    cfg = default_config()
    # Volles Programm fuer Generalisierung:
    cfg.scope.single_geometry_file = None
    cfg.scope.randomize_scale = True
    cfg.scope.randomize_rotation = True
    
    print(f"Starte Phase 6 Training (Generalisierung)")
    print(f"Randomisierung: Scale={cfg.scope.randomize_scale}, Rotation={cfg.scope.randomize_rotation}")
    
    out_dir = Path(__file__).resolve().parent / "checkpoints" / f"phase6_v{cfg.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_envs = 6
    print(f"Starte {n_envs} parallele Worker in SubprocVecEnv...")
    
    # SubprocVecEnv fuer echtes Multiprocessing
    vec_env = SubprocVecEnv([make_env_fn(cfg, i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    
    # Etwas tieferes Netz als in Phase 5, um mehr Geometrien zu fassen
    policy_kwargs = dict(
        features_extractor_class = NatureCNN,
        features_extractor_kwargs = dict(features_dim=512),
        net_arch = dict(pi=[128, 128], vf=[128, 128]),
        activation_fn = nn.ReLU
    )
    
    # PPO Modell
    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,  # pro Env, total batch size = 2048 * 8 = 16384
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(
            (Path(__file__).resolve().parent / "tensorboard").as_posix()
        ),
        verbose=1,
        device="auto"
    )
    
    # Callbacks aufsetzen
    # Eval_freq in Vektor-Envs bezieht sich auf step(), nicht auf total timesteps.
    # Bei 8 Envs entspricht eval_freq=12500 -> 100.000 Timesteps insgesamt.
    eval_freq_per_env = 100_000 // n_envs
    eval_callback = EvalCallback(cfg, eval_freq=eval_freq_per_env, save_dir=str(out_dir))
    
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // n_envs,
        save_path=str(out_dir),
        name_prefix="ppo_gen"
    )
    info_callback = InfoCallback()
    
    callbacks = CallbackList([info_callback, eval_callback, checkpoint_callback])
    
    total_timesteps = 5_000_000
    print(f"\nBeginne PPO-Training ({total_timesteps} Steps)...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=f"PPO_Phase6_Gen_v{cfg.version}"
        )
        print("Training abgeschlossen!")
        model.save(str(out_dir / "final_model"))
        print(f"Modell gespeichert unter {out_dir / 'final_model'}")
    except KeyboardInterrupt:
        print("Training vom Nutzer abgebrochen. Speichere aktuelles Modell...")
        model.save(str(out_dir / "interrupted_model_gen"))
    finally:
        vec_env.close()

if __name__ == "__main__":
    # Wichtig fuer SubprocVecEnv auf Windows
    import multiprocessing
    if multiprocessing.get_start_method() == 'spawn':
        pass
    train_phase6()

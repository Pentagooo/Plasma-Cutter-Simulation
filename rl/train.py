"""Phase 5b -- Re-Training mit Curriculum-Reward + Material-naher TCP-Init.

Phase 5 (Variante a) ist mechanisch durchgelaufen, hat aber das Lernziel
verfehlt: ep_coverage blieb auf 0, weil der Agent ein Local-Optimum
"Material meiden" gefunden hat. Diagnose:
  - random Bbox-Rand-Start  -> Agent muss blind zum Material wandern
  - Time-Penalty / cov-Reward -> Time dominiert frueh, weil cov ~0.01

Aenderungen gegenueber Phase 5:
  (b) `_sample_initial_tcp` startet jetzt nahe einem Materialpunkt
      (siehe env.py).
  (c) `cfg.reward.w_time_delta` deutlich kleiner (Curriculum-Stage 1).
      Time-Druck bleibt erhalten, ist aber im fruehen Training nicht
      mehr dominant gegenueber dem Coverage-Signal.

GPU/CPU-Last (User-Wunsch: deutlich mehr Auslastung):
  - 16 statt 8 parallele Env-Worker
  - 128x128 Observation statt 96x96
  - 1024-d Feature-Extractor + 512x512 MLP-Heads
  - Mini-Batch 2048, Rollout-Buffer 16k
  - leicht hoehere Entropie fuer mehr Exploration in der Stage-1-Phase
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
    from stable_baselines3.common.torch_layers import NatureCNN
except ImportError as e:
    print(f"FEHLER: stable-baselines3 oder torch fehlen ({e})")
    print("Bitte installieren: pip install stable-baselines3[extra] tensorboard")
    sys.exit(1)

import sys
from pathlib import Path

# Projektwurzel zum sys.path hinzufügen, um absolute Imports zu ermöglichen
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from plasma_cutter.rl.config import RLConfig, default_config
from plasma_cutter.rl.env import PlasmaCutterEnv


# ---------------------------------------------------------------------------
# Konfigurations-Konstanten
# ---------------------------------------------------------------------------

N_ENVS          = 16          # Reduziert auf 16 für moderate CPU-Last
N_STEPS_PER_ENV = 4096        # Massiver Rollout-Buffer
BATCH_SIZE      = 16384       # Angepasst an kleineren Buffer (16 * 4096 = 65536)
N_EPOCHS        = 15          # Mehr Epochen für intensiveres Lernen der neuen Reward-Struktur
TOTAL_TIMESTEPS = 2_000_000   
RESOLUTION      = 128         
FEATURES_DIM    = 1024        
MLP_HIDDEN      = 512         

# (c) Curriculum-Stage 1: Zeitstrafe auf NULL.
# Wir wollen, dass der Agent erst einmal NUR lernt, die Kontur zu finden.
# Zeitdruck wird erst in Stage 2 (Phase 6) wieder eingeführt.
CURRICULUM_W_TIME = 0.0
# Mehr Exploration im Stage 1, damit der Agent ueberhaupt erst
# lernt, dass es lohnt, das Material zu beruehren.
ENT_COEF          = 0.05


# ---------------------------------------------------------------------------
# Logging-Callback
# ---------------------------------------------------------------------------


class InfoCallback(BaseCallback):
    """Loggt Episoden-Infos in Tensorboard."""

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue
            infos = self.locals.get("infos", [])
            if i >= len(infos):
                continue
            info = infos[i]
            self.logger.record("rollout/ep_coverage", info.get("coverage", 0.0))
            self.logger.record("rollout/ep_time",     info.get("total_time", 0.0))
            self.logger.record("rollout/ep_pierces",  info.get("n_pierces", 0))
            self.logger.record("rollout/is_feasible", float(info.get("is_feasible", True)))
        return True


# ---------------------------------------------------------------------------
# Env-Factory fuer SubprocVecEnv
# ---------------------------------------------------------------------------


def _make_env_fn(cfg: RLConfig, worker_idx: int):
    """Closure-freie Factory -- SubprocVecEnv pickelt diese ueber Prozesse."""
    def _init():
        env = PlasmaCutterEnv(cfg, mode="train")
        # Jeder Worker bekommt einen eigenen Seed-Offset, damit die
        # Random-TCP-Posen pro Worker variieren und der Buffer divers wird.
        env.reset(seed=cfg.seed + worker_idx)
        return env
    return _init


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------


def train_phase5() -> None:
    # --- GPU-Setup -------------------------------------------------------
    if not torch.cuda.is_available():
        print("FEHLER: CUDA ist nicht verfuegbar. Pruefe torch-Installation.")
        sys.exit(1)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name} (compute capability {cap[0]}.{cap[1]})")
    print(f"CUDA: {torch.version.cuda} | torch {torch.__version__}")

    # --- Config ----------------------------------------------------------
    cfg = default_config()
    cfg.scope.single_geometry_file = "RLKontur.json"
    cfg.scope.randomize_scale      = False
    cfg.scope.randomize_rotation   = False
    cfg.observation.resolution     = RESOLUTION
    # (c) Curriculum-Stage 1
    cfg.reward.w_time_delta        = CURRICULUM_W_TIME
    cfg.description = (
        f"Phase 5b High-GPU: PPO/CnnPolicy auf RLKontur.json, "
        f"{RESOLUTION}x{RESOLUTION} Obs, {N_ENVS} parallele Envs, "
        f"w_time_delta={CURRICULUM_W_TIME} (Curriculum Stage 1), "
        f"LR=5e-4, TCP-Init nahe Material."
    )

    print(f"\nPhase 5b Training (single geometry: {cfg.scope.single_geometry_file})")
    print(f"Obs-Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Parallele Envs: {N_ENVS}")
    print(f"Rollout-Buffer: {N_STEPS_PER_ENV * N_ENVS} steps")
    print(f"Mini-Batch:     {BATCH_SIZE}")
    print(f"Total steps:    {TOTAL_TIMESTEPS:,}")
    print(f"Features-Dim:   {FEATURES_DIM}, MLP {MLP_HIDDEN}x{MLP_HIDDEN}")
    print(f"Curriculum:     w_time_delta={CURRICULUM_W_TIME}, ent_coef={ENT_COEF}")

    # --- VecEnv ----------------------------------------------------------
    vec_env = SubprocVecEnv(
        [_make_env_fn(cfg, i) for i in range(N_ENVS)],
        start_method="spawn",  # Windows-kompatibel
    )
    vec_env = VecMonitor(vec_env)

    # --- Policy ----------------------------------------------------------
    # NatureCNN expects float32 in [0,1] when normalize_images=False.
    # Net-Arch: 512-d Features, 256x256 MLP-Heads -- genug Kapazitaet
    # damit die GPU nicht idle laeuft.
    policy_kwargs = dict(
        features_extractor_class  = NatureCNN,
        features_extractor_kwargs = dict(features_dim=FEATURES_DIM),
        net_arch                  = dict(pi=[MLP_HIDDEN, MLP_HIDDEN],
                                         vf=[MLP_HIDDEN, MLP_HIDDEN]),
        activation_fn             = nn.ReLU,
        normalize_images          = False,   # Obs ist bereits float in [0,1]
    )

    # --- PPO -------------------------------------------------------------
    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate   = 5e-4,
        n_steps         = N_STEPS_PER_ENV,
        batch_size      = BATCH_SIZE,
        n_epochs        = N_EPOCHS,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = ENT_COEF,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = policy_kwargs,
        tensorboard_log = str((Path(__file__).resolve().parent / "tensorboard").as_posix()),
        verbose         = 1,
        device          = "cuda",
        seed            = cfg.seed,
    )

    # Verifizieren, dass die Policy wirklich auf der GPU sitzt.
    policy_device = next(model.policy.parameters()).device
    print(f"Policy device: {policy_device}")
    if policy_device.type != "cuda":
        print("WARNUNG: Policy ist NICHT auf der GPU. Abbruch.")
        sys.exit(1)

    # --- Callbacks -------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "checkpoints" / f"phase5b_v{cfg.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint-Verzeichnis: {out_dir}")

    checkpoint_cb = CheckpointCallback(
        save_freq      = max(1, 50_000 // N_ENVS),
        save_path      = str(out_dir),
        name_prefix    = "ppo_model",
    )
    info_cb = InfoCallback()
    callbacks = CallbackList([info_cb, checkpoint_cb])

    # --- Training --------------------------------------------------------
    print("\n=== TensorBoard Info ===")
    tb_url = "http://127.0.0.1:6006"
    print(f"TensorBoard Link: \033[4;34m{tb_url}\033[0m")
    
    import webbrowser
    # Kurze Verzögerung, damit das Training erst initialisiert
    webbrowser.open(tb_url)

    print("\n=== Beginne PPO-Training ===")
    t0 = time.perf_counter()
    try:
        model.learn(
            total_timesteps = TOTAL_TIMESTEPS,
            callback        = callbacks,
            tb_log_name     = f"PPO_Phase5b_v{cfg.version}",
        )
        elapsed = time.perf_counter() - t0
        print(f"\nTraining abgeschlossen in {elapsed/60:.1f} min")
        model.save(str(out_dir / "final_model"))
        print(f"Modell gespeichert: {out_dir / 'final_model.zip'}")
    except KeyboardInterrupt:
        print("\nTraining vom Nutzer abgebrochen. Speichere Zwischenstand...")
        model.save(str(out_dir / "interrupted_model"))
    finally:
        vec_env.close()


if __name__ == "__main__":
    # Wichtig fuer SubprocVecEnv auf Windows.
    import multiprocessing
    multiprocessing.freeze_support()
    train_phase5()

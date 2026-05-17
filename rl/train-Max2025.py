"""Phase 5c -- Neustart mit reduzierter Reward-Skala und stabileren PPO-Hypers.

Phase 5b hat nach ~5M Steps instabile Trainings-Metriken gezeigt:
  - train/std STEIGT statt zu fallen (Policy wird chaotisch)
  - train/value_loss explodiert
  - explained_variance sinkt
Ursache: Reward-Skala zu gross (w_coverage_delta=500, Returns bis ~550).
Die Value-Function konnte das MSE-Target nicht mehr tracken, Advantages
wurden Muell, Policy-Gradient instabil.

Aenderungen gegenueber Phase 5b:
  (1) Reward-Gewichte um Faktor 10 reduziert (config.py RewardConfig):
        w_coverage_delta 500 -> 50
        w_time_delta     0.5 -> 0.05
        w_completion     50  -> 5
        w_pierce         2   -> 0.2
        w_failure_penalty 50 -> 5
        infeasible_penalty -100 -> -10
      Neue Return-Range: ~-10 bis +55 (Value-Function kann das tracken).
  (2) max_steps_per_episode: 200 -> 400 (komplexere Trajektorien moeglich)
  (3) w_failure_penalty: 0 -> 5 (Signal bei Timeout ohne Erfolg)
  (4) PPO-Hypers stabiler:
        learning_rate 5e-4 -> 3e-4
        vf_coef       0.5  -> 1.0  (Value-Function staerker)
        clip_range_vf None -> 0.2  (Value-Clipping)
  (5) Beibehalten von Phase 5b:
        TCP-Init nahe Material, lineares Coverage-Potential, N_ENVS=20.

Hardware-Tuning (Ryzen 7 7800X3D / 32 GB RAM / RTX 5070 Ti)
-----------------------------------------------------------
Historie der Durchsatz-Fixes:

  Iteration 1 -- BLAS-Threads pro Worker auf 1 (oben im File, *vor*
  dem numpy-Import). Hat die frueheren 28 Worker * default-BLAS-
  Threads Oversubscription eliminiert, aber N_ENVS wurde gleichzeitig
  auf 16 reduziert. Messung: CPU ~10 %, GPU 10-35 %, RAM 60-70 %,
  steps/sec nur ~5 % besser, und -- kritisch -- Trainingsqualitaet hat
  gelitten (weniger parallele Rollout-Streams = korreliertere
  Gradienten). Diagnose: der eigentliche Bottleneck ist jetzt
  Python-seitige Pickle/Pipe-IPC ueber Named-Pipes, nicht mehr CPU.

  Iteration 2 -- Observation von float32 auf uint8 umgestellt (siehe
  mdp.render_observation / env.PlasmaCutterEnv.observation_space).
  Pro Step 49 kB statt 196 kB; bei 28 Worker * 512 Steps = 14336 Pickle-
  Transfers pro Rollout spart das ~10 GB pipe-/pickle-Traffic pro
  Rollout, entsprechend ~75 % weniger Master-Prozess-Idle. SB3 teilt
  auf der GPU durch 255 (`normalize_images=True` in CnnPolicy), die
  Lernqualitaet ist identisch zum float32-Pfad.

  Iteration 2 -- Hyperparameter auf die urspruengliche Konfiguration
  zurueckgesetzt, damit die Lernqualitaet wieder 1:1 zum Pre-Tuning
  Training ist:
    - N_ENVS         = 28   (maximale Rollout-Diversitaet)
    - N_STEPS_PER_ENV= 512  (14336 Rollout wie urspruenglich)
    - BATCH_SIZE     = 7168 (2 Mini-Batches pro Rollout)
    - N_EPOCHS       = 12
  Mit dem uint8-Obs reicht der RAM dafuer jetzt locker (Rollout-Buffer
  ist 4x kleiner: 840 MB statt 3.3 GB).

  - Observation 128x128, Feature-Dim 1024, MLP 512x512, ent_coef 0.05,
    LR 5e-4: alles unangetastet.
  - torch.set_float32_matmul_precision("high") nutzt die TF32-Tensor-
    Cores der Blackwell-5070Ti fuer MatMul-Ops (identisch zu
    `cudnn.allow_tf32=True`).
  - torch.set_num_threads(8) gibt dem Master-Prozess waehrend der PPO-
    Update-Phase seinen CPU-Thread-Pool zurueck (nur Master, Worker
    bleiben bei OMP=1).
"""

from __future__ import annotations

import os
import re
import sys
import time
import atexit
import zipfile
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# WICHTIG: BLAS-Thread-Limits MUESSEN vor `import numpy` / `import torch`
# gesetzt werden, sonst werden sie von den Libraries ignoriert. Diese
# Env-Vars werden an alle SubprocVecEnv-Worker (spawn) vererbt und
# verhindern die klassische Oversubscription unter Windows:
#   16 Worker * ~16 BLAS-Threads/Worker = 256 Threads auf 16 Kernen
#   -> massives Kontext-Switching, CPU haengt bei ~30%.
# Mit THREADS=1 pro Worker laeuft jeder Worker single-threaded auf genau
# einem Kern, der Scheduler kann sauber verteilen, CPU geht Richtung 100%.
# Der Master-Prozess bekommt seinen Thread-Pool spaeter per
# `torch.set_num_threads(...)` zurueck (siehe train_phase5()).
# ---------------------------------------------------------------------------
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

# pkg_resources-Deprecation-Warning aus TensorBoard 2.20 unterdruecken,
# bevor irgendein Import sie triggert. In PowerShell wird die Warnung
# sonst auf stderr gemeldet und als Fehler angezeigt, obwohl alles laeuft.
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::UserWarning:tensorboard.default",
)

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

# 20 Worker (reduziert von 28 fuer niedrigere RAM-/CPU-Last bei Stabilitaets-
# problemen). Rollout-Diversitaet bleibt gut, Rollout-Buffer ~600 MB.
N_ENVS          = 20
N_STEPS_PER_ENV = 512         # 20*512 = 10240 Steps pro Rollout
BATCH_SIZE      = 5120        # 2 Mini-Batches pro Rollout (10240 / 5120)
N_EPOCHS        = 10          # reduziert von 12 (weniger GPU-Last pro Update)
TOTAL_TIMESTEPS = 100_000_000
RESOLUTION      = 128
FEATURES_DIM    = 1024
MLP_HIDDEN      = 512
# Anzahl Threads, die der MASTER-Prozess fuer seine torch-CPU-Ops
# (Advantages, Logging, Minor-Ops) verwenden darf. Wird nur im Master
# aktiv -- Worker erben weiterhin OMP_NUM_THREADS=1 aus der os.environ.
MASTER_TORCH_THREADS = 4      # reduziert von 8 (entlastet CPU waehrend PPO-Update)

# (c) Curriculum-Stage 1: Zeitstrafe auf NULL.
# Wir wollen, dass der Agent erst einmal NUR lernt, die Kontur zu finden.
# Zeitdruck wird erst in Stage 2 (Phase 6) wieder eingeführt.
CURRICULUM_W_TIME = 0.0
# Mehr Exploration im Stage 1, damit der Agent ueberhaupt erst
# lernt, dass es lohnt, das Material zu beruehren.
ENT_COEF          = 0.05

# TensorBoard-Port. Fix belegt, damit der autogeoeffnete Browser immer
# auf denselben Tab zeigt.
TB_PORT = 6006


def _launch_tensorboard(logdir: Path, port: int) -> subprocess.Popen | None:
    """Startet tensorboard als Subprozess und registriert Cleanup.

    Vorher war in dieser Datei nur ein `webbrowser.open(...)` ohne Server
    -- das Ergebnis war eine tote URL. Wir starten das Binary aus dem
    aktiven venv (gleiche Python-Umgebung wie das Training), damit die
    Installation konsistent ist.
    """
    logdir.mkdir(parents=True, exist_ok=True)
    tb_exe = Path(sys.executable).with_name("tensorboard.exe")
    if not tb_exe.exists():
        tb_exe = Path(sys.executable).with_name("tensorboard")
    if not tb_exe.exists():
        print("WARNUNG: tensorboard-Binary nicht im venv gefunden "
              "-- ueberspringe Auto-Start.")
        return None

    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::UserWarning:tensorboard.default"
    try:
        proc = subprocess.Popen(
            [
                str(tb_exe),
                f"--logdir={logdir}",
                f"--port={port}",
                "--reload_multifile=true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except OSError as e:
        print(f"WARNUNG: tensorboard konnte nicht gestartet werden: {e}")
        return None

    atexit.register(lambda: proc.terminate() if proc.poll() is None else None)
    return proc


# ---------------------------------------------------------------------------
# Logging-Callback
# ---------------------------------------------------------------------------


def _find_latest_checkpoint(out_dir: Path) -> tuple[Path, int] | None:
    """Sucht den neuesten VALIDEN Checkpoint im Verzeichnis.

    Iteriert absteigend nach Step-Zahl und ueberspringt kaputte Zip-
    Dateien. Das kann passieren, wenn der PC waehrend eines
    `CheckpointCallback.save` abstuerzt -- dann steht die Datei zwar im
    Verzeichnis, ist aber unvollstaendig und `PPO.load` schmeisst
    `BadZipFile`. In so einem Fall nehmen wir einfach den naechstaelteren
    Checkpoint.
    """
    pattern = re.compile(r"^ppo_model_(\d+)_steps\.zip$")
    candidates: list[tuple[int, Path]] = []
    for p in out_dir.glob("ppo_model_*_steps.zip"):
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    # Absteigend nach Steps sortieren -- neuester zuerst.
    candidates.sort(key=lambda x: x[0], reverse=True)

    for steps, path in candidates:
        try:
            with zipfile.ZipFile(path) as _zf:
                # Wenn der Konstruktor die Central-Directory lesen kann,
                # ist der Zip strukturell intakt -- das ist exakt der
                # Check, den PPO.load als erstes macht.
                pass
            return path, steps
        except (zipfile.BadZipFile, OSError) as exc:
            print(
                f"WARNUNG: Checkpoint {path.name} beschaedigt "
                f"({type(exc).__name__}), ueberspringe."
            )

    return None


class InfoCallback(BaseCallback):
    """Sammelt Episoden-Infos pro Rollout und loggt Mittelwerte.

    `self.logger.record` ohne Aggregation würde bei mehreren Episoden pro
    Rollout nur den letzten Wert behalten. Wir puffern deshalb alle
    Episoden des aktuellen Rollouts und dumpen beim Rollout-Ende
    (`_on_rollout_end`) den Mittelwert, damit PPOs eigener Dump direkt
    danach die Scalars nach TensorBoard schreibt.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ep_coverage: list[float] = []
        self._ep_time:     list[float] = []
        self._ep_pierces:  list[int]   = []
        self._ep_feasible: list[float] = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue
            infos = self.locals.get("infos", [])
            if i >= len(infos):
                continue
            info = infos[i]
            self._ep_coverage.append(float(info.get("coverage", 0.0)))
            self._ep_time.append(float(info.get("total_time", 0.0)))
            self._ep_pierces.append(int(info.get("n_pierces", 0)))
            self._ep_feasible.append(float(info.get("is_feasible", True)))
        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_coverage:
            return
        self.logger.record("rollout/ep_coverage_mean", float(np.mean(self._ep_coverage)))
        self.logger.record("rollout/ep_coverage_max",  float(np.max(self._ep_coverage)))
        self.logger.record("rollout/ep_time_mean",     float(np.mean(self._ep_time)))
        self.logger.record("rollout/ep_pierces_mean",  float(np.mean(self._ep_pierces)))
        self.logger.record("rollout/feasible_rate",    float(np.mean(self._ep_feasible)))
        self.logger.record("rollout/episodes_in_rollout", len(self._ep_coverage))
        self._ep_coverage.clear()
        self._ep_time.clear()
        self._ep_pierces.clear()
        self._ep_feasible.clear()


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
    # Blackwell (sm_120) hat dedizierte TF32/BF16-Tensor-Cores. Mit
    # "high" nutzt torch diese automatisch fuer alle float32-MatMuls
    # (CNN-Conv, Linear-Layers) -- Speedup 1.3-1.8x ohne messbaren
    # Qualitaetsverlust.
    torch.set_float32_matmul_precision("high")

    # Master-Prozess darf mehrere Threads fuer torch-CPU-Ops nutzen
    # (PPO-Advantages, Logging). Worker-Prozesse erben OMP_NUM_THREADS=1
    # aus os.environ und sind davon unberuehrt.
    torch.set_num_threads(MASTER_TORCH_THREADS)

    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name} (compute capability {cap[0]}.{cap[1]})")
    print(f"CUDA: {torch.version.cuda} | torch {torch.__version__}")
    print(f"Master torch threads: {torch.get_num_threads()} | "
          f"Worker BLAS threads: {os.environ.get('OMP_NUM_THREADS', '?')}")

    # --- Config ----------------------------------------------------------
    cfg = default_config()
    cfg.scope.single_geometry_file = "RLKontur.json"
    cfg.scope.randomize_scale      = False
    cfg.scope.randomize_rotation   = False
    cfg.observation.resolution     = RESOLUTION
    # (c) Curriculum-Stage 1
    cfg.reward.w_time_delta        = CURRICULUM_W_TIME
    cfg.description = (
        f"Phase 5c (reduced reward scale x0.1): PPO/CnnPolicy auf "
        f"RLKontur.json, {RESOLUTION}x{RESOLUTION} Obs, {N_ENVS} Worker "
        f"(BLAS=1), rollout={N_ENVS*N_STEPS_PER_ENV}, batch={BATCH_SIZE}, "
        f"epochs={N_EPOCHS}, w_time_delta={CURRICULUM_W_TIME} "
        f"(Curriculum Stage 1), LR=3e-4, vf_coef=1.0, clip_vf=0.2, "
        f"max_steps=400, w_failure=5, TCP-Init nahe Material."
    )

    print(f"\nPhase 5c Training (single geometry: {cfg.scope.single_geometry_file})")
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

    tb_log = str((Path(__file__).resolve().parent / "tensorboard").as_posix())

    # --- Callbacks (benoetigt out_dir, daher vor PPO-Bau) ----------------
    out_dir = Path(__file__).resolve().parent / "checkpoints" / f"phase5c_v{cfg.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint-Verzeichnis: {out_dir}")

    checkpoint_cb = CheckpointCallback(
        save_freq      = max(1, 50_000 // N_ENVS),
        save_path      = str(out_dir),
        name_prefix    = "ppo_model",
    )
    info_cb = InfoCallback()
    callbacks = CallbackList([info_cb, checkpoint_cb])

    # --- PPO: neu oder aus Checkpoint fortsetzen -------------------------
    resume = _find_latest_checkpoint(out_dir)
    if resume is not None:
        ckpt_path, steps_done = resume
        print(f"\nCheckpoint gefunden -- setze fort ab {steps_done:,} Steps:")
        print(f"  {ckpt_path.name}")
        model = PPO.load(
            str(ckpt_path),
            env            = vec_env,
            device         = "cuda",
            # Ueberschreibe Parameter, die sich seit dem letzten Run
            # geaendert haben koennen (z.B. nach Lastreduzierung).
            n_steps        = N_STEPS_PER_ENV,
            batch_size     = BATCH_SIZE,
            n_epochs       = N_EPOCHS,
            ent_coef       = 0.05,        # zurueck auf Default -- 0.1 war zu hoch
            vf_coef        = 1.0,         # war 0.5 -- Value-Function staerker trainieren
            clip_range_vf  = 0.2,         # Value-Clipping fuer Stabilitaet
            learning_rate  = 3e-4,        # leicht senken fuer Stabilitaet beim Resume
            tensorboard_log= tb_log,
            verbose        = 1,
        )
        # Optimizer neu initialisieren: Der alte Adam-State kann mit den
        # neuen Hyperparametern (N_ENVS 28→20, BATCH_SIZE 7168→5120,
        # Reward-Skala reduziert um Faktor 10) inkompatibel sein und zu
        # instabilen Updates führen. Dadurch steigt train/std und
        # value_loss explodiert.
        model.policy.optimizer = torch.optim.Adam(
            model.policy.parameters(),
            lr=3e-4,
            eps=1e-5,  # SB3 Standard
        )
    else:
        print("\nKein Checkpoint -- starte neues Training.")
        steps_done = 0
        policy_kwargs = dict(
            features_extractor_class  = NatureCNN,
            features_extractor_kwargs = dict(features_dim=FEATURES_DIM),
            net_arch                  = dict(pi=[MLP_HIDDEN, MLP_HIDDEN],
                                             vf=[MLP_HIDDEN, MLP_HIDDEN]),
            activation_fn             = nn.ReLU,
            normalize_images          = True,    # uint8 obs -> /255 auf GPU
        )
        model = PPO(
            "CnnPolicy",
            vec_env,
            learning_rate   = 3e-4,   # Phase 5c: stabiler als 5e-4
            n_steps         = N_STEPS_PER_ENV,
            batch_size      = BATCH_SIZE,
            n_epochs        = N_EPOCHS,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            clip_range_vf   = 0.2,    # Phase 5c: Value-Clipping
            ent_coef        = ENT_COEF,
            vf_coef         = 1.0,    # Phase 5c: Value-Function staerker trainieren
            max_grad_norm   = 0.5,
            policy_kwargs   = policy_kwargs,
            tensorboard_log = tb_log,
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

    remaining = TOTAL_TIMESTEPS - steps_done
    if remaining <= 0:
        print(f"Training bereits abgeschlossen ({steps_done:,} >= {TOTAL_TIMESTEPS:,} Steps).")
        vec_env.close()
        return

    # --- TensorBoard -----------------------------------------------------
    tb_logdir = Path(__file__).resolve().parent / "tensorboard"
    tb_proc   = _launch_tensorboard(tb_logdir, TB_PORT)
    tb_url    = f"http://127.0.0.1:{TB_PORT}"
    print("\n=== TensorBoard Info ===")
    print(f"Logdir: {tb_logdir}")
    if tb_proc is not None:
        print(f"TensorBoard gestartet (PID {tb_proc.pid}): "
              f"\033[4;34m{tb_url}\033[0m")
        import webbrowser
        webbrowser.open(tb_url)
    else:
        print(f"Kein Auto-Start -- manuell: "
              f"tensorboard --logdir={tb_logdir} --port={TB_PORT}")

    if steps_done > 0:
        print(f"\n=== Setze PPO-Training fort ({steps_done:,} → {TOTAL_TIMESTEPS:,} Steps) ===")
    else:
        print("\n=== Beginne PPO-Training ===")
    t0 = time.perf_counter()
    try:
        model.learn(
            total_timesteps      = TOTAL_TIMESTEPS,
            callback             = callbacks,
            tb_log_name          = f"PPO_Phase5c_v{cfg.version}",
            reset_num_timesteps  = (steps_done == 0),  # False = Zaehler fortsetzen
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

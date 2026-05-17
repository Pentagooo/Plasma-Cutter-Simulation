"""Reinforcement-Learning-Pipeline fuer den Plasma-Cutter.

Dieses Package wird in Phasen aufgebaut (siehe Roadmap).

Status
------
  Phase 0  [DONE]  Ziel- und Erfolgsdefinition       --> config.py
  Phase 1  [DONE]  Heuristik-Baseline                --> baseline.py
  Phase 2  [DONE]  MDP-Design (State/Action/Reward)  --> mdp.py
  Phase 3  [DONE]  Gym-Environment (headless)        --> env.py
  Phase 4  [TODO]  Sanity-Checks                     --> sanity.py
  Phase 5  [TODO]  Erstes Training (single geometry) --> train.py
  Phase 6  [TODO]  Generalisierung                   --> train.py (variant)
  Phase 7  [TODO]  Optional: BC-Pretraining          --> bc.py
  Phase 8  [TODO]  Evaluation & Visualisierung       --> eval.py

Alle nachfolgenden Phasen lesen ihre Parameter aus `RLConfig`, damit
zentrale Designentscheidungen an genau EINER Stelle veraendert werden
koennen.


============================================================================
PLAN FUER DIE VERBLEIBENDEN PHASEN 4-8
============================================================================

Dieser Block ist eine DETAILLIERTE IMPLEMENTIERUNGSSKIZZE -- noch kein
Code. Er ist gedacht als Referenz beim Starten jeder Phase, damit die
Designentscheidungen nicht jedes Mal neu verhandelt werden muessen.
Aenderungen an diesem Plan bitte hier eintragen, bevor implementiert wird.

----------------------------------------------------------------------------
PHASE 4 -- Sanity-Checks (sanity.py)
----------------------------------------------------------------------------

Ziel
    Bevor ein einziger Trainingslauf startet, beweisen, dass die Env
    mechanisch fehlerfrei ist UND dass das Reward-Accounting stimmt.
    Diese Phase fangt ~70% aller spaeteren "Agent lernt nichts"-Bugs ab.

Wichtige Eigenschaft dieser Phase: sie braucht KEINE neuen Dependencies
(kein gymnasium, kein torch, kein SB3). Sie laeuft rein mit numpy +
der bereits vorhandenen Env.

Tests (jeder als eigene Funktion, via CLI `python -m plasma_cutter.rl.sanity`
einzeln oder komplett ausfuehrbar):

  1. `test_random_policy_smoke(cfg, n_episodes=100)`
       - 100 Episoden mit uniform-random Aktionen laufen lassen.
       - Aggregieren: return_mean, return_std, coverage_mean,
         termination_reason_histogram, avg_steps_per_episode.
       - Pruefungen:
           (a) Keine NaN/Inf in Obs, Action oder Reward.
           (b) Obs-Werte liegen in [0, cfg.observation.clip_max].
           (c) Return-Varianz > 0 (Agent sieht unterschiedliche Outcomes).
           (d) Kein "silent failure": mindestens 1 Episode mit cov > 0
               (sonst stimmt die Initialpose oder die Schrittweite nicht).
           (e) infeasible-Rate < 50% (sonst ist das Constraint zu hart).

  2. `test_baseline_replay(cfg)`
       - Fuer jede Trainings-Geometrie: die Baseline-Waypoints aus
         Phase 1 in Env-Aktionen konvertieren und durch env.step()
         replayen (via `options={"geometry_file": ...}` im reset, damit
         dieselbe Geometrie verwendet wird).
       - Kritische Pruefung: SUMME der step_rewards + terminal_reward
         muss dem Phase-1-Baseline-Reward aus results/baseline_v0.2.json
         entsprechen (Toleranz ~1e-3).
       - Wenn das abweicht, ist die Reward-Zerlegung in mdp.py falsch,
         NICHT die Baseline. Das ist der wichtigste Test der Phase.
       - Schwierigkeit: die Env erwartet normalisierte Delta-Aktionen,
         aber die Baseline liefert absolute Waypoints. Wir brauchen
         eine Konvertierung:
             action = (next_wp - current_wp) / max_step_mm
         und muessen u.U. Baseline-Schritte in mehrere kleine Env-Steps
         aufteilen, falls sie groesser als max_step_mm sind.

  3. `test_reward_decomposition(cfg)`
       - Laufe 10 Episoden (verschiedene Seeds). Fuer jede:
         summiere alle step_rewards + terminal_reward auf.
         Extrahiere aus info["coverage"] und info["total_time"] am
         Episodenende.
         Berechne das SOLL aus calculate_reward(final_result) wie in
         Phase 1.
         Differenz muss 0 sein (bis auf Floating-Point-Rauschen).
       - Schlaegt dieser Test fehl, ist die Potential-Shaping-Ableitung
         inkorrekt.

  4. `test_determinism(cfg)`
       - Zweimal dieselbe Env mit seed=42 starten, dieselbe
         Zufalls-Aktionssequenz durchschicken. Die komplette
         (obs, reward, done)-Sequenz muss bit-identisch sein.
       - Schlaegt das fehl, gibt es irgendwo einen unkontrollierten
         Zufallsgenerator (z.B. np.random statt self._rng).

  5. `test_env_speed(cfg, n_steps=5000)`
       - Reines Throughput-Benchmark: wieviele Env-Steps pro Sekunde?
       - Zielwert: >= 500 steps/s auf einer normalen CPU.
       - Falls drunter: Profiling per cProfile, wahrscheinlichste
         Engpaesse sind `plan_custom` (shapely-Unions) und
         `_rasterize_polygon`. Wenn noetig: die Bbox-Maske vorab
         berechnen und beim Rasterisieren per Region-of-Interest-Update
         arbeiten.

Output
    sanity.py loggt alle Test-Ergebnisse als JSON nach
    rl/results/sanity_v<version>.json und druckt eine PASS/FAIL-Uebersicht.
    Phase 5 darf nur starten wenn ALLE Tests passen.


----------------------------------------------------------------------------
PHASE 5 -- Erstes Training auf einer festen Geometrie (train.py)
----------------------------------------------------------------------------

Ziel
    Beweisen, dass das Problem ueberhaupt lernbar ist -- bewusst Overfitting
    auf EINE feste Kontur. Wenn der Agent das nicht schafft, hilft auch
    kein groesserer Run; dann ist das MDP/Reward kaputt und wir muessen
    zurueck zu Phase 2.

Abhaengigkeiten (pip install)
    gymnasium, stable-baselines3[extra], torch (CPU oder CUDA),
    tensorboard. Versionen in einer `requirements-rl.txt` fixieren,
    damit Trainings reproduzierbar sind.

Setup
    cfg = default_config()
    cfg.scope.single_geometry_file = "Test2.json"   # Baseline schafft 100%
    cfg.scope.randomize_scale      = False
    cfg.scope.randomize_rotation   = False

    env = PlasmaCutterEnv(cfg, mode="train")
    env = Monitor(env)                 # SB3 Episode-Logging
    env = DummyVecEnv([lambda: env])   # SB3 Vec-Wrapper

Algorithmus: PPO
    Gruende: kein Replay-Buffer noetig (Box-Obs mit 3x64x64 ist gerade
    noch handhabbar), robust gegen Reward-Skalierung, on-policy, gut
    dokumentiert. SAC ware eine Alternative, braucht aber mehr Tuning
    bei Bild-Inputs.

Policy
    CnnPolicy (SB3-Default fuer Bild-Obs).
    Anpassungen ueber `policy_kwargs`:
        features_extractor_class = NatureCNN   (3 Conv-Layer, 512-d feat)
        net_arch = [dict(pi=[64, 64], vf=[64, 64])]
        activation_fn = nn.ReLU

PPO-Hyperparameter (Startwerte)
    learning_rate   = 3e-4
    n_steps         = 2048     # Rollout-Laenge pro Env
    batch_size      = 64
    n_epochs        = 10
    gamma           = 0.99     # weit genug fuer ~100 Steps Horizont
    gae_lambda      = 0.95
    clip_range      = 0.2
    ent_coef        = 0.01     # bewusst etwas hoch fuer Exploration
    vf_coef         = 0.5
    max_grad_norm   = 0.5
    total_timesteps = 500_000  # skalieren falls 1 Geometrie zu einfach

Single-Step-Overfit-Test (VOR dem vollen Training)
    Ein Mini-Script, das eine feste Obs + Target-Action nimmt und die
    Policy bewusst overfittet (ohne RL, reiner SL-Loss). Wenn der Loss
    in 500 Iterationen nicht gegen 0 geht, ist die Policy-Architektur
    kaputt (z.B. Obs-Shape falsch, CNN erwartet andere Normierung).
    Dieser Test dauert Sekunden und fangt eine ganze Klasse von Bugs ab.

Logging
    Tensorboard via SB3-Callback. Log:
        - ep_rew_mean, ep_len_mean, ep_coverage_mean (via custom
          InfoCallback, liest info["coverage"] am Episodenende)
        - policy_loss, value_loss, entropy
        - infeasible_rate (Anteil der Episoden mit terminate_on_infeasible)
    Checkpoint alle 50k Steps in rl/checkpoints/phase5_v<version>/

Erfolgskriterium Phase 5
    - ep_coverage_mean > 0.99 auf der Trainings-Kontur innerhalb von
      500k Steps UND
    - ep_rew_mean > Baseline-Reward (aus baseline_v0.2.json, fuer Test2
      ca. 88) UND
    - infeasible_rate < 5%.
    Wenn erreicht: Weiter zu Phase 6.
    Wenn nicht erreicht: Zurueck zu Phase 2 und Reward/Action-Space
    diagnostizieren. NICHT einfach mehr Steps werfen.


----------------------------------------------------------------------------
PHASE 6 -- Generalisierung ueber Geometrien (train.py Variant)
----------------------------------------------------------------------------

Ziel
    Eine Policy trainieren, die ueber den kompletten Trainings-Pool
    generalisiert. Ab hier ist das Erfolgskriterium aus SuccessConfig
    relevant.

Setup-Aenderungen gegenueber Phase 5
    cfg.scope.single_geometry_file = None
    cfg.scope.randomize_scale      = True    # 0.8 - 1.2
    cfg.scope.randomize_rotation   = True    # -180 .. 180

    # VecEnv fuer parallele Rollouts
    n_envs = 8
    env = SubprocVecEnv([make_env(cfg, seed=i) for i in range(n_envs)])
    env = VecMonitor(env)

    # Groessere Netze helfen bei mehr Geometrien
    policy_kwargs = dict(
        features_extractor_class = NatureCNN,  # oder ResNet-lite
        features_extractor_kwargs = dict(features_dim=512),
        net_arch = [dict(pi=[128, 128], vf=[128, 128])],
    )
    total_timesteps = 5_000_000

Evaluation-Callback
    Alle 100k Steps: greedy Rollout auf dem TEST-Pool (mode="test"),
    per env.reset(options={"geometry_file": f}) jede Test-Geometrie
    fix besuchen. Speichere:
        - coverage pro Geometrie
        - total_time pro Geometrie
        - time_ratio vs Baseline (aus baseline_v<version>.json)
    Das beste Modell nach `success.min_success_rate` ablegen als
    best_model.zip.

Erfolgskriterium Phase 6 (aus cfg.success)
    Auf >= cfg.success.min_success_rate (95%) der Test-Geometrien:
        coverage >= cfg.success.coverage_target (0.99) UND
        total_time <= cfg.success.time_ratio_vs_baseline * baseline_time
                      (10% schneller als Baseline).

Falls nicht erreicht
    Reihenfolge der Interventionen:
      1. Trainingslauf verlaengern (x2 Schritte).
      2. ent_coef erhoehen (mehr Exploration).
      3. Observation-Resolution auf 96 oder 128 hochziehen.
      4. Action-Space pruefen: evtl. max_step_fraction senken
         (feinere Bewegungen).
      5. ERST DANN Phase 7 (BC-Pretraining) in Betracht ziehen.


----------------------------------------------------------------------------
PHASE 7 -- Optional: Behavior Cloning Pretraining (bc.py)
----------------------------------------------------------------------------

Nur wenn Phase 6 zu langsam konvergiert (< 20% Erfolgsrate nach 5M Steps).

Ziel
    Die Policy mit einem Supervised-Learning-Init warmstarten, damit RL
    ab einer halbwegs vernuenftigen Ausgangspolicy optimiert, nicht ab
    random.

Dataset-Erzeugung
    Fuer jede Trainings-Geometrie: die Baseline-Waypoints ueber die Env
    replayen und pro Step (obs, action_target) abspeichern. action_target
    entsteht durch die gleiche Konvertierung wie im Baseline-Replay-Test
    in Phase 4: delta_world / max_step_mm.
    Persistiert als .npz-Datei unter rl/results/bc_dataset_v<version>.npz.

BC-Training
    Reines Supervised Learning der Policy (MSE zwischen predicted und
    target action). ~50 Epochs auf dem Dataset. Speichert die
    Policy-Weights, die dann in Phase 6 als initial_policy an PPO
    uebergeben werden (SB3: policy.load_state_dict oder
    imitation-Package).

Erwarteter Effekt
    Trainingszeit von Phase 6 sollte um ~5-10x sinken. Wenn nicht,
    ist das BC-Dataset vermutlich zu klein (nur 4 Trainings-Geometrien)
    oder die Baseline-Trajektorien sind zu uniform.


----------------------------------------------------------------------------
PHASE 8 -- Evaluation & Visualisierung (eval.py)
----------------------------------------------------------------------------

Ziel
    Die trainierte Policy gegen die Baseline vergleichen, Reward-Hacking
    aufdecken, Erfolgskriterium formal pruefen.

Eingabe
    - best_model.zip aus Phase 6
    - baseline_v<version>.json aus Phase 1 (gleiche cfg.version!)

Schritte
    1. Config-Version-Check
         Geladenes Modell darf nur gegen Baseline mit gleicher version
         verglichen werden. Mismatch -> Abbruch mit klarer Fehlermeldung.
         Warum wichtig: Reward-Gewichtsaenderung macht Vergleiche
         unvergleichbar, aber leise statt laut.

    2. Deterministic-Rollout auf TEST-Pool
         `model.predict(obs, deterministic=True)` -- keine
         Action-Stochastik. Jede Test-Geometrie wird EINMAL mit fixem
         Seed besucht (reset(seed=..., options={"geometry_file": ...})).
         Pro Geometrie speichern:
             coverage, total_time, n_pierces, reward, segment_history
         Die segment_history ist wichtig, damit wir die Trajektorie
         spaeter visualisieren koennen.

    3. Metriken gegen Baseline
         Fuer jede Test-Geometrie:
             delta_coverage = rl.coverage - baseline.coverage
             time_ratio     = rl.total_time / baseline.total_time
             did_succeed    = (rl.coverage >= cfg.success.coverage_target
                               AND time_ratio <= cfg.success.time_ratio_vs_baseline)
         Gesamtscore:
             success_rate = mean(did_succeed)
             passed = success_rate >= cfg.success.min_success_rate

    4. Failure-Case-Collection
         Alle Episoden wo:
             (a) RL cov < baseline cov  (Regression)
             (b) RL cov >= 0.99 aber RL time > baseline time  (ineffizient)
             (c) RL hat is_feasible=False irgendwo bekommen (gelernt zu
                 schummeln aber erwischt)
         werden in eval_failures/ abgelegt (trajectory + reason).

    5. Visualisierung per env.render()
         Fuer jeden Failure-Case: die segment_history der trainierten
         Policy in ContinuousCutSimulation abspielen. Optisch pruefen
         auf:
             - unnoetige Verfahrbewegungen (Reward-Hacking Zeittrade-off)
             - staendige cut/rapid-Wechsel (Zuendungs-Abuse)
             - "kreiseln" an einer Stelle
             - Schnitte, die knapp am Constraint-Rand entlanggehen
         Diese visuelle Inspektion ist die EINZIGE zuverlaessige Art,
         Reward-Hacking zu finden. Tensorboard-Zahlen verraten es nicht.

    6. Report
         Erzeuge eval_v<version>.json mit:
             - passed: bool
             - success_rate: float
             - per-geometry Metriken
             - summary-Tabelle (mean/min/max)
             - config snapshot (cfg.to_dict())
         Optional: ein Markdown-Report `eval_v<version>.md` mit einer
         Vergleichstabelle RL vs Baseline fuer den User.

Erfolgskriterium Phase 8
    - passed == True  (= Erfolgskriterium aus cfg.success erfuellt)
    - Keine Reward-Hacking-Verdachtsfaelle in der visuellen Inspektion
    - infeasible_rate = 0 auf dem gesamten Test-Pool

Wenn passed == False
    Zurueck zu Phase 6 mit konkreten Hypothesen aus der Failure-Case-
    Collection. NICHT einfach weiter trainieren ohne das Root-Cause-
    Bild zu aktualisieren.


============================================================================
QUERSCHNITTSTHEMEN (gelten fuer alle Phasen)
============================================================================

Logging & Reproduzierbarkeit
    - Jeder Trainings-Run legt eine vollstaendige cfg.to_dict() in
      Tensorboard ab (als text-summary). Ohne das sind Laufvergleiche
      Glueckssache.
    - Seeds werden in der Ordnerstruktur sichtbar gemacht:
      `checkpoints/phase5_v0.2_seed42/`.
    - Jeder Lauf druckt cfg.version und cfg.description in den
      allerersten Log-Eintrag.

Versionierung der Reward-Funktion
    - Jede Reward-Aenderung hebt cfg.version ("0.2" -> "0.3") und
      aktualisiert cfg.description. Alte Trainings und alte Baseline-
      JSONs sind ab diesem Moment NICHT mehr vergleichbar -- neu zu
      generieren ist Pflicht, nicht Kuer.

Env-Geschwindigkeit
    - Wenn sanity.test_env_speed < 500 steps/s: Profiling und
      Optimierung VOR Phase 5. Ein 5M-Steps-Training mit 100 steps/s
      dauert 14 Stunden, mit 1000 steps/s 1.4 Stunden. Das lohnt sich.
    - Optimierungshebel: shapely-Operationen cachen, Rasterisierung
      auf Swept-Area-Bbox einschraenken, Coverage-Mask als uint8 statt
      bool (SIMD-freundlicher).

Seed-Sweeps
    - Jedes Erfolgskriterium muss gegen MEHRERE Seeds gepruft werden
      (mindestens 3). RL-Ergebnisse mit einem einzigen Seed sind
      statistisch nicht belastbar und haben in der RL-Community einen
      schlechten Ruf.
    - Standardseeds: 42, 1337, 2024.

Worin die Zeit investiert wird
    - Phase 4 (Sanity) sollte ~1 Tag kosten, Phase 5 ~2-3 Tage (inkl.
      Debugging), Phase 6 ~1 Woche (inkl. Seed-Sweeps), Phase 8 ~1 Tag.
    - Wenn eine Phase deutlich ueberzieht: zurueck in den Plan,
      nicht noch mehr Compute drauf werfen.
"""

from .config import (
    RLConfig,
    ScopeConfig,
    ObjectiveConfig,
    SuccessConfig,
    ConstraintsConfig,
    CutterConfig,
    ActionConfig,
    ObservationConfig,
    RewardConfig,
    default_config,
)
from .env import PlasmaCutterEnv

__all__ = [
    "RLConfig",
    "ScopeConfig",
    "ObjectiveConfig",
    "SuccessConfig",
    "ConstraintsConfig",
    "CutterConfig",
    "ActionConfig",
    "ObservationConfig",
    "RewardConfig",
    "default_config",
    "PlasmaCutterEnv",
]

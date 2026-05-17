"""Phase 3 -- Gymnasium-Environment fuer den Plasma-Cutter.

`PlasmaCutterEnv` ist ein duenner Wrapper um

  - `ContinuousPlanner.plan_custom()`  (aus cutter/continuous_path.py)
  - `PointGrid`                        (aus geometry/point_grid.py)
  - die MDP-Primitiven aus `rl/mdp.py`

und exportiert sie als Gymnasium-kompatible Environment fuer
Stable-Baselines3 / CleanRL / RLlib.

Verantwortlichkeiten
--------------------
Diese Datei macht AUSSCHLIESSLICH das Gym-Wrapping. Der gesamte
"interessante" Code (State/Action/Reward) lebt in `mdp.py`, der
gesamte Physik-Code in `cutter/` / `geometry/`. Wenn etwas am Reward
oder an der Observation geaendert werden soll: `config.py` + `mdp.py`.
Wenn etwas an der Gym-API geaendert werden soll: DIESE Datei.

Headless
--------
Die Env hat KEINE matplotlib-Abhaengigkeit im Hot-Path. `reset()` /
`step()` sind rein numerisch. Visualisierung gibt es nur via
`render()`, was die bestehende `ContinuousCutSimulation` benutzt und
explizit aufgerufen werden muss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from copy import deepcopy

import numpy as np

from plasma_cutter.geometry.point_grid import PointGrid
from plasma_cutter.cutter.cutter import Cutter
from plasma_cutter.cutter.continuous_path import ContinuousPlanner, ContinuousPathResult

from plasma_cutter.rl.config import RLConfig, default_config
from plasma_cutter.rl.mdp import (
    ObservationContext,
    action_space_shape,
    action_space_bounds,
    observation_space_shape,
    compute_max_step_mm,
    decode_action,
    build_observation_context,
    render_observation,
    compute_step_reward,
    compute_terminal_reward,
    _rasterize_polygon,
)
from plasma_cutter.rl.baseline import list_geometries, split_train_test  # reuse split logic


# ---------------------------------------------------------------------------
# gymnasium-Compat-Shim
# ---------------------------------------------------------------------------
#
# gymnasium ist DIE Standard-API fuer moderne RL-Bibliotheken (SB3,
# CleanRL). Die Env ist so gebaut, dass sie die gymnasium-API implementiert.
#
# Falls gymnasium nicht installiert ist (Phase 3 soll auch offline
# testbar sein), stellen wir einen minimalen Shim bereit, der genug API
# liefert, um die Env smoke-testen zu koennen. Sobald gymnasium
# installiert wird (`pip install gymnasium`), schaltet der Shim
# automatisch aus -- KEINE Code-Aenderung noetig.
# ---------------------------------------------------------------------------

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

    class _ShimBox:
        """Minimaler Stand-in fuer `gymnasium.spaces.Box`.

        Implementiert nur die Attribute, die von unserer Env selbst
        und von einfachem Smoke-Testing gebraucht werden. Beim Upgrade
        auf gymnasium verschwindet dieses Ding komplett.
        """
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low   = np.asarray(low, dtype=dtype)
            self.high  = np.asarray(high, dtype=dtype)
            self.shape = tuple(shape) if shape is not None else self.low.shape
            self.dtype = dtype

        def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
            r = rng if rng is not None else np.random.default_rng()
            return r.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x: np.ndarray) -> bool:
            x = np.asarray(x)
            return (
                x.shape == self.shape
                and bool(np.all(x >= self.low))
                and bool(np.all(x <= self.high))
            )

    class _ShimSpaces:
        Box = _ShimBox

    class _ShimEnv:
        """Duck-type-Basis fuer gymnasium.Env."""
        action_space = None
        observation_space = None
        metadata = {"render_modes": []}
        def reset(self, *args, **kwargs): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        def render(self):       return None
        def close(self):        pass

    class _ShimGym:
        Env = _ShimEnv

    gym = _ShimGym()       # type: ignore
    spaces = _ShimSpaces() # type: ignore


# ---------------------------------------------------------------------------
# Hilfsfunktionen: Geometrie-Transformation, initialer TCP, Cutter-Bau
# ---------------------------------------------------------------------------


class GeometryCache:
    """Per-Prozess-Cache für Geometrie-Grunddaten.
    Verhindert das redundante Einlesen und Parsen von JSON-Dateien in reset().
    """
    _cache: dict[Path, dict] = {}

    @classmethod
    def get_grid(cls, path: Path) -> PointGrid:
        if path not in cls._cache:
            # Einmalig laden und Grunddaten extrahieren
            grid = PointGrid.from_json(path)
            cls._cache[path] = {
                "coords": grid._coords.copy(),
                "status": grid._status.copy(),
                "point_spacing": grid.point_spacing,
                "contour_spacing": grid.contour_spacing,
            }
        
        data = cls._cache[path]
        # Neue Instanz mit Kopien der Arrays erzeugen, damit Transformationen
        # die Cache-Daten nicht korrumpieren.
        return PointGrid(
            coords=data["coords"].copy(),
            status=data["status"].copy(),
            point_spacing=data["point_spacing"],
            contour_spacing=data["contour_spacing"]
        )


def _build_cutter_from_config(cfg: RLConfig) -> Cutter:
    """Erzeugt einen Cutter mit den Werten aus `CutterConfig`.

    Wird pro Episode neu gebaut. Teuer ist das nicht (triviales Objekt),
    aber wir bleiben so konsistent zu Baseline und Sim -- falls jemand
    `CutterConfig` zur Laufzeit aendert, greifen die neuen Werte sofort.
    """
    return Cutter(
        max_depth     = cfg.cutter.max_depth,
        cutting_speed = cfg.cutter.cutting_speed,
        moving_speed  = cfg.cutter.moving_speed,
        rapid_speed   = cfg.cutter.rapid_speed,
        minimum_gap   = cfg.cutter.minimum_gap,
    )
def _apply_random_transform(
    grid: PointGrid,
    rng:  np.random.Generator,
    cfg:  RLConfig,
) -> dict:
    """Wendet Rotation + gleichmaessige Skalierung auf alle Gitterpunkte an.
    Vektorisiert über PointGrid.transform.
    """
    angle_deg = 0.0
    scale     = 1.0

    if cfg.scope.randomize_rotation:
        lo, hi = cfg.scope.rotation_range_deg
        angle_deg = float(rng.uniform(lo, hi))

    if cfg.scope.randomize_scale:
        lo, hi = cfg.scope.scale_range
        scale = float(rng.uniform(lo, hi))

    if angle_deg == 0.0 and scale == 1.0:
        return {"angle_deg": 0.0, "scale": 1.0}

    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Rotationsmatrix R (2, 2)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    
    grid.transform(R, scale)

    return {"angle_deg": angle_deg, "scale": scale}


def _sample_initial_tcp(
    ctx:  ObservationContext,
    grid: PointGrid,
    cfg:  RLConfig,
    rng:  np.random.Generator,
) -> np.ndarray:
    """Waehlt eine initiale TCP-Position DICHT am Material (Phase 5b).
    Vektorisiert über grid.coords.
    """
    pts = grid.coords
    if pts.size == 0:
        cx = 0.5 * (ctx.bbox_min[0] + ctx.bbox_max[0])
        cy = 0.5 * (ctx.bbox_min[1] + ctx.bbox_max[1])
        return np.array([cx, cy], dtype=np.float32)

    gap = float(cfg.cutter.minimum_gap)
    r_min = gap + 0.5
    r_max = gap + 5.0

    for _ in range(20):
        idx    = int(rng.integers(len(pts)))
        anchor = pts[idx]
        theta  = float(rng.uniform(0.0, 2 * np.pi))
        radius = float(rng.uniform(r_min, r_max))
        cand   = anchor + radius * np.array([np.cos(theta), np.sin(theta)])

        # Globale Clearance: Vektorisiert
        diffs = pts - cand
        min_dist = float(np.sqrt(np.sum(diffs**2, axis=1)).min())
        if min_dist >= gap:
            return cand.astype(np.float32)

    # Fallback: alter Bbox-Rand-Start.
    cx = 0.5 * (ctx.bbox_min[0] + ctx.bbox_max[0])
    cy = 0.5 * (ctx.bbox_min[1] + ctx.bbox_max[1])
    half = 0.5 * (ctx.bbox_max[0] - ctx.bbox_min[0])
    radius = 0.95 * half
    theta = float(rng.uniform(0.0, 2 * np.pi))
    return np.array(
        [cx + radius * np.cos(theta), cy + radius * np.sin(theta)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# PlasmaCutterEnv
# ---------------------------------------------------------------------------


class PlasmaCutterEnv(gym.Env):
    """Gym-Env fuer kontinuierliches Plasma-Schneiden.

    Lebenszyklus
    ------------
    __init__: liest `RLConfig`, baut Geometrie-Pool (train/test),
              definiert action_space und observation_space.
    reset():  sampled eine Geometrie, wendet Randomisierung an, baut
              Observation-Context + Planner, setzt initialen TCP,
              gibt erstes Observation + Info zurueck.
    step():   decodiert Action -> (next_tcp, is_cutting), fuehrt
              das Segment ueber `plan_custom` aus, aktualisiert
              Coverage-Maske + Reward, prueft Termination.
    render(): optional, nutzt die bestehende
              `ContinuousCutSimulation` fuer die Visualisierung.

    Threadsicherheit: die Env ist NICHT thread-safe. Fuer parallele
    Rollouts (SB3 `SubprocVecEnv`) wird je Worker eine eigene Instanz
    erzeugt -- das ist der uebliche SB3-Weg.
    """

    metadata = {"render_modes": ["matplotlib"], "render_fps": 30}

    def __init__(
        self,
        cfg:  RLConfig | None = None,
        mode: str             = "train",
    ) -> None:
        super().__init__()
        self.cfg  = cfg or default_config()
        self.mode = mode

        # Geometrie-Pool aufbauen. Nutzt die Train/Test-Split-Logik aus
        # der Baseline, damit ein bei Phase 1 gesehener Train-Pool
        # identisch hier wieder auftaucht.
        all_files = list_geometries(self.cfg)
        train_files, test_files = split_train_test(all_files, self.cfg)

        if self.cfg.scope.single_geometry_file:
            # Override: Sanity-Mode "immer dieselbe Kontur"
            override = self.cfg.geometry_dir_path() / self.cfg.scope.single_geometry_file
            if not override.exists():
                raise FileNotFoundError(
                    f"single_geometry_file nicht gefunden: {override}"
                )
            self._pool = [override]
        elif mode == "train":
            self._pool = train_files
        elif mode == "test":
            self._pool = test_files
        elif mode == "all":
            self._pool = all_files
        else:
            raise ValueError(
                f"Unbekannter mode='{mode}'. Erlaubt: 'train', 'test', 'all'."
            )

        if not self._pool:
            raise RuntimeError(
                f"Geometrie-Pool fuer mode='{mode}' ist leer. "
                f"Check cfg.scope.test_split und das Geometrie-Verzeichnis."
            )

        # --- Spaces ---
        low, high = action_space_bounds(self.cfg)
        self.action_space = spaces.Box(
            low   = low,
            high  = high,
            shape = action_space_shape(self.cfg),
            dtype = np.float32,
        )

        obs_shape = observation_space_shape(self.cfg)
        # uint8 statt float32: 4x kleinerer Pickle-/Pipe-Footprint pro
        # Step. SB3 CnnPolicy mit `normalize_images=True` teilt auf der
        # GPU durch 255 -- die Lernqualitaet bleibt 1:1 zum float32-Pfad.
        self.observation_space = spaces.Box(
            low   = 0,
            high  = 255,
            shape = obs_shape,
            dtype = np.uint8,
        )

        # --- Episode-State (wird in reset() gefuellt) ---
        self._rng:          np.random.Generator | None = None
        self._grid:         PointGrid           | None = None
        self._planner:      ContinuousPlanner   | None = None
        self._ctx:          ObservationContext  | None = None
        self._tcp:          np.ndarray          | None = None
        self._coverage_mask: np.ndarray         | None = None
        self._prev_coverage: float              = 0.0
        self._cum_time:      float              = 0.0
        self._n_pierces:     int                = 0
        self._last_is_cutting: bool             = False
        self._step_count:    int                = 0
        self._max_step_mm:   float              = 0.0
        self._current_geometry_file: Path       | None = None
        # Pro-Episode-Historie der Segmente (fuer render()). Jeder Eintrag
        # ist ein Tupel (waypoints, is_cutting). Wird von step() befuellt.
        self._segment_history: list[tuple[list[np.ndarray], bool]] = []

    # ---------------------------------------------------------------- reset

    def reset(
        self,
        *,
        seed:    int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Gym-API: startet eine neue Episode.

        Parameters
        ----------
        seed    : optionaler Seed. Wird der Gym-API gemaess nur beim
                  ERSTEN reset oder bei explizitem reseed angewendet.
        options : optional. Unterstuetzt derzeit:
                  {"geometry_file": <path or name>}
                  erzwingt eine bestimmte Geometrie (nuetzlich fuer
                  reproduzierbare Evaluation).
        """
        # gymnasium.Env.reset() ruft seed-setup auf. Wir halten uns an
        # die Konvention auch ohne gymnasium, damit der Shim-Pfad gleich
        # funktioniert.
        if _GYM_AVAILABLE:
            super().reset(seed=seed)

        # RNG initialisieren. Beim ersten reset aus `cfg.seed`, danach
        # deterministisch fortlaufend (jede Episode zieht aus demselben
        # RNG) -- das macht ein paralleles VecEnv-Training reproduzierbar.
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng(self.cfg.seed)

        # --- Geometrie auswaehlen ---
        if options and "geometry_file" in options:
            requested = options["geometry_file"]
            geom_path = Path(requested)
            if not geom_path.is_absolute():
                geom_path = self.cfg.geometry_dir_path() / geom_path
            if not geom_path.exists():
                raise FileNotFoundError(f"Geometrie nicht gefunden: {geom_path}")
        else:
            idx = int(self._rng.integers(len(self._pool)))
            geom_path = self._pool[idx]

        self._current_geometry_file = geom_path

        # --- Grid laden + Randomisierung ---
        # Nutzt den Cache, um redundantes Parsen pro Reset zu vermeiden.
        self._grid = GeometryCache.get_grid(geom_path)
        self._grid.reset()
        transform_info = _apply_random_transform(self._grid, self._rng, self.cfg)

        # --- Observation-Kontext + Planner bauen ---
        self._ctx = build_observation_context(self._grid, self.cfg)
        self._planner = ContinuousPlanner(
            grid       = self._grid,
            cutter     = _build_cutter_from_config(self.cfg),
            kerf_width = self.cfg.cutter.kerf_width,
        )

        # --- Episode-Zustand ruecksetzen ---
        self._coverage_mask    = np.zeros(
            (self._ctx.resolution, self._ctx.resolution), dtype=bool
        )
        self._tcp              = _sample_initial_tcp(
            self._ctx, self._grid, self.cfg, self._rng,
        )
        self._prev_coverage    = 0.0
        self._cum_time         = 0.0
        self._n_pierces        = 0
        # Vor dem ersten Step ist das Plasma aus -- der erste cut-Step
        # zaehlt damit korrekt als erste Zuendung.
        self._last_is_cutting  = False
        self._step_count       = 0
        self._max_step_mm      = compute_max_step_mm(
            self._ctx.bbox_diagonal, self.cfg
        )
        self._segment_history.clear()

        obs  = render_observation(
            self._ctx, self._coverage_mask, self._tcp, self.cfg
        )
        info = self._build_info(
            extra = {
                "transform":     transform_info,
                "geometry_file": str(geom_path.name),
                "max_step_mm":   self._max_step_mm,
                "total_points":  self._grid.total_points,
            }
        )
        return obs, info

    # ---------------------------------------------------------------- step

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Gym-API: fuehrt einen Step aus und liefert (obs, reward, terminated, truncated, info)."""
        if self._ctx is None or self._planner is None or self._grid is None:
            raise RuntimeError("step() vor reset() aufgerufen.")

        # --- Action decodieren ---
        next_tcp, is_cutting = decode_action(
            action      = action,
            current_tcp = self._tcp,
            max_step_mm = self._max_step_mm,
            cfg         = self.cfg,
        )

        # --- Segment ausfuehren ---
        # Minimaler Schritt-Check: wenn der Agent ~0 Bewegung schickt,
        # spart der Planner sich alles und gibt ein empty result zurueck.
        # Wir zaehlen den Step trotzdem -- sonst kann der Agent Reward
        # hacken, indem er auf der Stelle steht.
        segment_waypoints = [
            np.asarray(self._tcp,     dtype=np.float64).copy(),
            np.asarray(next_tcp,      dtype=np.float64).copy(),
        ]
        result: ContinuousPathResult = self._planner.plan_custom(
            segments = [(segment_waypoints, is_cutting)],
            apply    = True,
        )

        # --- Step-Statistiken extrahieren ---
        step_time = float(result.total_time)

        # Feasibility: alle erzeugten Cuts des Segments muessen ok sein.
        # Bei empty result (z.B. Segment zu kurz) gilt feasible = True.
        step_feasible = (
            all(c.is_feasible for c in result.continuous_cuts)
            if result.continuous_cuts
            else True
        )

        # Coverage Ground-Truth kommt aus dem Grid, NICHT aus dem Bild --
        # die rasterisierte Maske ist nur Observation, das Grid ist die
        # Wahrheit. n_cut / total_points gibt die aktuelle Abdeckung.
        curr_coverage = self._grid.n_cut / max(1, self._grid.total_points)

        # --- Coverage-Bildmaske updaten ---
        # Wir rasterisieren die neu geschnittenen Swept-Polygone und
        # ODERn sie in die kumulative Maske. Nur wenn wirklich geschnitten
        # wurde UND der Step feasible war -- sonst koennten wir Schnitte
        # zeichnen, die das Grid gar nicht applied hat.
        if is_cutting and step_feasible and result.continuous_cuts:
            for cc in result.continuous_cuts:
                sp = cc.swept_polygon
                if sp is None or sp.is_empty:
                    continue
                new_mask = _rasterize_polygon(sp, self._ctx)
                np.logical_or(self._coverage_mask, new_mask, out=self._coverage_mask)

        # --- Pierce-Counting ---
        # Transition rapid->cut = neue Zuendung. Der allererste cut-Step
        # einer Episode zaehlt, weil _last_is_cutting im reset auf False
        # steht.
        if is_cutting and not self._last_is_cutting:
            self._n_pierces += 1
        self._last_is_cutting = is_cutting

        # --- TCP aktualisieren (nur bei feasible Schritten) ---
        # Bei infeasible Schritten bleibt der TCP, wo er war -- die
        # Episode endet ohnehin gleich.
        if step_feasible:
            self._tcp = np.asarray(next_tcp, dtype=np.float32)
        self._segment_history.append((segment_waypoints, is_cutting))

        # --- Rewards ---
        step_r = compute_step_reward(
            prev_coverage = self._prev_coverage,
            curr_coverage = curr_coverage,
            step_time_s   = step_time,
            cfg           = self.cfg,
        )
        self._prev_coverage = curr_coverage
        self._cum_time     += step_time
        self._step_count   += 1

        # --- Termination ---
        terminated = False
        truncated  = False

        # Prioritaet 1: Infeasible -> harte Termination mit grosser Strafe.
        if not step_feasible and self.cfg.constraints.terminate_on_infeasible:
            terminated = True
            # Der step-Reward wird beibehalten, damit die Zeitstrafe gilt und die
            # mathematische Zerlegung (Potential Shaping) exakt aufgeht. 
            # Coverage-Zuwachs gibt es ohnehin nicht, da infeasible Schnitte 
            # vom Planner nicht angewendet werden.
            term_r     = compute_terminal_reward(
                final_coverage = curr_coverage,
                n_pierces      = self._n_pierces,
                is_feasible    = False,
                timed_out      = False,
                cfg            = self.cfg,
            )
        # Prioritaet 2: Coverage-Ziel erreicht -> Erfolg.
        elif curr_coverage >= self.cfg.objective.coverage_threshold:
            terminated = True
            term_r     = compute_terminal_reward(
                final_coverage = curr_coverage,
                n_pierces      = self._n_pierces,
                is_feasible    = True,
                timed_out      = False,
                cfg            = self.cfg,
            )
        # Prioritaet 3: Step-Budget aufgebraucht -> Timeout.
        elif self._step_count >= self.cfg.constraints.max_steps_per_episode:
            truncated  = True
            term_r     = compute_terminal_reward(
                final_coverage = curr_coverage,
                n_pierces      = self._n_pierces,
                is_feasible    = True,
                timed_out      = True,
                cfg            = self.cfg,
            )
        # Prioritaet 4: Zeitbudget aufgebraucht -> Timeout.
        elif (
            self.cfg.constraints.max_cutting_time is not None
            and self._cum_time >= self.cfg.constraints.max_cutting_time
        ):
            truncated  = True
            term_r     = compute_terminal_reward(
                final_coverage = curr_coverage,
                n_pierces      = self._n_pierces,
                is_feasible    = True,
                timed_out      = True,
                cfg            = self.cfg,
            )
        else:
            term_r = 0.0

        reward = float(step_r + term_r)

        # --- Observation + Info ---
        obs  = render_observation(
            self._ctx, self._coverage_mask, self._tcp, self.cfg
        )
        info = self._build_info(
            extra = {
                "step_reward":     float(step_r),
                "terminal_reward": float(term_r),
                "is_feasible":     bool(step_feasible),
                "is_cutting":      bool(is_cutting),
                "next_tcp":        np.asarray(next_tcp, dtype=np.float32).tolist(),
            }
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ info/utils

    def _build_info(self, extra: dict | None = None) -> dict:
        """Baut das info-dict. Konvention: keys, die sich ueber die
        Episode aufsummieren (coverage, total_time) sind IMMER drin,
        episodische Flags kommen als `extra` rein."""
        info: dict[str, Any] = {
            "coverage":   float(self._prev_coverage),
            "total_time": float(self._cum_time),
            "n_pierces":  int(self._n_pierces),
            "step_count": int(self._step_count),
            "tcp":        (
                np.asarray(self._tcp, dtype=np.float32).tolist()
                if self._tcp is not None else None
            ),
        }
        if extra:
            info.update(extra)
        return info

    # ---------------------------------------------------------------- render

    def render(self):
        """Optionale Visualisierung ueber `ContinuousCutSimulation`.

        Nicht fuer den Hot-Path -- nur zum Debuggen einzelner Episoden
        oder fuer End-of-Training-Evaluation. Fuehrt die bisher
        aufgezeichneten Segmente in der interaktiven Sim aus.
        """
        # Import lokal, um die matplotlib-Abhaengigkeit im Hot-Path zu
        # vermeiden (sonst laedt jeder Training-Worker matplotlib).
        from ..simulation.continuous_cut_simulation import ContinuousCutSimulation

        sim = ContinuousCutSimulation(
            grid         = deepcopy(self._grid),
            cutter       = _build_cutter_from_config(self.cfg),
            kerf_width   = self.cfg.cutter.kerf_width,
        )
        # Wir nutzen den low-level Planner-Aufruf von sim._planner,
        # um die aufgezeichnete History abzuspielen.
        result = sim._planner.plan_custom(self._segment_history, apply=True)
        sim._current_result = result
        sim._results.append(result)
        sim.run()  # blocking -- oeffnet das Fenster

    def close(self):
        """Ressourcen freigeben. Aktuell nichts zu tun -- wir halten
        keine offenen File-Handles oder GPU-Buffer."""
        pass

    # ------------------------------------------------------------ debug API

    @property
    def current_geometry_file(self) -> Optional[Path]:
        """Aktuell geladene Geometrie. None, wenn noch kein reset() lief."""
        return self._current_geometry_file

    @property
    def segment_history(self) -> list[tuple[list[np.ndarray], bool]]:
        """Alle Segmente der aktuellen Episode (fuer Evaluation/Debug)."""
        return list(self._segment_history)

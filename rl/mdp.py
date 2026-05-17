"""Phase 2 -- MDP-Design (Action / Observation / Reward).

Diese Datei enthaelt die KOMPLETTE MDP-Spezifikation in reiner
Funktionsform -- ohne Gym-Env-Wrapper, ohne Trainingscode. Der Zweck
ist Trennung der Verantwortlichkeiten:

    Phase 2 (HIER): *was* ist State, Action, Reward?
    Phase 3 (env.py): *wie* wird das an Gymnasium angebunden?
    Phase 5 (train.py): *welcher* Algorithmus trainiert auf der Env?

Die Funktionen hier sind ALLE pur (keine Klassen-State, keine
Globals). Das macht sie trivial testbar und wiederverwendbar in
Debug-Notebooks, Unit-Tests oder alternativen Env-Implementierungen.

Gliederung
----------
    1. Action-Space     -- Decoder von Agent-Ausgabe nach Weltkoordinaten
    2. Observation      -- Pro-Episode-Kontext + Multi-Channel-Bild-Rendering
    3. Reward           -- Per-Step- und Terminal-Reward-Funktionen

Alle drei Bloecke lesen ihre Parameter aus `RLConfig`. Wenn ein
Hyperparameter geaendert werden soll, wird die Config angepasst --
nicht dieser Code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.vectorized import contains as shp_contains

from ..geometry.point_grid import PointGrid
from .config import RLConfig


# ===========================================================================
# 1. ACTION-SPACE
# ===========================================================================
#
# Der Agent gibt pro Step einen Vector [-1, +1]^n aus. Die Dimension
# haengt davon ab, ob cut/rapid als zusaetzliche Dimension enthalten ist:
#
#   n = 2   -> (dx, dy)             -- immer schneiden
#   n = 3   -> (dx, dy, cut_flag)   -- Agent entscheidet cut/rapid
#
# Die Decodierung konvertiert (dx, dy) in eine Weltkoordinaten-Position
# in mm. Dabei wird die Schrittweite PRO EPISODE anhand der Geometrie-
# Bounding-Box bestimmt, damit die Policy skalierungs-invariant ist.
# ===========================================================================


def action_space_shape(cfg: RLConfig) -> tuple[int, ...]:
    """Gibt die Shape des Action-Vectors zurueck (fuer `gym.spaces.Box`)."""
    n = 2 + (1 if cfg.action.include_cut_flag else 0)
    return (n,)


def action_space_bounds(cfg: RLConfig) -> tuple[np.ndarray, np.ndarray]:
    """Liefert low/high-Arrays fuer `gym.spaces.Box`.

    Der gesamte Action-Vector liegt per Konstruktion in [-1, +1]. Das
    ist die Standardkonvention fuer SAC/PPO mit tanh-Output.
    """
    n = action_space_shape(cfg)[0]
    return -np.ones(n, dtype=np.float32), np.ones(n, dtype=np.float32)


def compute_max_step_mm(
    bbox_diagonal_mm: float,
    cfg: RLConfig,
) -> float:
    """Berechnet die maximale Schrittweite in mm fuer die aktuelle Geometrie.

    Skalierungs-invariante Formel:
        raw = bbox_diagonal * max_step_fraction
        step = clamp(raw, min_step_absolute_mm, max_step_absolute_mm)

    Die Min/Max-Clamps verhindern zwei Degenerations-Faelle:
      - Bei sehr kleinen Geometrien wuerde `raw` in Sub-mm-Bereich
        liegen und der Agent braeuchte Tausende Steps.
      - Bei extrem grossen Geometrien koennte ein einzelner Schritt
        die gesamte Kontur ueberspringen und der Planner wuerde
        nur ein grobes Segment erzeugen.
    """
    raw = bbox_diagonal_mm * cfg.action.max_step_fraction
    lo  = cfg.action.min_step_absolute_mm
    hi  = cfg.action.max_step_absolute_mm
    step = max(raw, lo)
    if hi is not None:
        step = min(step, hi)
    return float(step)


def decode_action(
    action: np.ndarray,
    current_tcp: np.ndarray,
    max_step_mm: float,
    cfg: RLConfig,
) -> tuple[np.ndarray, bool]:
    """Konvertiert den normalisierten Agent-Vector in eine konkrete
    (next_tcp, is_cutting)-Entscheidung.

    Parameters
    ----------
    action      : shape (2,) oder (3,) -- Agent-Ausgabe in [-1, +1]
    current_tcp : shape (2,) -- aktuelle TCP-Position in Weltkoordinaten (mm)
    max_step_mm : maximale Schrittweite fuer diese Episode (s. `compute_max_step_mm`)
    cfg         : RLConfig

    Returns
    -------
    next_tcp    : shape (2,) -- Ziel-Waypoint in Weltkoordinaten (mm)
    is_cutting  : True wenn das Segment im cut-Modus ausgefuehrt wird

    Hinweis: Diese Funktion validiert NICHT, ob der Zielpunkt
    clearance-konform ist. Die Clearance-Pruefung passiert weiter unten
    im `ContinuousPlanner._validate_grip_path`. Hier nur rein
    geometrische Decodierung.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    # Sicherheits-Clip: sollte durch die Policy-tanh eigentlich
    # garantiert sein, aber wir schuetzen uns vor NaNs/Outliers.
    action = np.clip(action, -1.0, 1.0)

    dx_norm = float(action[0])
    dy_norm = float(action[1])

    # Skalierung in mm. Die Action liegt in der Einheitskreis-Ecke
    # [-1,1]^2 -- die maximale Schrittweite wird also in x- und y-
    # Richtung GETRENNT gemessen. Das ist gewollt, weil sich die
    # resultierende 2D-Box gut durch die tanh-Ausgabe abbilden laesst.
    delta = np.array([dx_norm, dy_norm], dtype=np.float32) * max_step_mm

    next_tcp = np.asarray(current_tcp, dtype=np.float32) + delta

    # Cut-Flag decodieren (falls der Agent darueber entscheidet).
    if cfg.action.include_cut_flag and action.shape[0] >= 3:
        is_cutting = bool(action[2] > cfg.action.cut_threshold)
    else:
        is_cutting = True

    return next_tcp, is_cutting


# ===========================================================================
# 2. OBSERVATION
# ===========================================================================
#
# Die Observation ist ein (C, H, W)-float32-Array. Gerendert wird in
# einer pro-Episode berechneten Bounding-Box der Geometrie (mit Padding),
# damit das Bild immer maximalen Informationsgehalt hat -- unabhaengig
# von der absoluten Groesse des Werkstuecks.
#
# Eine `ObservationContext`-Datenklasse kapselt alles, was pro Episode
# EINMAL berechnet wird (Geometrie-Polygon, Bbox, statische
# geometry_mask). Damit kostet der Render-Aufruf pro Step nur das
# Update des coverage-Kanals und das Zeichnen des TCP-Blobs.
# ===========================================================================


@dataclass
class ObservationContext:
    """Frozen pro Episode. Enthaelt alles, was fuer `render_observation`
    wiederverwendet werden kann und sich nicht pro Step aendert.

    Fields
    ------
    polygon       : Shapely-Polygon der Geometrie (zum Rasterisieren neuer
                    Swept-Areas im Laufe der Episode).
    bbox_min      : (x_min, y_min) der gepadded Bounding-Box in Weltkoords.
    bbox_max      : (x_max, y_max)
    resolution    : Aufloesung H = W = resolution (Pixel).
    world_per_px  : Skalierungsfaktor (mm pro Pixel). Wird fuer Blob-
                    Groessen und TCP-Rasterisierung gebraucht.
    grid_x, grid_y: Flache Arrays der Pixel-Zentren in Weltkoordinaten
                    (fuer shapely.vectorized.contains).
    geometry_mask : (H, W) bool-Maske -- 1 wo Material existiert.
                    Wird einmal in `build_observation_context` berechnet
                    und dann nur noch gelesen.
    bbox_diagonal : Diagonalenlaenge der Bbox (fuer `compute_max_step_mm`).
    """
    polygon:       Polygon
    bbox_min:      np.ndarray
    bbox_max:      np.ndarray
    resolution:    int
    world_per_px:  float
    grid_x:        np.ndarray   # shape (H*W,)
    grid_y:        np.ndarray   # shape (H*W,)
    geometry_mask: np.ndarray   # shape (H, W), dtype bool
    bbox_diagonal: float


def build_polygon_from_grid(grid: PointGrid) -> Optional[Polygon]:
    """Baut ein Shapely-Polygon aus den Aussenpunkten des PointGrids.

    Gleiche Logik wie im `ContinuousPlanner._build_polygon_from_grid`,
    aber hier als freie Funktion, damit `ObservationContext` unabhaengig
    vom Planner gebaut werden kann. Wir duplizieren 5 Zeilen -- das ist
    billiger als einen Planner nur fuers Polygon zu instanziieren.
    """
    outer = grid.outer_points_ordered
    if len(outer) < 3:
        return None
    coords = [(float(p.x), float(p.y)) for p in outer]
    coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        # bei Fehlkorrekturen -- nimm den groessten Teil
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def build_observation_context(
    grid: PointGrid,
    cfg: RLConfig,
) -> ObservationContext:
    """Baut den Pro-Episode-Kontext fuer die Bild-Beobachtung.

    Wird EINMAL pro `reset()` der Env aufgerufen. Die statische
    `geometry_mask` wird hier berechnet, damit der Per-Step-Renderer
    nur noch die veraenderlichen Kanaele (coverage, TCP) zeichnen muss.
    """
    poly = build_polygon_from_grid(grid)
    if poly is None:
        raise ValueError(
            "Kann kein Polygon aus dem PointGrid bauen -- zu wenig "
            "Aussenpunkte."
        )

    # Bounding-Box mit Padding berechnen.
    min_x, min_y, max_x, max_y = poly.bounds
    width  = max_x - min_x
    height = max_y - min_y
    # Padding als Bruchteil der groesseren Bbox-Dimension (nicht der
    # Diagonale), damit der Rand proportional zur sichtbaren Kontur ist.
    pad = max(width, height) * cfg.observation.bbox_padding_fraction

    # Quadratische Bbox erzwingen: das Bild ist HxW mit H=W=resolution,
    # wir wollen keine Verzerrung der Geometrie. Wir nehmen das groessere
    # der beiden Seiten als gemeinsame Kantenlaenge und zentrieren.
    half = (max(width, height) + 2 * pad) / 2
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    bbox_min = np.array([cx - half, cy - half], dtype=np.float32)
    bbox_max = np.array([cx + half, cy + half], dtype=np.float32)

    resolution = int(cfg.observation.resolution)
    side_world = float(2 * half)
    world_per_px = side_world / resolution

    # Pixel-Zentren in Weltkoordinaten. Wir bilden das Bild so ab, dass
    # Zeile 0 (= "oben") dem GROESSTEN y-Wert entspricht -- Standard fuer
    # Bild-Konventionen. Das macht Visualisierung intuitiver.
    xs = bbox_min[0] + (np.arange(resolution) + 0.5) * world_per_px
    ys = bbox_max[1] - (np.arange(resolution) + 0.5) * world_per_px
    XX, YY = np.meshgrid(xs, ys)  # beide shape (H, W)

    grid_x = XX.ravel()
    grid_y = YY.ravel()

    # Statische Geometrie-Maske durch vektorisierten Point-in-Polygon-Test.
    # `shapely.vectorized.contains` ist in GEOS implementiert und
    # deutlich schneller als eine Python-Schleife.
    mask_flat = shp_contains(poly, grid_x, grid_y)
    geometry_mask = mask_flat.reshape(resolution, resolution).astype(bool)

    bbox_diagonal = float(np.sqrt(width ** 2 + height ** 2))

    return ObservationContext(
        polygon       = poly,
        bbox_min      = bbox_min,
        bbox_max      = bbox_max,
        resolution    = resolution,
        world_per_px  = world_per_px,
        grid_x        = grid_x,
        grid_y        = grid_y,
        geometry_mask = geometry_mask,
        bbox_diagonal = bbox_diagonal,
    )


def world_to_pixel(
    pos_world: np.ndarray,
    ctx: ObservationContext,
) -> tuple[int, int]:
    """Konvertiert eine Weltposition in (row, col) Pixel-Koordinaten.

    Achtung: row wird aus der y-Koordinate berechnet und INVERTIERT,
    weil Bild-Zeilen von oben nach unten laufen.
    """
    x, y = float(pos_world[0]), float(pos_world[1])
    col = (x - ctx.bbox_min[0]) / ctx.world_per_px
    row = (ctx.bbox_max[1] - y) / ctx.world_per_px
    # Auf gueltigen Index-Bereich clippen (TCP kann theoretisch ausserhalb
    # der Bbox sein, wenn der Agent stark herausfaehrt).
    col = int(np.clip(col, 0, ctx.resolution - 1))
    row = int(np.clip(row, 0, ctx.resolution - 1))
    return row, col


def _rasterize_polygon(
    poly: Polygon | MultiPolygon,
    ctx: ObservationContext,
) -> np.ndarray:
    """Rasterisiert ein Shapely-Polygon in eine (H, W)-bool-Maske.

    ROI-Optimierung
    ---------------
    Pro Step werden hier nur kleine Swept-Polygone gerastert
    (typischerweise ein paar Prozent der Gesamt-Bbox). Statt
    `shp_contains` ueber alle H*W Pixel laufen zu lassen, schraenken
    wir den Test auf die Polygon-Bounding-Box ein. Bei einer 64x64-
    Aufloesung und einer 4-Pixel-Swept-Area sinkt der Aufwand von 4096
    auf <100 Punkt-in-Polygon-Tests -- der dominante Hot-Path im
    Env-Step.

    Der erste, episodenweite Aufruf in `build_observation_context`
    benutzt diesen Pfad NICHT (er hat eine eigene Vektor-Schleife),
    sodass das Verhalten dort unveraendert bleibt.
    """
    H = W = ctx.resolution
    if poly is None or poly.is_empty:
        return np.zeros((H, W), dtype=bool)

    # Polygon-Bbox in Weltkoordinaten -> Pixel-Slice.
    min_x, min_y, max_x, max_y = poly.bounds

    # Pixel-Spalten: x waechst nach rechts.
    col_lo = int(np.floor((min_x - ctx.bbox_min[0]) / ctx.world_per_px))
    col_hi = int(np.ceil ((max_x - ctx.bbox_min[0]) / ctx.world_per_px))
    # Pixel-Zeilen: y waechst nach OBEN, Zeilen wachsen nach UNTEN.
    # row(y_max) ist die kleinste Zeile, row(y_min) die groesste.
    row_lo = int(np.floor((ctx.bbox_max[1] - max_y) / ctx.world_per_px))
    row_hi = int(np.ceil ((ctx.bbox_max[1] - min_y) / ctx.world_per_px))

    # Auf gueltigen Bildbereich clippen.
    col_lo = max(0, col_lo); col_hi = min(W, col_hi)
    row_lo = max(0, row_lo); row_hi = min(H, row_hi)
    if col_hi <= col_lo or row_hi <= row_lo:
        return np.zeros((H, W), dtype=bool)

    # Pixel-Zentren NUR im ROI berechnen -- gleiche Konvention wie in
    # build_observation_context (Zeile 0 = groesster y-Wert).
    xs = ctx.bbox_min[0] + (np.arange(col_lo, col_hi) + 0.5) * ctx.world_per_px
    ys = ctx.bbox_max[1] - (np.arange(row_lo, row_hi) + 0.5) * ctx.world_per_px
    XX, YY = np.meshgrid(xs, ys)

    mask_roi = shp_contains(poly, XX.ravel(), YY.ravel()).reshape(
        row_hi - row_lo, col_hi - col_lo
    )

    out = np.zeros((H, W), dtype=bool)
    out[row_lo:row_hi, col_lo:col_hi] = mask_roi
    return out


def _draw_tcp_blob(
    tcp_world: np.ndarray,
    ctx: ObservationContext,
    sigma_px: float,
) -> np.ndarray:
    """Zeichnet einen normalisierten Gauss-Blob an der TCP-Position.

    Warum Gauss und nicht ein einzelnes 1-Pixel?
    -------------------------------------------
    Ein einzelnes Pixel gibt dem CNN kaum Gradient -- minimale
    Positionsaenderungen veraendern nur einen Pixel. Ein Gauss-Blob
    verteilt die Information ueber mehrere Pixel, was (a) das
    Lernen stabilisiert und (b) subpixel-genaue TCP-Positionen
    (float-world -> integer-pixel) weniger verlustreich kodiert.

    Die Blob-Amplitude ist auf 1.0 normiert (max-Wert = 1 im Zentrum).
    """
    H = W = ctx.resolution
    blob = np.zeros((H, W), dtype=np.float32)

    # Pixel-Zentrum der TCP-Position (subpixel-genau).
    x, y = float(tcp_world[0]), float(tcp_world[1])
    col_f = (x - ctx.bbox_min[0]) / ctx.world_per_px - 0.5
    row_f = (ctx.bbox_max[1] - y) / ctx.world_per_px - 0.5

    # Wir zeichnen nur in einem Bounding-Quadrat von +/- 3 sigma,
    # der Rest waere numerisch vernachlaessigbar (Gauss faellt stark ab).
    half = int(np.ceil(3 * sigma_px))
    r0 = max(0, int(np.floor(row_f)) - half)
    r1 = min(H, int(np.ceil(row_f))  + half + 1)
    c0 = max(0, int(np.floor(col_f)) - half)
    c1 = min(W, int(np.ceil(col_f))  + half + 1)
    if r0 >= r1 or c0 >= c1:
        return blob  # TCP komplett ausserhalb des Bilds

    rr = np.arange(r0, r1).reshape(-1, 1)
    cc = np.arange(c0, c1).reshape(1, -1)
    d2 = (rr - row_f) ** 2 + (cc - col_f) ** 2
    blob[r0:r1, c0:c1] = np.exp(-d2 / (2 * sigma_px ** 2)).astype(np.float32)
    return blob


def render_observation(
    ctx:           ObservationContext,
    coverage_mask: np.ndarray,
    tcp_world:     np.ndarray,
    cfg:           RLConfig,
) -> np.ndarray:
    """Rendert die Observation als (C, H, W)-float32-Array.

    Parameters
    ----------
    ctx           : Pro-Episode-Kontext von `build_observation_context`
    coverage_mask : (H, W) bool -- kumulative Coverage (wird extern
                    gepflegt: nach jedem Cut wird das neu geschnittene
                    Polygon via `_rasterize_polygon` ge-ODERt).
    tcp_world     : (2,) -- aktuelle TCP-Position in mm
    cfg           : RLConfig

    Returns
    -------
    obs : (C, H, W) float32 mit C = Summe der aktivierten Kanaele
    """
    channels: list[np.ndarray] = []

    if cfg.observation.use_geometry_channel:
        channels.append(ctx.geometry_mask.astype(np.float32))

    if cfg.observation.use_coverage_channel:
        # coverage_mask kann vom Caller als leeres Array kommen -- wir
        # casten defensiv.
        cm = np.asarray(coverage_mask, dtype=bool)
        if cm.shape != ctx.geometry_mask.shape:
            cm = np.zeros_like(ctx.geometry_mask, dtype=bool)
        channels.append(cm.astype(np.float32))

    if cfg.observation.use_tcp_channel:
        blob = _draw_tcp_blob(
            tcp_world=tcp_world,
            ctx=ctx,
            sigma_px=cfg.observation.tcp_blob_sigma_px,
        )
        channels.append(blob)

    obs = np.stack(channels, axis=0)
    np.clip(obs, 0.0, cfg.observation.clip_max, out=obs)
    return obs


def observation_space_shape(cfg: RLConfig) -> tuple[int, int, int]:
    """Shape der Observation -- fuer `gym.spaces.Box` in Phase 3."""
    c = (
        int(cfg.observation.use_geometry_channel)
        + int(cfg.observation.use_coverage_channel)
        + int(cfg.observation.use_tcp_channel)
    )
    return (c, cfg.observation.resolution, cfg.observation.resolution)


# ===========================================================================
# 3. REWARD
# ===========================================================================
#
# Der Reward ist so zerlegt, dass die Summe ueber eine Episode exakt
# dem Phase-1-Reward entspricht (potential-basiertes Shaping):
#
#     sum_t [ w_cov * (cov_t² - cov_{t-1}²) - w_time * Δt_t ] + terminal
#   = w_cov * cov_T²  -  w_time * total_time  +  terminal
#   = Phase-1-Reward (bis auf Sonderfaelle)
#
# Sonderfaelle:
#   - Ein Cut mit `is_feasible=False` UEBERSCHREIBT den gesamten
#     Per-Step-Return mit `infeasible_penalty`. Der Agent soll keinen
#     positiven Reward aus einem halbausgefuehrten infeasiblen Cut
#     mitnehmen.
#   - Timeout (max_steps erreicht ohne Erfolg): optionale
#     `w_failure_penalty`-Strafe, default 0.
# ===========================================================================


def compute_step_reward(
    prev_coverage: float,
    curr_coverage: float,
    step_time_s:   float,
    cfg:           RLConfig,
) -> float:
    """Per-Step-Reward (ohne Terminal-Terme).

    Parameters
    ----------
    prev_coverage : Coverage VOR diesem Step [0..1]
    curr_coverage : Coverage NACH diesem Step [0..1]
    step_time_s   : Zeitdauer dieses Steps in Sekunden
                    (Schnitt- + Verfahrzeit des erzeugten Segments)
    """
    # Lineares Potential: belohnt jeden Fortschritt am Anfang gleich stark.
    # Das hilft, das "Plateau" bei 0% Coverage zu überwinden.
    cov_reward = cfg.reward.w_coverage_delta * (
        curr_coverage - prev_coverage
    )
    time_penalty = cfg.reward.w_time_delta * step_time_s
    return float(cov_reward - time_penalty)


def compute_terminal_reward(
    final_coverage: float,
    n_pierces:      int,
    is_feasible:    bool,
    timed_out:      bool,
    cfg:            RLConfig,
) -> float:
    """Terminal-Reward am Episodenende.

    Regeln (in Prioritaetsreihenfolge):
      1. `is_feasible=False`: sofort `constraints.infeasible_penalty`,
         alles andere wird ignoriert. Der Agent soll diesen Fehler
         unter allen Umstaenden vermeiden.
      2. Coverage >= completion_threshold: Completion-Bonus.
      3. Pierce-Strafe (jede zusaetzliche Zuendung).
      4. Timeout-Strafe (proportional zur fehlenden Coverage), falls
         eingeschaltet.
    """
    if not is_feasible:
        # Kein Bonus, keine anderen Terme -- nur die Strafe.
        # Wir lesen den Wert aus `constraints`, nicht aus `reward`,
        # damit ein Aendern der Strafe nur an einer Stelle noetig ist.
        return float(cfg.constraints.infeasible_penalty)

    r = 0.0

    if final_coverage >= cfg.reward.completion_threshold:
        r += cfg.reward.w_completion

    # Pierce-Strafe: jede Zuendung ueber die erste hinaus kostet.
    r -= cfg.reward.w_pierce * max(0, n_pierces - 1)

    if timed_out and final_coverage < cfg.reward.completion_threshold:
        r -= cfg.reward.w_failure_penalty * (1.0 - final_coverage)

    return float(r)


def episode_return_lower_bound(cfg: RLConfig) -> float:
    """Theoretisches Minimum eines Episoden-Returns.

    Nuetzlich fuer Sanity-Checks in Phase 4: wenn ein trainierter Agent
    niedriger als diese Schranke landet, stimmt etwas nicht (entweder
    im Reward oder im Reporting).
    """
    # Worst case: infeasible sofort im ersten Step. Dann gibt es nur
    # den infeasible_penalty.
    return float(cfg.constraints.infeasible_penalty)


def episode_return_upper_bound(cfg: RLConfig) -> float:
    """Theoretisches Maximum eines Episoden-Returns.

    Obergrenze: 100% Coverage in Null-Zeit mit einer Zuendung.
    Realistisch nicht erreichbar, aber setzt den "perfekten" Wert.
    """
    return float(
        cfg.reward.w_coverage_delta * 1.0  # cov^2 = 1
        + cfg.reward.w_completion          # Completion-Bonus
        - 0.0                              # Zeit = 0
        - 0.0                              # Pierces = 1 -> keine Strafe
    )

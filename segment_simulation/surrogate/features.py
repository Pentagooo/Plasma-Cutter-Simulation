from __future__ import annotations

"""Feature-Berechnung für den Learned Surrogate Planner 
Alle Features werden mit cKDTree-Distanzen berechnet.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

try:
    from ..segments import SegmentedContour, Segment
    from ...cutter.cutter import Cutter
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, Segment,
    )
    from plasma_cutter.cutter.cutter import Cutter


# ---------------------------------------------------------------------- -----
# Geschwindigkeits-Grenzen
# ---------------------------------------------------------------------------
V_MIN_DEFAULT = 19.4   # = v_cut, minimale Schnittgeschwindigkeit (Projektwert 17.09.2026)
V_MAX_DEFAULT = 34.7   # Base, wenn cutter.max_cutting_speed fehlt


T_SWITCH_DEFAULT = 0.0


# ---------------------------------------------------------------------------
# Physik-Parameter aus dem Cutter ableiten (ohne simulation.py zu laden)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysParams:
    """Aus einem ``Cutter`` abgeleitete kenngrößen.
    """
    blade0: float        # Klingenlänge L(v=0) [mm]
    slope: float         # |dL/dv| [mm/(mm/s)] (positiv)
    gap: float           # Mindestabstand TCP<->Material [mm]
    v_cut: float         # Basis-Schnittgeschwindigkeit [mm/s]
    v_min: float
    v_max: float
    # Zeitaufschlag je Geschwindigkeitswechsel im laufenden Schnitt [s]
    # (siehe T_SWITCH_DEFAULT; Wert ist als Projektwert festzulegen).
    t_switch: float = T_SWITCH_DEFAULT

    @property
    def full_eff_depth(self) -> float:
        """Effektive Schnitttiefe bei v->0"""
        return self.blade0 - self.gap

    @property
    def base_eff_depth(self) -> float:
        """Effektive Schnitttiefe bei der Basisgeschwindigkeit v_cut [mm].
        """
        return max(0.0, self.blade0 - self.slope * self.v_cut - self.gap)

    def blade_at(self, v: float) -> float:
        """Klingenlänge L(v) = clip(L0 - slope*v, 0, L0) [mm]."""
        return float(np.clip(self.blade0 - self.slope * float(v),
                             0.0, self.blade0))

    def eff_depth_at(self, v: float) -> float:
        """Effektive Schnitttiefe bei v [mm] (= L(v) - gap, >= 0)."""
        return max(0.0, self.blade_at(v) - self.gap)

    def speed_for_depth(self, depth_req: float) -> float:
        """Geschwindigkeitsregel schnellste zulässige v.
        """
        if self.slope <= 1e-9:
            return self.v_max
        v = (self.blade0 - self.gap - float(depth_req)) / self.slope
        return float(np.clip(v, self.v_min, self.v_max))


def phys_from_cutter(cutter: Cutter | None,
                     v_min: float = V_MIN_DEFAULT,
                     v_max: float | None = None,
                     t_switch: float | None = None) -> PhysParams:
    """Leitet ``PhysParams`` aus einem Cutter ab (ohne simulation.py).

    ``slope`` wird numerisch aus dem L(v)-Modell bestimmt
    (L(0) - L(1)); das ist robust gegenüber der internen Darstellung im
    ``BladeLengthModel`` und braucht keinen Zugriff auf private Felder.
    ``t_switch`` (None) kommt aus ``cutter.t_switch`` bzw. dem
    Default (0.0 s, offener Projektwert).
    """
    if cutter is None:
        cutter = _default_cutter()
    blade0 = float(cutter.blade_length(0.0))
    slope = float(cutter.blade_length(0.0) - cutter.blade_length(1.0))
    if slope < 0:
        slope = 0.0
    if v_max is None:
        v_max = float(cutter.max_cutting_speed or V_MAX_DEFAULT)
    if t_switch is None:
        t_switch = float(getattr(cutter, "t_switch",
                                 T_SWITCH_DEFAULT))
    return PhysParams(
        blade0=blade0,
        slope=slope,
        gap=float(cutter.minimum_gap),
        v_cut=float(cutter.cutting_speed),
        v_min=float(v_min),
        v_max=float(v_max),
        t_switch=float(t_switch),
    )


def _default_cutter() -> Cutter:
    #Standard-Cutter der Segment-Simulation
    try:
        from ..simulation import make_default_cutter
    except ImportError:
        from plasma_cutter.segment_simulation.simulation import (
            make_default_cutter,
        )
    return make_default_cutter()


# ---------------------------------------------------------------------------
# Geteilte Distanz-/KDTree-Helfer (auch von dataset.py + planner.py genutzt)
# Vorbereitung für KFTree (2D Array)
# ---------------------------------------------------------------------------

def _contour_point_index(
    contour: SegmentedContour,
) -> tuple[np.ndarray, np.ndarray]:
    """Alle Konturpunkte aller Segmente + zugehörige Segment-ID.

    Ein Konturpunkt kann zu zwei Segmenten gehören (gemeinsamer Knoten);
    er wird deterministisch dem Segment mit KLEINERER seg_id zugeordnet,
    damit die nächste-Nachbar-Zuordnung reproduzierbar ist.

    Returns
    -------
    pts    : (M, 2) Konturkoordinaten
    seg_of : (M,)   seg_id je Konturpunkt
    """
    seg_of_pos: dict[tuple[int, int], int] = {}
    coords: list[np.ndarray] = []
    seg_ids: list[int] = []
    for seg in sorted(contour.segments, key=lambda s: s.seg_id, reverse=True):
        loop = contour.loop_by_id(seg.loop_id)
        for pos in seg.positions:
            key = (seg.loop_id, pos)
            # kleinste seg_id (deterministisch).
            seg_of_pos[key] = seg.seg_id
    #  In Segment-/Positionsreihenfolge einsammeln
    seen: set[tuple[int, int]] = set()
    for seg in contour.segments:
        loop = contour.loop_by_id(seg.loop_id)
        for pos in seg.positions:
            key = (seg.loop_id, pos)
            if key in seen:
                continue
            seen.add(key)
            coords.append(loop.points[pos])
            seg_ids.append(seg_of_pos[key])
    return np.asarray(coords, dtype=float), np.asarray(seg_ids, dtype=int)

#
def assign_points(
    coords: np.ndarray,
    contour: SegmentedContour,
) -> tuple[np.ndarray, np.ndarray]:
    """Weist jeden Materialpunkt seinem nächsten Segment zu.

    Returns
    -------
    seg_idx : (P,) seg_id des nächsten Konturpunkts je Materialpunkt
    depth   : (P,) Distanz zum nächsten Konturpunkt [mm]
              (= Distanztransformationswert, "benätigte Tiefe")
    """
    pts, seg_of = _contour_point_index(contour)
    tree = cKDTree(pts)
    depth, idx = tree.query(coords, k=1)
    return seg_of[idx], np.asarray(depth, dtype=float)


def coverability_matrix(
    coords: np.ndarray,
    contour: SegmentedContour,
    radius: float,
) -> np.ndarray:
    """Binäre, distanzbasierte Abdeckbarkeits-Schätzung A_hat[p, s].
    Welche Segmente erreichen überhaupt einen Punkt?

    A_hat[p, s] = True  <=>  dist(p, Polylinie(s)) <= radius

     Genutzt vom Lehrer (dataset) zum
    schnellen Verwerfen unvollständiger Subsets und vom Planner im
    Repair-Schritt. Billiger Check
    """
    n_seg = len(contour.segments)
    A = np.zeros((len(coords), n_seg), dtype=bool)
    for seg in contour.segments:
        loop = contour.loop_by_id(seg.loop_id)
        poly = loop.points[np.asarray(seg.positions, dtype=int)]
        tree = cKDTree(poly)
        d, _ = tree.query(coords, k=1)
        A[:, seg.seg_id] = d <= radius
    return A


# ---------------------------------------------------------------------------
# Feature-Definition
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "arc_length",          # Bogenlänge des Segments [mm]
    "n_points",            # Anzahl Konturpunkte im Segment
    "depth_req_max",       # max. benötigte Tiefe zugewiesener Punkte [mm]
    "depth_req_mean",      # mittlere benötigte Tiefe [mm]
    "n_assigned",          # Anzahl zugewiesener Materialpunkte
    "coverable_count",     # abdeckbare Punkte bei v->0
    "exclusive_fraction",  # Anteil exklusiv (nur s) abdeckbarer Punkte
    "wall_thickness",      # lokale Wanddicke ~ 2*max(DT) [mm]
    "has_opposing_wall",   # 1, wenn Gegenwand in Basis-Reichweite
    "corner_angle_start",  # Eckwinkel am Startknoten [rad]
    "corner_angle_end",    # Eckwinkel am Endknoten [rad]
    "mean_abs_turn",       # mittlere |Richtungsänderung| entlang s [rad]
    "loop_perimeter",      # Umfang des Loops [mm]
    "is_outer",            # 1 = Außenkontur, 0 = Lochkontur
    "n_loops",             # Anzahl Loops der Geometrie
    "rel_position",        # relative Lage des Segmentstarts im Loop [0..1]
    "v_hat",               # zugewiesene Geschwindigkeit [mm/s]
    "time_est",            # Zeitschätzung ell/v_hat [s]
    # --- Überlappungs-/Ersetzbarkeitsmerkmale (19.09.2026) -------------
    # Die alte exclusive_fraction rechnet bei maximaler Reichweite (v->0);
    # dort erreicht fast jeden Punkt mehr als ein Segment, das Merkmal ist
    # praktisch immer 0. Die neuen Merkmale beschreiben, wie leicht die
    # NACHBARN die Punkte eines Segments übernehmen können.
    "coverable_base_count",   # abdeckbare Punkte bei v_cut (Basis-Reichweite)
    "exclusive_base_frac",    # Anteil davon, die bei v_cut NUR s erreicht
    "overlap_max",            # max. Anteil von C_s, den EIN anderes Segment auch erreicht
    "n_overlap_base",         # Anzahl anderer Segmente mit gemeinsamen Punkten bei v_cut
    "alt_depth_mean",         # mittlere Tiefe der zugewiesenen Punkte ab dem NÄCHSTEN FREMDEN Segment [mm]
    "alt_depth_max",          # max. Tiefe ab dem nächsten fremden Segment [mm]
    "alt_replaceable_frac",   # Anteil zugewiesener Punkte, die ein fremdes Segment bei v_cut erreicht
    "n_segments",             # Segmentzahl der Instanz (Kontext)
    "coverable_rel",          # coverable_count / Anzahl Materialpunkte
    "alt_extra_depth_mean",   # mittlere Mehrtiefe, wenn ein fremdes Segment übernimmt [mm]
    "alt_v_hat",              # Geschwindigkeit, die der Ersatz für alt_depth_max bräuchte [mm/s]
    # --- Lage in der Kante (drehungs- und startinvariant, ersetzt rel_position)
    "edge_len",               # Länge der Kante (Segmentkette zwischen zwei Ecken) [mm]
    "edge_n_seg",             # Segmente in dieser Kante
    "edge_rel_pos_sym",       # Lage der Segmentmitte in der Kante, 0 = an der Ecke, 0.5 = Kantenmitte
    "dist_corner",            # Bogenlänge Segmentmitte -> nächste Ecke [mm]
    # --- Kantenkontext: WELCHE Kante des Profils (invariant statt rel_position)
    "edge_len_rel",           # Kantenlänge / Loop-Umfang
    "edge_nb_len_min",        # kürzere Nachbarkante [mm]
    "edge_nb_len_max",        # längere Nachbarkante [mm]
    "edge_depth_mean",        # mittlere benötigte Tiefe aller Punkte der Kante [mm]
]

N_FEATURES = len(FEATURE_NAMES)
N_FEATURES_V1 = 18            # Spalten des Modells vor dem 19.09.2026

# Merkmale, mit denen das Modell tatsächlich trainiert wird (Teilmenge von
# FEATURE_NAMES). ``segment_features`` rechnet immer alle Spalten; Labels
# und Datensätze speichern alle Spalten; das Modell wählt seine Spalten
# über ``select_features`` anhand der gespeicherten ``feature_names``.
# Weggelassen (19.09.2026): Konstanten im Katalog (has_opposing_wall,
# is_outer, n_loops), Duplikate (wall_thickness = 2*depth_req_max,
# time_est/n_points ~ arc_length, v_hat = f(depth_req_max), alt_depth_mean
# ~ alt_depth_max, coverable_base_count ~ coverable_count, n_assigned ~
# coverable_count, alt_replaceable_frac ~ -exclusive_base_frac), tote
# Merkmale (exclusive_fraction, mean_abs_turn, dist_corner, alt_v_hat,
# exclusive_base_frac) und rel_position (Lage ab Loop-Start: hängt von
# der Konstruktionsreihenfolge des Katalogs ab, nicht von der Physik;
# ersetzt durch die edge_*-Merkmale).
DEFAULT_MODEL_FEATURES: list[str] = [
    # Kantenkontext (welche Kante des Profils, invariant gegen Drehung/Start)
    "edge_len_rel",
    "edge_depth_mean",
    "edge_nb_len_max",
    "edge_nb_len_min",
    "edge_len",
    "edge_n_seg",
    "edge_rel_pos_sym",
    # Ersetzbarkeit / Überlappung mit den Nachbarn
    "alt_depth_max",
    "alt_extra_depth_mean",
    "overlap_max",
    "n_overlap_base",
    # benötigte Tiefe und Reichweite
    "depth_req_max",
    "depth_req_mean",
    "coverable_count",
    "coverable_rel",
    # Segmentgeometrie
    "arc_length",
    "corner_angle_start",
    "corner_angle_end",
    # Instanzkontext
    "loop_perimeter",
    "n_segments",
]


def select_features(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Wählt aus einer VOLLEN Merkmalsmatrix (Spalten = FEATURE_NAMES) die
    Spalten ``names``. Hat X bereits genau len(names) Spalten, wird sie
    unverändert zurückgegeben (Datensatz schon reduziert)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2 and X.shape[1] == len(names):
        return X
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"Merkmalsmatrix hat {X.shape} Spalten, erwartet "
                         f"{N_FEATURES} (alle) oder {len(names)} (Modell).")
    idx = [FEATURE_NAMES.index(n) for n in names]
    return X[:, idx]

# Nachbarn je Materialpunkt für die Suche nach dem nächsten FREMDEN
# Konturpunkt (Segmente bis ~30 Punkte -> 64 reicht praktisch immer).
_ALT_K = 64


def alt_segment_depth(coords: np.ndarray, contour: SegmentedContour,
                      cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Je Materialpunkt: nächstes Segment und die Distanz zum nächsten
    Konturpunkt eines ANDEREN Segments (``cap``, wenn keins unter den
    ``_ALT_K`` nächsten Konturpunkten liegt).

    Returns
    -------
    seg_idx : (P,) seg_id des nächsten Segments
    d_alt   : (P,) Distanz zum nächsten fremden Segment [mm]
    """
    pts, seg_of = _contour_point_index(contour)
    k = int(min(_ALT_K, len(pts)))
    tree = cKDTree(pts)
    dist, idx = tree.query(coords, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    segs = seg_of[idx]
    first = segs[:, :1]
    diff = segs != first
    has = diff.any(axis=1)
    j = np.argmax(diff, axis=1)
    d_alt = dist[np.arange(len(coords)), j]
    d_alt = np.where(has, d_alt, float(cap))
    return segs[:, 0], np.asarray(d_alt, dtype=float)


def _corner_angle(loop, pos: int) -> float:
    """Richtungsänderung (Knick) der Kontur an Position pos [rad]."""
    n = loop.n
    v1 = loop.points[pos] - loop.points[(pos - 1) % n]
    v2 = loop.points[(pos + 1) % n] - loop.points[pos]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.acos(cosang)


def edge_features(contour: SegmentedContour,
                  corner_deg: float = 30.0) -> np.ndarray:
    """Je Segment: (edge_len, edge_n_seg, edge_rel_pos_sym, dist_corner,
    edge_len_rel, edge_nb_len_min, edge_nb_len_max) und die Kanten-ID.

    Eine Kante ist die Segmentkette zwischen zwei Eckknoten (Knick >=
    ``corner_deg``). Die Lage wird symmetrisch gemessen (Abstand zur
    näheren Ecke), damit sie weder vom Loop-Start noch von der
    Umlaufrichtung abhängt. Ein Loop ohne Eckknoten (Kreis) ist EINE
    geschlossene Kante: Lage 0.5, Eckabstand = halber Umfang.
    """
    n_seg = len(contour.segments)
    out = np.zeros((n_seg, 7), dtype=float)
    edge_id = np.zeros(n_seg, dtype=int)
    next_id = 0
    for loop in contour.loops:
        segs = [s for s in contour.segments if s.loop_id == loop.loop_id]
        if not segs:
            continue
        n0 = contour.nodes[loop.loop_id][0]
        segs.sort(key=lambda s: loop.arc_length(n0, s.start_pos, +1))
        corners = set(loop.corner_positions(corner_deg))
        chains: list[list] = []
        cur: list = []
        for s in segs:
            if cur and s.start_pos in corners:
                chains.append(cur)
                cur = []
            cur.append(s)
        if cur:
            chains.append(cur)
        if len(chains) > 1 and segs[0].start_pos not in corners:
            chains[0] = chains[-1] + chains[0]      # Kette über den Loop-Start
            chains.pop()
        closed = (len(chains) == 1 and segs[0].start_pos not in corners)
        lens = [float(sum(s.length for s in ch)) for ch in chains]
        m = len(chains)
        for k, ch in enumerate(chains):
            L = lens[k]
            nb = (lens[(k - 1) % m], lens[(k + 1) % m])
            acc = 0.0
            for s in ch:
                c = acc + 0.5 * s.length
                acc += s.length
                if closed or L <= 1e-9:
                    sym, dist = 0.5, 0.5 * L
                else:
                    rel = c / L
                    sym = min(rel, 1.0 - rel)
                    dist = min(c, L - c)
                out[s.seg_id] = (L, len(ch), sym, dist,
                                 L / loop.length if loop.length > 1e-9 else 0.0,
                                 min(nb), max(nb))
                edge_id[s.seg_id] = next_id
            next_id += 1
    return out, edge_id


def _mean_abs_turn(loop, positions: list[int]) -> float:
    """Mittlere absolute Richtungsänderung entlang einer Punktfolge [rad]
    (Rauheitsmaß)."""
    if len(positions) < 3:
        return 0.0
    pts = loop.points[np.asarray(positions, dtype=int)]
    seg = np.diff(pts, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = np.abs(np.diff(ang))
    dang = np.minimum(dang, 2 * math.pi - dang)  # kleinster Winkel
    if len(dang) == 0:
        return 0.0
    return float(np.mean(dang))


def segment_features(
    grid,
    contour: SegmentedContour,
    cutter: Cutter | None = None,
    phys: PhysParams | None = None,
) -> np.ndarray:
    """Feature-Matrix ``X[n_segments, N_FEATURES]`` für eine Geometrie.

    Parameters
    ----------
    grid    : PointGrid (liefert die Materialpunkte ``grid.coords``)
    contour : SegmentedContour mit den Primitiv-Segmenten
    cutter  : Cutter für die Physik-Parameter (None -> Default)
    phys    : optionale, vorab gebaute ``PhysParams`` (spart Aufbau)

    Returns
    -------
    X : (n_segments, N_FEATURES) float64, deterministisch, ohne NaN.
        Zeile i entspricht ``contour.segments[i]`` (seg_id == i).
    """
    if phys is None:
        phys = phys_from_cutter(cutter)

    coords = np.asarray(grid.coords, dtype=float)
    n_seg = len(contour.segments)
    X = np.zeros((n_seg, N_FEATURES), dtype=np.float64)
    if n_seg == 0:
        return X

    # Zuordnung Punkt -> Segment + benötigte Tiefe (Distanztransformation)
    seg_idx, depth = assign_points(coords, contour)

    # Abdeckbarkeit bei maximaler Reichweite (v -> 0)
    A = coverability_matrix(coords, contour, radius=phys.full_eff_depth)
    coverable_count = A.sum(axis=0).astype(np.int64)          # je Segment
    n_covering = A.sum(axis=1).astype(np.int64)               # je Punkt
    exclusive_pt = (n_covering == 1)                          # nur 1 Segment

    # Abdeckbarkeit bei Basis-Reichweite (v_cut): dort unterscheiden sich
    # die Segmente wirklich (bei v->0 erreicht fast alles jeder Nachbar).
    A_base = coverability_matrix(coords, contour, radius=phys.base_eff_depth)
    coverable_base = A_base.sum(axis=0).astype(np.int64)
    exclusive_base_pt = (A_base.sum(axis=1) == 1)
    # Überlappung Segment x Segment (gemeinsam erreichbare Punkte)
    Af = A.astype(np.int32)
    overlap_full = Af.T @ Af                                  # (S, S)
    np.fill_diagonal(overlap_full, 0)
    Ab = A_base.astype(np.int32)
    overlap_base = Ab.T @ Ab
    np.fill_diagonal(overlap_base, 0)
    # Distanz zum nächsten FREMDEN Segment je Punkt (Ersetzbarkeit)
    _, d_alt = alt_segment_depth(coords, contour, cap=2.0 * phys.full_eff_depth)

    n_loops = len(contour.loops)
    n_pts = len(coords)
    E, edge_id = edge_features(contour)
    # mittlere benötigte Tiefe je Kante (über alle zugewiesenen Punkte)
    edge_depth = np.zeros(n_seg, dtype=float)
    for e in np.unique(edge_id):
        m = np.isin(seg_idx, np.flatnonzero(edge_id == e))
        edge_depth[edge_id == e] = float(np.mean(depth[m])) if m.any() else 0.0

    for seg in contour.segments:
        s = seg.seg_id
        loop = contour.loop_by_id(seg.loop_id)

        assigned = (seg_idx == s)
        n_assigned = int(np.count_nonzero(assigned))
        if n_assigned > 0:
            d_assigned = depth[assigned]
            d_max = float(np.max(d_assigned))
            d_mean = float(np.mean(d_assigned))
        else:
            d_max = 0.0
            d_mean = 0.0

        col = A[:, s]
        cov = int(coverable_count[s])
        excl = int(np.count_nonzero(col & exclusive_pt))
        excl_frac = excl / cov if cov > 0 else 0.0

        wall_thickness = 2.0 * d_max
        has_opposing = 1.0 if d_max <= phys.base_eff_depth else 0.0

        v_hat = phys.speed_for_depth(d_max)
        time_est = seg.length / v_hat if v_hat > 1e-9 else 0.0

        # relative Lage: Bogenlänge vom ersten Knoten bis Segmentstart
        rel_pos = 0.0
        if loop.length > 1e-9:
            rel_pos = loop.arc_length(
                contour.nodes[seg.loop_id][0], seg.start_pos, +1) / loop.length

        X[s, 0] = seg.length
        X[s, 1] = len(seg.positions)
        X[s, 2] = d_max
        X[s, 3] = d_mean
        X[s, 4] = n_assigned
        X[s, 5] = cov
        X[s, 6] = excl_frac
        X[s, 7] = wall_thickness
        X[s, 8] = has_opposing
        X[s, 9] = _corner_angle(loop, seg.start_pos)
        X[s, 10] = _corner_angle(loop, seg.end_pos)
        X[s, 11] = _mean_abs_turn(loop, seg.positions)
        X[s, 12] = loop.length
        X[s, 13] = 1.0 if loop.kind == "outer" else 0.0
        X[s, 14] = n_loops
        X[s, 15] = float(np.clip(rel_pos, 0.0, 1.0))
        X[s, 16] = v_hat
        X[s, 17] = time_est

        # Überlappung / Ersetzbarkeit
        cov_b = int(coverable_base[s])
        excl_b = int(np.count_nonzero(A_base[:, s] & exclusive_base_pt))
        X[s, 18] = cov_b
        X[s, 19] = excl_b / cov_b if cov_b > 0 else 0.0
        X[s, 20] = float(overlap_full[s].max()) / cov if cov > 0 else 0.0
        X[s, 21] = int(np.count_nonzero(overlap_base[s] > 0))
        X[s, 25] = n_seg
        X[s, 26] = cov / n_pts if n_pts > 0 else 0.0
        if n_assigned > 0:
            da = d_alt[assigned]
            X[s, 22] = float(np.mean(da))
            X[s, 23] = float(np.max(da))
            X[s, 24] = float(np.mean(da <= phys.base_eff_depth))
            X[s, 27] = float(np.mean(da - d_assigned))
            X[s, 28] = phys.speed_for_depth(float(np.max(da)))
        else:
            X[s, 28] = phys.v_max
        X[s, 29:36] = E[s]
        X[s, 36] = edge_depth[s]

    # Sicherheitsnetz: keine NaN/Inf ins Modell
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X

"""Per-segment features for the surrogate model.

- only from contour, material points and distances (cKDTree), no swept area
- physics (blade length, speed rule) via ``PhysParams``
- ``segment_features`` always computes all columns (``FEATURE_NAMES``), the
  model picks its subset (``DEFAULT_MODEL_FEATURES``)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

try:
    from ..segments import SegmentedContour, Segment
    from ...cutter.cutter import Cutter
except ImportError:  # direct run without package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, Segment,
    )
    from plasma_cutter.cutter.cutter import Cutter


# ---------------------------------------------------------------------------
# Speed limits
# ---------------------------------------------------------------------------
V_MIN_DEFAULT = 19.4   # = v_cut, minimum cutting speed [mm/s]
V_MAX_DEFAULT = 34.7   # if cutter.max_cutting_speed is missing


T_SWITCH_DEFAULT = 0.0


# ---------------------------------------------------------------------------
# Derive physics parameters from the Cutter (without loading simulation.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysParams:
    """Parameters derived from a ``Cutter``."""
    blade0: float        # blade length L(v=0) [mm]
    slope: float         # |dL/dv| [mm/(mm/s)] (positive)
    gap: float           # minimum gap TCP<->material [mm]
    v_cut: float         # base speed [mm/s]
    v_min: float
    v_max: float
    # penalty per speed change within a running cut [s]
    t_switch: float = T_SWITCH_DEFAULT

    @property
    def full_eff_depth(self) -> float:
        """Effective depth at v -> 0 [mm]."""
        return self.blade0 - self.gap

    @property
    def base_eff_depth(self) -> float:
        """Effective depth at v_cut [mm]."""
        return max(0.0, self.blade0 - self.slope * self.v_cut - self.gap)

    def blade_at(self, v: float) -> float:
        """Blade length L(v) = clip(L0 - slope*v, 0, L0) [mm]."""
        return float(np.clip(self.blade0 - self.slope * float(v),
                             0.0, self.blade0))

    def eff_depth_at(self, v: float) -> float:
        """Effective depth at v [mm] (= L(v) - gap, >= 0)."""
        return max(0.0, self.blade_at(v) - self.gap)

    def speed_for_depth(self, depth_req: float) -> float:
        """Speed rule: fastest v that still reaches ``depth_req``,
        clamped to [v_min, v_max].
        """
        if self.slope <= 1e-9:
            return self.v_max
        v = (self.blade0 - self.gap - float(depth_req)) / self.slope
        return float(np.clip(v, self.v_min, self.v_max))


def phys_from_cutter(cutter: Cutter | None,
                     v_min: float = V_MIN_DEFAULT,
                     v_max: float | None = None,
                     t_switch: float | None = None) -> PhysParams:
    """Derives ``PhysParams`` from a Cutter (without simulation.py).

    - slope numerically: L(0) - L(1)
    - t_switch: argument, else ``cutter.t_switch``, else 0
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
    # default cutter of the simulator (lazy, avoids import cycle)
    try:
        from ..simulation import make_default_cutter
    except ImportError:
        from plasma_cutter.segment_simulation.simulation import (
            make_default_cutter,
        )
    return make_default_cutter()


# ---------------------------------------------------------------------------
# Distance helpers (cKDTree)
# ---------------------------------------------------------------------------

def _contour_point_index(
    contour: SegmentedContour,
) -> tuple[np.ndarray, np.ndarray]:
    """All contour points + seg_id per point.

    Shared node of two segments -> smaller seg_id (reproducible).

    Returns
    -------
    pts    : (M, 2) contour coordinates
    seg_of : (M,)   seg_id per contour point
    """
    seg_of_pos: dict[tuple[int, int], int] = {}
    coords: list[np.ndarray] = []
    seg_ids: list[int] = []
    for seg in sorted(contour.segments, key=lambda s: s.seg_id, reverse=True):
        loop = contour.loop_by_id(seg.loop_id)
        for pos in seg.positions:
            key = (seg.loop_id, pos)
            # descending -> smallest seg_id wins
            seg_of_pos[key] = seg.seg_id
    # collect in segment/position order
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


def assign_points(
    coords: np.ndarray,
    contour: SegmentedContour,
) -> tuple[np.ndarray, np.ndarray]:
    """Assigns each material point to the nearest segment.

    Returns
    -------
    seg_idx : (P,) seg_id of the nearest contour point
    depth   : (P,) distance to the nearest contour point = required depth [mm]
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
    """Distance-based estimate of which segments reach a point.

    A_hat[p, s] = True  <=>  dist(p, polyline(s)) <= radius
    (cheap, without swept area)
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
# Feature definition
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "arc_length",          # arc length of the segment [mm]
    "n_points",            # number of contour points in the segment
    "depth_req_max",       # max. required depth of assigned points [mm]
    "depth_req_mean",      # mean required depth [mm]
    "n_assigned",          # number of assigned material points
    "coverable_count",     # coverable points at v->0
    "exclusive_fraction",  # fraction of points coverable exclusively (only s)
    "wall_thickness",      # local wall thickness ~ 2*max(DT) [mm]
    "has_opposing_wall",   # 1 if opposing wall within base reach
    "corner_angle_start",  # corner angle at the start node [rad]
    "corner_angle_end",    # corner angle at the end node [rad]
    "mean_abs_turn",       # mean |direction change| along s [rad]
    "loop_perimeter",      # perimeter of the loop [mm]
    "is_outer",            # 1 = outer contour, 0 = hole contour
    "n_loops",             # number of loops of the geometry
    "rel_position",        # relative position of segment start in the loop [0..1]
    "v_hat",               # assigned speed [mm/s]
    "time_est",            # time estimate ell/v_hat [s]
    # --- overlap / replaceability: how easily the neighbors take over
    #     the points of a segment (at v_cut instead of v -> 0)
    "coverable_base_count",   # coverable points at v_cut (base reach)
    "exclusive_base_frac",    # fraction of these reached ONLY by s at v_cut
    "overlap_max",            # max. fraction of C_s reached by another segment
    "n_overlap_base",         # other segments with shared points at v_cut
    "alt_depth_mean",         # mean depth from the nearest other segment [mm]
    "alt_depth_max",          # max. depth from the nearest other segment [mm]
    "alt_replaceable_frac",   # fraction reached by another segment at v_cut
    "n_segments",             # segment count of the instance (context)
    "coverable_rel",          # coverable_count / number of material points
    "alt_extra_depth_mean",   # mean extra depth at the other segment [mm]
    "alt_v_hat",              # speed for alt_depth_max [mm/s]
    # --- position in the edge (rotation- and start-invariant, replaces rel_position)
    "edge_len",               # edge length (segments between two corners) [mm]
    "edge_n_seg",             # segments in this edge
    "edge_rel_pos_sym",       # segment center position: 0 = corner, 0.5 = edge center
    "dist_corner",            # arc length segment center -> nearest corner [mm]
    # --- edge context: WHICH edge of the profile (invariant instead of rel_position)
    "edge_len_rel",           # edge length / loop perimeter
    "edge_nb_len_min",        # shorter neighboring edge [mm]
    "edge_nb_len_max",        # longer neighboring edge [mm]
    "edge_depth_mean",        # mean required depth of the edge [mm]
]

N_FEATURES = len(FEATURE_NAMES)

# Model features (subset of FEATURE_NAMES)
#   - labels/datasets store all columns, ``select_features`` picks the subset
#   - omitted: constants in the catalog (has_opposing_wall, is_outer,
#     n_loops), duplicates (wall_thickness, time_est, n_points, v_hat,
#     alt_depth_mean, coverable_base_count, n_assigned, alt_replaceable_frac),
#     ineffective features (exclusive_fraction, mean_abs_turn, dist_corner,
#     alt_v_hat, exclusive_base_frac)
#   - rel_position depends on the order in the catalog -> edge_*
DEFAULT_MODEL_FEATURES: list[str] = [
    # edge context (which edge of the profile, invariant to rotation/start)
    "edge_len_rel",
    "edge_depth_mean",
    "edge_nb_len_max",
    "edge_nb_len_min",
    "edge_len",
    "edge_n_seg",
    "edge_rel_pos_sym",
    # replaceability / overlap with the neighbors
    "alt_depth_max",
    "alt_extra_depth_mean",
    "overlap_max",
    "n_overlap_base",
    # required depth and reach
    "depth_req_max",
    "depth_req_mean",
    "coverable_count",
    "coverable_rel",
    # segment geometry
    "arc_length",
    "corner_angle_start",
    "corner_angle_end",
    # instance context
    "loop_perimeter",
    "n_segments",
]


def select_features(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Columns ``names`` from the full feature matrix (unchanged if
    X already has exactly these columns).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2 and X.shape[1] == len(names):
        return X
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"Merkmalsmatrix hat {X.shape} Spalten, erwartet "
                         f"{N_FEATURES} (alle) oder {len(names)} (Modell).")
    idx = [FEATURE_NAMES.index(n) for n in names]
    return X[:, idx]

# k neighbors for finding the nearest contour point of another segment
# (well above the points per segment)
_ALT_K = 64


def alt_segment_depth(coords: np.ndarray, contour: SegmentedContour,
                      cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Per material point: nearest segment + distance to the nearest
    contour point of another segment (``cap`` if none lies among the
    ``_ALT_K`` nearest).

    Returns
    -------
    seg_idx : (P,) seg_id of the nearest segment
    d_alt   : (P,) distance to the nearest other segment [mm]
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
    """Direction change (kink) of the contour at position pos [rad]."""
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
    """Edge features per segment + edge ID.

    - edge = segment chain between two corner nodes (kink >= ``corner_deg``)
    - position symmetric (distance to the nearer corner): independent of
      loop start and traversal direction
    - loop without corner (circle) = one closed edge, position 0.5

    Columns: edge_len, edge_n_seg, edge_rel_pos_sym, dist_corner,
    edge_len_rel, edge_nb_len_min, edge_nb_len_max
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
            chains[0] = chains[-1] + chains[0]      # chain across the loop start
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
    """Mean |direction change| along the point sequence [rad]
    (roughness).
    """
    if len(positions) < 3:
        return 0.0
    pts = loop.points[np.asarray(positions, dtype=int)]
    seg = np.diff(pts, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = np.abs(np.diff(ang))
    dang = np.minimum(dang, 2 * math.pi - dang)  # smallest angle
    if len(dang) == 0:
        return 0.0
    return float(np.mean(dang))


def segment_features(
    grid,
    contour: SegmentedContour,
    cutter: Cutter | None = None,
    phys: PhysParams | None = None,
) -> np.ndarray:
    """Feature matrix ``X[n_segments, N_FEATURES]`` of a geometry.

    grid    : PointGrid (material points ``grid.coords``)
    contour : SegmentedContour (row i = seg_id i)
    cutter  : for the physics (None -> default)
    phys    : precomputed ``PhysParams`` (optional)

    Deterministic, no NaN/Inf.
    """
    if phys is None:
        phys = phys_from_cutter(cutter)

    coords = np.asarray(grid.coords, dtype=float)
    n_seg = len(contour.segments)
    X = np.zeros((n_seg, N_FEATURES), dtype=np.float64)
    if n_seg == 0:
        return X

    # assignment point -> segment + required depth (distance transform)
    seg_idx, depth = assign_points(coords, contour)

    # coverability at maximum reach (v -> 0)
    A = coverability_matrix(coords, contour, radius=phys.full_eff_depth)
    coverable_count = A.sum(axis=0).astype(np.int64)          # per segment
    n_covering = A.sum(axis=1).astype(np.int64)               # per point
    exclusive_pt = (n_covering == 1)                          # only 1 segment

    # coverability at base reach (v_cut)
    A_base = coverability_matrix(coords, contour, radius=phys.base_eff_depth)
    coverable_base = A_base.sum(axis=0).astype(np.int64)
    exclusive_base_pt = (A_base.sum(axis=1) == 1)
    # overlap segment x segment (jointly reachable points)
    Af = A.astype(np.int32)
    overlap_full = Af.T @ Af                                  # (S, S)
    np.fill_diagonal(overlap_full, 0)
    Ab = A_base.astype(np.int32)
    overlap_base = Ab.T @ Ab
    np.fill_diagonal(overlap_base, 0)
    # distance to the nearest OTHER segment per point (replaceability)
    _, d_alt = alt_segment_depth(coords, contour, cap=2.0 * phys.full_eff_depth)

    n_loops = len(contour.loops)
    n_pts = len(coords)
    E, edge_id = edge_features(contour)
    # mean required depth per edge (over all assigned points)
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

        # relative position: arc length from the first node to the segment start
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

        # overlap / replaceability
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

    # safety net: no NaN/Inf into the model
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X

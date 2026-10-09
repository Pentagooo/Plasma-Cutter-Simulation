"""Shared planning stages for teacher, Automatic Planner and surrogate.

- build runs: contiguous segments -> CutRun, kinematics at the
  assigned speed v (blade L(v) -> swept area)
- DP split (``split_group_for_speed``, ``build_speed_chains``,
  ``ChainedRun``): a chain splits into blocks with their own
  speed, penalty ``t_switch`` per speed change; the Sequencer sees
  one node per chain
- ``build_plan_with_speeds``: plan with variable speeds; non-linkable
  runs are dropped instead of crashing
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree

try:
    from ..segments import SegmentedContour, CutRun
    from ..planning import (
        RunKinematics, LinkPlanner, Sequencer, LinkInfeasibleError, CHAIN_TOL,
    )
    from ...cutter.cutter import Cutter
    from .features import PhysParams, phys_from_cutter
except ImportError:  # direct run without package context
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, CutRun,
    )
    from plasma_cutter.segment_simulation.planning import (
        RunKinematics, LinkPlanner, Sequencer, LinkInfeasibleError, CHAIN_TOL,
    )
    from plasma_cutter.cutter.cutter import Cutter
    from plasma_cutter.segment_simulation.surrogate.features import (
        PhysParams, phys_from_cutter,
    )


# ---------------------------------------------------------------------------
# Exact Sequencer
# ---------------------------------------------------------------------------

# alias: makes "exactly sequenced" (Held-Karp) visible at the call sites
ExactSequencer = Sequencer


# ---------------------------------------------------------------------------
# Loop order of the segments
# ---------------------------------------------------------------------------

def loop_segment_order(contour: SegmentedContour) -> dict[int, list[int]]:
    """seg_id per loop in contour/node order (for adjacency)."""
    order: dict[int, list[int]] = {}
    for seg in contour.segments:
        order.setdefault(seg.loop_id, []).append(seg.seg_id)
    return order


def group_contiguous(
    contour: SegmentedContour,
    selected: set[int],
) -> list[tuple[int, list[int]]]:
    """Splits the selected segments into cyclically contiguous groups.

    Returns [(loop_id, [seg_id, ...])], one CutRun per group (one
    pierce).
    """
    order_by_loop = loop_segment_order(contour)
    groups: list[tuple[int, list[int]]] = []
    for loop_id, order in order_by_loop.items():
        chosen = [s for s in order if s in selected]
        if not chosen:
            continue
        k = len(order)
        if len(chosen) == k:
            groups.append((loop_id, list(order)))       # complete loop
            continue
        for i, s in enumerate(order):
            if s not in selected or order[(i - 1) % k] in selected:
                continue  # no group start
            group = [s]
            j = i
            while order[(j + 1) % k] in selected and len(group) < k:
                j += 1
                group.append(order[j % k])
            groups.append((loop_id, group))
    return groups


# ---------------------------------------------------------------------------
# CutRun construction + kinematics at the assigned speed
# ---------------------------------------------------------------------------

def make_group_run(
    contour: SegmentedContour,
    loop_id: int,
    group: list[int],
    run_id: int,
) -> CutRun:
    """Builds the CutRun of a segment group (no kinematics yet)."""
    order = loop_segment_order(contour)[loop_id]
    full_loop = len(group) == len(order)
    first = contour.segments[group[0]]
    last = contour.segments[group[-1]]
    loop = contour.loop_by_id(loop_id)
    start_pos = first.start_pos
    end_pos = first.start_pos if full_loop else last.end_pos
    positions = loop.arc_positions(start_pos, end_pos, +1)
    length = (loop.length if full_loop
              else loop.arc_length(start_pos, end_pos, +1))
    return CutRun(
        run_id=run_id, loop_id=loop_id,
        start_pos=start_pos, end_pos=end_pos, direction=+1,
        positions=positions, polyline=loop.polyline(positions),
        length=length,
        node_positions=contour.node_positions_of(loop_id, positions),
    )


def attach_at_speed(
    run: CutRun,
    material,
    phys: PhysParams,
    v: float,
    kerf: float,
) -> CutRun:
    """Attach kinematics at speed ``v`` (in-place).

    Swept-area error (GEOS on invalid geometry) -> run marked as not
    executable instead of crashing.
    """
    try:
        kin = RunKinematics(material, clearance=phys.gap,
                            blade_length=phys.blade_at(v), kerf=kerf)
        kin.attach(run)
    except Exception:
        run.is_feasible = False
        run.swept_polygon = None
        run.tcp_polyline = None
        run.reason = "Kinematik-Fehler (ungueltige Geometrie)"
    return run


# ---------------------------------------------------------------------------
# Coverage-preserving merge (like AutoPlanner._runs_for_group)
# ---------------------------------------------------------------------------

def build_singletons(
    contour: SegmentedContour,
    seg_ids,
    material,
    phys: PhysParams,
    v: float,
    kerf: float,
    coords: np.ndarray,
) -> tuple[dict, dict]:
    """Singleton run per segment at ``v`` + boolean point mask.

    Returns (seg_run, seg_mask): seg_id -> CutRun or mask.
    """
    seg_run: dict[int, CutRun] = {}
    seg_mask: dict[int, np.ndarray] = {}
    for s in seg_ids:
        seg = contour.segments[s]
        r = make_group_run(contour, seg.loop_id, [s], run_id=s + 1)
        attach_at_speed(r, material, phys, v, kerf)
        seg_run[s] = r
        if r.is_feasible and r.swept_polygon is not None:
            seg_mask[s] = shapely.contains_xy(
                r.swept_polygon, coords[:, 0], coords[:, 1])
        else:
            seg_mask[s] = np.zeros(len(coords), dtype=bool)
    return seg_run, seg_mask


def _singletons_for(contour, group, material, phys, v, kerf, coords):
    """Fresh singleton runs + union mask of a segment group."""
    singles: list[CutRun] = []
    target = np.zeros(len(coords), dtype=bool)
    for s in group:
        seg = contour.segments[s]
        r = make_group_run(contour, seg.loop_id, [s], run_id=0)
        attach_at_speed(r, material, phys, v, kerf)
        singles.append(r)
        if r.is_feasible and r.swept_polygon is not None:
            target |= shapely.contains_xy(
                r.swept_polygon, coords[:, 0], coords[:, 1])
    return singles, target


def merge_covering_runs(
    contour: SegmentedContour,
    selected,
    material,
    phys: PhysParams,
    v: float,
    kerf: float,
    coords: np.ndarray,
    check_partial: bool = False,
) -> list[CutRun]:
    """Merges contiguous segments; if the merge loses coverage, the
    individual segments are used.

    - full-loop merge: always checked exactly (closing the ring partly
      sweeps less than the individual segments)
    - partial-arc merge: checked only with ``check_partial`` (like the
      teacher, reproduces its runs; costs one singleton build per group)
    - all runs freshly attached
    """
    order_by_loop = loop_segment_order(contour)
    runs: list[CutRun] = []
    rid = 1
    for loop_id, group in group_contiguous(contour, set(selected)):
        merged = make_group_run(contour, loop_id, list(group), run_id=rid)
        attach_at_speed(merged, material, phys, v, kerf)
        feasible = merged.is_feasible and merged.swept_polygon is not None
        is_full = len(group) == len(order_by_loop.get(loop_id, []))

        if feasible and (is_full or check_partial):
            singles, target = _singletons_for(
                contour, group, material, phys, v, kerf, coords)
            merged_mask = shapely.contains_xy(
                merged.swept_polygon, coords[:, 0], coords[:, 1])
            if not bool(np.all(target <= merged_mask)):
                for r in singles:  # merge loses coverage
                    if r.is_feasible and r.swept_polygon is not None:
                        r.run_id = rid
                        runs.append(r)
                        rid += 1
                continue

        if feasible:
            runs.append(merged)
            rid += 1
        else:
            for s in group:  # merge itself infeasible -> individual segments
                seg = contour.segments[s]
                r = make_group_run(contour, seg.loop_id, [s], run_id=rid)
                attach_at_speed(r, material, phys, v, kerf)
                if r.is_feasible and r.swept_polygon is not None:
                    runs.append(r)
                    rid += 1
    return runs


# ---------------------------------------------------------------------------
# Traverse reachability (exclude fully enclosed holes)
# ---------------------------------------------------------------------------

def linkable_segments(
    contour: SegmentedContour,
    feasible,
    material,
    phys: PhysParams,
    cutter: Cutter,
    kerf: float,
) -> set[int]:
    """Segments whose loop is reachable from the outer contour by rapid traverse.

    Fully enclosed holes are dropped so that the teacher's
    coverage target stays reachable.
    """
    feas = set(feasible)
    order_by_loop = loop_segment_order(contour)
    outer_ids = [l.loop_id for l in contour.loops if l.kind == "outer"]

    def full_run(loop_id, rid):
        order = [s for s in order_by_loop.get(loop_id, []) if s in feas]
        if not order:
            return None
        r = make_group_run(contour, loop_id, order, run_id=rid)
        return attach_at_speed(r, material, phys, phys.v_cut, kerf)

    link = LinkPlanner(material, clearance=phys.gap)
    seq = Sequencer(cutter, contour, link)
    linkable: set[int] = set()
    outer_run = full_run(outer_ids[0], 1) if outer_ids else None

    for l in contour.loops:
        if l.loop_id not in order_by_loop:
            continue
        if l.kind == "outer" or outer_run is None or not outer_run.is_feasible:
            linkable.add(l.loop_id)
            continue
        hole_run = full_run(l.loop_id, 2)
        if hole_run is None or not hole_run.is_feasible:
            linkable.add(l.loop_id)  # not testable -> keep
            continue
        try:
            seq.order_runs([outer_run, hole_run])
            linkable.add(l.loop_id)
        except LinkInfeasibleError:
            pass  # fully enclosed -> exclude
    return {s for s in feas if contour.segments[s].loop_id in linkable}


# ---------------------------------------------------------------------------
# Coverage-preserving speed increase
# ---------------------------------------------------------------------------

def fastest_safe_speed(
    run: CutRun,
    coords: np.ndarray,
    material,
    phys: PhysParams,
    kerf: float,
    v_lo: float,
    v_hi: float,
) -> float:
    """Fastest speed that keeps the base coverage of the run.

    - run must be attached at ``v_lo``
    - analytic instead of binary search: d_r = largest distance of a
      covered point to the run contour, v_r = speed_for_depth(d_r)
    - attach once at v_r and check exactly, else back to v_lo
      -> plan only gets faster, coverage stays

    Run stays attached at the chosen speed.
    """
    if run.swept_polygon is None or v_hi <= v_lo + 1e-9:
        return v_lo
    base_mask = shapely.contains_xy(run.swept_polygon, coords[:, 0], coords[:, 1])
    if not base_mask.any():
        return v_lo

    # required depth d_r = deepest covered point (distance to the contour)
    covered = coords[base_mask]
    tree = cKDTree(np.asarray(run.polyline, dtype=float))
    d_r = float(tree.query(covered, k=1)[0].max())
    v = phys.speed_for_depth(d_r)
    if v <= v_lo + 1e-9:
        return v_lo  # no speed-up possible

    attach_at_speed(run, material, phys, v, kerf)
    if (run.swept_polygon is not None and run.is_feasible):
        m = shapely.contains_xy(run.swept_polygon, coords[:, 0], coords[:, 1])
        if bool(np.all(base_mask <= m)):
            return v
    # approximation error -> back to base speed
    attach_at_speed(run, material, phys, v_lo, kerf)
    return v_lo


def speed_up_runs(
    runs: list[CutRun],
    coords: np.ndarray,
    material,
    phys: PhysParams,
    kerf: float,
) -> dict[int, float]:
    """Raises each run to its fastest coverage-preserving speed
    (runs attached at v_cut). Returns run_id -> v.
    """
    speeds: dict[int, float] = {}
    for run in runs:
        if not run.is_feasible or run.swept_polygon is None:
            speeds[run.run_id] = phys.v_cut
            continue
        v = fastest_safe_speed(run, coords, material, phys, kerf,
                               v_lo=phys.v_cut, v_hi=phys.v_max)
        speeds[run.run_id] = v
    return speeds


# ---------------------------------------------------------------------------
# DP split: time-minimal chain decomposition (speed rule + t_switch)
# ---------------------------------------------------------------------------

def clamp_rule_speed(phys: PhysParams, depth: float) -> float:
    """Rule speed for a depth, clamped to [v_cut, v_max]
    (never slower than v_cut).
    """
    v_hi = max(phys.v_max, phys.v_cut)
    return float(min(max(phys.speed_for_depth(float(depth)), phys.v_cut),
                     v_hi))


def run_required_depth(run: CutRun, coords: np.ndarray,
                       mask: np.ndarray | None = None) -> float:
    """Required depth [mm]: distance of the farthest covered
    point to the run contour. No attach; ``mask`` = precomputed
    point mask.
    """
    if mask is None:
        if run.swept_polygon is None:
            return 0.0
        mask = shapely.contains_xy(run.swept_polygon,
                                   coords[:, 0], coords[:, 1])
    if not mask.any():
        return 0.0
    tree = cKDTree(np.asarray(run.polyline, dtype=float))
    return float(tree.query(coords[mask], k=1)[0].max())


def split_group_for_speed(
    lengths,
    depths,
    phys: PhysParams,
    t_switch: float | None = None,
) -> tuple[list[tuple[int, int]], list[float], float]:
    """Time-minimal decomposition of a segment chain into blocks with one
    speed each (DP, O(m^2)).

    - instead of running the whole chain at the speed of the deepest spot,
      blocks may run faster
    - penalty ``t_switch`` per block transition (no lift-off, no pierce)
    - full-loop chains: linear chain from the group start (same for all
      planners)

    lengths  : cut length per chain segment [mm]
    depths   : required depth per chain segment [mm]
    t_switch : penalty [s]; None -> ``phys.t_switch``; inf -> one block

    Returns (blocks, speeds, total_time):
      blocks     : (start, end_exclusive) in chain indices
      speeds     : speed per block [mm/s]
      total_time : cut + switch time of the chain [s]

    Edge cases (tests/test_speed_split.py): t_switch = inf -> 1 block;
    t_switch = 0 -> sum(len_i / v_i).
    """
    m = len(lengths)
    if t_switch is None:
        t_switch = float(phys.t_switch)
    if m == 0:
        return [], [], 0.0
    lengths = [float(x) for x in lengths]
    depths = [float(x) for x in depths]
    if not math.isfinite(t_switch):
        v = clamp_rule_speed(phys, max(depths))
        return [(0, m)], [v], sum(lengths) / max(v, 1e-9)

    # dp_cost[i] = minimum time of the first i segments; tie-break: fewer
    # blocks (avoids useless splits at t_switch = 0).
    INF = math.inf
    dp_cost = [INF] * (m + 1)
    dp_blocks = [0] * (m + 1)
    parent = [-1] * (m + 1)
    dp_cost[0] = 0.0
    for i in range(1, m + 1):
        d_max = 0.0
        length = 0.0
        for j in range(i - 1, -1, -1):      # block = chain segments j..i-1
            d_max = max(d_max, depths[j])
            length += lengths[j]
            v = clamp_rule_speed(phys, d_max)
            c = dp_cost[j] + length / max(v, 1e-9) + (t_switch if j > 0
                                                      else 0.0)
            b = dp_blocks[j] + 1
            if (c < dp_cost[i] - 1e-12
                    or (abs(c - dp_cost[i]) <= 1e-12 and b < dp_blocks[i])):
                dp_cost[i], dp_blocks[i], parent[i] = c, b, j

    blocks: list[tuple[int, int]] = []
    i = m
    while i > 0:
        j = parent[i]
        blocks.append((j, i))
        i = j
    blocks.reverse()
    speeds = [clamp_rule_speed(phys, max(depths[a:b])) for a, b in blocks]
    return blocks, speeds, float(dp_cost[m])


# ---------------------------------------------------------------------------
# ChainedRun: chain as ONE Sequencer macro node + sub-run rollout
# ---------------------------------------------------------------------------

@dataclass
class ChainedRun:
    """Contiguous segment chain as one Sequencer node.

    - Held-Karp sees only ``macro`` (chain ends, direction free)
    - afterwards the sub-runs (DP blocks) are rolled out from the chosen
      end; transitions seamless (no link, no pierce, only t_switch)

    macro    : light run for sequencing only (without swept area)
    sub_runs : attached sub-runs in chain order
    speeds   : speed per sub-run [mm/s]
    cut_time / switch_time / n_switches : precomputed times for
               chains without sub-runs
    """
    macro: CutRun
    sub_runs: list[CutRun] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    cut_time: float = 0.0
    switch_time: float = 0.0
    n_switches: int = 0

    def rolled_out(self, ordered_macro: CutRun) -> tuple[list[CutRun],
                                                         list[float]]:
        """Roll out the sub-runs in the direction chosen by the Sequencer."""
        if ordered_macro.direction == self.macro.direction:
            return list(self.sub_runs), list(self.speeds)
        return ([r.reversed() for r in reversed(self.sub_runs)],
                list(reversed(self.speeds)))


def _chain_macro(contour, loop_id: int, group: list[int], run_id: int,
                 tcp_start: np.ndarray, tcp_end: np.ndarray,
                 tcp_length: float) -> CutRun:
    """Light macro run (endpoints only, no attach). Endpoints from the
    sub-runs, so Held-Karp evaluates exactly the transitions of the rollout.
    """
    loop = contour.loop_by_id(loop_id)
    first = contour.segments[group[0]]
    last = contour.segments[group[-1]]
    full_loop = first.start_pos == last.end_pos
    return CutRun(
        run_id=run_id, loop_id=loop_id,
        start_pos=first.start_pos,
        end_pos=first.start_pos if full_loop else last.end_pos,
        direction=+1, positions=[],
        polyline=np.asarray([loop.points[first.start_pos],
                             loop.points[first.start_pos if full_loop
                                         else last.end_pos]], dtype=float),
        length=float(sum(contour.segments[s].length for s in group)),
        tcp_polyline=np.asarray([tcp_start, tcp_end], dtype=float),
        tcp_length=float(tcp_length),
    )


def _covers(run: CutRun, coords: np.ndarray, target: np.ndarray) -> bool:
    """Does the (attached) run cover all ``target`` points exactly?"""
    if not target.any():
        return run.is_feasible and run.swept_polygon is not None
    if not run.is_feasible or run.swept_polygon is None:
        return False
    hit = shapely.contains_xy(run.swept_polygon, coords[target, 0],
                              coords[target, 1])
    return bool(np.all(hit))


def build_speed_chains(
    contour: SegmentedContour,
    selected,
    material,
    phys: PhysParams,
    kerf: float,
    coords: np.ndarray,
    seg_run: dict[int, CutRun] | None = None,
    seg_mask: dict[int, np.ndarray] | None = None,
    t_switch: float | None = None,
) -> list[ChainedRun]:
    """Shared, exactly checked DP split stage of all planners (same
    decision space -> the teacher dominates).

    Per contiguous group:
      1. depth/length per segment from the singleton runs (v_cut) -> DP split
      2. attach each block at its speed, check against the singleton
         masks; missed -> v_cut; still missed ->
         individual segments with ``fastest_safe_speed``
      3. macro run per chain for the Sequencer

    Promised coverage = union of the singleton masks. Checking can only
    lower speeds -> DP time is a lower bound.
    """
    if t_switch is None:
        t_switch = float(phys.t_switch)
    v_cut = phys.v_cut
    v_hi = max(phys.v_max, v_cut)
    sel = sorted(set(selected))
    if seg_run is None or seg_mask is None:
        seg_run, seg_mask = build_singletons(
            contour, sel, material, phys, v_cut, kerf, coords)

    chains: list[ChainedRun] = []
    rid = 1
    for loop_id, group in group_contiguous(contour, set(sel)):
        depths: list[float] = []
        lengths: list[float] = []
        for s in group:
            r = seg_run.get(s)
            m = seg_mask.get(s)
            if r is None or not r.is_feasible or m is None or not m.any():
                depths.append(0.0)
                lengths.append(float(contour.segments[s].length))
            else:
                depths.append(run_required_depth(r, coords, m))
                lengths.append(float(r.tcp_length if r.tcp_length > 0
                                     else r.length))
        blocks, speeds, _ = split_group_for_speed(lengths, depths, phys,
                                                  t_switch)

        sub_runs: list[CutRun] = []
        sub_speeds: list[float] = []
        for (a, b), v_blk in zip(blocks, speeds):
            block = list(group[a:b])
            target = np.zeros(len(coords), dtype=bool)
            for s in block:
                if s in seg_mask:
                    target |= seg_mask[s]
            run = make_group_run(contour, loop_id, block, run_id=0)
            attach_at_speed(run, material, phys, v_blk, kerf)
            if _covers(run, coords, target):
                sub_runs.append(run)
                sub_speeds.append(v_blk)
                continue
            if abs(v_blk - v_cut) > 1e-9:
                attach_at_speed(run, material, phys, v_cut, kerf)
                if _covers(run, coords, target):
                    sub_runs.append(run)
                    sub_speeds.append(v_cut)
                    continue
            # missed even at v_cut -> individual segments, each checked separately
            for s in block:
                single = make_group_run(contour, loop_id, [s], run_id=0)
                attach_at_speed(single, material, phys, v_cut, kerf)
                if not single.is_feasible or single.swept_polygon is None:
                    continue
                v_s = fastest_safe_speed(single, coords, material, phys,
                                         kerf, v_lo=v_cut, v_hi=v_hi)
                sub_runs.append(single)
                sub_speeds.append(v_s)
        if not sub_runs:
            continue
        for r in sub_runs:
            r.run_id = rid
            rid += 1
        macro = _chain_macro(
            contour, loop_id, list(group), run_id=rid,
            tcp_start=sub_runs[0].tcp_start, tcp_end=sub_runs[-1].tcp_end,
            tcp_length=float(sum(r.tcp_length for r in sub_runs)))
        rid += 1
        chains.append(ChainedRun(macro=macro, sub_runs=sub_runs,
                                 speeds=sub_speeds))
    return chains


# ---------------------------------------------------------------------------
# Robust plan construction with variable speeds
# ---------------------------------------------------------------------------

@dataclass
class SpeedPlan:
    """Result of ``build_plan_with_speeds``; ``switch_time``/``n_switches``
    = speed changes within a running cut (reported separately like
    ``pierce_time``).
    """
    ordered_runs: list[CutRun] = field(default_factory=list)
    run_speeds: list[float] = field(default_factory=list)
    total_time: float = 0.0
    cut_time: float = 0.0
    travel_time: float = 0.0
    pierce_time: float = 0.0
    switch_time: float = 0.0
    n_pierces: int = 0
    n_switches: int = 0
    dropped_run_ids: list[int] = field(default_factory=list)
    is_optimal: bool = False


def _pairwise_linkable(seq: Sequencer, runs: list[CutRun]) -> dict[int, int]:
    """Number of linkable partners per run (in either direction)."""
    variants = [(r, r.reversed()) for r in runs]
    conn = {r.run_id: 0 for r in runs}
    for i, (fi, ri) in enumerate(variants):
        for j, (fj, rj) in enumerate(variants):
            if i == j:
                continue
            ok = any(math.isfinite(seq._trans_cost(a.tcp_end, b.tcp_start))
                     for a in (fi, ri) for b in (fj, rj))
            if ok:
                conn[runs[i].run_id] += 1
    return conn


def build_plan_with_speeds(
    sequencer: Sequencer,
    runs: list,
    run_speed: dict[int, float],
    cutter: Cutter,
    drop_unlinkable: bool = True,
    t_switch: float = 0.0,
) -> SpeedPlan:
    """Time-minimal ordered plan with variable cutting speeds.

    runs may mix CutRun and ChainedRun (run_ids unique):
      CutRun     : speed from ``run_speed``
      ChainedRun : one Held-Karp node per chain, then roll out the
                   sub-runs; without sub-runs the precomputed times
                   are used

    - seamless transition (< CHAIN_TOL): no link, no pierce, t_switch
      on speed change; else rapid traverse + pierce
    - ``LinkInfeasibleError``: non-linkable runs are dropped,
      their points remain uncut
    """
    plan = SpeedPlan()
    chains: dict[int, ChainedRun] = {}
    macros: list[CutRun] = []
    for r in runs:
        if isinstance(r, ChainedRun):
            chains[r.macro.run_id] = r
            macros.append(r.macro)
        elif r.is_feasible and r.swept_polygon is not None:
            macros.append(r)
    if not macros:
        return plan

    # not linkable -> drop the worst node, retry
    while macros:
        try:
            ordered, is_opt = sequencer.order_runs(macros)
            break
        except LinkInfeasibleError:
            if not drop_unlinkable or len(macros) <= 1:
                # a single node always works (no transitions)
                if len(macros) == 1:
                    ordered, is_opt = [macros[0]], True
                    break
                return plan
            conn = _pairwise_linkable(sequencer, macros)
            # the node with the fewest partners is dropped; tie -> the shorter one
            # (less coverage loss; for test_lochjson this keeps the
            # outer contour instead of the enclosed hole)
            worst = min(macros, key=lambda r: (
                conn[r.run_id],
                r.tcp_length if r.tcp_length > 0 else r.length))
            plan.dropped_run_ids.append(worst.run_id)
            macros = [r for r in macros if r is not worst]
    else:
        return plan

    plan.is_optimal = is_opt
    pierce = cutter.pierce_time()

    prev_end = None
    prev_v: float | None = None
    for macro in ordered:
        chain = chains.get(macro.run_id)
        if chain is not None and not chain.sub_runs:
            # chain without sub-runs: times precomputed
            plan.cut_time += chain.cut_time
            plan.switch_time += chain.switch_time
            plan.n_switches += chain.n_switches
            items: list[tuple[CutRun, float | None]] = [(macro, None)]
        elif chain is not None:
            items = list(zip(*chain.rolled_out(macro)))
        else:
            items = [(macro, run_speed.get(macro.run_id,
                                           cutter.cutting_speed))]

        for run, v in items:
            if v is not None:
                cut_len = run.tcp_length if run.tcp_length > 0 else run.length
                plan.cut_time += cut_len / max(v, 1e-9)
                plan.run_speeds.append(float(v))
            else:
                plan.run_speeds.append(float("nan"))
            plan.ordered_runs.append(run)

            needs_pierce = True
            if prev_end is not None:
                gap = float(np.linalg.norm(run.tcp_start - prev_end))
                if gap < CHAIN_TOL:
                    # seamless: torch stays on, only t_switch on v change
                    needs_pierce = False
                    if (v is not None and prev_v is not None
                            and abs(v - prev_v) > 1e-9):
                        plan.switch_time += t_switch
                        plan.n_switches += 1
                else:
                    link = sequencer.link_between(prev_end, run.tcp_start)
                    if link is not None:
                        plan.travel_time += link.length / cutter.rapid_speed
            if needs_pierce:
                plan.pierce_time += pierce
                plan.n_pierces += 1
            prev_end = run.tcp_end
            prev_v = v

    plan.total_time = (plan.cut_time + plan.travel_time
                       + plan.pierce_time + plan.switch_time)
    return plan

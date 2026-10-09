"""Surrogate planner and Automatic Planner.

``surrogate_plan(grid, model)``
  1. features -> model p(s) -> segments sorted by p(s)
  2. Greedy Set Cover in this order on exact singleton masks
     (lazy, only touched segments)
  3. optional pruning of redundant segments
  4. one plan construction (DP split + Held-Karp), one exact verify
  5. reachable points missing -> classic fallback (greedy selection)

  The model only sets the order; coverage never depends on the model.

``greedy_plus_plan(grid)``
  Greedy selection of the ``AutoPlanner`` with the same speed stage
  (DP split) as the surrogate and the teacher -> fair comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

try:
    from ...geometry.point_grid import PointGrid
    from ..segments import SegmentedContour, compute_grid_coverage
    from ..planning import LinkPlanner, RunKinematics
    from ..autoplan import AutoPlanner
    from .features import segment_features, phys_from_cutter
    from .instances import default_cutter
    from .params import KERF
    from .runutils import (
        ExactSequencer, build_plan_with_speeds, SpeedPlan,
        build_speed_chains, build_singletons,
    )
except ImportError:  # direct run without package context
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.geometry.point_grid import PointGrid
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, compute_grid_coverage,
    )
    from plasma_cutter.segment_simulation.planning import (
        LinkPlanner, RunKinematics,
    )
    from plasma_cutter.segment_simulation.autoplan import AutoPlanner
    from plasma_cutter.segment_simulation.surrogate.features import (
        segment_features, phys_from_cutter,
    )
    from plasma_cutter.segment_simulation.surrogate.instances import (
        default_cutter,
    )
    from plasma_cutter.segment_simulation.surrogate.params import KERF
    from plasma_cutter.segment_simulation.surrogate.runutils import (
        ExactSequencer, build_plan_with_speeds, SpeedPlan,
        build_speed_chains, build_singletons,
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SurrogatePlanResult:
    """Result of ``surrogate_plan``.

    runs          : ordered CutRuns of the plan
    plan          : SpeedPlan (time breakdown, variable speeds)
    selected      : selected seg_ids (after pruning, or those of the fallback)
    coverage      : exact coverage [0..1]
    T             : execution time [s]
    t_plan        : total planning time [s]
    n_pruned      : pruned (redundant) segments
    used_fallback : True = greedy selection adopted
    n_unreachable : unreachable grid points (also for greedy)
    n_candidates  : primitive segments
    n_selected    : selected segments
    timings       : stage times [s]
    """
    runs: list = field(default_factory=list)
    plan: SpeedPlan = field(default_factory=SpeedPlan)
    selected: list[int] = field(default_factory=list)
    coverage: float = 0.0
    T: float = 0.0
    t_plan: float = 0.0
    n_pruned: int = 0
    used_fallback: bool = False
    n_unreachable: int = 0
    n_candidates: int = 0
    n_selected: int = 0
    timings: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Short overview incl. stage times [ms]."""
        tg = self.timings
        stages = " ".join(f"{k}={tg.get(k, 0) * 1e3:.1f}ms"
                          for k in ("features", "predict", "masks", "prune",
                                    "speedup", "sequence", "verify", "repair"))
        return (
            f"Surrogat: {self.n_selected}/{self.n_candidates} Segmente -> "
            f"{len(self.runs)} Schnitt(e), Coverage {self.coverage:.1%}"
            + (f" ({self.n_unreachable} unerreichbar)" if self.n_unreachable else "")
            + f", T={self.T:.1f}s, geplant in {self.t_plan * 1e3:.1f}ms"
            + (" [FALLBACK]" if self.used_fallback else "")
            + f", gepruned={self.n_pruned}\n  Stufen: {stages}"
        )


# ---------------------------------------------------------------------------
# Robust classic fallback (crash-free against LinkInfeasibleError)
# ---------------------------------------------------------------------------

def _fallback_plan(grid: PointGrid, contour, material, cutter, phys, kerf):
    """Classic greedy selection of the ``AutoPlanner`` at v_cut.

    - robust against non-linkable runs
    - also the reference for genuine reachability
    - returns: (runs, mask, SpeedPlan, selected)
    """
    blade = cutter.blade_length(cutter.cutting_speed)
    kin = RunKinematics(material, clearance=phys.gap,
                        blade_length=blade, kerf=kerf)
    link = LinkPlanner(material, clearance=phys.gap)
    seq = ExactSequencer(cutter, contour, link)
    selected: list[int] = []
    try:
        ap = AutoPlanner(grid, contour, cutter, kin, seq)
        selected = sorted(ap._prune(ap._greedy()))
        runs = [r for r in ap._merge_to_runs(selected)
                if r.is_feasible and r.swept_polygon is not None]
        run_speed = {r.run_id: cutter.cutting_speed for r in runs}  # all v_cut
        plan = build_plan_with_speeds(seq, runs, run_speed, cutter,
                                      drop_unlinkable=True)
    except Exception:
        # invalid geometry -> empty reference instead of crash
        plan = SpeedPlan()
    kept = plan.ordered_runs  # linkable runs
    mask = compute_grid_coverage(grid, kept).mask  # reachable points
    return kept, mask, plan, selected


# ---------------------------------------------------------------------------
# Automatic Planner: greedy selection + speed rule
# ---------------------------------------------------------------------------

def greedy_plus_plan(grid: PointGrid, cutter=None, kerf: float = KERF,
                     contour: SegmentedContour | None = None):
    """Greedy selection of the ``AutoPlanner`` + shared DP split stage.

    Returns: dict with
      T         execution time [s] (None if the selection crashes)
      coverage  exact coverage
      n_runs    number of runs
      t_plan    total planning time [s]
      timings   stages [s]: candidates (swept areas), select (greedy +
                pruning), sequence (Held-Karp + LinkPlanner), speedup
                (DP split), verify (exact coverage)

    contour: given segmentation, else ``from_grid``.
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {"candidates": 0.0, "select": 0.0,
                                 "sequence": 0.0, "speedup": 0.0,
                                 "verify": 0.0}

    def _fail():
        return {"T": None, "coverage": 0.0, "n_runs": 0,
                "t_plan": time.perf_counter() - t_start, "timings": timings}

    if cutter is None:
        cutter = default_cutter()
    phys = phys_from_cutter(cutter)
    try:
        if contour is None:
            contour = SegmentedContour.from_grid(grid)
        material = contour.material_polygon()
    except Exception:
        return _fail()
    if material is None:
        return _fail()

    coords = np.asarray(grid.coords, dtype=float)

    # classic greedy selection + base plan as reference
    blade = cutter.blade_length(cutter.cutting_speed)
    kin = RunKinematics(material, clearance=phys.gap,
                        blade_length=blade, kerf=kerf)
    seq = ExactSequencer(cutter, contour,
                         LinkPlanner(material, clearance=phys.gap))
    try:
        t0 = time.perf_counter()
        ap = AutoPlanner(grid, contour, cutter, kin, seq)
        timings["candidates"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        selected = ap._prune(ap._greedy())
        timings["select"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        base_runs = [r for r in ap._merge_to_runs(selected)
                     if r.is_feasible and r.swept_polygon is not None]
        base_speed = {r.run_id: cutter.cutting_speed for r in base_runs}
        base_plan = build_plan_with_speeds(seq, base_runs, base_speed,
                                           cutter, drop_unlinkable=True)
        timings["sequence"] += time.perf_counter() - t0
    except Exception:
        return _fail()
    if not base_plan.ordered_runs:
        return _fail()
    t0 = time.perf_counter()
    base_mask = compute_grid_coverage(grid, base_plan.ordered_runs).mask
    timings["verify"] += time.perf_counter() - t0

    # shared DP split stage (identical to teacher and surrogate)
    try:
        t0 = time.perf_counter()
        chains = build_speed_chains(contour, selected, material, phys,
                                    kerf, coords)
        timings["speedup"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        plan = build_plan_with_speeds(seq, chains, {}, cutter,
                                      drop_unlinkable=True,
                                      t_switch=phys.t_switch)
        timings["sequence"] += time.perf_counter() - t0
        if plan.ordered_runs:
            t0 = time.perf_counter()
            mask = compute_grid_coverage(grid, plan.ordered_runs).mask
            timings["verify"] += time.perf_counter() - t0
            # no points lost and not slower than the base plan
            if (bool(np.all(base_mask <= mask))
                    and plan.total_time <= base_plan.total_time + 1e-9):
                return {"T": plan.total_time, "coverage": float(mask.mean()),
                        "n_runs": len(plan.ordered_runs),
                        "t_plan": time.perf_counter() - t_start,
                        "timings": timings}
    except Exception:
        pass
    return {"T": base_plan.total_time, "coverage": float(base_mask.mean()),
            "n_runs": len(base_plan.ordered_runs),
            "t_plan": time.perf_counter() - t_start, "timings": timings}


# ---------------------------------------------------------------------------
# Core: surrogate_plan
# ---------------------------------------------------------------------------

def surrogate_plan(
    grid: PointGrid,
    model,
    cutter=None,
    kerf: float = KERF,
    prune: bool = True,
    proba_override=None,
    contour: SegmentedContour | None = None,
    speed_rule: bool = True,
) -> SurrogatePlanResult:
    """Surrogate planner (flow: see module header).

    prune          : False -> selection of the Greedy Set Cover unchanged
    proba_override : replaces the model output (ablation)
    contour        : segmentation of the seg_ids (simulator: its contour)
    speed_rule     : False -> v_max = v_cut for plan construction and fallback;
                     features still use the training parameters
    """
    t_start = time.perf_counter()
    timings = {"features": 0.0, "predict": 0.0, "masks": 0.0, "prune": 0.0,
               "speedup": 0.0, "sequence": 0.0, "verify": 0.0, "repair": 0.0}
    if cutter is None:
        cutter = default_cutter()
    phys = phys_from_cutter(cutter)
    phys_plan = phys if speed_rule else replace(phys, v_max=phys.v_cut)
    if contour is None:
        contour = SegmentedContour.from_grid(grid)
    material = contour.material_polygon()
    n_seg = len(contour.segments)
    coords = np.asarray(grid.coords, dtype=float)
    n_pts = len(coords)
    result = SurrogatePlanResult(n_candidates=n_seg, timings=timings)

    if n_seg == 0 or material is None:
        result.t_plan = time.perf_counter() - t_start
        return result

    # A: features + model ranking
    t0 = time.perf_counter()
    X = segment_features(grid, contour, cutter, phys=phys)
    timings["features"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = (np.asarray(proba_override, dtype=float) if proba_override is not None
             else np.asarray(model.predict_proba(X), dtype=float))
    order = np.argsort(-proba)
    timings["predict"] = time.perf_counter() - t0

    # B: Greedy Set Cover in p(s) order, masks lazy
    t0 = time.perf_counter()
    seg_run, seg_mask = {}, {}
    covered = np.zeros(n_pts, dtype=bool)
    selected: set[int] = set()
    for s in order:
        if covered.all():
            break
        s = int(s)
        r1, m1 = build_singletons(contour, [s], material, phys_plan,
                                  phys_plan.v_cut, kerf, coords)
        seg_run.update(r1)
        seg_mask.update(m1)
        if bool((seg_mask[s] & ~covered).any()):  # new points?
            selected.add(s)
            covered |= seg_mask[s]
    timings["masks"] = time.perf_counter() - t0

    # C: pruning: drop a segment if all its points are covered at least twice;
    #    smallest p(s) first
    t0 = time.perf_counter()
    n_pruned = 0
    if prune and len(selected) > 1:
        cover_count = np.zeros(n_pts, dtype=int)
        for s in selected:
            cover_count += seg_mask[s]
        for s in sorted(selected, key=lambda i: float(proba[i])):
            col = seg_mask[s]
            if bool(col.any()) and bool(np.all(cover_count[col] >= 2)):
                selected.discard(s)
                cover_count[col] -= 1
                n_pruned += 1
    timings["prune"] = time.perf_counter() - t0
    result.n_pruned = n_pruned

    # D: one plan construction (singleton runs reused) + sequencing
    seq = ExactSequencer(cutter, contour,
                         LinkPlanner(material, clearance=phys_plan.gap))
    t0 = time.perf_counter()
    chains = build_speed_chains(contour, selected, material, phys_plan, kerf,
                                coords, seg_run=seg_run, seg_mask=seg_mask,
                                t_switch=phys_plan.t_switch)
    timings["speedup"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    plan = build_plan_with_speeds(seq, chains, {}, cutter,
                                  drop_unlinkable=True,
                                  t_switch=phys_plan.t_switch)
    timings["sequence"] = time.perf_counter() - t0

    # E: exact verify
    t0 = time.perf_counter()
    mask = compute_grid_coverage(grid, plan.ordered_runs).mask
    timings["verify"] = time.perf_counter() - t0

    # F: fallback if reachable points are missing
    used_fallback = False
    n_unreachable = 0
    if not mask.all():
        t0 = time.perf_counter()
        fb_runs, fb_mask, fb_plan, fb_sel = _fallback_plan(
            grid, contour, material, cutter, phys_plan, kerf)
        if int(((~mask) & fb_mask).sum()) > 0:
            used_fallback = True
            plan, mask, selected = fb_plan, fb_mask, set(fb_sel)
        n_unreachable = int(np.count_nonzero(~mask))
        timings["repair"] = time.perf_counter() - t0

    result.runs = plan.ordered_runs
    result.plan = plan
    result.selected = sorted(int(s) for s in selected)
    result.coverage = float(mask.mean()) if n_pts else 0.0
    result.T = plan.total_time
    result.used_fallback = used_fallback
    result.n_unreachable = n_unreachable
    result.n_selected = len(selected)
    result.t_plan = time.perf_counter() - t_start
    return result

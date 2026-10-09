"""Dominance: T_teacher <= T_AutomaticPlanner <= T_Greedy.

- all planners share the speed stage (DP split + t_switch), so the
  teacher must dominate each of them
- Automatic Planner (Greedy + DP split) never slower than Greedy (v_cut)
- comparison only at equal coverage; instances above ``K_MAX`` segments
  are skipped
"""

from __future__ import annotations

import math

import pytest

from plasma_cutter.segment_simulation.autoplan import auto_plan
from plasma_cutter.segment_simulation.surrogate.instances import (
    generate_instances, make_catalog_instance,
)
from plasma_cutter.segment_simulation.surrogate.planner import greedy_plus_plan
from plasma_cutter.segment_simulation.surrogate.teacher import (
    TeacherSkipped, exhaustive_plan,
)
from ._helpers import real_geometries, small_flat_bar

EPS = 1e-6
K_MAX = 13     # skip larger instances (runtime)


def _cases():
    yield "small_flat_bar", small_flat_bar()
    for idx in range(3):
        g = make_catalog_instance(idx, seed=11)
        if g is not None:
            yield f"cat_{idx}", g
    for g in real_geometries():
        name = (g._source_path.stem if hasattr(g, "_source_path")
                else "real")
        yield f"real_{name}", g
    # one benchmark instance (seed 7), historical regression case
    ib = next((g for g in generate_instances(30, 7)
               if getattr(g, "_instance_name", "") == "ibeam_00002"), None)
    if ib is not None:
        yield "ibeam_00002_seed7", ib


@pytest.mark.slow
@pytest.mark.parametrize("name,grid", list(_cases()))
def test_teacher_dominates_greedy_plus(name, grid):
    try:
        tr = exhaustive_plan(grid, k_max=K_MAX, keep_covers=False)
    except TeacherSkipped:
        pytest.skip(f"{name}: zu viele Segmente fuer den Brute-Force-Lehrer")
    if not tr.selected:
        pytest.skip(f"{name}: Lehrer fand keine gueltige Auswahl")

    gp = greedy_plus_plan(grid)
    if gp["T"] is None:
        pytest.skip(f"{name}: Automatic Planner nicht verfuegbar")

    # Compare only on the same task (identical achieved coverage).
    if not math.isclose(tr.coverage, gp["coverage"], abs_tol=1e-6):
        pytest.skip(f"{name}: Coverage differiert "
                    f"({tr.coverage:.4f} vs {gp['coverage']:.4f})")

    assert tr.total_time <= gp["T"] + EPS, (
        f"{name}: Lehrer ({tr.total_time:.3f}s) schlechter als Automatic Planner "
        f"({gp['T']:.3f}s) -- Zielfunktions-Mismatch?")


@pytest.mark.parametrize("name,grid", list(_cases()))
def test_greedy_plus_not_worse_than_greedy(name, grid):
    """The speed rule may only make Greedy faster."""
    try:
        res = auto_plan(grid)
    except Exception:
        pytest.skip(f"{name}: Greedy nicht planbar")
    gp = greedy_plus_plan(grid)
    if gp["T"] is None:
        pytest.skip(f"{name}: Automatic Planner nicht verfuegbar")
    assert gp["T"] <= res.plan.total_time + EPS, (
        f"{name}: Automatic Planner ({gp['T']:.3f}s) langsamer als Greedy "
        f"({res.plan.total_time:.3f}s)")

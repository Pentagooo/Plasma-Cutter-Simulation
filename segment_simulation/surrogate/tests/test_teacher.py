"""Teacher test: Brute Force <= Greedy on a small geometry.

The teacher optimizes selection x order x direction + speed rule,
so it must never be worse than the classic Greedy planner (v_cut).
"""

from __future__ import annotations

from plasma_cutter.segment_simulation.segments import SegmentedContour
from plasma_cutter.segment_simulation.autoplan import auto_plan
from plasma_cutter.segment_simulation.surrogate.teacher import (
    TeacherSkipped, exhaustive_plan,
)
from ._helpers import small_flat_bar


def test_teacher_not_worse_than_greedy():
    grid = small_flat_bar()
    contour = SegmentedContour.from_grid(grid)
    assert len(contour.segments) <= 15

    tr = exhaustive_plan(grid, k_max=15, keep_covers=False)
    greedy = auto_plan(grid)

    # Both reach 100 % (small solid part, everything reachable)
    assert tr.coverage >= 0.999
    assert greedy.report.fraction >= 0.999

    # The teacher is optimal in the decision space -> T_opt <= T_greedy
    assert tr.total_time <= greedy.plan.total_time + 1e-6, (
        f"Lehrer T={tr.total_time:.3f} > Greedy T={greedy.plan.total_time:.3f}")
    assert tr.selected, "Lehrer hat kein Subset gewaehlt"
    # all subsets enumerated, the selection is one of the covers
    assert tr.n_covers >= 1
    assert tr.n_subsets == (1 << tr.n_feasible) - 1


def test_teacher_speed_rule_off_not_faster():
    """Speed rule off (v_max = v_cut) = subspace -> optimum can only be
    equal or slower.
    """
    grid = small_flat_bar()
    on = exhaustive_plan(grid, k_max=15, keep_covers=False, speed_rule=True)
    off = exhaustive_plan(grid, k_max=15, keep_covers=False, speed_rule=False)
    assert on.total_time <= off.total_time + 1e-6


def test_teacher_skips_large():
    # Artificially small k_max -> TeacherSkipped
    grid = small_flat_bar()
    try:
        exhaustive_plan(grid, k_max=1)
    except TeacherSkipped:
        return
    raise AssertionError("TeacherSkipped wurde nicht ausgeloest")

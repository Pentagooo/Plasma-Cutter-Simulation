"""Lehrer-Test: Brute Force <= Greedy auf einer kleinen Geometrie.

Der Lehrer optimiert Auswahl x Reihenfolge x Richtung + Geschwindigkeitsregel,
darf also nie schlechter sein als der klassische Greedy-Planer (v_cut).
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

    # Beide erreichen 100 % (kleiner Vollkörper, alles erreichbar)
    assert tr.coverage >= 0.999
    assert greedy.report.fraction >= 0.999

    # Der Lehrer ist im Entscheidungsraum optimal -> T_opt <= T_greedy
    assert tr.total_time <= greedy.plan.total_time + 1e-6, (
        f"Lehrer T={tr.total_time:.3f} > Greedy T={greedy.plan.total_time:.3f}")
    assert tr.selected, "Lehrer hat kein Subset gewaehlt"
    # alle Teilmengen aufgezählt, die Auswahl ist eine der Abdeckungen
    assert tr.n_covers >= 1
    assert tr.n_subsets == (1 << tr.n_feasible) - 1


def test_teacher_speed_rule_off_not_faster():
    """Geschwindigkeitsregel aus (v_max = v_cut) = Teilraum -> Optimum nur
    gleich oder langsamer.
    """
    grid = small_flat_bar()
    on = exhaustive_plan(grid, k_max=15, keep_covers=False, speed_rule=True)
    off = exhaustive_plan(grid, k_max=15, keep_covers=False, speed_rule=False)
    assert on.total_time <= off.total_time + 1e-6


def test_teacher_skips_large():
    # Künstlich kleines k_max -> TeacherSkipped
    grid = small_flat_bar()
    try:
        exhaustive_plan(grid, k_max=1)
    except TeacherSkipped:
        return
    raise AssertionError("TeacherSkipped wurde nicht ausgeloest")

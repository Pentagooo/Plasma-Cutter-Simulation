"""Dominanz-Test: T_Lehrer <= T_Greedy+ <= T_Greedy (konsistente Zielfunktion).

Die Geschwindigkeitszuweisung (Kap. 4.5 + t_switch, DP-Split je Kette)
ist Teil des GEMEINSAMEN Entscheidungsraums aller Planer. Der
Brute-Force-Lehrer bewertet jede vollstaendige Abdeckung mit genau
dieser Pipeline und muss deshalb jeden klassischen Planer dominieren;
Greedy+ (Greedy-Auswahl + DP-Split) darf nie langsamer sein als Greedy
(Greedy-Auswahl bei Basisgeschwindigkeit).

Toleranz: kleine Epsilons fuer Gleitkomma; Vergleich nur bei identischer
Coverage (sonst vergleichen wir verschiedene Aufgaben). Instanzen ueber
``K_MAX`` Segmenten werden uebersprungen (Lehrerzeit waechst ~2^k).
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
K_MAX = 13     # ~1-2 min fuer den ganzen Test; groessere Instanzen skippen


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
    # eine Benchmark-Instanz (Seed 7), historischer Regressionsfall
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
        pytest.skip(f"{name}: Greedy+ nicht verfuegbar")

    # Nur bei gleicher Aufgabe vergleichen (identische erreichte Coverage).
    if not math.isclose(tr.coverage, gp["coverage"], abs_tol=1e-6):
        pytest.skip(f"{name}: Coverage differiert "
                    f"({tr.coverage:.4f} vs {gp['coverage']:.4f})")

    assert tr.total_time <= gp["T"] + EPS, (
        f"{name}: Lehrer ({tr.total_time:.3f}s) schlechter als Greedy+ "
        f"({gp['T']:.3f}s) -- Zielfunktions-Mismatch?")


@pytest.mark.parametrize("name,grid", list(_cases()))
def test_greedy_plus_not_worse_than_greedy(name, grid):
    """Die Geschwindigkeitsregel darf Greedy nur schneller machen."""
    try:
        res = auto_plan(grid)
    except Exception:
        pytest.skip(f"{name}: Greedy nicht planbar")
    gp = greedy_plus_plan(grid)
    if gp["T"] is None:
        pytest.skip(f"{name}: Greedy+ nicht verfuegbar")
    assert gp["T"] <= res.plan.total_time + EPS, (
        f"{name}: Greedy+ ({gp['T']:.3f}s) langsamer als Greedy "
        f"({res.plan.total_time:.3f}s)")

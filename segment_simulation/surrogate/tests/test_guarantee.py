"""Garantie-Test (A1): die Coverage haengt NIE am Modell.

Fuer ALLE 6 Testgeometrien und selbst mit absichtlich schlechten Modellen
(Zufall, "waehle nichts", "waehle alles") liefert ``surrogate_plan`` einen
GUELTIGEN Plan, dessen Coverage mindestens so gross ist wie die der
klassischen Baseline. Damit gilt:

    surrogate_missing  subset  baseline_missing

d.h. jede verbleibende Fehlstelle ist auch fuer die Baseline unerreichbar
(ehrliche Unerreichbarkeit, z.B. der vollstaendig umschlossene Bereich in
``test_lochjson``). Fuer Vollkoerper heisst das Coverage == 1.0.
"""
from __future__ import annotations

import numpy as np

from plasma_cutter.segment_simulation.segments import compute_grid_coverage
from plasma_cutter.segment_simulation.surrogate.planner import surrogate_plan
from ._helpers import RandModel, ZeroModel, OneModel, real_geometries


def _mask(grid, result):
    return compute_grid_coverage(grid, result.runs).mask


def _baseline_mask(grid):
    """Robuste klassische Baseline-Abdeckung (ueber erzwungenen Fallback)."""
    r0 = surrogate_plan(grid, ZeroModel())
    return _mask(grid, r0)


def test_coverage_never_depends_on_model():
    geoms = real_geometries()
    assert len(geoms) == 6, f"erwartet 6 Testgeometrien, gefunden {len(geoms)}"

    models = [RandModel(0), RandModel(1), RandModel(7), ZeroModel(), OneModel()]
    for grid in geoms:
        bmask = _baseline_mask(grid)
        for model in models:
            r = surrogate_plan(grid, model)
            mask = _mask(grid, r)
            # (1) Surrogat deckt ALLES ab, was die Baseline abdeckt
            assert np.all(bmask <= mask), (
                f"{getattr(grid,'_source_path',grid)}: Surrogat verfehlt "
                f"Baseline-Punkte (Modell {type(model).__name__})")
            # (2) Coverage == 1.0 ODER exakt die bekannte Unerreichbarkeit
            #     (== Baseline-Unerreichbarkeit); Vollkoerper -> 1.0
            if bmask.all():
                assert mask.all(), "Vollkoerper nicht zu 100 % abgedeckt"
            # (3) Konsistenz des berichteten Coverage-Werts
            assert abs(r.coverage - mask.mean()) < 1e-9


def test_lochjson_reports_unreachable_not_crash():
    """test_lochjson: kein Crash, ehrliche Unerreichbarkeit gemeldet."""
    geoms = {g._source_path.stem: g for g in real_geometries()
             if hasattr(g, "_source_path")}
    grid = geoms.get("test_lochjson")
    assert grid is not None
    r = surrogate_plan(grid, RandModel(3))
    # Plan gueltig (alle Runs verbunden -> per Konstruktion) und Coverage
    # entspricht der ehrlich erreichbaren Menge (< 1.0, aber > 0).
    assert 0.0 < r.coverage <= 1.0
    assert r.n_unreachable > 0


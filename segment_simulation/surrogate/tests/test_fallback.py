"""Fallback-Test: ein Modell, das nichts selektiert, muss die Coverage
ueber Repair ODER Fallback trotzdem herstellen.
"""
from __future__ import annotations

import numpy as np

from plasma_cutter.segment_simulation.segments import compute_grid_coverage
from plasma_cutter.segment_simulation.surrogate.planner import surrogate_plan
from ._helpers import ZeroModel, small_flat_bar, real_geometries


def test_zero_model_recovers_coverage():
    # Vollkoerper: auch ein nutzloses Modell (p(s)=0) muss 100 % liefern.
    # Die 'ranked cover-to-completion'-Auswahl deckt unabhaengig vom Modell
    # vollstaendig ab; falls doch Luecken bleiben, schliessen Repair/Fallback
    # sie. Entscheidend ist die COVERAGE, nicht welcher Mechanismus griff.
    grid = small_flat_bar()
    r = surrogate_plan(grid, ZeroModel())
    mask = compute_grid_coverage(grid, r.runs).mask
    assert mask.all(), f"Coverage nur {mask.mean():.3f}"


def test_zero_model_on_all_geometries_full_or_honest():
    # Ein nutzloses Modell liefert auf jeder Geometrie einen gueltigen Plan;
    # Vollkoerper -> 1.0, sonst ehrlich unerreichbare Restmenge.
    for grid in real_geometries():
        r = surrogate_plan(grid, ZeroModel())
        assert r.coverage > 0.0
        mask = compute_grid_coverage(grid, r.runs).mask
        assert abs(r.coverage - mask.mean()) < 1e-9


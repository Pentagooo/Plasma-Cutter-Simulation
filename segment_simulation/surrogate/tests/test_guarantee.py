"""Guarantee: coverage never depends on the model.

On all test geometries and with deliberately bad models (random,
nothing, everything), ``surrogate_plan`` returns a valid plan with at least
the coverage of the classic baseline:

    surrogate_missing  subset of  baseline_missing

Solid part -> coverage 1.0; gaps only where the baseline cannot reach either
(e.g. the enclosed region in ``test_lochjson``).
"""

from __future__ import annotations

import numpy as np

from plasma_cutter.segment_simulation.segments import compute_grid_coverage
from plasma_cutter.segment_simulation.surrogate.planner import surrogate_plan
from ._helpers import (
    RandModel, ZeroModel, OneModel, real_geometries, small_flat_bar,
)


def _mask(grid, result):
    return compute_grid_coverage(grid, result.runs).mask


def test_zero_model_covers_full_body():
    # useless model (p(s) = 0) on a solid part -> still 100 %
    grid = small_flat_bar()
    mask = _mask(grid, surrogate_plan(grid, ZeroModel()))
    assert mask.all(), f"Coverage nur {mask.mean():.3f}"


def _baseline_mask(grid):
    """Robust classic baseline coverage (via forced fallback)."""
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
            # (1) surrogate covers EVERYTHING the baseline covers
            assert np.all(bmask <= mask), (
                f"{getattr(grid,'_source_path',grid)}: Surrogat verfehlt "
                f"Baseline-Punkte (Modell {type(model).__name__})")
            # (2) coverage == 1.0 OR exactly the known unreachability
            #     (== baseline unreachability); solid part -> 1.0
            if bmask.all():
                assert mask.all(), "Vollkoerper nicht zu 100 % abgedeckt"
            # (3) consistency of the reported coverage value
            assert abs(r.coverage - mask.mean()) < 1e-9


def test_lochjson_reports_unreachable_not_crash():
    """test_lochjson: no crash, genuine unreachability reported."""
    geoms = {g._source_path.stem: g for g in real_geometries()
             if hasattr(g, "_source_path")}
    grid = geoms.get("test_lochjson")
    assert grid is not None
    r = surrogate_plan(grid, RandModel(3))
    # Plan valid (all runs linked -> by construction) and coverage
    # equals the genuinely reachable set (< 1.0, but > 0).
    assert 0.0 < r.coverage <= 1.0
    assert r.n_unreachable > 0


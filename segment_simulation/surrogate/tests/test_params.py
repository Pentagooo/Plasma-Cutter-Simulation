"""Parameterstempel: deterministisch, sensitiv fuer Physik und Segmentierung."""
from __future__ import annotations

from dataclasses import replace

from plasma_cutter.segment_simulation.segments import SegmentedContour
from plasma_cutter.segment_simulation.simulation import make_default_cutter
from plasma_cutter.segment_simulation.surrogate.dataset import cache_key
from plasma_cutter.segment_simulation.surrogate.params import (
    LABEL_VERSION, label_params, make_contour, params_hash, phys_hash, stamp,
)
from ._helpers import small_flat_bar


def test_hash_deterministic():
    a, b = label_params(), label_params()
    assert a == b
    assert phys_hash(a) == phys_hash(b) and params_hash(a) == params_hash(b)
    assert len(phys_hash(a)) == 8 and len(params_hash(a)) == 8


def test_physics_change_changes_phys_hash():
    base = label_params()
    other = label_params(cutter=make_default_cutter(speed_switch_time=0.5))
    assert other.t_switch == 0.5
    assert phys_hash(other) != phys_hash(base)
    assert params_hash(other) != params_hash(base)


def test_segmentation_change_changes_only_params_hash():
    base = label_params()
    fine = label_params(seg_divisor=24.0, seg_min_spacings=3.0)
    assert phys_hash(fine) == phys_hash(base)
    assert params_hash(fine) != params_hash(base)
    assert cache_key(("cat", 0, 42), base) != cache_key(("cat", 0, 42), fine)
    assert cache_key(("cat", 0, 42), base).endswith("_exact")
    assert f"_L{LABEL_VERSION}_" in cache_key(("cat", 0, 42), base)


def test_float_noise_does_not_change_hash():
    base = label_params()
    noisy = replace(base, blade0=base.blade0 + 1e-12)
    assert phys_hash(noisy) == phys_hash(base)


def test_stamp_roundtrip():
    p = label_params()
    st = stamp(p)
    assert st["label_version"] == LABEL_VERSION
    assert st["phys_hash"] == phys_hash(p)
    assert st["params"]["seg_divisor"] == 12.0


def test_make_contour_matches_default_and_fine():
    grid = small_flat_bar()
    default = SegmentedContour.from_grid(grid)
    assert make_contour(grid, label_params()).nodes == default.nodes
    fine = make_contour(grid, label_params(seg_divisor=24.0, seg_min_spacings=3.0))
    assert len(fine.segments) > len(default.segments)

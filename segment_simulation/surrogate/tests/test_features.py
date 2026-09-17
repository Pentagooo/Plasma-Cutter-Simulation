"""Feature-Tests: deterministisch, keine NaN, Spaltenzahl == FEATURE_NAMES."""
from __future__ import annotations

import numpy as np

from plasma_cutter.segment_simulation.segments import SegmentedContour
from plasma_cutter.segment_simulation.surrogate.features import (
    segment_features, FEATURE_NAMES, N_FEATURES,
)
from ._helpers import small_flat_bar, real_geometries


def _grids():
    return [small_flat_bar()] + real_geometries()


def test_feature_count_matches_names():
    grid = small_flat_bar()
    contour = SegmentedContour.from_grid(grid)
    X = segment_features(grid, contour)
    assert X.shape[1] == len(FEATURE_NAMES) == N_FEATURES
    assert X.shape[0] == len(contour.segments)


def test_features_deterministic():
    for grid in _grids():
        contour = SegmentedContour.from_grid(grid)
        X1 = segment_features(grid, contour)
        X2 = segment_features(grid, contour)
        assert np.array_equal(X1, X2), "Features nicht deterministisch"


def test_features_no_nan_or_inf():
    for grid in _grids():
        contour = SegmentedContour.from_grid(grid)
        X = segment_features(grid, contour)
        assert np.isfinite(X).all(), "Features enthalten NaN/Inf"


def test_features_shapes_and_ranges():
    grid = small_flat_bar()
    contour = SegmentedContour.from_grid(grid)
    X = segment_features(grid, contour)
    names = FEATURE_NAMES
    # is_outer ist 0/1
    io = X[:, names.index("is_outer")]
    assert set(np.unique(io)).issubset({0.0, 1.0})
    # v_hat > 0 und <= v_max-Grenze; time_est >= 0
    assert (X[:, names.index("v_hat")] > 0).all()
    assert (X[:, names.index("time_est")] >= 0).all()
    # Bogenlaenge > 0
    assert (X[:, names.index("arc_length")] > 0).all()


if __name__ == "__main__":
    test_feature_count_matches_names()
    test_features_deterministic()
    test_features_no_nan_or_inf()
    test_features_shapes_and_ranges()
    print("test_features: OK")

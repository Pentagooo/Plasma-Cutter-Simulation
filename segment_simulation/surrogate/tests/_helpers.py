"""Gemeinsame Test-Hilfen: Zufalls-/Konstant-Modelle, kleine Geometrien."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from plasma_cutter.geometry.point_grid import PointGrid
from plasma_cutter.segment_simulation.surrogate.instances import (
    grid_from_polygon, TEST_GEOMETRY_DIR,
)
from shapely.geometry import box


class RandModel:
    """Absichtlich schlechtes Modell: rein zufaellige p(s)."""
    def __init__(self, seed: int = 0, tau: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.tau = tau

    def predict_proba(self, X):
        return self.rng.random(len(X))


class ZeroModel:
    """Waehlt NICHTS (p(s)=0) -> erzwingt Repair/Fallback."""
    tau = 0.5

    def predict_proba(self, X):
        return np.zeros(len(X))


class OneModel:
    """Waehlt ALLES (p(s)=1)."""
    tau = 0.5

    def predict_proba(self, X):
        return np.ones(len(X))


def small_flat_bar() -> PointGrid:
    """Kleiner Flachstahl (wenige Segmente) fuer den Lehrer-Test."""
    return grid_from_polygon(box(-40.0, -6.0, 40.0, 6.0))


def real_geometries() -> list[PointGrid]:
    """Die geprueften Testgeometrien."""
    out = []
    for p in sorted(Path(TEST_GEOMETRY_DIR).glob("*.json")):
        out.append(PointGrid.from_json(p))
    return out

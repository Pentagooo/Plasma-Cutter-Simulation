from __future__ import annotations

import json
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class PointStatus(Enum):
    """Classification of a grid point.

    OUTER : outer boundary of the geometry (accessible for torch piercing).
    HOLE  : inner boundary (hole contour) – also a boundary, but inside the
            profile (bores/cutouts).
    INNER : volume point in the material interior.
    """
    OUTER = "outer"   # outer boundary (entry allowed)
    HOLE  = "hole"    # inner boundary (hole contour) – hole boundary point
    INNER = "inner"   # material interior


@dataclass
class GridPoint:
    """A point in the point grid with position and status.

    Parameters
    ----------
    x, y   : position [mm]
    status : OUTER (boundary) or INNER (interior)
    index  : unique index within the PointGrid
    """
    x: float
    y: float
    status: PointStatus
    index: int

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def is_outer(self) -> bool:
        return self.status == PointStatus.OUTER

    @property
    def is_hole(self) -> bool:
        """True if the point lies on a hole contour."""
        return self.status == PointStatus.HOLE

    @property
    def is_inner(self) -> bool:
        return self.status == PointStatus.INNER

    @property
    def is_boundary(self) -> bool:
        """True for outer AND hole boundary points (all contour points)."""
        return self.status in (PointStatus.OUTER, PointStatus.HOLE)

    def __repr__(self) -> str:
        return (f"GridPoint(idx={self.index}, "
                f"x={self.x:.2f}, y={self.y:.2f}, "
                f"{self.status.value})")


# ---------------------------------------------------------------------------
# PointGrid
# ---------------------------------------------------------------------------

class PointGrid:
    """Collection of grid points with outer/inner classification.
    """

    # Default path to the folder with checked geometries
    DEFAULT_DIR: Path = Path(__file__).parent / "Geometrie_Konturen_geprüft"

    # Internal status encoding (int8)
    _STATUS_OUTER: int = 0
    _STATUS_INNER: int = 1
    _STATUS_HOLE:  int = 2

    def __init__(
        self,
        coords: np.ndarray,
        status: np.ndarray,
        point_spacing: float,
        contour_spacing: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        coords        : (N, 2) float64 array of point positions
        status        : (N,) int8 array (0=OUTER, 1=INNER, 2=HOLE)
        point_spacing : characteristic spacing [mm]
        """
        self._coords = np.asarray(coords, dtype=np.float64)
        self._status = np.asarray(status, dtype=np.int8)
        self.point_spacing = float(point_spacing)
        self.contour_spacing: float = float(contour_spacing) if contour_spacing else point_spacing * 2.0

        # GridPoint cache, filled on demand
        self._grid_point_cache: list[GridPoint | None] = [None] * len(self._coords)

    @property
    def points(self) -> list[GridPoint]:
        """All points as a GridPoint list (slow for many points)."""
        return [self._get_grid_point(i) for i in range(len(self._coords))]

    def _get_grid_point(self, i: int) -> GridPoint:
        if self._grid_point_cache[i] is None:
            s = int(self._status[i])
            if s == self._STATUS_OUTER:
                status = PointStatus.OUTER
            elif s == self._STATUS_HOLE:
                status = PointStatus.HOLE
            else:
                status = PointStatus.INNER
            self._grid_point_cache[i] = GridPoint(
                x=float(self._coords[i, 0]),
                y=float(self._coords[i, 1]),
                status=status,
                index=i
            )
        return self._grid_point_cache[i]

    @property
    def coords(self) -> np.ndarray:
        """Direct access to the (N, 2) coordinate array."""
        return self._coords

    # ------------------------------------------------------------------
    # Loading from JSON
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> PointGrid:
        """Load a PointGrid from a processed geometry JSON."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        grid_spacing = float(data.get("grid_spacing", 2.5))
        contour_spacing = float(data.get("contour_spacing", grid_spacing * 2.0))

        n = len(data["points"])
        coords = np.zeros((n, 2), dtype=np.float64)
        status = np.zeros(n, dtype=np.int8)

        for i, p in enumerate(data["points"]):
            coords[i, 0] = float(p["x"])
            coords[i, 1] = float(p["y"])
            t = p["type"]
            if t == "inner":
                status[i] = cls._STATUS_INNER
            elif t == "hole":
                status[i] = cls._STATUS_HOLE
            else:
                status[i] = cls._STATUS_OUTER

        grid = cls(coords=coords, status=status, point_spacing=grid_spacing,
                   contour_spacing=contour_spacing)
        grid._source_path = path
        return grid

    @classmethod
    def from_json_dialog(cls, initial_dir: str | Path | None = None) -> PointGrid | None:
        """Open a file dialog to select a geometry JSON."""
        import tkinter as tk
        from tkinter import filedialog

        if initial_dir is None:
            initial_dir = cls.DEFAULT_DIR

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Geometrie-JSON auswählen",
            filetypes=[("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")],
            initialdir=str(initial_dir),
        )
        root.destroy()

        if not chosen:
            print("Kein Datei ausgewählt.")
            return None

        print(f"Lade: {chosen}")
        return cls.from_json(chosen)

    # ------------------------------------------------------------------
    # Point access
    # ------------------------------------------------------------------

    @property
    def total_points(self) -> int:
        return len(self._coords)

    @property
    def outer_points(self) -> list[GridPoint]:
        indices = np.where(self._status == self._STATUS_OUTER)[0]
        return [self._get_grid_point(i) for i in indices]

    @property
    def hole_points(self) -> list[GridPoint]:
        """Points on the hole contours."""
        indices = np.where(self._status == self._STATUS_HOLE)[0]
        return [self._get_grid_point(i) for i in indices]

    # ------------------------------------------------------------------
    # Contour geometry (for angle computation)
    # ------------------------------------------------------------------

    @property
    def outer_points_ordered(self) -> list[GridPoint]:
        # The GeometryProcessor writes outer points first, so sorted by
        # index they are already in contour order.
        outer_idx = np.where(self._status == self._STATUS_OUTER)[0]
        return [self._get_grid_point(i) for i in outer_idx]

    @property
    def hole_points_ordered(self) -> list[GridPoint]:
        """Hole contour points in contour order. The GeometryProcessor
        writes hole points after the outer points and before the inner
        points, so index order matches contour order.
        """
        hole_idx = np.where(self._status == self._STATUS_HOLE)[0]
        return [self._get_grid_point(i) for i in hole_idx]

    def __repr__(self) -> str:
        n_outer = int(np.sum(self._status == self._STATUS_OUTER))
        n_hole  = int(np.sum(self._status == self._STATUS_HOLE))
        n_inner = int(np.sum(self._status == self._STATUS_INNER))
        return (
            f"PointGrid(gesamt={self.total_points}, "
            f"außen={n_outer}, lochrand={n_hole}, innen={n_inner}, "
            f"abstand={self.point_spacing} mm)"
        )


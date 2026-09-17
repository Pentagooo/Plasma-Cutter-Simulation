from __future__ import annotations

import json
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Enums & Datenklassen
# ---------------------------------------------------------------------------

class PointStatus(Enum):
    """Klassifizierung eines Gitterpunkts.

    OUTER : Außenrand der Geometrie (zugänglich für Brennereinstich).
    HOLE  : Innenrand (Lochkontur) – auch ein Rand, aber innerhalb des
            Profils (Bohrungen/Aussparungen).
    INNER : Volumen-Punkt im Materialinneren.
    """
    OUTER = "outer"   # Außenrand (Eintritt erlaubt)
    HOLE  = "hole"    # Innenrand (Lochkontur) – Innenrandpunkt
    INNER = "inner"   # Materialinneres


@dataclass
class GridPoint:
    """Ein Punkt im Punktgitter mit Position und Status.

    Parameters
    ----------
    x, y   : Position [mm]
    status : OUTER (Rand) oder INNER (Innenbereich)
    index  : eindeutiger Index innerhalb des PointGrid
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
        """True, wenn der Punkt auf einer Innenrandkontur (Loch) liegt."""
        return self.status == PointStatus.HOLE

    @property
    def is_inner(self) -> bool:
        return self.status == PointStatus.INNER

    @property
    def is_boundary(self) -> bool:
        """True für Außen- UND Innenrandpunkte (alle Konturpunkte)."""
        return self.status in (PointStatus.OUTER, PointStatus.HOLE)

    def __repr__(self) -> str:
        return (f"GridPoint(idx={self.index}, "
                f"x={self.x:.2f}, y={self.y:.2f}, "
                f"{self.status.value})")


# ---------------------------------------------------------------------------
# PointGrid
# ---------------------------------------------------------------------------

class PointGrid:
    """Sammlung von Gitterpunkten mit Außen-/Innen-Klassifizierung.
    """

    # Standard-Pfad zum Ordner mit geprüften Geometrien
    DEFAULT_DIR: Path = Path(__file__).parent / "Geometrie_Konturen_geprüft"

    # Interne Status-Kodierung (int8)
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
        coords        : (N, 2) float64 Array der Punkt-Positionen
        status        : (N,) int8 Array (0=OUTER, 1=INNER, 2=HOLE)
        point_spacing : charakteristischer Abstand [mm]
        """
        self._coords = np.asarray(coords, dtype=np.float64)
        self._status = np.asarray(status, dtype=np.int8)
        self.point_spacing = float(point_spacing)
        self.contour_spacing: float = float(contour_spacing) if contour_spacing else point_spacing * 2.0

        # Lazy-Cache für GridPoint-Objekte (nur bei Bedarf erstellt)
        self._grid_point_cache: list[GridPoint | None] = [None] * len(self._coords)

    @property
    def points(self) -> list[GridPoint]:
        """Kompatibilitäts-Property: Gibt alle Punkte als GridPoint-Liste zurück.
        Achtung: Erzeugt Tausende Python-Objekte!
        """
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
        """Direkter Zugriff auf das (N, 2) Koordinaten-Array."""
        return self._coords

    # ------------------------------------------------------------------
    # Laden aus JSON
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> PointGrid:
        """Lade ein PointGrid aus einer verarbeiteten Geometrie-JSON."""
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
        """Öffnet einen Datei-Dialog zur Auswahl einer Geometrie-JSON."""
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
    # Punkt-Zugriff
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
        """Innenrandpunkte (Lochkonturen)."""
        indices = np.where(self._status == self._STATUS_HOLE)[0]
        return [self._get_grid_point(i) for i in indices]

    # ------------------------------------------------------------------
    # Kontur-Geometrie (für Winkelberechnung)
    # ------------------------------------------------------------------

    @property
    def outer_points_ordered(self) -> list[GridPoint]:
        # Da der GeometryProcessor Außenpunkte zuerst schreibt,
        # sind sie nach Index sortiert bereits in Kontur-Reihenfolge.
        outer_idx = np.where(self._status == self._STATUS_OUTER)[0]
        return [self._get_grid_point(i) for i in outer_idx]

    @property
    def hole_points_ordered(self) -> list[GridPoint]:
        """Innenrandpunkte in Kontur-Reihenfolge. Der GeometryProcessor
        schreibt Lochpunkte nach den Außenpunkten und vor den Innenpunkten,
        sodass die Index-Reihenfolge der Kontur-Reihenfolge entspricht.
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


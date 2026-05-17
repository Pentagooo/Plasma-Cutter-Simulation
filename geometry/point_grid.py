from __future__ import annotations

import json
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from shapely.geometry import LineString, Point as ShapelyPoint


# ---------------------------------------------------------------------------
# Enums & Datenklassen
# ---------------------------------------------------------------------------

class PointStatus(Enum):
    """Klassifizierung eines Gitterpunkts.

    OUTER : Außenrand der Geometrie (zugänglich für Brennereinstich).
    HOLE  : Innenrand (Lochkontur) – auch ein Rand, aber innerhalb des
            Profils. Für Bohrungen/Aussparungen typischerweise mit
            Y-Fase (45°) zur Schweißnahtvorbereitung geschnitten.
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
# CutPoint – neue Klasse für geschnittene Punkte
# ---------------------------------------------------------------------------

class CutPoint:
    """Ein Gitterpunkt der von einem Schnitt getroffen wurde.

    Wird erzeugt wenn ein Schnitt einen GridPoint trifft.
    Speichert Position, Originalstatus und welcher Schnitt ihn erzeugt hat.

    Parameters
    ----------
    x, y              : Position [mm] (identisch mit dem GridPoint)
    original_status   : OUTER / INNER – Klassifizierung vor dem Schnitt
    index             : Index des Quell-GridPoint im PointGrid
    cut_index         : 0-basierter Index des Schnitts der diesen Punkt erzeugt hat
    """

    def __init__(
        self,
        x: float,
        y: float,
        original_status: PointStatus,
        index: int,
        cut_index: int,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.original_status = original_status
        self.index = index
        self.cut_index = cut_index

    # ------------------------------------------------------------------

    @classmethod
    def from_grid_point(cls, pt: GridPoint, cut_index: int) -> CutPoint:
        """CutPoint aus einem GridPoint erzeugen."""
        return cls(
            x=pt.x,
            y=pt.y,
            original_status=pt.status,
            index=pt.index,
            cut_index=cut_index,
        )

    # ------------------------------------------------------------------

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def was_outer(self) -> bool:
        return self.original_status == PointStatus.OUTER

    @property
    def was_hole(self) -> bool:
        return self.original_status == PointStatus.HOLE

    @property
    def was_inner(self) -> bool:
        return self.original_status == PointStatus.INNER

    @property
    def was_boundary(self) -> bool:
        return self.original_status in (PointStatus.OUTER, PointStatus.HOLE)

    def __repr__(self) -> str:
        return (f"CutPoint(idx={self.index}, "
                f"x={self.x:.2f}, y={self.y:.2f}, "
                f"orig={self.original_status.value}, "
                f"cut={self.cut_index})")


# ---------------------------------------------------------------------------
# PointGrid
# ---------------------------------------------------------------------------

class PointGrid:
    """Sammlung von Gitterpunkten mit Außen-/Innen-Klassifizierung.

    Optimiert auf niedrigen Speicherverbrauch und hohe Performance im
    RL-Training durch interne Speicherung in NumPy-Arrays statt Listen
    von Python-Objekten.
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

        self._is_cut = np.zeros(len(self._coords), dtype=bool)
        self._cut_points: list[CutPoint] = []
        
        # Lazy-Cache für GridPoint-Objekte (nur bei Bedarf erstellt)
        self._grid_point_cache: list[GridPoint | None] = [None] * len(self._coords)

    @property
    def points(self) -> list[GridPoint]:
        """Kompatibilitäts-Property: Gibt alle Punkte als GridPoint-Liste zurück.
        Achtung: Erzeugt Tausende Python-Objekte – im Hot-Path vermeiden!
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

    def transform(self, R: np.ndarray, scale: float) -> None:
        """Wendet eine Rotationsmatrix R und Skalierung in-place an.
        Vektorisiert für maximale Performance.
        """
        self._coords = (self._coords @ R.T) * scale
        self.point_spacing *= scale
        self.contour_spacing *= scale
        # Cache entleeren, da sich die Koordinaten geändert haben
        self._grid_point_cache = [None] * len(self._coords)

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
    # Tiefenbeschränkung
    # ------------------------------------------------------------------

    def max_points_per_cut(self, max_depth: float) -> int:
        return int(max_depth / self.point_spacing) + 1

    # ------------------------------------------------------------------
    # Punkt-Zugriff
    # ------------------------------------------------------------------

    @property
    def total_points(self) -> int:
        return len(self._coords)

    @property
    def n_cut(self) -> int:
        return int(np.sum(self._is_cut))

    @property
    def cut_percentage(self) -> float:
        return 100.0 * self.n_cut / max(1, self.total_points)

    @property
    def cut_points(self) -> list[CutPoint]:
        return list(self._cut_points)

    @property
    def remaining_points(self) -> list[GridPoint]:
        """GridPoints die noch nicht geschnitten wurden.
        Achtung: Erzeugt Python-Objekte.
        """
        indices = np.where(~self._is_cut)[0]
        return [self._get_grid_point(i) for i in indices]

    @property
    def outer_points(self) -> list[GridPoint]:
        indices = np.where((~self._is_cut) & (self._status == self._STATUS_OUTER))[0]
        return [self._get_grid_point(i) for i in indices]

    @property
    def hole_points(self) -> list[GridPoint]:
        """Verbleibende Innenrandpunkte (Lochkonturen). Diese qualifizieren
        sich für 45°-Y-Fasen-Schnitte zusätzlich zum 90°-Standardschnitt.
        """
        indices = np.where((~self._is_cut) & (self._status == self._STATUS_HOLE))[0]
        return [self._get_grid_point(i) for i in indices]

    @property
    def boundary_points(self) -> list[GridPoint]:
        """Alle Konturpunkte (Außen + Lochrand)."""
        mask = (~self._is_cut) & (
            (self._status == self._STATUS_OUTER)
            | (self._status == self._STATUS_HOLE)
        )
        return [self._get_grid_point(i) for i in np.where(mask)[0]]

    @property
    def inner_points(self) -> list[GridPoint]:
        indices = np.where((~self._is_cut) & (self._status == self._STATUS_INNER))[0]
        return [self._get_grid_point(i) for i in indices]

    # ------------------------------------------------------------------
    # Geometrie: Punkte auf Linie finden
    # ------------------------------------------------------------------

    def points_on_line(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        tolerance: float | None = None,
    ) -> list[GridPoint]:
        if tolerance is None:
            tolerance = self.point_spacing * 0.5

        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        d = p2 - p1
        length = np.linalg.norm(d)
        if length < 1e-12:
            return []
        d_unit = d / length

        # Vektorisierte Berechnung
        rem_idx = np.where(~self._is_cut)[0]
        if rem_idx.size == 0:
            return []
            
        rem_coords = self._coords[rem_idx]
        v = rem_coords - p1
        proj = np.dot(v, d_unit)
        perp = np.linalg.norm(v - proj[:, np.newaxis] * d_unit, axis=1)
        
        mask = perp <= tolerance
        hit_indices = rem_idx[mask]
        hit_projs = proj[mask]
        
        # Sortieren nach Projektion
        sort_idx = np.argsort(hit_projs)
        return [self._get_grid_point(i) for i in hit_indices[sort_idx]]

    # ------------------------------------------------------------------
    # Mutation: Schnitt anwenden
    # ------------------------------------------------------------------

    def apply_cut(
        self, points: list[GridPoint], cut_index: int
    ) -> list[CutPoint]:
        new_cut_points: list[CutPoint] = []
        for pt in points:
            if self._is_cut[pt.index]:
                continue
            cp = CutPoint.from_grid_point(pt, cut_index)
            self._cut_points.append(cp)
            self._is_cut[pt.index] = True
            new_cut_points.append(cp)
        return new_cut_points

    def apply_trajectory(
        self,
        waypoints: list[np.ndarray],
        kerf_width: float,
        cut_index: int | None = None,
    ) -> list[CutPoint]:
        if len(waypoints) < 2:
            return []

        coords = [(float(w[0]), float(w[1])) for w in waypoints]
        line = LineString(coords)
        swept = line.buffer(kerf_width / 2)
        return self.apply_swept_area(swept, cut_index)

    def apply_swept_area(
        self,
        polygon,
        cut_index: int | None = None,
    ) -> list[CutPoint]:
        """Markiert alle Gitterpunkte innerhalb eines Shapely-Polygons als geschnitten.
        Optimierter Hot-Path: Nutzt shapely.vectorized.contains auf den internen Arrays.
        """
        if polygon is None or polygon.is_empty:
            return []

        if cut_index is None:
            cut_index = self.n_cut

        rem_idx = np.where(~self._is_cut)[0]
        if rem_idx.size == 0:
            return []

        from shapely.vectorized import contains as shp_contains_vec
        xs = self._coords[rem_idx, 0]
        ys = self._coords[rem_idx, 1]
        inside = shp_contains_vec(polygon, xs, ys)

        new_cut_indices = rem_idx[inside]
        new_cut_points: list[CutPoint] = []
        
        for idx in new_cut_indices:
            pt = self._get_grid_point(idx)
            cp = CutPoint.from_grid_point(pt, cut_index)
            self._cut_points.append(cp)
            self._is_cut[idx] = True
            new_cut_points.append(cp)

        return new_cut_points

    def reset(self) -> None:
        """Setzt alle Schnitte zurück."""
        self._cut_points.clear()
        self._is_cut.fill(False)

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

    def outer_neighbors(
        self, pt: GridPoint
    ) -> tuple[GridPoint, GridPoint]:
        ordered_idx = np.where(self._status == self._STATUS_OUTER)[0]
        # Finde Position im ordered_idx
        pos_list = np.where(ordered_idx == pt.index)[0]
        if pos_list.size == 0:
            raise ValueError(f"Punkt {pt} ist kein Außenpunkt.")
        pos = pos_list[0]
        n = len(ordered_idx)
        return (self._get_grid_point(ordered_idx[(pos - 1) % n]),
                self._get_grid_point(ordered_idx[(pos + 1) % n]))

    def contour_normal_at(self, pt: GridPoint) -> np.ndarray:
        prev_pt, next_pt = self.outer_neighbors(pt)
        tangent = next_pt.coords - prev_pt.coords
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            return np.array([1.0, 0.0])
        tangent /= norm
        return np.array([-tangent[1], tangent[0]])

    def __repr__(self) -> str:
        n_outer = int(np.sum(self._status == self._STATUS_OUTER))
        n_hole  = int(np.sum(self._status == self._STATUS_HOLE))
        n_inner = int(np.sum(self._status == self._STATUS_INNER))
        return (
            f"PointGrid(gesamt={self.total_points}, "
            f"außen={n_outer}, lochrand={n_hole}, innen={n_inner}, "
            f"geschnitten={self.n_cut}, "
            f"abstand={self.point_spacing} mm)"
        )


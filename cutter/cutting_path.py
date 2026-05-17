from __future__ import annotations

"""Schneidpfad entlang der Aussenkontur eines PointGrid.

Der Schneider faehrt von Punkt A nach Punkt B entlang der Kontur und setzt
in regelmaessigen Abstaenden senkrechte Schnitte ins Material.

``CuttingPath.compute()`` gibt ein ``PathResult`` zurueck, das die komplette
Ausfuehrung beschreibt und von einem Optimierer oder ML-Modell ausgewertet
werden kann.

Typischer Ablauf (manuell)::

    path_engine = CuttingPath(grid, cutter)
    result = path_engine.compute(start_pt, end_pt)
    print(result.summary())

Fuer ML / Optimierung (ohne Grid-Modifikation)::

    result = path_engine.compute(start_pt, end_pt, apply=False)
    score  = my_reward_fn(result)
"""

from dataclasses import dataclass, field
import numpy as np

from ..geometry.point_grid import PointGrid, GridPoint, CutPoint
from .cutter import Cutter


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------

@dataclass
class PathCut:
    """Ein einzelner Schnitt senkrecht zur Kontur.

    Attributes
    ----------
    position       : (x, y) Startposition des Schnitts (auf der Kontur) [mm]
    direction      : Einheitsvektor nach innen (Schnittrichtung)
    path_distance  : kumulierte Distanz entlang der Kontur vom Pfadstart [mm]
    points_hit     : getroffene GridPoints (sortiert nach Tiefe)
    cut_length     : Tiefe des Schnitts (Projektion des fernsten Treffers) [mm]
    is_feasible    : True wenn der Schnitt die Gegenseite erreicht (endet an Aussenpunkt)
    cutting_time   : Zeitbedarf fuer diesen Schnitt [s]
    """
    position:      np.ndarray
    direction:     np.ndarray
    path_distance: float
    points_hit:    list[GridPoint] = field(default_factory=list)
    cut_length:    float = 0.0
    is_feasible:   bool = True
    cutting_time:  float = 0.0


@dataclass
class PathResult:
    """Komplettes Ergebnis einer Schnittpfad-Ausfuehrung.

    Entworfen fuer einfache Auswertung durch Optimierer / ML.
    """
    contour_points:     list[np.ndarray]   # Kontur-Positionen [mm]
    cuts:               list[PathCut]       # einzelne Schnitte
    total_path_length:  float               # Kontur-Pfadlaenge [mm]
    total_travel_time:  float               # Verfahrzeit [s]
    total_cutting_time: float               # Schneidzeit [s]
    total_time:         float               # Gesamt (Verfahren + Schneiden) [s]
    all_cut_points:     list[CutPoint]      # erzeugte CutPoints
    n_successful:       int                 # Anzahl erfolgreicher Schnitte
    n_total:            int                 # Anzahl Schnitte gesamt

    # -- Metriken fuer ML / Optimierung --

    @property
    def success_rate(self) -> float:
        """Anteil erfolgreicher Schnitte [0..1]."""
        return self.n_successful / max(1, self.n_total)

    @property
    def total_points_cut(self) -> int:
        """Gesamtanzahl geschnittener Punkte."""
        return len(self.all_cut_points)

    def summary(self) -> str:
        return (
            f"PathResult: {self.n_total} Schnitte ({self.n_successful} OK) | "
            f"Pfad: {self.total_path_length:.1f} mm | "
            f"Zeit: {self.total_time:.1f} s | "
            f"Punkte: {self.total_points_cut}"
        )


# ---------------------------------------------------------------------------
# CuttingPath
# ---------------------------------------------------------------------------

class CuttingPath:
    """Berechnet und fuehrt einen Schneidpfad entlang der Aussenkontur aus.

    Parameters
    ----------
    grid            : PointGrid mit der Geometrie
    cutter          : Cutter mit physikalischen Parametern
    cut_spacing     : Abstand zwischen Schnitten entlang der Kontur [mm].
                      Standard: 2x contour_spacing.
    cut_tolerance   : Treff-Toleranz fuer Innenpunkte [mm]
    outer_tolerance : Treff-Toleranz fuer Aussenpunkte [mm]
    """

    def __init__(
        self,
        grid: PointGrid,
        cutter: Cutter,
        cut_spacing: float | None = None,
        cut_tolerance: float | None = None,
        outer_tolerance: float | None = None,
    ) -> None:
        self.grid = grid
        self.cutter = cutter
        self.cut_spacing = cut_spacing or grid.contour_spacing * 2.0
        self.cut_tolerance = cut_tolerance or grid.point_spacing * 0.55
        self.outer_tolerance = outer_tolerance or grid.contour_spacing * 0.60

    # ------------------------------------------------------------------
    # Kontur-Teilpfad
    # ------------------------------------------------------------------

    def contour_subpath(
        self, start: GridPoint, end: GridPoint,
    ) -> list[GridPoint]:
        """Geordnete Aussenpunkte von *start* nach *end* (kuerzere Richtung).

        Beide Wege um die Kontur-Schleife werden berechnet; der kuerzere
        wird zurueckgegeben.
        """
        ordered = self.grid.outer_points_ordered
        n = len(ordered)
        if n == 0:
            return []

        si = next((i for i, p in enumerate(ordered) if p.index == start.index), None)
        ei = next((i for i, p in enumerate(ordered) if p.index == end.index), None)
        if si is None or ei is None:
            return []

        # Vorwaerts: si -> ei (Umlauf)
        fwd: list[GridPoint] = []
        i = si
        while True:
            fwd.append(ordered[i])
            if i == ei:
                break
            i = (i + 1) % n

        # Rueckwaerts: si -> ei (Umlauf)
        bwd: list[GridPoint] = []
        i = si
        while True:
            bwd.append(ordered[i])
            if i == ei:
                break
            i = (i - 1) % n

        fwd_len = polyline_length([p.coords for p in fwd])
        bwd_len = polyline_length([p.coords for p in bwd])
        return fwd if fwd_len <= bwd_len else bwd

    # ------------------------------------------------------------------
    # Hauptberechnung
    # ------------------------------------------------------------------

    def compute(
        self,
        start: GridPoint,
        end: GridPoint,
        apply: bool = True,
    ) -> PathResult:
        """Berechnet den Schnittpfad von *start* nach *end*.

        Parameters
        ----------
        apply : wenn False, wird das Grid NICHT veraendert (Dry-Run fuer ML).
        """
        # Grid-Zustand sichern falls Dry-Run
        if not apply:
            saved_ci = self.grid._cut_indices.copy()
            saved_cp = list(self.grid._cut_points)

        path_pts = self.contour_subpath(start, end)
        empty = PathResult([], [], 0, 0, 0, 0, [], 0, 0)
        if len(path_pts) < 2:
            if not apply:
                self.grid._cut_indices = saved_ci
                self.grid._cut_points = saved_cp
            return empty

        coords = [p.coords for p in path_pts]
        path_length = polyline_length(coords)
        cum = cumulative_dists(coords)

        # Schnitte platzieren
        path_cuts = self._place_cuts(path_pts, coords, cum, path_length)

        # Schnitte ausfuehren (in Reihenfolge – spaetere sehen weniger Punkte)
        all_cp: list[CutPoint] = []
        n_ok = 0
        base_idx = self.grid.n_cut

        for i, pc in enumerate(path_cuts):
            cands = self._ray_candidates(pc.position, pc.direction)
            pc.points_hit = cands

            if cands:
                projs = [float(np.dot(c.coords - pc.position, pc.direction))
                         for c in cands]
                pc.cut_length = max(projs)
                # Ein Schnitt ist "feasible" wenn die Gegenseite ein
                # Konturpunkt ist (Aussenrand ODER Lochrand) -- d.h. der
                # Schnitt erreicht eine Materialgrenze. Innere Punkte
                # bedeuten, dass der Schnitt mitten im Material endet.
                pc.is_feasible = cands[-1].is_boundary
                pc.cutting_time = self.cutter.time_for_length(
                    pc.cut_length, mode="cut",
                )
            else:
                pc.cut_length = 0.0
                pc.is_feasible = True
                pc.cutting_time = 0.0

            if pc.is_feasible and cands:
                n_ok += 1

            new_cps = self.grid.apply_cut(cands, base_idx + i)
            all_cp.extend(new_cps)

        travel_time = self.cutter.time_for_length(path_length, mode="move")
        cutting_time = sum(pc.cutting_time for pc in path_cuts)

        result = PathResult(
            contour_points=coords,
            cuts=path_cuts,
            total_path_length=path_length,
            total_travel_time=travel_time,
            total_cutting_time=cutting_time,
            total_time=travel_time + cutting_time,
            all_cut_points=all_cp,
            n_successful=n_ok,
            n_total=len(path_cuts),
        )

        # Dry-Run: Grid zuruecksetzen
        if not apply:
            self.grid._cut_indices = saved_ci
            self.grid._cut_points = saved_cp

        return result

    # ------------------------------------------------------------------
    # Schnitt-Platzierung
    # ------------------------------------------------------------------

    def _place_cuts(
        self,
        path_pts: list[GridPoint],
        coords: list[np.ndarray],
        cum: list[float],
        path_length: float,
    ) -> list[PathCut]:
        """Verteilt Schnitte gleichmaessig entlang der Kontur."""
        if path_length < self.cut_spacing * 0.5:
            return []

        n_cuts = max(1, round(path_length / self.cut_spacing))
        spacing = path_length / (n_cuts + 1)

        cuts: list[PathCut] = []
        for k in range(1, n_cuts + 1):
            d = k * spacing
            pos = interpolate_at(coords, cum, d)
            normal = self._inward_normal_at(coords, cum, d)
            cuts.append(PathCut(position=pos, direction=normal, path_distance=d))
        return cuts

    def _inward_normal_at(
        self,
        coords: list[np.ndarray],
        cum: list[float],
        dist: float,
    ) -> np.ndarray:
        """Einheits-Normalenvektor der Kontur, nach innen gerichtet."""
        # Segment-Index finden
        seg_idx = 0
        for i in range(len(cum) - 1):
            if cum[i] <= dist <= cum[i + 1]:
                seg_idx = i
                break

        # Tangente aus Nachbarpunkten
        i0 = max(0, seg_idx - 1)
        i1 = min(len(coords) - 1, seg_idx + 2)
        tangent = np.array(coords[i1] - coords[i0], dtype=float)
        norm = np.linalg.norm(tangent)
        if norm < 1e-9:
            return np.array([1.0, 0.0])
        tangent /= norm

        # 90-Grad-Drehung -> Normale
        normal = np.array([-tangent[1], tangent[0]])

        # Richtung nach innen (zum Schwerpunkt aller Gitterpunkte)
        pos = interpolate_at(coords, cum, dist)
        all_c = np.array([[p.x, p.y] for p in self.grid.points])
        centroid = all_c.mean(axis=0)
        if np.dot(normal, centroid - pos) < 0:
            normal = -normal

        return normal

    # ------------------------------------------------------------------
    # Strahl-Kandidaten
    # ------------------------------------------------------------------

    def _ray_candidates(
        self,
        start: np.ndarray,
        d_unit: np.ndarray,
    ) -> list[GridPoint]:
        """Verbleibende Gitterpunkte entlang des Schnittstrahls, nach Tiefe sortiert."""
        remaining = self.grid.remaining_points
        if not remaining:
            return []

        coords = np.array([[p.x, p.y] for p in remaining])
        # HOLE-Punkte (Lochrand) sind ebenfalls Konturpunkte und teilen
        # sich die Toleranz mit den Aussenpunkten.
        is_boundary = np.array([p.is_boundary for p in remaining])

        v = coords - start
        proj = v @ d_unit
        perp = np.linalg.norm(v - np.outer(proj, d_unit), axis=1)

        eps = min(self.cut_tolerance, self.outer_tolerance) * 0.15
        mask = (
            (proj >= -eps)
            & (proj <= self.cutter.max_depth)
            & np.where(is_boundary,
                       perp <= self.outer_tolerance,
                       perp <= self.cut_tolerance)
        )

        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []

        order = idx[np.argsort(proj[idx])]
        return [remaining[i] for i in order]


# ---------------------------------------------------------------------------
# Hilfsfunktionen (oeffentlich – werden auch von der Simulation genutzt)
# ---------------------------------------------------------------------------

def polyline_length(coords: list[np.ndarray]) -> float:
    """Gesamtlaenge einer Polylinie."""
    if len(coords) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(coords[i + 1] - coords[i]))
        for i in range(len(coords) - 1)
    )


def cumulative_dists(coords: list[np.ndarray]) -> list[float]:
    """Kumulierte Abstande entlang einer Polylinie."""
    cum = [0.0]
    for i in range(len(coords) - 1):
        cum.append(cum[-1] + float(np.linalg.norm(coords[i + 1] - coords[i])))
    return cum


def interpolate_at(
    coords: list[np.ndarray],
    cum: list[float],
    dist: float,
) -> np.ndarray:
    """Interpolierte Position bei kumulierter Distanz *dist*."""
    for i in range(len(cum) - 1):
        if cum[i] <= dist <= cum[i + 1]:
            seg = cum[i + 1] - cum[i]
            if seg < 1e-12:
                return np.array(coords[i], dtype=float)
            frac = (dist - cum[i]) / seg
            return (np.array(coords[i], dtype=float)
                    + frac * (np.array(coords[i + 1], dtype=float)
                              - np.array(coords[i], dtype=float)))
    return np.array(coords[-1], dtype=float)
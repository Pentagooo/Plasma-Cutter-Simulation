from __future__ import annotations

"""Konturen, Segmentierung und Coverage fuer die Segment-Simulation.

Begriffe
--------
ContourLoop : Eine geschlossene Kontur (Aussenrand oder Lochrand) als
              geordnete Punktfolge. Positionen sind Indizes 0..N-1
              innerhalb des Loops (zyklisch, modulo N).
Segment     : Ein automatisch erzeugtes Primitiv-Segment = Sammlung
              aufeinanderfolgender Konturpunkte zwischen zwei Knoten.
              Die Knoten (Segmentgrenzen) sind die "moeglichen Start-
              und Endpunkte", auf die Benutzer-Klicks gesnappt werden.
CutRun      : Ein vom Benutzer gewaehlter Schnitt von einem Knoten zu
              einem anderen entlang der Kontur (kann mehrere
              Primitiv-Segmente umfassen). Richtung +1/-1.
Coverage    : Anteil der Konturpunkte, die von den gewaehlten CutRuns
              abgedeckt werden. 100 % = Objekt vollstaendig
              durchgeschnitten.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

try:
    from ..geometry.point_grid import PointGrid
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.geometry.point_grid import PointGrid


# ---------------------------------------------------------------------------
# ContourLoop
# ---------------------------------------------------------------------------

@dataclass
class ContourLoop:
    """Eine geschlossene Kontur als geordnete Punktfolge.

    Parameters
    ----------
    loop_id : eindeutige ID innerhalb der SegmentedContour
    kind    : "outer" (Aussenrand) oder "hole" (Lochrand)
    points  : (N, 2) Array der Punktpositionen in Kontur-Reihenfolge
    """
    loop_id: int
    kind: str
    points: np.ndarray

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        n = len(self.points)
        # Kantenlaenge von Punkt i zu Punkt i+1 (zyklisch)
        nxt = np.roll(self.points, -1, axis=0)
        self._edge_len = np.linalg.norm(nxt - self.points, axis=1)
        self._n = n

    @property
    def n(self) -> int:
        return self._n

    @property
    def length(self) -> float:
        """Gesamtumfang des Loops [mm]."""
        return float(self._edge_len.sum())

    # ------------------------------------------------------------------
    # Bogen-Operationen (zyklisch)
    # ------------------------------------------------------------------

    def arc_positions(self, a: int, b: int, direction: int) -> list[int]:
        """Positionen von a nach b in der gegebenen Richtung (inklusive).

        a == b bedeutet: kompletter Loop (einmal herum, Endpunkt = a).
        """
        n = self._n
        a, b = a % n, b % n
        out = [a]
        pos = a
        if a == b:
            for _ in range(n):
                pos = (pos + direction) % n
                out.append(pos)
            return out
        while pos != b:
            pos = (pos + direction) % n
            out.append(pos)
        return out

    def arc_length(self, a: int, b: int, direction: int) -> float:
        """Bogenlaenge von a nach b in der gegebenen Richtung [mm]."""
        positions = self.arc_positions(a, b, direction)
        total = 0.0
        for i in range(len(positions) - 1):
            p, q = positions[i], positions[i + 1]
            if direction == +1:
                total += self._edge_len[p]
            else:
                total += self._edge_len[q]
        return total

    def polyline(self, positions: list[int]) -> np.ndarray:
        """(M, 2) Koordinaten-Array fuer eine Positionsfolge."""
        return self.points[np.asarray(positions, dtype=int)]

    # ------------------------------------------------------------------
    # Geometrie
    # ------------------------------------------------------------------

    def tangent_at(self, pos: int) -> np.ndarray:
        """Einheits-Tangente an Position pos (zentrale Differenz)."""
        n = self._n
        t = self.points[(pos + 1) % n] - self.points[(pos - 1) % n]
        norm = np.linalg.norm(t)
        if norm < 1e-9:
            return np.array([1.0, 0.0])
        return t / norm

    def outward_normal(
        self,
        pos: int,
        material: Polygon | MultiPolygon | None,
        probe: float = 1.0,
    ) -> np.ndarray:
        """Einheits-Normale an Position pos, die vom Material WEG zeigt.

        Fuer Aussenkonturen zeigt sie nach aussen, fuer Lochkonturen in
        das Loch hinein (= freier Raum). Die Richtung wird per
        Punkt-im-Polygon-Test bestimmt.
        """
        t = self.tangent_at(pos)
        normal = np.array([-t[1], t[0]])
        if material is None or material.is_empty:
            return normal
        from shapely.geometry import Point as ShapelyPoint
        p = self.points[pos]
        test = ShapelyPoint(p[0] + normal[0] * probe, p[1] + normal[1] * probe)
        if material.contains(test):
            normal = -normal
        return normal

    def corner_positions(self, angle_thresh_deg: float = 40.0) -> list[int]:
        """Positionen mit starkem Richtungswechsel (Ecken der Kontur)."""
        n = self._n
        corners: list[int] = []
        thresh = math.radians(angle_thresh_deg)
        for i in range(n):
            v1 = self.points[i] - self.points[(i - 1) % n]
            v2 = self.points[(i + 1) % n] - self.points[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            if math.acos(cosang) >= thresh:
                corners.append(i)
        return corners

    def __repr__(self) -> str:
        return (f"ContourLoop(id={self.loop_id}, {self.kind}, "
                f"n={self._n}, L={self.length:.1f} mm)")


# ---------------------------------------------------------------------------
# Segment (Primitiv-Segment, automatisch erzeugt)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """Automatisch erzeugtes Primitiv-Segment einer Kontur.

    Eine Sammlung aufeinanderfolgender Konturpunkte zwischen zwei
    Knoten. Die Knoten sind die moeglichen Start-/Endpunkte fuer
    Benutzer-Auswahlen.
    """
    seg_id: int
    loop_id: int
    start_pos: int           # Knoten (Position im Loop)
    end_pos: int             # Knoten (Position im Loop, Richtung +1)
    positions: list[int]     # alle Positionen inkl. beider Knoten
    length: float            # Bogenlaenge [mm]

    def __repr__(self) -> str:
        return (f"Segment(S{self.seg_id}, loop={self.loop_id}, "
                f"{self.start_pos}->{self.end_pos}, "
                f"{len(self.positions)} Punkte, {self.length:.1f} mm)")


# ---------------------------------------------------------------------------
# CutRun (Benutzer-Auswahl)
# ---------------------------------------------------------------------------

@dataclass
class CutRun:
    """Ein gewaehlter Schnitt entlang der Kontur: Startknoten ->
    Endknoten in einer Richtung.

    Die Kontur-Geometrie (positions/polyline) beschreibt, WO an der
    Oberflaeche geschnitten wird. Die Lichtschwert-Kinematik
    (tcp_polyline/tip_polyline/swept_polygon) wird nachtraeglich von
    ``planning.RunKinematics.attach()`` berechnet: Der TCP faehrt auf
    dem Offset-Pfad (Mindestabstand zum Material), die Klinge ragt mit
    L(v) Richtung Objekt und ueberstreicht die Querschnittsflaeche.

    Attributes
    ----------
    positions     : Loop-Positionen in Schnittreihenfolge (inkl. Endpunkte)
    polyline      : (M, 2) Konturkoordinaten in Schnittreihenfolge
    length        : Konturlaenge des Schnitts [mm]
    tcp_polyline  : (K, 2) TCP-Wegpunkte auf dem Offset-Pfad
    tip_polyline  : (K, 2) Klingenspitzen-Wegpunkte
    swept_polygon : Shapely-Polygon der ueberstrichenen Flaeche
    tcp_length    : Laenge des TCP-Pfads [mm] (massgeblich fuer die Zeit)
    """
    run_id: int
    loop_id: int
    start_pos: int
    end_pos: int
    direction: int
    positions: list[int]
    polyline: np.ndarray
    length: float
    is_feasible: bool = True
    reason: str = ""
    tcp_polyline: np.ndarray | None = None
    tip_polyline: np.ndarray | None = None
    swept_polygon: object = None
    tcp_length: float = 0.0

    @property
    def start_xy(self) -> np.ndarray:
        return self.polyline[0]

    @property
    def end_xy(self) -> np.ndarray:
        return self.polyline[-1]

    @property
    def tcp_start(self) -> np.ndarray:
        """TCP-Startpunkt (Offset-Pfad); Fallback: Konturpunkt."""
        if self.tcp_polyline is not None and len(self.tcp_polyline):
            return self.tcp_polyline[0]
        return self.polyline[0]

    @property
    def tcp_end(self) -> np.ndarray:
        if self.tcp_polyline is not None and len(self.tcp_polyline):
            return self.tcp_polyline[-1]
        return self.polyline[-1]

    @property
    def is_full_loop(self) -> bool:
        return self.start_pos == self.end_pos

    def reversed(self) -> CutRun:
        """Derselbe Schnitt in Gegenrichtung (fuer den Sequencer)."""
        return CutRun(
            run_id=self.run_id,
            loop_id=self.loop_id,
            start_pos=self.end_pos,
            end_pos=self.start_pos,
            direction=-self.direction,
            positions=list(reversed(self.positions)),
            polyline=self.polyline[::-1].copy(),
            length=self.length,
            is_feasible=self.is_feasible,
            reason=self.reason,
            tcp_polyline=(self.tcp_polyline[::-1].copy()
                          if self.tcp_polyline is not None else None),
            tip_polyline=(self.tip_polyline[::-1].copy()
                          if self.tip_polyline is not None else None),
            swept_polygon=self.swept_polygon,
            tcp_length=self.tcp_length,
        )

    def __repr__(self) -> str:
        d = "+" if self.direction > 0 else "-"
        return (f"CutRun(R{self.run_id}, loop={self.loop_id}, "
                f"{self.start_pos}->{self.end_pos} ({d}), "
                f"{self.length:.1f} mm)")


# ---------------------------------------------------------------------------
# SegmentedContour
# ---------------------------------------------------------------------------

class SegmentedContour:
    """Alle Konturen einer Geometrie inkl. automatischer Segmentierung.

    Segmentierungs-Strategie ("sinnvolle Groesse"):

      1. Ecken der Kontur (Richtungswechsel >= corner_angle_deg) werden
         immer Knoten -- Segmente enden an geometrischen Ecken.
      2. Bogen zwischen zwei Ecken, die laenger als die Ziellaenge
         sind, werden in gleich grosse Teile unterteilt.
      3. Ziellaenge = clamp(Umfang / 12, 4 * Punktabstand, Umfang / 4)
         (ueberschreibbar via target_segment_length).

    Parameters
    ----------
    loops                  : Liste der ContourLoops
    target_segment_length  : Ziel-Segmentlaenge [mm] (None = automatisch)
    corner_angle_deg       : Schwellwinkel fuer Eckenerkennung [Grad]
    point_spacing          : Konturpunkt-Abstand [mm] (fuer Min-Groesse)
    """

    def __init__(
        self,
        loops: list[ContourLoop],
        target_segment_length: float | None = None,
        corner_angle_deg: float = 40.0,
        point_spacing: float = 5.0,
    ) -> None:
        self.loops = loops
        self.point_spacing = float(point_spacing)
        self.corner_angle_deg = float(corner_angle_deg)
        self.target_segment_length = target_segment_length

        # Knoten + Segmente pro Loop erzeugen
        self.nodes: dict[int, list[int]] = {}
        self.segments: list[Segment] = []
        for loop in loops:
            self._segment_loop(loop)

    # ------------------------------------------------------------------
    # Konstruktion aus PointGrid
    # ------------------------------------------------------------------

    @classmethod
    def from_grid(
        cls,
        grid: PointGrid,
        target_segment_length: float | None = None,
        corner_angle_deg: float = 40.0,
    ) -> SegmentedContour:
        """Baut die SegmentedContour aus einem PointGrid.

        Aussenpunkte und Lochpunkte des Grids sind in Kontur-Reihenfolge
        gespeichert (siehe PointGrid.outer_points_ordered).
        """
        loops: list[ContourLoop] = []

        outer = grid.outer_points_ordered
        if len(outer) >= 3:
            pts = np.array([[p.x, p.y] for p in outer])
            loops.append(ContourLoop(loop_id=0, kind="outer", points=pts))

        hole = grid.hole_points_ordered
        if len(hole) >= 3:
            pts = np.array([[p.x, p.y] for p in hole])
            loops.append(ContourLoop(loop_id=len(loops), kind="hole",
                                     points=pts))

        if not loops:
            raise ValueError("Geometrie enthaelt keine Kontur mit >= 3 Punkten.")

        return cls(
            loops=loops,
            target_segment_length=target_segment_length,
            corner_angle_deg=corner_angle_deg,
            point_spacing=grid.contour_spacing,
        )

    # ------------------------------------------------------------------
    # Material-Polygon (fuer Kollisionspruefung)
    # ------------------------------------------------------------------

    def material_polygon(self) -> Polygon | MultiPolygon | None:
        """Material = Aussenkontur abzueglich aller Lochkonturen."""
        outer_poly = None
        holes: list[Polygon] = []
        for loop in self.loops:
            coords = [tuple(p) for p in loop.points]
            coords.append(coords[0])
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly is None or poly.is_empty:
                continue
            if loop.kind == "outer":
                outer_poly = poly
            else:
                holes.append(poly)

        if outer_poly is None:
            return None
        for h in holes:
            try:
                outer_poly = outer_poly.difference(h)
            except Exception:
                pass
        return outer_poly if not outer_poly.is_empty else None

    # ------------------------------------------------------------------
    # Segmentierung eines Loops
    # ------------------------------------------------------------------

    def _segment_loop(self, loop: ContourLoop) -> None:
        target = self.target_segment_length
        if target is None:
            target = loop.length / 12.0
        # Nicht zu klein, nicht zu gross
        target = float(np.clip(
            target, 4.0 * self.point_spacing, loop.length / 4.0))

        corners = loop.corner_positions(self.corner_angle_deg)
        if not corners:
            corners = [0]  # mindestens ein Anker-Knoten

        node_set: set[int] = set(corners)

        # Boegen zwischen aufeinanderfolgenden Ecken unterteilen
        n_corners = len(corners)
        for ci in range(n_corners):
            a = corners[ci]
            b = corners[(ci + 1) % n_corners]
            positions = loop.arc_positions(a, b, +1)
            if len(positions) < 2:
                continue
            arc_len = loop.arc_length(a, b, +1)
            n_parts = max(1, int(round(arc_len / target)))
            if n_parts <= 1:
                continue
            # Knoten an gleichmaessigen Bogenlaengen-Abstaenden einfuegen
            step = arc_len / n_parts
            accum = 0.0
            next_split = step
            for i in range(len(positions) - 1):
                p = positions[i]
                accum += loop._edge_len[p]
                if accum >= next_split - 1e-9 and i + 1 < len(positions) - 1:
                    node_set.add(positions[i + 1])
                    next_split += step

        nodes = sorted(node_set)

        # Zu dicht liegende Knoten ausduennen (Mindestabstand 2 Punkte),
        # Ecken haben Vorrang vor eingefuegten Teilungs-Knoten.
        corner_set = set(corners)
        filtered: list[int] = []
        for nd in nodes:
            if not filtered:
                filtered.append(nd)
                continue
            if nd - filtered[-1] < 2:
                if nd in corner_set and filtered[-1] not in corner_set:
                    filtered[-1] = nd
                continue
            filtered.append(nd)
        # Zyklischen Abstand zwischen letztem und erstem Knoten pruefen
        if len(filtered) >= 2 and (filtered[0] + loop.n) - filtered[-1] < 2:
            if filtered[-1] not in corner_set:
                filtered.pop()
            elif filtered[0] not in corner_set:
                filtered.pop(0)
        nodes = filtered

        self.nodes[loop.loop_id] = nodes

        # Segmente zwischen aufeinanderfolgenden Knoten
        m = len(nodes)
        for i in range(m):
            a = nodes[i]
            b = nodes[(i + 1) % m]
            positions = loop.arc_positions(a, b, +1)
            self.segments.append(Segment(
                seg_id=len(self.segments),
                loop_id=loop.loop_id,
                start_pos=a,
                end_pos=b,
                positions=positions,
                length=loop.arc_length(a, b, +1),
            ))

    # ------------------------------------------------------------------
    # Klick-Snapping
    # ------------------------------------------------------------------

    def snap_node(
        self,
        xy: np.ndarray,
        max_dist: float | None = None,
    ) -> tuple[int, int] | None:
        """Findet den naechstgelegenen Segment-Knoten zu einem Klick.

        Returns
        -------
        (loop_id, position) oder None wenn kein Knoten in max_dist liegt.
        """
        xy = np.asarray(xy, dtype=float)
        best: tuple[int, int] | None = None
        best_d = math.inf
        for loop in self.loops:
            for pos in self.nodes.get(loop.loop_id, []):
                d = float(np.linalg.norm(loop.points[pos] - xy))
                if d < best_d:
                    best_d = d
                    best = (loop.loop_id, pos)
        if best is None:
            return None
        if max_dist is not None and best_d > max_dist:
            return None
        return best

    def loop_by_id(self, loop_id: int) -> ContourLoop:
        return self.loops[loop_id]

    # ------------------------------------------------------------------
    # CutRun-Erzeugung (Bogen-Auswahl)
    # ------------------------------------------------------------------

    def make_run(
        self,
        run_id: int,
        loop_id: int,
        start_pos: int,
        end_pos: int,
        covered: set[int] | None = None,
    ) -> CutRun:
        """Erzeugt einen CutRun von start_pos nach end_pos.

        Zwischen zwei Knoten gibt es zwei moegliche Boegen (im / gegen
        den Uhrzeigersinn). Gewaehlt wird der Bogen mit den meisten
        noch NICHT abgedeckten Punkten; bei Gleichstand der kuerzere.

        start_pos == end_pos waehlt den kompletten Loop.
        """
        loop = self.loop_by_id(loop_id)
        covered_set: set[int] = covered if covered is not None else set()

        if start_pos == end_pos:
            positions = loop.arc_positions(start_pos, end_pos, +1)
            return CutRun(
                run_id=run_id, loop_id=loop_id,
                start_pos=start_pos, end_pos=end_pos, direction=+1,
                positions=positions,
                polyline=loop.polyline(positions),
                length=loop.length,
            )

        candidates: list[CutRun] = []
        for direction in (+1, -1):
            positions = loop.arc_positions(start_pos, end_pos, direction)
            candidates.append(CutRun(
                run_id=run_id, loop_id=loop_id,
                start_pos=start_pos, end_pos=end_pos, direction=direction,
                positions=positions,
                polyline=loop.polyline(positions),
                length=loop.arc_length(start_pos, end_pos, direction),
            ))

        def score(run: CutRun) -> tuple[float, float]:
            # Hoechster ANTEIL neuer Punkte zuerst (so wird bei zwei
            # unberuehrten Boegen der kuerzere gewaehlt, bei teilweise
            # abgedeckter Kontur aber der noch fehlende Bogen),
            # dann kuerzere Laenge.
            uncovered = len(set(run.positions) - covered_set)
            fraction = uncovered / max(1, len(run.positions))
            return -fraction, run.length

        candidates.sort(key=score)
        return candidates[0]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    """Abdeckungs-Bericht: Welcher Anteil der Kontur wird geschnitten?

    Attributes
    ----------
    per_loop          : loop_id -> (abgedeckte Punkte, Punkte gesamt)
    missing_positions : loop_id -> Positionen die NICHT abgedeckt sind
    """
    per_loop: dict[int, tuple[int, int]] = field(default_factory=dict)
    missing_positions: dict[int, list[int]] = field(default_factory=dict)

    @property
    def total_points(self) -> int:
        return sum(t for _, t in self.per_loop.values())

    @property
    def covered_points(self) -> int:
        return sum(c for c, _ in self.per_loop.values())

    @property
    def fraction(self) -> float:
        """Abdeckung [0..1] ueber alle Konturen."""
        t = self.total_points
        return self.covered_points / t if t > 0 else 0.0

    @property
    def missing_fraction(self) -> float:
        return 1.0 - self.fraction

    @property
    def is_complete(self) -> bool:
        return self.covered_points >= self.total_points

    def summary(self) -> str:
        lines = [
            f"Coverage: {self.fraction:.1%} "
            f"({self.covered_points}/{self.total_points} Konturpunkte)"
        ]
        for loop_id, (c, t) in sorted(self.per_loop.items()):
            missing = t - c
            status = "OK" if missing == 0 else f"{missing} Punkte fehlen"
            lines.append(f"  Loop {loop_id}: {c}/{t} ({c / max(1, t):.1%}) -- {status}")
        if not self.is_complete:
            lines.append(
                f"  => Objekt wird NICHT vollstaendig durchgeschnitten, "
                f"es fehlen {self.missing_fraction:.1%}."
            )
        return "\n".join(lines)


def compute_coverage(
    contour: SegmentedContour,
    runs: list[CutRun],
) -> CoverageReport:
    """Berechnet die Konturabdeckung der gewaehlten CutRuns."""
    report = CoverageReport()
    for loop in contour.loops:
        covered: set[int] = set()
        for run in runs:
            if run.loop_id == loop.loop_id:
                covered.update(run.positions)
        missing = [p for p in range(loop.n) if p not in covered]
        report.per_loop[loop.loop_id] = (loop.n - len(missing), loop.n)
        report.missing_positions[loop.loop_id] = missing
    return report


def covered_positions(
    contour: SegmentedContour,
    runs: list[CutRun],
    loop_id: int,
) -> set[int]:
    """Abgedeckte Positionen eines Loops (Hilfsfunktion fuer UI)."""
    covered: set[int] = set()
    for run in runs:
        if run.loop_id == loop_id:
            covered.update(run.positions)
    return covered


# ---------------------------------------------------------------------------
# Querschnitts-Coverage (alle Gitterpunkte: innen + aussen)
# ---------------------------------------------------------------------------

@dataclass
class GridCoverageReport:
    """Abdeckung der gesamten Querschnittsflaeche (alle Gitterpunkte).

    Das eigentliche Ziel: ALLE Punkte (Innen- und Aussenpunkte) muessen
    von der Klinge ueberstrichen werden, damit das Objekt vollstaendig
    zertrennt ist. 100 % = alle Gitterpunkte in der Swept Area.

    Attributes
    ----------
    mask : (N,) bool-Array ueber grid.coords -- True = geschnitten
    """
    total: int
    mask: np.ndarray

    @property
    def covered(self) -> int:
        return int(np.sum(self.mask))

    @property
    def fraction(self) -> float:
        return self.covered / self.total if self.total > 0 else 0.0

    @property
    def missing_fraction(self) -> float:
        return 1.0 - self.fraction

    @property
    def is_complete(self) -> bool:
        return self.covered >= self.total

    @property
    def missing_indices(self) -> np.ndarray:
        return np.where(~self.mask)[0]

    def summary(self) -> str:
        line = (f"Querschnitt-Coverage: {self.fraction:.1%} "
                f"({self.covered}/{self.total} Punkte)")
        if not self.is_complete:
            line += (f"\n  => NICHT vollstaendig durchtrennt, es fehlen "
                     f"{self.missing_fraction:.1%} "
                     f"({self.total - self.covered} Punkte).")
        return line


def compute_grid_coverage(grid: PointGrid, runs: list[CutRun]) -> GridCoverageReport:
    """Welche Gitterpunkte (innen + aussen) liegen in den Swept Areas?

    Beruecksichtigt nur feasible Runs mit berechneter Kinematik
    (siehe planning.RunKinematics.attach).
    """
    import shapely
    coords = grid.coords
    mask = np.zeros(len(coords), dtype=bool)
    for run in runs:
        if not run.is_feasible or run.swept_polygon is None:
            continue
        poly = run.swept_polygon
        if poly.is_empty:
            continue
        mask |= shapely.contains_xy(poly, coords[:, 0], coords[:, 1])
    return GridCoverageReport(total=len(coords), mask=mask)

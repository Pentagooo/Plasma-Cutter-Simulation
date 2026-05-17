from __future__ import annotations

"""Kontinuierliche Schneidpfade fuer Plasma-Roboter (Lichtschwert-Modell).

Der Brenner wird als Lichtschwert modelliert:

  - **Griff** (TCP): Die Position des Roboter-Endeffektors.
    Der Griff faehrt entlang der Aussenkontur oder eines Offset-Pfads.
  - **Klinge** (Plasmastrahl): Ragt senkrecht vom Griff nach innen
    ins Material.  Laenge = ``cutter.max_depth``, Breite = ``kerf_width``.

Waehrend der Griff sich bewegt, "wischt" die Klinge wie ein Besen
durch das Material und traegt alles in der Swept Area ab.

Die Swept Area wird geometrisch als Polygon berechnet:
  Links-Rand  = Griff-Pfad
  Rechts-Rand = Klingenspitzen-Pfad (Griff + Normale * blade_length)
  Die Flaeche dazwischen (+ kerf_width/2 Puffer) ist der Materialabtrag.

Zwei Modi:

  1) **Manueller Pfad** -- Benutzer gibt Griff-Waypoints vor.

Typischer Ablauf::

    planner = ContinuousPlanner(grid, cutter, kerf_width=3.0)
    result  = planner.plan_custom(segments)
    reward  = calculate_reward(result)
"""

from dataclasses import dataclass, field
import math
import numpy as np
from shapely.geometry import (
    LineString, Polygon, Point, MultiPolygon, MultiLineString,
)
from shapely.ops import unary_union

from ..geometry.point_grid import PointGrid, GridPoint, CutPoint, PointStatus
from .cutter import Cutter
from .assumptions import CuttingAssumptions


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------

@dataclass
class ContinuousCut:
    """Eine zusammenhaengende Trajektorie (ein Lichtschwert-Durchgang).

    Attributes
    ----------
    grip_waypoints  : (x,y)-Positionen des Griffs (TCP) [mm]
    blade_waypoints : (x,y)-Positionen der Klingenspitze [mm]
    cutting_time    : Zeitbedarf inkl. Pierce-Pauschale [s]
    pierce_time     : Anteil der Wiedereintrittspauschale [s]
    is_feasible     : True wenn keine Kollision
    cut_points      : GridPoints die von der Klinge getroffen wurden
    swept_polygon   : Shapely Polygon der Swept Area (Klinge + Kerf)
    is_cutting      : True wenn Brenner an, False wenn Brenner aus (Eilgang)
    tilt_angles     : pro Waypoint angewendeter Brennertilt [rad] (Laenge n)
    bevel_angle     : globaler Y-Fasenwinkel des Schnitts [rad].
                      0 = senkrecht, >0 = Fase (Standard 45deg bei Lochrand).
    is_bevel        : True wenn dies ein 45deg-Fasen-Schnitt ist
                      (Y-Groove an Innenrandpunkt, vgl. Lit. Liu 2024)
    """
    grip_waypoints:  list[np.ndarray]
    blade_waypoints: list[np.ndarray] = field(default_factory=list)
    cutting_time:    float = 0.0
    pierce_time:     float = 0.0
    is_feasible:     bool = True
    cut_points:      list[CutPoint] = field(default_factory=list)
    swept_polygon:   Polygon | MultiPolygon | None = None
    is_cutting:      bool = True
    tilt_angles:     list[float] = field(default_factory=list)
    bevel_angle:     float = 0.0
    is_bevel:        bool = False

    @property
    def n_waypoints(self) -> int:
        return len(self.grip_waypoints)

    @property
    def path_length(self) -> float:
        """Laenge des Griff-Pfads [mm]."""
        return _polyline_length(self.grip_waypoints)


@dataclass
class ContinuousPathResult:
    """Ergebnis einer kontinuierlichen Schneidplanung."""
    continuous_cuts:    list[ContinuousCut]
    total_cutting_time: float
    total_travel_time:  float
    total_time:         float
    all_cut_points:     list[CutPoint]
    material_removed:   float
    strategy:           str  # z.B. 'custom'
    total_area:         float = 0.0  # Gesamtflaeche des Querschnitts [mm^2]
    total_points:       int   = 0    # Gesamtanzahl Grid-Punkte

    @property
    def coverage(self) -> float:
        """Anteil der geschnittenen Punkte [0..1]."""
        if self.total_points <= 0:
            return 0.0
        return min(1.0, self.total_points_cut / self.total_points)

    @property
    def n_pierces(self) -> int:
        pierces = 0
        was_cutting = False
        for c in self.continuous_cuts:
            if c.is_cutting and not was_cutting:
                pierces += 1
            was_cutting = c.is_cutting
        return pierces

    @property
    def total_pierce_time(self) -> float:
        """Aufsummierte Wiedereintrittspauschale ueber alle Zuendungen [s]."""
        return sum(c.pierce_time for c in self.continuous_cuts if c.is_cutting)

    @property
    def n_bevel_cuts(self) -> int:
        """Anzahl 45deg-Y-Fasen-Schnitte (an Innenrandpunkten)."""
        return sum(1 for c in self.continuous_cuts if c.is_bevel)

    @property
    def total_points_cut(self) -> int:
        return len(self.all_cut_points)

    @property
    def total_path_length(self) -> float:
        return sum(c.path_length for c in self.continuous_cuts)

    def summary(self) -> str:
        bevel_info = f" | Fasen: {self.n_bevel_cuts}" if self.n_bevel_cuts else ""
        pierce_info = (
            f" | Pierce: {self.total_pierce_time:.2f}s"
            if self.total_pierce_time > 0 else ""
        )
        return (
            f"ContinuousPathResult: {self.n_pierces} Zuendung(en) | "
            f"Strategie: {self.strategy} | "
            f"Griff-Pfad: {self.total_path_length:.1f} mm | "
            f"Zeit: {self.total_time:.1f} s{pierce_info}{bevel_info} | "
            f"Punkte: {self.total_points_cut} | "
            f"Material: {self.material_removed:.1f} mm^2 | "
            f"Coverage: {self.coverage * 100:.1f}%"
        )


# ---------------------------------------------------------------------------
# ContinuousPlanner
# ---------------------------------------------------------------------------

class ContinuousPlanner:
    """Lichtschwert-Planer fuer kontinuierliche Schneidpfade.

    Parameters
    ----------
    grid            : PointGrid mit der Geometrie
    cutter          : Cutter (max_depth = Klingen-Laenge, minimum_gap)
    kerf_width      : Breite des Schnittspalts [mm]
    rng             : numpy.random.Generator fuer reproduzierbare
                      Brennerwinkel-Streuung (None -> default_rng())
    """

    def __init__(
        self,
        grid: PointGrid,
        cutter: Cutter,
        kerf_width: float = 3.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.grid = grid
        self.cutter = cutter
        self.kerf_width = kerf_width
        self.minimum_gap = cutter.minimum_gap
        self._rng = rng or np.random.default_rng()

        # Annahmen-Bundle vom Cutter durchreichen
        self.assumptions: CuttingAssumptions = cutter.assumptions

        # Effektive Klingen-Laenge: nutzt L(v) wenn aktiviert
        self.blade_length = cutter.blade_length()

        # Effektive Schnitttiefe pro Ring = Klinge minus Sicherheitsabstand
        self._effective_depth = max(0.0, self.blade_length - self.minimum_gap)

    # ------------------------------------------------------------------
    # Geometrie aus PointGrid
    # ------------------------------------------------------------------

    def _build_polygon_from_grid(self) -> Polygon | MultiPolygon | None:
        """Baut das Material-Polygon (Aussenkontur abzueglich Loch) auf.

        Robust gegen ungueltige (selbst-schneidende) Konturen:
        ``buffer(0)`` repariert die Aussenkontur (kann MultiPolygon
        zurueckgeben, das wird beibehalten). Wenn eine Lochkontur
        existiert, wird sie davon abgezogen.

        Wichtig: Rueckwaerts-kompatibel zur Original-Implementierung,
        die MultiPolygon-Resultate erlaubt -- ``contains()``,
        ``distance()`` etc. funktionieren auch darauf.
        """
        ordered = self.grid.outer_points_ordered
        if len(ordered) < 3:
            return None
        outer_coords = [(p.x, p.y) for p in ordered]
        outer_coords.append(outer_coords[0])
        outer_poly = Polygon(outer_coords)
        if not outer_poly.is_valid:
            outer_poly = outer_poly.buffer(0)
        if outer_poly is None or outer_poly.is_empty:
            return None

        hole_pts = self.grid.hole_points_ordered
        if len(hole_pts) >= 3:
            hcoords = [(p.x, p.y) for p in hole_pts]
            hcoords.append(hcoords[0])
            hole_poly = Polygon(hcoords)
            if not hole_poly.is_valid:
                hole_poly = hole_poly.buffer(0)
            if hole_poly is not None and not hole_poly.is_empty:
                try:
                    outer_poly = outer_poly.difference(hole_poly)
                except Exception:
                    pass

        return outer_poly if not outer_poly.is_empty else None

    # ------------------------------------------------------------------
    # Normalen-Berechnung entlang eines Pfads
    # ------------------------------------------------------------------

    def _compute_inward_normals(
        self,
        waypoints: list[np.ndarray],
        poly: Polygon | None = None,
    ) -> list[np.ndarray]:
        """Berechnet fuer jeden Waypoint den Einheits-Normalenvektor zur Geometrie.

        1. Tangenten-basierte Normale (90-Grad-Drehung der lokalen Tangente)
        2. Schwerpunkt-Check: Normale zeigt zum Schwerpunkt
        3. Polygon-Validierung: prueft ob die Klingenspitze tatsaechlich
           naeher am / im Polygon liegt als die Gegenrichtung.
        """
        n = len(waypoints)
        if n < 2:
            return [np.array([0.0, 1.0])] * n

        # Schwerpunkt fuer Richtungsbestimmung
        if poly is not None and not poly.is_empty:
            centroid = np.array([poly.centroid.x, poly.centroid.y])
        else:
            # Nutzt das neue coords-Property (Vektorisierter Zugriff)
            centroid = self.grid.coords.mean(axis=0)

        normals: list[np.ndarray] = []
        for i in range(n):
            # Tangente aus Nachbarpunkten
            i0 = max(0, i - 1)
            i1 = min(n - 1, i + 1)
            tangent = np.asarray(waypoints[i1], dtype=float) - np.asarray(waypoints[i0], dtype=float)
            length = np.linalg.norm(tangent)
            if length < 1e-9:
                normals.append(np.array([0.0, 1.0]))
                continue
            tangent /= length

            # 90-Grad-Drehung -> Normale
            normal = np.array([-tangent[1], tangent[0]])

            # Richtung nach innen (zum Schwerpunkt)
            pos = np.asarray(waypoints[i], dtype=float)
            if np.dot(normal, centroid - pos) < 0:
                normal = -normal

            normals.append(normal)

        # Polygon-basierte Validierung: Klingenspitze muss naeher am
        # Polygon liegen als die Gegenrichtung
        if poly is not None and not poly.is_empty:
            for i in range(n):
                pos = np.asarray(waypoints[i], dtype=float)
                normal = normals[i]
                tip = pos + normal * self.blade_length
                alt = pos - normal * self.blade_length

                tip_pt = Point(float(tip[0]), float(tip[1]))
                alt_pt = Point(float(alt[0]), float(alt[1]))

                tip_inside = poly.contains(tip_pt)
                alt_inside = poly.contains(alt_pt)

                # Flip wenn Gegenrichtung im Polygon liegt, Spitze aber nicht
                if alt_inside and not tip_inside:
                    normals[i] = -normal
                # Flip wenn beide aussen, aber Gegenrichtung naeher am Polygon
                elif not tip_inside and not alt_inside:
                    if poly.distance(tip_pt) > poly.distance(alt_pt):
                        normals[i] = -normal

        return normals

    # ------------------------------------------------------------------
    # Tangenten entlang des Pfads (fuer Tilt-Berechnung)
    # ------------------------------------------------------------------

    def _compute_tangents(
        self,
        waypoints: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Einheits-Tangentenvektor an jedem Waypoint."""
        n = len(waypoints)
        if n == 0:
            return []
        if n == 1:
            return [np.array([1.0, 0.0])]

        tangents: list[np.ndarray] = []
        for i in range(n):
            i0 = max(0, i - 1)
            i1 = min(n - 1, i + 1)
            t = np.asarray(waypoints[i1], dtype=float) - np.asarray(waypoints[i0], dtype=float)
            nrm = np.linalg.norm(t)
            if nrm < 1e-9:
                tangents.append(np.array([1.0, 0.0]))
            else:
                tangents.append(t / nrm)
        return tangents

    # ------------------------------------------------------------------
    # TCP-Pfad Validierung (Clearance + Kollision)
    # ------------------------------------------------------------------

    def _validate_grip_path(
        self,
        grip_wps: list[np.ndarray],
        poly: Polygon | None,
    ) -> tuple[list[np.ndarray], bool]:
        """Stellt sicher, dass der TCP-Pfad minimum_gap vom Material haelt
        und nicht durch das Material geht.

        Waypoints die zu nah am Material sind oder im Material liegen,
        werden nach aussen auf die sichere Distanz projiziert.

        Returns
        -------
        validated_wps : korrigierte Waypoints
        is_feasible   : True wenn der Pfad ohne Korrektur gueltig war
        """
        if poly is None or poly.is_empty or self.minimum_gap <= 0:
            return grip_wps, True

        centroid = np.array([poly.centroid.x, poly.centroid.y])
        validated: list[np.ndarray] = []
        was_modified = False

        for wp in grip_wps:
            pos = np.asarray(wp, dtype=float)
            pt = Point(float(pos[0]), float(pos[1]))

            inside = poly.contains(pt)
            dist_to_boundary = poly.boundary.distance(pt)

            if inside or dist_to_boundary < self.minimum_gap:
                was_modified = True
                # Naechster Punkt auf der Polygon-Grenze
                proj = poly.boundary.project(pt)
                nearest_pt = poly.boundary.interpolate(proj)
                nearest = np.array([nearest_pt.x, nearest_pt.y])

                # Auswarts-Richtung bestimmen
                if inside:
                    # Im Material: Richtung von Schwerpunkt weg
                    outward = nearest - centroid
                else:
                    # Zu nah: Richtung vom Polygon weg (von nearest zu pos)
                    outward = pos - nearest

                norm = np.linalg.norm(outward)
                if norm < 1e-9:
                    outward = nearest - centroid
                    norm = np.linalg.norm(outward)
                    if norm < 1e-9:
                        outward = np.array([1.0, 0.0])
                        norm = 1.0

                outward /= norm
                validated.append(nearest + outward * self.minimum_gap)
            else:
                validated.append(pos)

        # Pruefen ob der gesamte Pfad das Material schneidet
        if len(validated) >= 2:
            grip_line = LineString(
                [(float(w[0]), float(w[1])) for w in validated])
            if grip_line.intersects(poly):
                was_modified = True

        return validated, not was_modified

    # ------------------------------------------------------------------
    # Lichtschwert Swept Area berechnen
    # ------------------------------------------------------------------

    def _compute_blade_sweep(
        self,
        grip_wps: list[np.ndarray],
        normals: list[np.ndarray],
        tilt_angles: list[float] | np.ndarray | None = None,
        bevel_angle: float = 0.0,
    ) -> tuple[list[np.ndarray], Polygon | None]:
        """Berechnet Klingenspitzen-Pfad und Swept-Area-Polygon.

        Fuer jeden Griff-Punkt:

            klinge_spitze = griff
                         + cos(tilt) * normal * L
                         + sin(tilt) * tangent * L

        Der Tilt setzt sich zusammen aus
          - dem konstanten Y-Fasenwinkel ``bevel_angle`` (Standard 0
            = senkrecht; 45deg bei Innenrand-Fase, B5), und
          - waypoint-spezifischen Stoertilts ``tilt_angles[i]``
            (Wahrscheinlichkeitsverteilung aus assumptions.tilt).

        Der Tangent zeigt entlang der Pfadrichtung; positives Tilt
        bedeutet "Klinge kippt in Bewegungsrichtung".

        Returns
        -------
        blade_wps     : Klingenspitzen-Positionen
        swept_polygon : Polygon der gesamten Swept Area
        """
        n = len(grip_wps)
        if tilt_angles is None:
            tilt_angles = np.zeros(n)
        else:
            tilt_angles = np.asarray(tilt_angles, dtype=float)
            if tilt_angles.shape[0] != n:
                tilt_angles = np.zeros(n)

        # Tangenten aus Nachbarpunkten (fuer Tilt-Richtung)
        tangents = self._compute_tangents(grip_wps)

        blade_wps: list[np.ndarray] = []
        for i in range(n):
            g = np.asarray(grip_wps[i], dtype=float)
            normal = np.asarray(normals[i], dtype=float)
            tangent = tangents[i]

            theta = float(tilt_angles[i]) + bevel_angle
            tip = g + math.cos(theta) * normal * self.blade_length \
                    + math.sin(theta) * tangent * self.blade_length
            blade_wps.append(tip)

        if n < 2:
            return blade_wps, None

        # Polygon bauen: Griff-Pfad vorwaerts + Klingenspitzen-Pfad rueckwaerts
        # Das ergibt ein Band-Polygon (wie ein Streifen)
        grip_coords = [(float(w[0]), float(w[1])) for w in grip_wps]
        blade_coords = [(float(w[0]), float(w[1])) for w in blade_wps]

        # Band: Griff vorwaerts -> Klinge rueckwaerts -> schliessen
        band_coords = grip_coords + list(reversed(blade_coords))
        band_coords.append(band_coords[0])  # Ring schliessen

        try:
            band = Polygon(band_coords)
            if not band.is_valid:
                band = band.buffer(0)
            # Kerf-Puffer: die Klinge hat auch eine Breite
            swept = band.buffer(self.kerf_width / 2)
            if isinstance(swept, MultiPolygon):
                swept = unary_union(swept)
            if isinstance(swept, MultiPolygon):
                swept = max(swept.geoms, key=lambda g: g.area)
        except Exception:
            # Fallback: Buffer um den Griff-Pfad
            line = LineString(grip_coords)
            swept = line.buffer(self.blade_length + self.kerf_width / 2)

        return blade_wps, swept if isinstance(swept, Polygon) else None

    # ------------------------------------------------------------------
    # Einzelnen Cut ausfuehren
    # ------------------------------------------------------------------

    def _execute_cut(
        self,
        grip_wps:    list[np.ndarray],
        poly:        Polygon | None = None,
        apply:       bool  = True,
        validate:    bool  = True,
        bevel_angle: float = 0.0,
        is_bevel:    bool  = False,
    ) -> ContinuousCut:
        """Fuehrt einen Lichtschwert-Durchgang aus.

        1. TCP-Pfad validieren (Clearance + Kollision)
        2. Normalen berechnen (Klinge zeigt zur Geometrie)
        3. Brennertilt entlang des Pfads sampeln (B/A2*)
        4. Klingenspitzen-Pfad + Swept Area berechnen
        5. Alle Grid-Punkte in der Swept Area als geschnitten markieren
        6. Zeitbedarf inkl. Pierce-Pauschale (B3)

        Parameters
        ----------
        bevel_angle : globaler Tilt (Y-Fasenwinkel) [rad]. 0 = senkrecht,
                      pi/4 = 45deg-Fase (z.B. an Innenrand, B5).
        is_bevel    : Flag, ob dieser Schnitt eine Fase ist.
        """
        # TCP-Validierung: Clearance einhalten, nicht durch Material fahren
        if validate and poly is not None:
            _, is_feasible = self._validate_grip_path(grip_wps, poly)
        else:
            is_feasible = True

        # Bei Verletzung der Constraints: Schnitt NICHT ausfuehren
        if not is_feasible:
            normals = self._compute_inward_normals(grip_wps, poly)
            blade_wps = [
                np.asarray(g, dtype=float) + n * self.blade_length
                for g, n in zip(grip_wps, normals)
            ]
            return ContinuousCut(
                grip_waypoints=grip_wps,
                blade_waypoints=blade_wps,
                cutting_time=0.0,
                pierce_time=0.0,
                is_feasible=False,
                cut_points=[],
                swept_polygon=None,
                bevel_angle=bevel_angle,
                is_bevel=is_bevel,
            )

        normals = self._compute_inward_normals(grip_wps, poly)

        # A2*: Brennerwinkel-Streuung als glatte Trajektorie
        path_len = _polyline_length(grip_wps)
        n_wps = len(grip_wps)
        seg_len = max(path_len / max(1, n_wps - 1), 1e-6)
        tilt_angles = self.assumptions.sample_tilts(
            n_wps, segment_length=seg_len, rng=self._rng,
        )

        blade_wps, swept_poly = self._compute_blade_sweep(
            grip_wps, normals,
            tilt_angles=tilt_angles,
            bevel_angle=bevel_angle,
        )

        # Schneidzeit + Wiedereintrittspauschale (B3)
        cutting_time = self.cutter.time_for_length(path_len, mode="cut")
        pierce_time  = self.cutter.pierce_time()
        total_time   = cutting_time + pierce_time

        cut_points: list[CutPoint] = []
        if apply and swept_poly is not None:
            cut_points = self.grid.apply_swept_area(swept_poly)

        return ContinuousCut(
            grip_waypoints=grip_wps,
            blade_waypoints=blade_wps,
            cutting_time=total_time,
            pierce_time=pierce_time,
            is_feasible=True,
            cut_points=cut_points,
            swept_polygon=swept_poly,
            tilt_angles=list(tilt_angles),
            bevel_angle=bevel_angle,
            is_bevel=is_bevel,
        )

    # ------------------------------------------------------------------
    # Manueller Pfad
    # ------------------------------------------------------------------

    def plan_custom(
        self,
        segments: list,
        apply: bool = True,
    ) -> ContinuousPathResult:
        """Fuehrt einen benutzerdefinierten Griff-Pfad (oder mehrere) aus.

        Akzeptiert zwei Segment-Formate fuer Rueckwaertskompatibilitaet:

          (a) (waypoints, is_cutting)                  -- alt
          (b) (waypoints, is_cutting, bevel_angle_rad) -- mit Y-Fase

        ``bevel_angle_rad`` ist 0 fuer senkrechte Schnitte (Standard)
        und z.B. ``pi/4`` fuer eine 45deg-Y-Fase an Innenrandpunkten
        (siehe HoleBevelPolicy / B5).

        Wenn ein TCP-Pfad die Clearance-Grenze verletzt (TCP zu nah am
        oder im Material), wird der Schnitt NICHT ausgefuehrt und das
        Ergebnis hat is_feasible=False.
        """
        if not apply:
            saved_ci = self.grid._cut_indices.copy()
            saved_cp = list(self.grid._cut_points)

        all_cuts: list[ContinuousCut] = []
        all_cut_points: list[CutPoint] = []
        total_cutting_time = 0.0
        total_travel_time = 0.0
        total_material_removed = 0.0

        if not segments:
            result = self._empty_result("custom")
            if not apply:
                self.grid._cut_indices = saved_ci
                self.grid._cut_points = saved_cp
            return result

        # Vorabnormalisierung: Segmente koennen 2- oder 3-Tuple sein
        _check_norm = [self._normalize_segment(s) for s in segments]
        if all(len(wps) < 2 for wps, _, _ in _check_norm):
            result = self._empty_result("custom")
            if not apply:
                self.grid._cut_indices = saved_ci
                self.grid._cut_points = saved_cp
            return result

        poly = self._build_polygon_from_grid()
        t_area = poly.area if poly is not None and not poly.is_empty else 0.0

        # Normalisierung: jedes Segment auf (wps, is_cutting, bevel) bringen
        norm_segments = [self._normalize_segment(s) for s in segments]

        for i, (wps, is_cutting, bevel) in enumerate(norm_segments):
            if len(wps) < 2:
                continue

            if is_cutting:
                cut = self._execute_cut(
                    wps, poly=poly, apply=apply,
                    bevel_angle=bevel,
                    is_bevel=(abs(bevel) > 1e-6),
                )
                cut.is_cutting = True
                total_cutting_time += cut.cutting_time
                if cut.swept_polygon:
                    total_material_removed += cut.swept_polygon.area
            else:
                _, rapid_feasible = self._validate_grip_path(wps, poly)
                air_dist = _polyline_length(wps)
                t_rapid = self.cutter.time_for_length(air_dist, mode="rapid")
                cut = ContinuousCut(
                    grip_waypoints=wps,
                    blade_waypoints=wps, # Dummy Klinge
                    cutting_time=t_rapid,
                    pierce_time=0.0,
                    is_feasible=rapid_feasible,
                    cut_points=[],
                    swept_polygon=None,
                    is_cutting=False,
                    bevel_angle=0.0,
                    is_bevel=False,
                )
                total_travel_time += t_rapid

            all_cuts.append(cut)
            all_cut_points.extend(cut.cut_points)

            # Verfahrzeit (rapid mode) zwischen den Segmenten
            if i > 0:
                prev_end = norm_segments[i-1][0][-1]
                curr_start = wps[0]
                air_dist = float(np.linalg.norm(np.asarray(curr_start) - np.asarray(prev_end)))
                if air_dist > 1e-6:
                    total_travel_time += self.cutter.time_for_length(air_dist, mode="rapid")
                    _, rapid_feasible = self._validate_grip_path([prev_end, curr_start], poly)
                    if not rapid_feasible:
                        cut.is_feasible = False

        result = ContinuousPathResult(
            continuous_cuts=all_cuts,
            total_cutting_time=total_cutting_time,
            total_travel_time=total_travel_time,
            total_time=total_cutting_time + total_travel_time,
            all_cut_points=all_cut_points,
            material_removed=total_material_removed,
            strategy="custom",
            total_area=t_area,
            total_points=self.grid.total_points,
        )

        if not apply:
            self.grid._cut_indices = saved_ci
            self.grid._cut_points = saved_cp

        return result

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_segment(seg) -> tuple[list[np.ndarray], bool, float]:
        """Wandelt ein Segment in das interne (wps, is_cutting, bevel)-Format.

        Akzeptiert (wps, is_cutting), (wps, is_cutting, bevel) oder
        ein Dict {"wps": ..., "is_cutting": ..., "bevel": ...}.
        """
        if isinstance(seg, dict):
            return (
                seg["wps"],
                bool(seg.get("is_cutting", True)),
                float(seg.get("bevel", 0.0)),
            )
        if len(seg) == 2:
            wps, is_cutting = seg
            return wps, bool(is_cutting), 0.0
        if len(seg) == 3:
            wps, is_cutting, bevel = seg
            return wps, bool(is_cutting), float(bevel)
        raise ValueError(f"Unbekanntes Segmentformat: {seg!r}")

    # ------------------------------------------------------------------
    # Innenrand-Y-Fase (B5)
    # ------------------------------------------------------------------

    def plan_hole_bevel_segments(
        self,
        margin: float | None = None,
    ) -> list[tuple[list[np.ndarray], bool, float]]:
        """Erzeugt automatisch Y-Fasen-Segmente entlang der Lochkontur.

        Ablauf laut Best-Practice (Lit. [5], Hypertherm "Bevel cutting"):
        1) Senkrechter Schnitt entlang der Lochkontur (Land first)
        2) 45deg-Y-Fase entlang derselben Kontur (Bevel second)

        Die Reihenfolge wird durch ``assumptions.hole.cut_perp_first``
        bestimmt; wenn False wird der Y-Fasen-Schnitt zuerst gemacht.

        Parameters
        ----------
        margin : minimaler Abstand des TCP zur Lochkante [mm].
                 None -> minimum_gap des Cutters.

        Returns
        -------
        Liste von Segmenten im Format (waypoints, is_cutting, bevel_rad).
        Leer wenn kein Loch existiert oder Y-Fasen deaktiviert sind.
        """
        if not (self.assumptions.use_hole_bevel and self.assumptions.hole.enabled):
            return []

        hole_pts = self.grid.hole_points_ordered
        if len(hole_pts) < 3:
            return []

        # Etwas groesserer Sicherheitsabstand gegen numerische Diskrepanz
        # zwischen Lochpunkten und dem via buffer(0) reparierten
        # Material-Polygonrand (sonst kann der Wegpunkt knapp im
        # minimum_gap-Bereich landen).
        margin = (self.minimum_gap + self.kerf_width) if margin is None else margin

        # TCP-Pfad: Lochkontur, nach **innen** ins Loch versetzt.
        # Der Brenner schwebt also in der Lochoeffnung (freie Luft) und
        # die Klinge greift radial nach aussen ins Material -- analog
        # zum Standard-Aussenkontur-Schnitt, nur "von innen".
        hole_coords = np.array([[p.x, p.y] for p in hole_pts])
        hole_centroid = hole_coords.mean(axis=0)

        offset_wps: list[np.ndarray] = []
        for i in range(len(hole_coords)):
            p = hole_coords[i]
            # Vektor von Lochpunkt zum Lochzentrum (= "ins Loch hinein")
            inward = hole_centroid - p
            norm = np.linalg.norm(inward)
            if norm > 1e-9:
                offset_wps.append(p + inward / norm * margin)
            else:
                offset_wps.append(p.copy())
        # Ring schliessen
        offset_wps.append(offset_wps[0].copy())

        bevel = self.assumptions.hole.bevel_angle
        segments: list[tuple[list[np.ndarray], bool, float]] = []
        if self.assumptions.hole.cut_perp_first:
            segments.append((offset_wps, True, 0.0))     # land
            segments.append((offset_wps, True, bevel))   # bevel
        else:
            segments.append((offset_wps, True, bevel))
            segments.append((offset_wps, True, 0.0))
        return segments

    def _empty_result(self, strategy: str) -> ContinuousPathResult:
        poly = self._build_polygon_from_grid()
        t_area = poly.area if poly is not None and not poly.is_empty else 0.0
        return ContinuousPathResult(
            continuous_cuts=[], total_cutting_time=0, total_travel_time=0,
            total_time=0, all_cut_points=[], material_removed=0,
            strategy=strategy, total_area=t_area,
            total_points=self.grid.total_points,
        )


# ---------------------------------------------------------------------------
# Reward-Funktion
# ---------------------------------------------------------------------------

def calculate_reward(
    result: ContinuousPathResult,
    coverage_override: float | None = None,
    w_coverage: float = 100.0,
    w_completion: float = 50.0,
    w_time: float = 0.5,
    w_pierce: float = 2.0,
) -> float:
    """Reward fuer ML-Training – optimiert auf 100% Punkt-Abdeckung.

    Primaeres Ziel : 100% aller Punkte schneiden.
    Sekundaere Ziele: geringe Gesamtzeit, wenige Zuendungen (Schnitte).

    Parameters
    ----------
    result            : Ergebnis einer Schneidplanung
    coverage_override : Optionaler Coverage-Wert [0..1], z.B. vom Grid
                        (kumulativ ueber mehrere Ergebnisse).
                        Wenn None, wird result.coverage verwendet.

    Berechnung
    ----------
    coverage = geschnittene_punkte / gesamt_punkte   (0..1)

    reward = w_coverage  * coverage^2           (quadratisch: letzte % zaehlen mehr)
           + w_completion * bonus               (Bonus bei >= 99.5% Abdeckung)
           - w_time      * total_time           (Zeitstrafe)
           - w_pierce    * (n_pierces - 1)      (Zuendungsstrafe)

    Die quadratische Skalierung sorgt dafuer, dass die letzten Prozent
    ueberproportional belohnt werden:
        0% -> 50%   bringt  25 / 100 Punkte
       50% -> 100%  bringt  75 / 100 Punkte
    """
    coverage = coverage_override if coverage_override is not None else result.coverage
    if result.total_points <= 0 and coverage_override is None:
        return 0.0

    # Hauptbelohnung: quadratische Coverage
    reward = w_coverage * coverage ** 2

    # Grosser Bonus fuer (nahezu) vollstaendige Abdeckung
    if coverage >= 0.995:
        reward += w_completion

    # Strafen (nur relevant zum Differenzieren bei aehnlicher Coverage)
    reward -= w_time * result.total_time
    reward -= w_pierce * max(0, result.n_pierces - 1)

    return reward


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _polyline_length(waypoints: list[np.ndarray]) -> float:
    if len(waypoints) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(
            np.asarray(waypoints[i + 1]) - np.asarray(waypoints[i])
        ))
        for i in range(len(waypoints) - 1)
    )


def _largest_polygon(geom) -> Polygon | None:
    """Aus einem evtl. MultiPolygon/Geometry-Mix das groesste Polygon
    nach Flaeche zurueckgeben. None wenn nichts brauchbares dabei ist."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if geom.area > 0 else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if isinstance(g, Polygon) and g.area > 0]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
    # GeometryCollection o.ae.: nach Polygonen filtern
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    if not polys:
        return None
    return max(polys, key=lambda g: g.area)

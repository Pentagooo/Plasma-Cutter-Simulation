"""Verbindungs- und Reihenfolgeplanung.

RunKinematics : TCP-Pfad, Klingenspitzen und Swept Area eines CutRuns
LinkPlanner   : kollisionsfreie Eilgangwege zwischen TCP-Punkten (direkt
                oder Sichtbarkeitsgraph + Dijkstra); ohne Weg ist der
                Übergang nicht ausführbar, kein Überflug über das Material
Sequencer     : Reihenfolge + Richtung der Runs, exakt per Held-Karp;
                minimiert Eilgang + Pierce (Schnittzeiten sind konstant)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep

try:
    from ..cutter.cutter import Cutter
    from .segments import SegmentedContour, CutRun
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.cutter.cutter import Cutter
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, CutRun,
    )


# Endpunkte näher als CHAIN_TOL = nahtlos (kein Eilgang, keine Zündung)
CHAIN_TOL = 1e-6

# max. Abstand der TCP-Stützpunkte [mm]: Klinge dreht an Ecken stetig mit
# (sonst fehlt der Eckfächer in der Swept Area)
TCP_SAMPLE_STEP = 3.0

# Eckenregel: An einer Konturecke ist die Projektion auf den Offset-Ring
# mehrdeutig (der ganze Eckbogen ist gleich weit entfernt). Daher
# TCP-Punkt = Bogenmitte (p + clearance * Winkelhalbierende):
#   - eindeutig und richtungsunabhängig
#   - gleiche Station für beide Nachbarsegmente (nahtloses Verketten)
#   - Eckbogen hälftig geteilt, Summe der Segmentzeiten = durchgehender Schnitt

# ---------------------------------------------------------------------------
# Punktezahl (Score)
# ---------------------------------------------------------------------------
# weniger Zeit = mehr Punkte, skaliert mit der Coverage
SCORE_BASE = 1000.0        # Punkte bei Zeit 0 und voller Coverage
SCORE_TIME_WEIGHT = 1.0    # Abzug pro Sekunde


def compute_score(total_time: float, coverage: float) -> float:
    """Punktezahl eines Plans.

    score = coverage * max(0, SCORE_BASE - SCORE_TIME_WEIGHT * Zeit)

    Parameters
    ----------
    total_time : Gesamtzeit des Plans [s]
    coverage   : Querschnitts-Coverage [0..1]
    """
    return float(coverage) * max(
        0.0, SCORE_BASE - SCORE_TIME_WEIGHT * float(total_time))


# ---------------------------------------------------------------------------
# RunKinematics
# ---------------------------------------------------------------------------

class RunKinematics:
    """TCP-Pfad, Klingenspitzen und Swept Area eines CutRuns.

      - TCP auf dem Offset-Pfad (Rand von ``material.buffer(clearance)``)
      - Klinge vom TCP Richtung Objekt, Länge L(v); Schnitttiefe
        L(v) - clearance
      - Swept Area = Vereinigung der Vierecke (TCP_i, TCP_i+1,
        Spitze_i+1, Spitze_i), mit kerf/2 gepuffert

    Parameters
    ----------
    material     : Material-Polygon (Außenkontur minus Löcher)
    clearance    : konstanter Mindestabstand TCP-Material [mm]
    blade_length : Klingenlänge L(v) bei der Schnittgeschwindigkeit [mm]
    kerf         : Schnittspaltbreite [mm]
    """

    def __init__(
        self,
        material: Polygon | MultiPolygon | None,
        clearance: float,
        blade_length: float,
        kerf: float = 3.0,
    ) -> None:
        self.material = material
        self.clearance = float(clearance)
        self.blade_length = float(blade_length)
        self.kerf = float(kerf)

        self._rings: list[LineString] = []
        self._mat_boundary = None
        self._bnd_rings: list[np.ndarray] = []
        if material is not None and not material.is_empty:
            self._mat_boundary = material.boundary
            offset_boundary = material.buffer(self.clearance).boundary
            if offset_boundary.geom_type == "LineString":
                self._rings = [offset_boundary]
            else:
                self._rings = list(offset_boundary.geoms)
            # Konturecken für die Eckenregel (ohne doppelten Schlusspunkt)
            geoms = (self._mat_boundary.geoms
                     if hasattr(self._mat_boundary, "geoms")
                     else [self._mat_boundary])
            self._bnd_rings = [np.asarray(g.coords, dtype=float)[:-1]
                               for g in geoms if len(g.coords) > 2]

    @property
    def effective_depth(self) -> float:
        """Schnitttiefe ab Materialoberfläche [mm]."""
        return self.blade_length - self.clearance

    # ------------------------------------------------------------------

    def attach(self, run: CutRun) -> CutRun:
        """Berechnet die Kinematik und schreibt sie in den Run (in-place).

        is_feasible=False, wenn kein Offset-Pfad existiert (z.B. Loch zu
        klein) oder die Klinge das Material nicht erreicht.
        """
        if self._mat_boundary is None or not self._rings:
            run.is_feasible = False
            run.reason = "kein Material-Polygon"
            return run
        if self.effective_depth <= 0:
            run.is_feasible = False
            run.reason = (f"Klinge zu kurz: L(v)={self.blade_length:.1f} mm "
                          f"<= Mindestabstand {self.clearance:.1f} mm")
            return run

        pts = run.polyline
        # Offset-Ring = Ring mit dem kleinsten max. Abstand über eine
        # Stichprobe (der Mittelpunkt allein trifft bei langen Bögen ggf.
        # den Loch-Ring)
        sample = pts[::max(1, len(pts) // 8)]
        if len(sample) == 0 or not np.array_equal(sample[-1], pts[-1]):
            sample = list(sample) + [pts[-1]]
        ring = min(self._rings, key=lambda r: max(
            r.distance(Point(tuple(p))) for p in sample))
        mid = Point(tuple(pts[len(pts) // 2]))
        if ring.distance(mid) > self.clearance * 1.5:
            run.is_feasible = False
            run.reason = ("kein Offset-Pfad im Mindestabstand erreichbar "
                          "(Kontur zu eng fuer den TCP)")
            return run

        # Stationen auf dem Ring (Eckenregel) als Stützstellen des TCP-Pfads:
        #   - Stützstellen = Segmentknoten (ohne node_positions: alle Punkte)
        #   - verschachtelt: Merge-Pfad enthält die Knoten der Einzelsegmente
        #     -> Swept Area des Merges = Vereinigung, kein Coverage-Verlust
        #   - alle Konturpunkte bestimmen die Laufrichtung
        n_pts = len(pts)
        if run.node_positions is None:
            key_idx = list(range(n_pts))
        else:
            nodes = set(run.node_positions)
            key_idx = [i for i, pos in enumerate(run.positions)
                       if i in (0, n_pts - 1) or pos in nodes]
        key_set = set(key_idx)
        q_pts = [self._offset_query(np.asarray(p, dtype=float))
                 if i in key_set else np.asarray(p, dtype=float)
                 for i, p in enumerate(pts)]
        s_all = [float(ring.project(Point(tuple(p)))) for p in q_pts]
        s_key = [s_all[i] for i in key_idx]

        if run.is_full_loop:
            tcp = self._full_ring_path(ring, s_key)
        else:
            tcp = self._walk_ring(ring, s_all, s_key)
        tcp = self._densify(tcp, TCP_SAMPLE_STEP)

        # Klinge vom TCP zum nächsten Oberflächenpunkt, Länge L(v)
        tips: list[np.ndarray] = []
        last_dir = np.array([1.0, 0.0])
        for t in tcp:
            q = nearest_points(self._mat_boundary, Point(tuple(t)))[0]
            d = np.array([q.x, q.y]) - t
            norm = float(np.linalg.norm(d))
            if norm < 1e-9:
                d, norm = last_dir, 1.0
            d = d / norm
            last_dir = d
            tips.append(t + d * self.blade_length)

        tcp_arr = np.asarray(tcp, dtype=float)
        tip_arr = np.asarray(tips, dtype=float)

        run.tcp_polyline = tcp_arr
        run.tip_polyline = tip_arr
        run.tcp_length = float(
            np.linalg.norm(np.diff(tcp_arr, axis=0), axis=1).sum())
        run.swept_polygon = self._swept_area(tcp_arr, tip_arr)
        run.is_feasible = True
        run.reason = ""
        return run

    # ------------------------------------------------------------------

    def _offset_query(self, p: np.ndarray) -> np.ndarray:
        """Eindeutiger TCP-Punkt zur Konturstelle ``p`` (Eckenregel).

        Ecke: Bogenmitte; sonst bleibt ``p`` (Projektion dort eindeutig).
        """
        for c in self._bnd_rings:
            k = len(c)
            d2 = (c[:, 0] - p[0]) ** 2 + (c[:, 1] - p[1]) ** 2
            i = int(np.argmin(d2))
            if d2[i] > 1e-12:
                continue
            e_prev = c[i] - c[(i - 1) % k]
            e_next = c[(i + 1) % k] - c[i]
            b = self._unit_normal(e_prev) + self._unit_normal(e_next)
            nb = float(np.linalg.norm(b))
            if nb < 1e-9:              # 180deg-Spitze: keine Halbierende
                return p
            q = p + (b / nb) * self.clearance
            # vom Material weg (Löcher laufen andersherum)
            if self.material is not None and self.material.contains(
                    Point(tuple(q))):
                q = p - (b / nb) * self.clearance
            return q
        return p

    @staticmethod
    def _unit_normal(edge: np.ndarray) -> np.ndarray:
        """Einheitsnormale (rechts der Kantenrichtung) -- Vorzeichen egal,
        weil ``_offset_query`` die Richtung anschließend prüft."""
        n = float(np.linalg.norm(edge))
        if n < 1e-12:
            return np.zeros(2)
        return np.array([edge[1], -edge[0]]) / n

    @staticmethod
    def _densify(
        path: list[np.ndarray], max_step: float
    ) -> list[np.ndarray]:
        """Fügt Zwischenpunkte ein, bis kein Abschnitt länger als
        max_step ist (Klingenrichtung dreht dann kontinuierlich mit)."""
        if len(path) < 2:
            return path
        out: list[np.ndarray] = [path[0]]
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            d = float(np.linalg.norm(b - a))
            n_parts = max(1, math.ceil(d / max_step))
            for k in range(1, n_parts + 1):
                out.append(a + (b - a) * (k / n_parts))
        return out

    @staticmethod
    def _ring_stations(ring: LineString) -> tuple[np.ndarray, np.ndarray]:
        """Ring-Vertices + deren Bogenlängen-Stationen."""
        coords = np.asarray(ring.coords, dtype=float)
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        stations = np.concatenate([[0.0], np.cumsum(seg)])
        return coords, stations

    def _walk_ring(
        self, ring: LineString, s_vals: list[float],
        s_key: list[float] | None = None,
    ) -> list[np.ndarray]:
        """TCP-Pfad entlang des Rings von s_vals[0] nach s_vals[-1].

          - Laufrichtung per Mehrheitsentscheid über ``s_vals``
          - Stützstellen = Ring-Vertices + Knoten-Stationen ``s_key``
        """
        if s_key is None:
            s_key = s_vals
        S = float(ring.length)
        coords, stations = self._ring_stations(ring)
        s0, sN = s_vals[0], s_vals[-1]

        diffs = [(s_vals[i + 1] - s_vals[i]) % S
                 for i in range(len(s_vals) - 1)]
        forward_votes = sum(1 for d in diffs if d < S / 2)
        forward = forward_votes >= len(diffs) / 2

        inner = np.asarray(s_key[1:-1], dtype=float)
        if forward:
            span = (sN - s0) % S
            rel = (stations[:-1] - s0) % S          # letzter Vertex == erster
            rel_in = (inner - s0) % S
        else:
            span = (s0 - sN) % S
            rel = (s0 - stations[:-1]) % S
            rel_in = (s0 - inner) % S
        cand = []
        for r, c in zip(rel, coords[:-1]):
            if 1e-9 < r < span - 1e-9:
                cand.append((float(r), np.asarray(c, dtype=float)))
        for r, sv in zip(rel_in, inner):
            if 1e-9 < r < span - 1e-9:
                q = ring.interpolate(float(sv))
                cand.append((float(r), np.array([q.x, q.y])))
        cand.sort(key=lambda t: t[0])

        p_start = ring.interpolate(s0)
        p_end = ring.interpolate(sN)
        path = [np.array([p_start.x, p_start.y])]
        last_r = -1.0
        for r, pt in cand:
            if r - last_r > 1e-9:                   # doppelte Stationen einmal
                path.append(pt)
                last_r = r
        path.append(np.array([p_end.x, p_end.y]))
        return path

    def _full_ring_path(self, ring: LineString,
                        s_vals: list[float]) -> list[np.ndarray]:
        """Kompletter Ring ab s_vals[0]; Stützstellen = Ring-Vertices + alle
        Stationen.
        """
        S = float(ring.length)
        s0 = float(s_vals[0])
        coords, stations = self._ring_stations(ring)
        cand = [(float((st - s0) % S), np.asarray(c, dtype=float))
                for st, c in zip(stations[:-1], coords[:-1])]
        for sv in s_vals[1:]:
            q = ring.interpolate(float(sv))
            cand.append((float((float(sv) - s0) % S), np.array([q.x, q.y])))
        cand.sort(key=lambda t: t[0])
        p0 = ring.interpolate(s0)
        start = np.array([p0.x, p0.y])
        path = [start]
        last_r = 0.0
        for r, pt in cand:
            if r - last_r > 1e-9:
                path.append(pt)
                last_r = r
        path.append(start.copy())
        return path

    def _swept_area(
        self, tcp: np.ndarray, tips: np.ndarray
    ) -> Polygon | MultiPolygon | None:
        """Swept Area = Vereinigung der Vierecke pro Pfadkante + Kerf."""
        quads = []
        for i in range(len(tcp) - 1):
            if float(np.linalg.norm(tcp[i + 1] - tcp[i])) < 1e-9:
                continue
            q = Polygon([tuple(tcp[i]), tuple(tcp[i + 1]),
                         tuple(tips[i + 1]), tuple(tips[i])])
            if not q.is_valid:
                q = q.buffer(0)
            if not q.is_empty:
                quads.append(q)
        if not quads:
            # nur Zündung: Klingenlinie als Schnitt
            line = LineString([tuple(tcp[0]), tuple(tips[0])])
            return line.buffer(self.kerf / 2)
        swept = unary_union(quads).buffer(self.kerf / 2)
        return swept


# ---------------------------------------------------------------------------
# LinkPath
# ---------------------------------------------------------------------------

@dataclass
class LinkPath:
    """Ein geplanter Verfahrweg zwischen zwei Konturpunkten.

    Attributes
    ----------
    points  : (M, 2) TCP-Wegpunkte inkl. Start- und Zielkonturpunkt
    length  : Weglänge [mm]
    """
    points: np.ndarray
    length: float

    @classmethod
    def from_points(cls, pts: list[np.ndarray]) -> LinkPath:
        arr = np.asarray(pts, dtype=float)
        seg = np.diff(arr, axis=0)
        length = float(np.linalg.norm(seg, axis=1).sum()) if len(arr) > 1 else 0.0
        return cls(points=arr, length=length)


class LinkInfeasibleError(RuntimeError):
    """Kein kollisionsfreier Verfahrweg (kein Überflug über das Material)."""


# ---------------------------------------------------------------------------
# LinkPlanner
# ---------------------------------------------------------------------------

class LinkPlanner:
    """Plant kollisionsfreie Eilgang-Pfade um das Material herum.

    Parameters
    ----------
    material  : Material-Polygon (Außenkontur minus Löcher)
    clearance : Mindestabstand TCP zu Material [mm] (cutter.minimum_gap)
    """

    def __init__(
        self,
        material: Polygon | MultiPolygon | None,
        clearance: float,
    ) -> None:
        self.material = material
        self.clearance = float(clearance)

        if material is None or material.is_empty or clearance <= 0:
            self._forbidden = None
            self._forbidden_prep = None
            self._route_nodes: list[np.ndarray] = []
            self._node_visibility: dict[tuple[int, int], float] = {}
            return

        # verbotene Zone: 0.9 * clearance (Toleranz für Punkte genau auf
        # dem Abstand)
        self._forbidden = material.buffer(clearance * 0.90)
        self._forbidden_prep = prep(self._forbidden)

        # Routing-Knoten: Ecken des gepufferten, vereinfachten Materials
        # (Außen- und Lochringe)
        route_poly = material.buffer(clearance * 1.15)
        route_poly = route_poly.simplify(clearance * 0.15)
        self._route_nodes = self._collect_ring_nodes(route_poly)

        # Sichtbarkeit zwischen Routing-Knoten einmalig vorberechnen
        self._node_visibility = {}
        n = len(self._route_nodes)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._edge_if_free(self._route_nodes[i],
                                       self._route_nodes[j])
                if d is not None:
                    self._node_visibility[(i, j)] = d

    # ------------------------------------------------------------------

    @staticmethod
    def _collect_ring_nodes(poly: Polygon | MultiPolygon) -> list[np.ndarray]:
        nodes: list[np.ndarray] = []
        geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for g in geoms:
            rings = [g.exterior] + list(g.interiors)
            for ring in rings:
                coords = list(ring.coords)[:-1]  # letzter == erster
                nodes.extend(np.asarray(c, dtype=float) for c in coords)
        return nodes

    def _edge_if_free(self, a: np.ndarray, b: np.ndarray) -> float | None:
        """Kantenlänge wenn die Strecke a->b kollisionsfrei ist, sonst None."""
        d = float(np.linalg.norm(b - a))
        if d < 1e-9:
            return 0.0
        if self._forbidden_prep is None:
            return d
        line = LineString([tuple(a), tuple(b)])
        if self._forbidden_prep.intersects(line):
            return None
        return d

    def _is_point_free(self, p: np.ndarray) -> bool:
        if self._forbidden_prep is None:
            return True
        return not self._forbidden_prep.contains(Point(float(p[0]), float(p[1])))

    # ------------------------------------------------------------------
    # Pfadsuche
    # ------------------------------------------------------------------

    def plan(self, start: np.ndarray, goal: np.ndarray) -> LinkPath | None:
        """Verfahrweg zwischen zwei TCP-Punkten (beide schon auf dem Offset-Pfad).

        Returns
        -------
        LinkPath, oder None, wenn kein kollisionsfreier 2D-Pfad existiert
        (z.B. vollständig umschlossene Lochkontur).
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if float(np.linalg.norm(goal - start)) < CHAIN_TOL:
            return LinkPath.from_points([start, goal])

        if self._forbidden is None:
            return LinkPath.from_points([start, goal])

        # Start/Ziel nicht frei (bei ausführbaren Runs nicht erwartet)
        if not self._is_point_free(start) or not self._is_point_free(goal):
            return None

        # 1) Direktverbindung frei?
        if self._edge_if_free(start, goal) is not None:
            return LinkPath.from_points([start, goal])

        # 2) Sichtbarkeitsgraph + Dijkstra
        path = self._dijkstra(start, goal)
        if path is not None:
            return LinkPath.from_points(path)

        # 3) Kein 2D-Pfad -> infeasible (kein Überflug erlaubt)
        return None

    def _dijkstra(
        self, a: np.ndarray, b: np.ndarray
    ) -> list[np.ndarray] | None:
        """Kürzester Pfad von a nach b über die Routing-Knoten."""
        nodes = [a, b] + self._route_nodes
        n = len(nodes)
        if n < 2:
            return None

        # Adjazenz: vorberechnete Sichtbarkeit, nur Kanten zu a (0) / b (1) neu
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
        for (i, j), d in self._node_visibility.items():
            adj[i + 2].append((j + 2, d))
            adj[j + 2].append((i + 2, d))
        for src in (0, 1):
            for k in range(n):
                if k == src or (src == 0 and k == 1) or (src == 1 and k == 0):
                    continue
                d = self._edge_if_free(nodes[src], nodes[k])
                if d is not None:
                    adj[src].append((k, d))
                    adj[k].append((src, d))
        # a <-> b wurde bereits vorher als nicht-frei erkannt

        dist = [math.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        pq: list[tuple[float, int]] = [(0.0, 0)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == 1:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if not math.isfinite(dist[1]):
            return None

        path_idx = [1]
        while path_idx[-1] != 0:
            path_idx.append(prev[path_idx[-1]])
        path_idx.reverse()
        return [nodes[i] for i in path_idx]


# ---------------------------------------------------------------------------
# CutPlan (Ergebnis des Sequencers)
# ---------------------------------------------------------------------------

@dataclass
class PlannedStep:
    """Ein Schritt des fertigen Plans: Schnitt oder Verfahrweg."""
    kind: str                    # "cut" | "link"
    run: CutRun | None = None    # bei kind == "cut"
    link: LinkPath | None = None # bei kind == "link"
    needs_pierce: bool = False   # bei kind == "cut": neue Zündung nötig
    duration: float = 0.0        # Zeit dieses Schritts [s]


@dataclass
class CutPlan:
    """Ausführungsplan: geordnete Schnitte und Verbindungen.

    ``switch_time``/``n_switches``: Aufschläge t_switch zwischen nahtlosen
    Sub-Runs einer Kette.
    """
    steps: list[PlannedStep] = field(default_factory=list)
    cut_time: float = 0.0
    travel_time: float = 0.0
    pierce_time: float = 0.0
    switch_time: float = 0.0
    n_pierces: int = 0
    n_switches: int = 0
    cut_length: float = 0.0
    travel_length: float = 0.0
    is_optimal: bool = False   # True = exakte (zeitminimale) Reihenfolge

    @property
    def total_time(self) -> float:
        return (self.cut_time + self.travel_time + self.pierce_time
                + self.switch_time)

    @property
    def runs_in_order(self) -> list[CutRun]:
        return [s.run for s in self.steps if s.kind == "cut" and s.run]

    def summary(self) -> str:
        opt_info = "exakt zeitminimal" if self.is_optimal else "heuristisch"
        switch = (f" | Wechsel: {self.switch_time:.1f} s"
                  if self.switch_time > 0 else "")
        return (
            f"CutPlan ({opt_info}): {len(self.runs_in_order)} Segment(e) | "
            f"Zuendungen: {self.n_pierces} | "
            f"Schnitt: {self.cut_length:.0f} mm / {self.cut_time:.1f} s | "
            f"Eilgang: {self.travel_length:.0f} mm / {self.travel_time:.1f} s | "
            f"Pierce: {self.pierce_time:.1f} s{switch} | "
            f"Gesamt: {self.total_time:.1f} s"
        )


# ---------------------------------------------------------------------------
# Sequencer
# ---------------------------------------------------------------------------

class Sequencer:
    """Optimale Reihenfolge + Richtung für eine Menge von CutRuns (exakt,
    Held-Karp; die Run-Zahl ist nach dem Verschmelzen klein).

    Parameters
    ----------
    cutter       : Cutter (Geschwindigkeiten + Pierce-Pauschale)
    contour      : SegmentedContour
    link_planner : LinkPlanner für die Verbindungswege
    """

    def __init__(
        self,
        cutter: Cutter,
        contour: SegmentedContour,
        link_planner: LinkPlanner,
    ) -> None:
        self.cutter = cutter
        self.contour = contour
        self.link_planner = link_planner
        self._material = link_planner.material
        # Cache der Verbindungswege (symmetrisch, oft abgefragt)
        self._link_cache: dict[tuple, LinkPath | None] = {}

    # ------------------------------------------------------------------
    # Übergangskosten (kollisionsfreie Verfahrwege, gecacht)
    # ------------------------------------------------------------------

    @staticmethod
    def _pt_key(p: np.ndarray) -> tuple:
        return (round(float(p[0]), 6), round(float(p[1]), 6))

    def link_between(self, a: np.ndarray, b: np.ndarray) -> LinkPath | None:
        """Kollisionsfreier Verfahrweg a -> b (gecacht, symmetrisch).

        None = kein kollisionsfreier Pfad möglich (infeasible).
        """
        ka, kb = self._pt_key(a), self._pt_key(b)
        if (ka, kb) in self._link_cache:
            return self._link_cache[(ka, kb)]
        if (kb, ka) in self._link_cache:
            rev = self._link_cache[(kb, ka)]
            link = (None if rev is None else
                    LinkPath(points=rev.points[::-1].copy(),
                             length=rev.length))
        else:
            link = self.link_planner.plan(a, b)
        self._link_cache[(ka, kb)] = link
        return link

    def _trans_cost(self, end_xy: np.ndarray, start_xy: np.ndarray) -> float:
        """Übergangszeit [s]: Verfahrweg + Pierce; nahtlos = 0, kein Weg = inf."""
        if float(np.linalg.norm(start_xy - end_xy)) < CHAIN_TOL:
            return 0.0  # nahtlos: kein Eilgang, keine neue Zündung
        link = self.link_between(end_xy, start_xy)
        if link is None:
            return math.inf
        return self._link_time(link) + self.cutter.pierce_time()

    def _sequence_cost(self, ordered: list[CutRun]) -> float:
        cost = 0.0
        for i in range(len(ordered) - 1):
            cost += self._trans_cost(ordered[i].tcp_end,
                                     ordered[i + 1].tcp_start)
        return cost

    # ------------------------------------------------------------------
    # Reihenfolge-Optimierung
    # ------------------------------------------------------------------

    def order_runs(
        self,
        runs: list[CutRun],
        start_xy: np.ndarray | None = None,
    ) -> tuple[list[CutRun], bool]:
        """Zeitminimale Reihenfolge + Richtung aller Runs (exakt, Held-Karp).

        Minimiert die Übergänge (Eilgang + Zündungen); jeder Run darf auch
        rückwärts geschnitten werden.

        Returns
        -------
        (geordnete Runs, is_optimal), is_optimal ist immer True.
        """
        if not runs:
            return [], True
        if len(runs) == 1:
            return list(runs), True
        return self._order_exact(runs), True

    # ------------------------------------------------------------------
    # Exakt: Held-Karp über (Teilmenge, letzter Run, Richtung)
    # ------------------------------------------------------------------

    def _order_exact(self, runs: list[CutRun]) -> list[CutRun]:
        n = len(runs)
        variants: list[tuple[CutRun, CutRun]] = [
            (r, r.reversed()) for r in runs
        ]

        # cost[i, oi, j, oj]: Ende Run i (Richtung oi) -> Start Run j (oj)
        cost = np.full((n, 2, n, 2), np.inf)
        for i in range(n):
            for oi in (0, 1):
                e = variants[i][oi].tcp_end
                for j in range(n):
                    if i == j:
                        continue
                    for oj in (0, 1):
                        s = variants[j][oj].tcp_start
                        cost[i, oi, j, oj] = self._trans_cost(e, s)

        # dp[mask, j, oj]: min. Übergangszeit, Runs in mask besucht, Ende bei j (oj)
        full = (1 << n) - 1
        dp = np.full((1 << n, n, 2), np.inf)
        parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for i in range(n):
            dp[1 << i, i, 0] = 0.0
            dp[1 << i, i, 1] = 0.0

        for mask in range(1, full + 1):
            for j in range(n):
                if not mask & (1 << j):
                    continue
                for oj in (0, 1):
                    cur = dp[mask, j, oj]
                    if not math.isfinite(cur):
                        continue
                    for k in range(n):
                        if mask & (1 << k):
                            continue
                        nmask = mask | (1 << k)
                        for ok in (0, 1):
                            nd = cur + cost[j, oj, k, ok]
                            if nd < dp[nmask, k, ok]:
                                dp[nmask, k, ok] = nd
                                parent[(nmask, k, ok)] = (mask, j, oj)

        # Bestes Ende suchen und Pfad rekonstruieren
        end_idx = np.unravel_index(np.argmin(dp[full]), dp[full].shape)
        if not math.isfinite(dp[full][end_idx]):
            self._raise_infeasible(variants)
        state = (full, int(end_idx[0]), int(end_idx[1]))
        order_rev: list[tuple[int, int]] = []
        while True:
            _, j, oj = state
            order_rev.append((j, oj))
            if state not in parent:
                break
            state = parent[state]
        order_rev.reverse()
        return [variants[j][oj] for j, oj in order_rev]

    # ------------------------------------------------------------------
    # Infeasibility-Meldung
    # ------------------------------------------------------------------

    def _raise_infeasible(
        self, variants: list[tuple[CutRun, CutRun]]
    ) -> None:
        """Wirft LinkInfeasibleError und nennt die Übergänge ohne Verfahrweg."""
        blocked: list[str] = []
        for i, (fwd_i, rev_i) in enumerate(variants):
            for j, (fwd_j, rev_j) in enumerate(variants):
                if i == j:
                    continue
                feasible = any(
                    math.isfinite(self._trans_cost(a.tcp_end, b.tcp_start))
                    for a in (fwd_i, rev_i) for b in (fwd_j, rev_j))
                if not feasible:
                    blocked.append(
                        f"R{fwd_i.run_id} -> R{fwd_j.run_id}")
        if blocked:
            detail = ", ".join(blocked)
            raise LinkInfeasibleError(
                f"Kein kollisionsfreier Verfahrweg fuer die "
                f"Uebergaenge: {detail}. Der Brenner kann nicht ueber "
                f"das Material springen -- diese Segmentkombination ist "
                f"nicht planbar (z.B. vollstaendig umschlossene "
                f"Lochkontur).")
        raise LinkInfeasibleError(
            "Keine zulaessige Schnittreihenfolge: die kollisionsfreien "
            "Verfahrwege lassen sich nicht zu einer Tour ueber alle "
            "Segmente verbinden (Ueberflug ist nicht moeglich).")

    # ------------------------------------------------------------------
    # Plan bauen
    # ------------------------------------------------------------------

    def build_plan(
        self,
        runs: list[CutRun],
        start_xy: np.ndarray | None = None,
    ) -> CutPlan:
        """Ordnet die Runs und plant die Verbindungswege.

        Returns
        -------
        CutPlan mit alternierenden cut/link-Schritten und Zeitbilanz.
        """
        plan = CutPlan()
        ordered, is_optimal = self.order_runs(runs, start_xy=start_xy)
        plan.is_optimal = is_optimal
        if not ordered:
            return plan

        for i, run in enumerate(ordered):
            needs_pierce = True
            if i > 0:
                prev = ordered[i - 1]
                gap = float(np.linalg.norm(run.tcp_start - prev.tcp_end))
                if gap < CHAIN_TOL:
                    # Nahtloser Anschluss: Brenner bleibt an
                    needs_pierce = False
                else:
                    link = self.link_between(prev.tcp_end, run.tcp_start)
                    t_link = self._link_time(link)
                    plan.steps.append(PlannedStep(
                        kind="link", link=link, duration=t_link))
                    plan.travel_time += t_link
                    plan.travel_length += link.length

            # Maßgeblich ist der TCP-Pfad (Offset), nicht die Kontur
            cut_len = run.tcp_length if run.tcp_length > 0 else run.length
            t_cut = self.cutter.time_for_length(cut_len, mode="cut")
            t_pierce = self.cutter.pierce_time() if needs_pierce else 0.0
            plan.steps.append(PlannedStep(
                kind="cut", run=run, needs_pierce=needs_pierce,
                duration=t_cut + t_pierce))
            plan.cut_time += t_cut
            plan.cut_length += cut_len
            plan.pierce_time += t_pierce
            if needs_pierce:
                plan.n_pierces += 1

        return plan

    def _link_time(self, link: LinkPath) -> float:
        """Zeit eines Verfahrwegs [s], komplett im Eilgang."""
        if link.length <= 0:
            return 0.0
        return link.length / self.cutter.rapid_speed

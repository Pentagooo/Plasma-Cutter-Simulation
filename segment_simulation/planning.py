from __future__ import annotations

"""Automatische Verbindungs- und Reihenfolgeplanung.

LinkPlanner
-----------
Plant kollisionsfreie Verfahrwege (Eilgang) zwischen Segment-Endpunkten.
Der TCP-Pfad muss ``clearance`` (= cutter.minimum_gap) Abstand zum
Material halten -- wie in der bisherigen Simulation, jetzt aber
AUTOMATISCH statt manuell:

  1. Abheben:   vom Konturpunkt entlang der Aussennormale auf
                Sicherheitsabstand (Lead-out).
  2. Verfahren: Sichtbarkeitsgraph um das gepufferte Material-Polygon,
                kuerzester Pfad via Dijkstra.
  3. Anfahren:  von aussen auf den naechsten Konturpunkt (Lead-in).

Existiert kein kollisionsfreier 2D-Pfad (z.B. Ziel auf einer
Lochkontur, die vollstaendig von Material umschlossen ist), ist der
Uebergang INFEASIBLE: ``plan()`` gibt None zurueck. Der Brenner kann
NICHT ueber das Material springen (Modellentscheidung, kein Z-Hub).
Der Sequencer bewertet solche Uebergaenge mit unendlichen Kosten und
wirft eine LinkInfeasibleError, wenn keine zulaessige Reihenfolge
existiert.

Sequencer
---------
Bringt die gewaehlten CutRuns in eine optimale Reihenfolge und
Richtung. Zielfunktion = Gesamtzeit:

    sum(Schnittzeiten)            -- konstant
  + sum(Eilgang-Zeiten)           -- abhaengig von Reihenfolge
  + sum(Pierce-Pauschalen)        -- entfaellt wenn zwei Runs nahtlos
                                     aneinander anschliessen

Loesung: Nearest-Neighbour-Konstruktion (alle Startkandidaten)
+ lokale Verbesserung (2-opt + Richtungs-Flips). Die Problemgroesse
ist klein (typisch < 20 Runs), das genuegt.
"""

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


# Endpunkte naeher als CHAIN_TOL gelten als "nahtlos verbunden":
# kein Abheben, kein Eilgang, keine neue Zuendung.
CHAIN_TOL = 1e-6

# Maximaler Abstand zweier TCP-Stuetzpunkte fuer die Swept-Area-
# Berechnung [mm]. Der TCP-Pfad wird auf diese Schrittweite verdichtet,
# damit die Klingenrichtung auch an Ecken kontinuierlich mitdreht
# (sonst fehlt der "Faecher", den die Klinge beim Umfahren einer Ecke
# ueberstreicht, und die Coverage wird unterschaetzt).
TCP_SAMPLE_STEP = 3.0

# ---------------------------------------------------------------------------
# Punktezahl (Score)
# ---------------------------------------------------------------------------
# Die Punktezahl besteht (vorerst) vor allem aus der Zeit: weniger Zeit
# = mehr Punkte. Skaliert mit der Coverage, damit "nichts schneiden"
# keine Punkte bringt.
SCORE_BASE = 1000.0        # Punkte bei (theoretischer) Zeit 0 und 100 % Coverage
SCORE_TIME_WEIGHT = 1.0    # Punktabzug pro Sekunde Gesamtzeit


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
# RunKinematics -- Lichtschwert-Kinematik fuer einen CutRun
# ---------------------------------------------------------------------------

class RunKinematics:
    """Berechnet TCP-Pfad, Klingenspitzen und Swept Area eines CutRuns.

    Modell (wie in den bisherigen Simulationen):

      - Der TCP (Brennergriff) haelt IMMER den konstanten Mindestabstand
        ``clearance`` zum Material und faehrt auf dem Offset-Pfad
        (Rand von ``material.buffer(clearance)``).
      - Die Klinge (Plasmastrahl) ragt vom TCP in Richtung Objekt mit
        Laenge ``blade_length`` = L(v). Effektive Schnitttiefe ab
        Oberflaeche: L(v) - clearance.
      - Die ueberstrichene Flaeche (Swept Area) wird als Vereinigung
        der Vierecke (TCP_i, TCP_i+1, Spitze_i+1, Spitze_i) berechnet,
        mit kerf/2 gepuffert.

    Parameters
    ----------
    material     : Material-Polygon (Aussenkontur minus Loecher)
    clearance    : konstanter Mindestabstand TCP-Material [mm]
    blade_length : Klingenlaenge L(v) bei der Schnittgeschwindigkeit [mm]
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
        if material is not None and not material.is_empty:
            self._mat_boundary = material.boundary
            offset_boundary = material.buffer(self.clearance).boundary
            if offset_boundary.geom_type == "LineString":
                self._rings = [offset_boundary]
            else:
                self._rings = list(offset_boundary.geoms)

    @property
    def effective_depth(self) -> float:
        """Schnitttiefe ab Materialoberflaeche [mm]."""
        return self.blade_length - self.clearance

    # ------------------------------------------------------------------

    def attach(self, run: CutRun) -> CutRun:
        """Berechnet die Kinematik und schreibt sie in den Run (in-place).

        Setzt is_feasible=False wenn kein gueltiger Offset-Pfad
        existiert (z.B. Loch zu klein fuer den Mindestabstand) oder
        die Klinge das Material nicht erreicht.
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
        mid = Point(tuple(pts[len(pts) // 2]))
        ring = min(self._rings, key=lambda r: r.distance(mid))
        if ring.distance(mid) > self.clearance * 1.5:
            run.is_feasible = False
            run.reason = ("kein Offset-Pfad im Mindestabstand erreichbar "
                          "(Kontur zu eng fuer den TCP)")
            return run

        # Projektions-Stationen der Konturpunkte auf dem Offset-Ring
        s_vals = [float(ring.project(Point(tuple(p)))) for p in pts]

        if run.is_full_loop:
            tcp = self._full_ring_path(ring, s_vals[0])
        else:
            tcp = self._walk_ring(ring, s_vals)
        tcp = self._densify(tcp, TCP_SAMPLE_STEP)

        # Klingenspitzen: Klinge zeigt vom TCP zum naechsten Punkt der
        # Materialoberflaeche (Richtung Objekt), Laenge L(v).
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

    @staticmethod
    def _densify(
        path: list[np.ndarray], max_step: float
    ) -> list[np.ndarray]:
        """Fuegt Zwischenpunkte ein, bis kein Abschnitt laenger als
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
        """Ring-Vertices + deren Bogenlaengen-Stationen."""
        coords = np.asarray(ring.coords, dtype=float)
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        stations = np.concatenate([[0.0], np.cumsum(seg)])
        return coords, stations

    def _walk_ring(
        self, ring: LineString, s_vals: list[float]
    ) -> list[np.ndarray]:
        """TCP-Pfad entlang des Rings von s_vals[0] nach s_vals[-1].

        Die Laufrichtung wird per Mehrheitsentscheid aus den
        aufeinanderfolgenden Projektions-Stationen bestimmt (damit der
        TCP der Kontur folgt statt aussen herum zu laufen).
        """
        S = float(ring.length)
        coords, stations = self._ring_stations(ring)
        s0, sN = s_vals[0], s_vals[-1]

        diffs = [(s_vals[i + 1] - s_vals[i]) % S
                 for i in range(len(s_vals) - 1)]
        forward_votes = sum(1 for d in diffs if d < S / 2)
        forward = forward_votes >= len(diffs) / 2

        if forward:
            span = (sN - s0) % S
            rel = (stations[:-1] - s0) % S          # letzter Vertex == erster
        else:
            span = (s0 - sN) % S
            rel = (s0 - stations[:-1]) % S
        inside = (rel > 1e-9) & (rel < span - 1e-9)
        order = np.argsort(rel[inside])
        mids = coords[:-1][inside][order]

        p_start = ring.interpolate(s0)
        p_end = ring.interpolate(sN)
        path = [np.array([p_start.x, p_start.y])]
        path.extend(np.asarray(m, dtype=float) for m in mids)
        path.append(np.array([p_end.x, p_end.y]))
        return path

    def _full_ring_path(self, ring: LineString, s0: float) -> list[np.ndarray]:
        """Kompletter Ring, umsortiert so dass er bei s0 beginnt/endet."""
        S = float(ring.length)
        coords, stations = self._ring_stations(ring)
        rel = (stations[:-1] - s0) % S
        order = np.argsort(rel)
        p0 = ring.interpolate(s0)
        start = np.array([p0.x, p0.y])
        path = [start]
        path.extend(np.asarray(coords[:-1][i], dtype=float)
                    for i in order if rel[i] > 1e-9)
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
            # Nur Zuendung an einer Stelle: Klingen-Linie als Schnitt
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
    length  : Weglaenge [mm]
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
    """Kein kollisionsfreier Verfahrweg moeglich -- der Plan ist nicht
    ausfuehrbar, weil der Brenner nicht ueber das Material springen kann."""


# ---------------------------------------------------------------------------
# LinkPlanner
# ---------------------------------------------------------------------------

class LinkPlanner:
    """Plant kollisionsfreie Eilgang-Pfade um das Material herum.

    Parameters
    ----------
    material  : Material-Polygon (Aussenkontur minus Loecher)
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

        # Verbotene Zone fuer den Verfahrweg (etwas kleiner als clearance,
        # damit Punkte AUF dem Sicherheitsabstand nicht faelschlich
        # kollidieren -- numerische Toleranz).
        self._forbidden = material.buffer(clearance * 0.90)
        self._forbidden_prep = prep(self._forbidden)

        # Routing-Knoten: Ecken des auf clearance*1.15 gepufferten
        # Materials (vereinfacht). Aussenring + Innenringe (Loecher).
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
        """Kantenlaenge wenn die Strecke a->b kollisionsfrei ist, sonst None."""
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
        """Plant den Verfahrweg zwischen zwei TCP-Punkten.

        Beide Punkte liegen bereits auf dem Offset-Pfad (Mindestabstand
        zum Material), daher ist kein Abheben noetig -- nur die Route
        dazwischen muss kollisionsfrei sein.

        Returns
        -------
        LinkPath -- oder None, wenn kein kollisionsfreier 2D-Pfad
        existiert (z.B. Ziel auf einer vollstaendig umschlossenen
        Lochkontur). Der Brenner kann nicht ueber das Material
        springen, der Uebergang ist dann infeasible.
        """
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        if float(np.linalg.norm(goal - start)) < CHAIN_TOL:
            return LinkPath.from_points([start, goal])

        if self._forbidden is None:
            return LinkPath.from_points([start, goal])

        # Wenn Start/Ziel nicht im freien Raum liegen (sollte bei
        # feasiblen Runs nicht vorkommen): kein Verfahrweg moeglich.
        if not self._is_point_free(start) or not self._is_point_free(goal):
            return None

        # 1) Direktverbindung frei?
        if self._edge_if_free(start, goal) is not None:
            return LinkPath.from_points([start, goal])

        # 2) Sichtbarkeitsgraph + Dijkstra
        path = self._dijkstra(start, goal)
        if path is not None:
            return LinkPath.from_points(path)

        # 3) Kein 2D-Pfad -> infeasible (kein Ueberflug erlaubt)
        return None

    def _dijkstra(
        self, a: np.ndarray, b: np.ndarray
    ) -> list[np.ndarray] | None:
        """Kuerzester Pfad von a nach b ueber die Routing-Knoten."""
        nodes = [a, b] + self._route_nodes
        n = len(nodes)
        if n < 2:
            return None

        # Adjazenz aufbauen: vorberechnete Knoten-Sichtbarkeit nutzen,
        # nur Kanten von/zu a (Index 0) und b (Index 1) neu testen.
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
    needs_pierce: bool = False   # bei kind == "cut": neue Zuendung noetig
    duration: float = 0.0        # Zeit dieses Schritts [s]


@dataclass
class CutPlan:
    """Vollstaendiger Ausfuehrungsplan: geordnete Schnitte + Verbindungen."""
    steps: list[PlannedStep] = field(default_factory=list)
    cut_time: float = 0.0
    travel_time: float = 0.0
    pierce_time: float = 0.0
    n_pierces: int = 0
    cut_length: float = 0.0
    travel_length: float = 0.0
    is_optimal: bool = False   # True = exakte (zeitminimale) Reihenfolge

    @property
    def total_time(self) -> float:
        return self.cut_time + self.travel_time + self.pierce_time

    @property
    def runs_in_order(self) -> list[CutRun]:
        return [s.run for s in self.steps if s.kind == "cut" and s.run]

    def summary(self) -> str:
        opt_info = "exakt zeitminimal" if self.is_optimal else "heuristisch"
        return (
            f"CutPlan ({opt_info}): {len(self.runs_in_order)} Segment(e) | "
            f"Zuendungen: {self.n_pierces} | "
            f"Schnitt: {self.cut_length:.0f} mm / {self.cut_time:.1f} s | "
            f"Eilgang: {self.travel_length:.0f} mm / {self.travel_time:.1f} s | "
            f"Pierce: {self.pierce_time:.1f} s | "
            f"Gesamt: {self.total_time:.1f} s"
        )


# ---------------------------------------------------------------------------
# Sequencer
# ---------------------------------------------------------------------------

class Sequencer:
    """Optimale Reihenfolge + Richtung fuer eine Menge von CutRuns.

    Parameters
    ----------
    cutter       : Cutter (Geschwindigkeiten + Pierce-Pauschale)
    contour      : SegmentedContour (fuer Normalen an den Endpunkten)
    link_planner : LinkPlanner fuer die finalen Verbindungswege
    """

    # Bis zu dieser Anzahl Runs wird EXAKT optimiert (Held-Karp-DP
    # ueber Reihenfolge x Richtung). Darueber Heuristik (NN + 2-opt).
    EXACT_MAX_RUNS = 11

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
        # Cache fuer geplante Verbindungswege: ein Punktpaar wird in der
        # Optimierung viele Male bewertet, der Routing-Pfad ist aber
        # symmetrisch und aendert sich nicht.
        self._link_cache: dict[tuple, LinkPath | None] = {}

    # ------------------------------------------------------------------
    # Uebergangs-Kosten: ECHTE kollisionsfreie Verfahrwege (gecacht)
    # ------------------------------------------------------------------

    @staticmethod
    def _pt_key(p: np.ndarray) -> tuple:
        return (round(float(p[0]), 6), round(float(p[1]), 6))

    def link_between(self, a: np.ndarray, b: np.ndarray) -> LinkPath | None:
        """Kollisionsfreier Verfahrweg a -> b (gecacht, symmetrisch).

        None = kein kollisionsfreier Pfad moeglich (infeasible).
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
        """Zeitkosten [s] fuer den Uebergang zwischen zwei Runs:
        echter Verfahrweg + Pierce-Zeit. Nahtloser Anschluss (gleicher
        Punkt) kostet nichts. Existiert kein kollisionsfreier
        Verfahrweg, ist der Uebergang infeasible -> unendliche Kosten.
        """
        if float(np.linalg.norm(start_xy - end_xy)) < CHAIN_TOL:
            return 0.0  # nahtlos: kein Eilgang, keine neue Zuendung
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
        """Findet die zeitminimale Reihenfolge + Richtung fuer alle Runs.

        Reihenfolge UND Schnittrichtung jedes Segments sind frei: ein
        Segment darf auch vom Endpunkt zum Startpunkt geschnitten
        werden. Die Schnittzeiten selbst sind konstant, minimiert wird
        also die Summe der Uebergaenge (Eilgang + Zuendungen), bewertet
        mit den ECHTEN kollisionsfreien Verfahrwegen.

        Bis EXACT_MAX_RUNS Segmente: exakte Loesung per Held-Karp-DP
        ueber (besuchte Menge, letztes Segment, Richtung).
        Darueber: Nearest-Neighbour + 2-opt-Heuristik.

        Returns
        -------
        (geordnete Runs, is_optimal) -- is_optimal=True bei exakter Loesung.
        """
        if not runs:
            return [], True
        if len(runs) == 1:
            return list(runs), True

        if len(runs) <= self.EXACT_MAX_RUNS:
            return self._order_exact(runs), True
        return self._order_heuristic(runs), False

    # ------------------------------------------------------------------
    # Exakt: Held-Karp ueber (Teilmenge, letzter Run, Richtung)
    # ------------------------------------------------------------------

    def _order_exact(self, runs: list[CutRun]) -> list[CutRun]:
        n = len(runs)
        variants: list[tuple[CutRun, CutRun]] = [
            (r, r.reversed()) for r in runs
        ]

        # Uebergangs-Kosten-Matrix: cost[i][oi][j][oj] =
        # Ende von Run i (Richtung oi) -> Start von Run j (Richtung oj)
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

        # DP: dp[mask][j][oj] = minimale Uebergangszeit, wenn die Runs
        # in `mask` besucht sind und der Pfad bei Run j (Richtung oj) endet.
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
        """Wirft LinkInfeasibleError mit den Uebergaengen, fuer die in
        KEINER Richtungs-Kombination ein kollisionsfreier Verfahrweg
        existiert (der Brenner kann nicht ueber das Material springen)."""
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
    # Heuristik (Fallback fuer viele Runs)
    # ------------------------------------------------------------------

    def _order_heuristic(self, runs: list[CutRun]) -> list[CutRun]:
        """Nearest-Neighbour von allen Startkandidaten + 2-opt."""
        variants: list[tuple[CutRun, CutRun]] = [
            (r, r.reversed()) for r in runs
        ]

        best_seq: list[CutRun] | None = None
        best_cost = math.inf

        for first_i in range(len(variants)):
            for first_dir in (0, 1):
                seq = [variants[first_i][first_dir]]
                used = {first_i}
                while len(used) < len(variants):
                    cur_end = seq[-1].tcp_end
                    best_j, best_d, best_v = -1, math.inf, None
                    for j in range(len(variants)):
                        if j in used:
                            continue
                        for v in variants[j]:
                            c = self._trans_cost(cur_end, v.tcp_start)
                            if c < best_d:
                                best_j, best_d, best_v = j, c, v
                    if best_v is None:
                        break  # alle Rest-Uebergaenge infeasible
                    seq.append(best_v)
                    used.add(best_j)
                if len(used) < len(variants):
                    continue  # dieser Startkandidat fuehrt nicht zum Ziel
                cost = self._sequence_cost(seq)
                if math.isfinite(cost) and cost < best_cost:
                    best_cost = cost
                    best_seq = seq

        if best_seq is None:
            self._raise_infeasible(variants)
        best_seq = self._local_search(best_seq)
        return best_seq

    def _local_search(self, seq: list[CutRun]) -> list[CutRun]:
        """2-opt (Teilfolgen-Umkehr) + Richtungs-Flips bis keine
        Verbesserung mehr gefunden wird."""
        n = len(seq)
        if n < 2:
            return seq
        cost = self._sequence_cost(seq)
        improved = True
        guard = 0
        while improved and guard < 200:
            improved = False
            guard += 1
            # Richtungs-Flips einzelner Runs
            for i in range(n):
                cand = list(seq)
                cand[i] = cand[i].reversed()
                c = self._sequence_cost(cand)
                if c < cost - 1e-12:
                    seq, cost, improved = cand, c, True
            # 2-opt: Teilfolge umkehren (inkl. Richtungs-Umkehr der Runs)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = (
                        seq[:i]
                        + [r.reversed() for r in reversed(seq[i:j + 1])]
                        + seq[j + 1:]
                    )
                    c = self._sequence_cost(cand)
                    if c < cost - 1e-12:
                        seq, cost, improved = cand, c, True
        return seq

    # ------------------------------------------------------------------
    # Plan bauen (mit echten kollisionsfreien Verbindungen)
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

            # Massgeblich ist der TCP-Pfad (Offset), nicht die Kontur
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
        """Zeit fuer einen Verfahrweg [s] (Eilgang).

        Der TCP bleibt durchgehend auf Sicherheitsabstand, daher
        komplett im Eilgang.
        """
        if link.length <= 0:
            return 0.0
        return link.length / self.cutter.rapid_speed

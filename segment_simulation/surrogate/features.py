from __future__ import annotations

"""Feature-Berechnung fuer den Learned Surrogate Planner (BA-Kap. 5.2).

Alle Features werden AUSSCHLIESSLICH aus der Kontur und einer
Distanztransformation des Materialgitters abgeleitet -- es werden KEINE
Swept Areas gebaut. Genau das ist der Hebel des Surrogats: die teure
Shapely-Geometrie (``RunKinematics.attach``) entfaellt in der
Merkmalsstufe vollstaendig, es bleiben nur billige cKDTree-Distanzen.

Zentrale Groessen je Primitiv-Segment s:

  * Bogenlaenge, Punktzahl, relative Lage im Loop
  * benoetigte Tiefe der (per naechster-Nachbar) zugewiesenen
    Materialpunkte (max/mittel) -- steuert die Geschwindigkeitsregel
  * Anteil exklusiv abdeckbarer Punkte (Punkte, die nur s bei v->0
    erreicht) -- ein starkes Auswahlsignal
  * lokale Wanddicke + Gegenwand-Flag
  * Eckwinkel an beiden Enden, mittlere Richtungsaenderung (Rauheit)
  * Loop-Kontext (Umfang, innen/aussen, Anzahl Loops)
  * v_hat(s) nach der Geschwindigkeitsregel (Kap. 4.5) und die
    Zeitschaetzung ell(s)/v_hat(s)

Die Reihenfolge der Spalten ist deterministisch durch ``FEATURE_NAMES``
festgelegt.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

try:
    from ..segments import SegmentedContour, Segment
    from ...cutter.cutter import Cutter
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, Segment,
    )
    from plasma_cutter.cutter.cutter import Cutter


# ---------------------------------------------------------------------- -----
# Geschwindigkeits-Grenzen (Kap. 4.5)
# ---------------------------------------------------------------------------
# Untere/obere Schranke fuer die zugewiesene Schnittgeschwindigkeit v_hat.
# TODO(Projektwerte): v_min/v_max final aus den Prozessgrenzen des realen
# Brenners uebernehmen; v_max faellt sonst auf cutter.max_cutting_speed
# zurueck. v_min sollte > 0 bleiben (v=0 => Stillstand).
V_MIN_DEFAULT = 19.4   # = v_cut, minimale Schnittgeschwindigkeit (Projektwert 17.09.2026)
V_MAX_DEFAULT = 34.7   # Rueckfall, wenn cutter.max_cutting_speed fehlt

# Zeitaufschlag je Geschwindigkeitswechsel IM laufenden Schnitt [s]
# (Roboter-Rampe + Qualitaetstransient) -- EIN abstrakter Parameter analog
# sigma_TCP, der Roboter bleibt unspezifiziert. Faellt NUR zwischen
# Sub-Runs derselben Kette an (kein Abheben, kein Pierce).
# TODO(Projektwerte): t_switch aus dem Cut Chart / der Rampenzeit des
# realen Systems begruenden. Default 0.0 s = Wechsel kostenlos -- das ist
# ein OFFENER PUNKT, kein belegter Wert.
T_SWITCH_DEFAULT = 0.0


# ---------------------------------------------------------------------------
# Physik-Parameter aus dem Cutter ableiten (ohne simulation.py zu laden)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhysParams:
    """Aus einem ``Cutter`` abgeleitete physikalische Kenngroessen.

    Wird einmal je Feature-/Planungslauf gebaut und dann wiederverwendet.
    Alle Laengen in mm, Geschwindigkeiten in mm/s.
    """
    blade0: float        # Klingenlaenge L(v=0) [mm]
    slope: float         # |dL/dv| [mm/(mm/s)] (positiv)
    gap: float           # Mindestabstand TCP<->Material [mm]
    v_cut: float         # Basis-Schnittgeschwindigkeit [mm/s]
    v_min: float
    v_max: float
    # Zeitaufschlag je Geschwindigkeitswechsel im laufenden Schnitt [s]
    # (siehe T_SWITCH_DEFAULT; Wert ist als Projektwert festzulegen).
    t_switch: float = T_SWITCH_DEFAULT

    @property
    def full_eff_depth(self) -> float:
        """Effektive Schnitttiefe bei v->0 (maximale Reichweite) [mm]."""
        return self.blade0 - self.gap

    @property
    def base_eff_depth(self) -> float:
        """Effektive Schnitttiefe bei der Basisgeschwindigkeit v_cut [mm].

        Entspricht exakt der Reichweite, mit der die Baseline
        (``auto_plan``) rechnet -- daher der Referenzwert fuer
        Erreichbarkeit/Gegenwand.
        """
        return max(0.0, self.blade0 - self.slope * self.v_cut - self.gap)

    def blade_at(self, v: float) -> float:
        """Klingenlaenge L(v) = clip(L0 - slope*v, 0, L0) [mm]."""
        return float(np.clip(self.blade0 - self.slope * float(v),
                             0.0, self.blade0))

    def eff_depth_at(self, v: float) -> float:
        """Effektive Schnitttiefe bei v [mm] (= L(v) - gap, >= 0)."""
        return max(0.0, self.blade_at(v) - self.gap)

    def speed_for_depth(self, depth_req: float) -> float:
        """Geschwindigkeitsregel (Kap. 4.5): schnellste zulaessige v, bei
        der die Klinge die benoetigte Tiefe ``depth_req`` noch erreicht.

            v_hat = clip((L0 - gap - depth_req) / slope, v_min, v_max)

        Kleine benoetigte Tiefe -> hohe Geschwindigkeit (kurze Klinge
        genuegt); grosse Tiefe -> langsam (lange Klinge). Bei slope<=0
        (geschwindigkeitsunabhaengige Klinge) wird v_max zurueckgegeben.
        """
        if self.slope <= 1e-9:
            return self.v_max
        v = (self.blade0 - self.gap - float(depth_req)) / self.slope
        return float(np.clip(v, self.v_min, self.v_max))


def phys_from_cutter(cutter: Cutter | None,
                     v_min: float = V_MIN_DEFAULT,
                     v_max: float | None = None,
                     t_switch: float | None = None) -> PhysParams:
    """Leitet ``PhysParams`` aus einem Cutter ab (ohne simulation.py).

    ``slope`` wird numerisch aus dem L(v)-Modell bestimmt
    (L(0) - L(1)); das ist robust gegenueber der internen Darstellung im
    ``BladeLengthModel`` und braucht keinen Zugriff auf private Felder.
    ``t_switch`` (None) kommt aus ``cutter.speed_switch_time`` bzw. dem
    Default (0.0 s, offener Projektwert).
    """
    if cutter is None:
        cutter = _default_cutter()
    blade0 = float(cutter.blade_length(0.0))
    slope = float(cutter.blade_length(0.0) - cutter.blade_length(1.0))
    if slope < 0:
        slope = 0.0
    if v_max is None:
        v_max = float(cutter.max_cutting_speed or V_MAX_DEFAULT)
    if t_switch is None:
        t_switch = float(getattr(cutter, "speed_switch_time",
                                 T_SWITCH_DEFAULT))
    return PhysParams(
        blade0=blade0,
        slope=slope,
        gap=float(cutter.minimum_gap),
        v_cut=float(cutter.cutting_speed),
        v_min=float(v_min),
        v_max=float(v_max),
        t_switch=float(t_switch),
    )


def _default_cutter() -> Cutter:
    """Standard-Cutter der Segment-Simulation (lazy, ohne matplotlib-Zwang
    beim Modul-Import)."""
    try:
        from ..simulation import make_default_cutter
    except ImportError:
        from plasma_cutter.segment_simulation.simulation import (
            make_default_cutter,
        )
    return make_default_cutter()


# ---------------------------------------------------------------------------
# Geteilte Distanz-/KDTree-Helfer (auch von dataset.py + planner.py genutzt)
# ---------------------------------------------------------------------------

def _contour_point_index(
    contour: SegmentedContour,
) -> tuple[np.ndarray, np.ndarray]:
    """Alle Konturpunkte aller Segmente + zugehoerige Segment-ID.

    Ein Konturpunkt kann zu zwei Segmenten gehoeren (gemeinsamer Knoten);
    er wird deterministisch dem Segment mit KLEINERER seg_id zugeordnet,
    damit die naechste-Nachbar-Zuordnung reproduzierbar ist.

    Returns
    -------
    pts    : (M, 2) Konturkoordinaten
    seg_of : (M,)   seg_id je Konturpunkt
    """
    seg_of_pos: dict[tuple[int, int], int] = {}
    coords: list[np.ndarray] = []
    seg_ids: list[int] = []
    for seg in sorted(contour.segments, key=lambda s: s.seg_id, reverse=True):
        loop = contour.loop_by_id(seg.loop_id)
        for pos in seg.positions:
            key = (seg.loop_id, pos)
            # reverse=True + Ueberschreiben => am Ende gewinnt die
            # kleinste seg_id (deterministisch).
            seg_of_pos[key] = seg.seg_id
    # Stabil in Segment-/Positionsreihenfolge einsammeln
    seen: set[tuple[int, int]] = set()
    for seg in contour.segments:
        loop = contour.loop_by_id(seg.loop_id)
        for pos in seg.positions:
            key = (seg.loop_id, pos)
            if key in seen:
                continue
            seen.add(key)
            coords.append(loop.points[pos])
            seg_ids.append(seg_of_pos[key])
    return np.asarray(coords, dtype=float), np.asarray(seg_ids, dtype=int)


def assign_points(
    coords: np.ndarray,
    contour: SegmentedContour,
) -> tuple[np.ndarray, np.ndarray]:
    """Weist jeden Materialpunkt seinem naechsten Segment zu.

    Returns
    -------
    seg_idx : (P,) seg_id des naechsten Konturpunkts je Materialpunkt
    depth   : (P,) Distanz zum naechsten Konturpunkt [mm]
              (= Distanztransformationswert, "benoetigte Tiefe")
    """
    pts, seg_of = _contour_point_index(contour)
    tree = cKDTree(pts)
    depth, idx = tree.query(coords, k=1)
    return seg_of[idx], np.asarray(depth, dtype=float)


def coverability_matrix(
    coords: np.ndarray,
    contour: SegmentedContour,
    radius: float,
) -> np.ndarray:
    """Binaere, distanzbasierte Abdeckbarkeits-Schaetzung A_hat[p, s].

    A_hat[p, s] = True  <=>  dist(p, Polylinie(s)) <= radius

    Das ist die BILLIGE Schaetzung der Swept-Area-Abdeckung (nur
    Distanzen, keine Klingen-Kinematik). Genutzt vom Lehrer (dataset) zum
    schnellen Verwerfen unvollstaendiger Subsets und vom Planner im
    Repair-Schritt.
    """
    n_seg = len(contour.segments)
    A = np.zeros((len(coords), n_seg), dtype=bool)
    for seg in contour.segments:
        loop = contour.loop_by_id(seg.loop_id)
        poly = loop.points[np.asarray(seg.positions, dtype=int)]
        tree = cKDTree(poly)
        d, _ = tree.query(coords, k=1)
        A[:, seg.seg_id] = d <= radius
    return A


# ---------------------------------------------------------------------------
# Feature-Definition
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "arc_length",          # Bogenlaenge des Segments [mm]
    "n_points",            # Anzahl Konturpunkte im Segment
    "depth_req_max",       # max. benoetigte Tiefe zugewiesener Punkte [mm]
    "depth_req_mean",      # mittlere benoetigte Tiefe [mm]
    "n_assigned",          # Anzahl zugewiesener Materialpunkte
    "coverable_count",     # abdeckbare Punkte bei v->0
    "exclusive_fraction",  # Anteil exklusiv (nur s) abdeckbarer Punkte
    "wall_thickness",      # lokale Wanddicke ~ 2*max(DT) [mm]
    "has_opposing_wall",   # 1, wenn Gegenwand in Basis-Reichweite
    "corner_angle_start",  # Eckwinkel am Startknoten [rad]
    "corner_angle_end",    # Eckwinkel am Endknoten [rad]
    "mean_abs_turn",       # mittlere |Richtungsaenderung| entlang s [rad]
    "loop_perimeter",      # Umfang des Loops [mm]
    "is_outer",            # 1 = Aussenkontur, 0 = Lochkontur
    "n_loops",             # Anzahl Loops der Geometrie
    "rel_position",        # relative Lage des Segmentstarts im Loop [0..1]
    "v_hat",               # zugewiesene Geschwindigkeit [mm/s]
    "time_est",            # Zeitschaetzung ell/v_hat [s]
]

N_FEATURES = len(FEATURE_NAMES)


def _corner_angle(loop, pos: int) -> float:
    """Richtungsaenderung (Knick) der Kontur an Position pos [rad]."""
    n = loop.n
    v1 = loop.points[pos] - loop.points[(pos - 1) % n]
    v2 = loop.points[(pos + 1) % n] - loop.points[pos]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.acos(cosang)


def _mean_abs_turn(loop, positions: list[int]) -> float:
    """Mittlere absolute Richtungsaenderung entlang einer Punktfolge [rad]
    (Rauheitsmass)."""
    if len(positions) < 3:
        return 0.0
    pts = loop.points[np.asarray(positions, dtype=int)]
    seg = np.diff(pts, axis=0)
    ang = np.arctan2(seg[:, 1], seg[:, 0])
    dang = np.abs(np.diff(ang))
    dang = np.minimum(dang, 2 * math.pi - dang)  # kleinster Winkel
    if len(dang) == 0:
        return 0.0
    return float(np.mean(dang))


def segment_features(
    grid,
    contour: SegmentedContour,
    cutter: Cutter | None = None,
    phys: PhysParams | None = None,
) -> np.ndarray:
    """Feature-Matrix ``X[n_segments, N_FEATURES]`` fuer eine Geometrie.

    Parameters
    ----------
    grid    : PointGrid (liefert die Materialpunkte ``grid.coords``)
    contour : SegmentedContour mit den Primitiv-Segmenten
    cutter  : Cutter fuer die Physik-Parameter (None -> Default)
    phys    : optionale, vorab gebaute ``PhysParams`` (spart Aufbau)

    Returns
    -------
    X : (n_segments, N_FEATURES) float64, deterministisch, ohne NaN.
        Zeile i entspricht ``contour.segments[i]`` (seg_id == i).
    """
    if phys is None:
        phys = phys_from_cutter(cutter)

    coords = np.asarray(grid.coords, dtype=float)
    n_seg = len(contour.segments)
    X = np.zeros((n_seg, N_FEATURES), dtype=np.float64)
    if n_seg == 0:
        return X

    # Zuordnung Punkt -> Segment + benoetigte Tiefe (Distanztransformation)
    seg_idx, depth = assign_points(coords, contour)

    # Abdeckbarkeit bei maximaler Reichweite (v -> 0)
    A = coverability_matrix(coords, contour, radius=phys.full_eff_depth)
    coverable_count = A.sum(axis=0).astype(np.int64)          # je Segment
    n_covering = A.sum(axis=1).astype(np.int64)               # je Punkt
    exclusive_pt = (n_covering == 1)                          # nur 1 Segment

    n_loops = len(contour.loops)

    for seg in contour.segments:
        s = seg.seg_id
        loop = contour.loop_by_id(seg.loop_id)

        assigned = (seg_idx == s)
        n_assigned = int(np.count_nonzero(assigned))
        if n_assigned > 0:
            d_assigned = depth[assigned]
            d_max = float(np.max(d_assigned))
            d_mean = float(np.mean(d_assigned))
        else:
            d_max = 0.0
            d_mean = 0.0

        col = A[:, s]
        cov = int(coverable_count[s])
        excl = int(np.count_nonzero(col & exclusive_pt))
        excl_frac = excl / cov if cov > 0 else 0.0

        wall_thickness = 2.0 * d_max
        has_opposing = 1.0 if d_max <= phys.base_eff_depth else 0.0

        v_hat = phys.speed_for_depth(d_max)
        time_est = seg.length / v_hat if v_hat > 1e-9 else 0.0

        # relative Lage: Bogenlaenge vom ersten Knoten bis Segmentstart
        rel_pos = 0.0
        if loop.length > 1e-9:
            rel_pos = loop.arc_length(
                contour.nodes[seg.loop_id][0], seg.start_pos, +1) / loop.length

        X[s, 0] = seg.length
        X[s, 1] = len(seg.positions)
        X[s, 2] = d_max
        X[s, 3] = d_mean
        X[s, 4] = n_assigned
        X[s, 5] = cov
        X[s, 6] = excl_frac
        X[s, 7] = wall_thickness
        X[s, 8] = has_opposing
        X[s, 9] = _corner_angle(loop, seg.start_pos)
        X[s, 10] = _corner_angle(loop, seg.end_pos)
        X[s, 11] = _mean_abs_turn(loop, seg.positions)
        X[s, 12] = loop.length
        X[s, 13] = 1.0 if loop.kind == "outer" else 0.0
        X[s, 14] = n_loops
        X[s, 15] = float(np.clip(rel_pos, 0.0, 1.0))
        X[s, 16] = v_hat
        X[s, 17] = time_est

    # Sicherheitsnetz: keine NaN/Inf ins Modell
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X

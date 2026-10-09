"""Gemeinsame Planungsstufen für Lehrer, Automatic Planner und Surrogat.

- Runs bauen: zusammenhängende Segmente -> CutRun, Kinematik bei
  zugewiesener Geschwindigkeit v (Klinge L(v) -> Swept Area)
- DP-Split (``split_group_for_speed``, ``build_speed_chains``,
  ``ChainedRun``): eine Kette zerfällt in Blöcke mit eigener
  Geschwindigkeit, Aufschlag ``t_switch`` je Wechsel; der Sequencer sieht
  einen Knoten je Kette
- ``build_plan_with_speeds``: Plan mit variablen Geschwindigkeiten; nicht
  verbindbare Runs werden verworfen statt zu crashen
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree

try:
    from ..segments import SegmentedContour, CutRun
    from ..planning import (
        RunKinematics, LinkPlanner, Sequencer, LinkInfeasibleError, CHAIN_TOL,
    )
    from ...cutter.cutter import Cutter
    from .features import PhysParams, phys_from_cutter
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, CutRun,
    )
    from plasma_cutter.segment_simulation.planning import (
        RunKinematics, LinkPlanner, Sequencer, LinkInfeasibleError, CHAIN_TOL,
    )
    from plasma_cutter.cutter.cutter import Cutter
    from plasma_cutter.segment_simulation.surrogate.features import (
        PhysParams, phys_from_cutter,
    )


# ---------------------------------------------------------------------------
# Exakter Sequencer
# ---------------------------------------------------------------------------

# Alias: macht "exakt sequenziert" (Held-Karp) an den Aufrufstellen sichtbar
ExactSequencer = Sequencer


# ---------------------------------------------------------------------------
# Loop-Reihenfolge der Segmente
# ---------------------------------------------------------------------------

def loop_segment_order(contour: SegmentedContour) -> dict[int, list[int]]:
    """seg_id je Loop in Kontur-/Knotenreihenfolge (für Nachbarschaft)."""
    order: dict[int, list[int]] = {}
    for seg in contour.segments:
        order.setdefault(seg.loop_id, []).append(seg.seg_id)
    return order


def group_contiguous(
    contour: SegmentedContour,
    selected: set[int],
) -> list[tuple[int, list[int]]]:
    """Zerlegt die gewählten Segmente in zyklisch zusammenhängende Gruppen.

    Rückgabe [(loop_id, [seg_id, ...])], je Gruppe ein CutRun (eine
    Zündung).
    """
    order_by_loop = loop_segment_order(contour)
    groups: list[tuple[int, list[int]]] = []
    for loop_id, order in order_by_loop.items():
        chosen = [s for s in order if s in selected]
        if not chosen:
            continue
        k = len(order)
        if len(chosen) == k:
            groups.append((loop_id, list(order)))       # kompletter Loop
            continue
        for i, s in enumerate(order):
            if s not in selected or order[(i - 1) % k] in selected:
                continue  # kein Gruppenstart
            group = [s]
            j = i
            while order[(j + 1) % k] in selected and len(group) < k:
                j += 1
                group.append(order[j % k])
            groups.append((loop_id, group))
    return groups


# ---------------------------------------------------------------------------
# CutRun-Bau + Kinematik bei zugewiesener Geschwindigkeit
# ---------------------------------------------------------------------------

def make_group_run(
    contour: SegmentedContour,
    loop_id: int,
    group: list[int],
    run_id: int,
) -> CutRun:
    """Baut den (noch kinematiklosen) CutRun einer Segmentgruppe."""
    order = loop_segment_order(contour)[loop_id]
    full_loop = len(group) == len(order)
    first = contour.segments[group[0]]
    last = contour.segments[group[-1]]
    loop = contour.loop_by_id(loop_id)
    start_pos = first.start_pos
    end_pos = first.start_pos if full_loop else last.end_pos
    positions = loop.arc_positions(start_pos, end_pos, +1)
    length = (loop.length if full_loop
              else loop.arc_length(start_pos, end_pos, +1))
    return CutRun(
        run_id=run_id, loop_id=loop_id,
        start_pos=start_pos, end_pos=end_pos, direction=+1,
        positions=positions, polyline=loop.polyline(positions),
        length=length,
        node_positions=contour.node_positions_of(loop_id, positions),
    )


def attach_at_speed(
    run: CutRun,
    material,
    phys: PhysParams,
    v: float,
    kerf: float,
) -> CutRun:
    """Kinematik bei Geschwindigkeit ``v`` anheften (in-place).

    Swept-Area-Fehler (GEOS bei ungültiger Geometrie) -> Run als nicht
    ausführbar markiert statt Crash.
    """
    try:
        kin = RunKinematics(material, clearance=phys.gap,
                            blade_length=phys.blade_at(v), kerf=kerf)
        kin.attach(run)
    except Exception:
        run.is_feasible = False
        run.swept_polygon = None
        run.tcp_polyline = None
        run.reason = "Kinematik-Fehler (ungueltige Geometrie)"
    return run


# ---------------------------------------------------------------------------
# Coverage-erhaltendes Verschmelzen (wie AutoPlanner._runs_for_group)
# ---------------------------------------------------------------------------

def build_singletons(
    contour: SegmentedContour,
    seg_ids,
    material,
    phys: PhysParams,
    v: float,
    kerf: float,
    coords: np.ndarray,
) -> tuple[dict, dict]:
    """Singleton-Run je Segment bei ``v`` + boolesche Punktmaske.

    Rückgabe (seg_run, seg_mask): seg_id -> CutRun bzw. Maske.
    """
    seg_run: dict[int, CutRun] = {}
    seg_mask: dict[int, np.ndarray] = {}
    for s in seg_ids:
        seg = contour.segments[s]
        r = make_group_run(contour, seg.loop_id, [s], run_id=s + 1)
        attach_at_speed(r, material, phys, v, kerf)
        seg_run[s] = r
        if r.is_feasible and r.swept_polygon is not None:
            seg_mask[s] = shapely.contains_xy(
                r.swept_polygon, coords[:, 0], coords[:, 1])
        else:
            seg_mask[s] = np.zeros(len(coords), dtype=bool)
    return seg_run, seg_mask


def _singletons_for(contour, group, material, phys, v, kerf, coords):
    """Frische Singleton-Runs + Vereinigungsmaske einer Segmentgruppe."""
    singles: list[CutRun] = []
    target = np.zeros(len(coords), dtype=bool)
    for s in group:
        seg = contour.segments[s]
        r = make_group_run(contour, seg.loop_id, [s], run_id=0)
        attach_at_speed(r, material, phys, v, kerf)
        singles.append(r)
        if r.is_feasible and r.swept_polygon is not None:
            target |= shapely.contains_xy(
                r.swept_polygon, coords[:, 0], coords[:, 1])
    return singles, target


def merge_covering_runs(
    contour: SegmentedContour,
    selected,
    material,
    phys: PhysParams,
    v: float,
    kerf: float,
    coords: np.ndarray,
    check_partial: bool = False,
) -> list[CutRun]:
    """Verschmilzt zusammenhängende Segmente; verliert der Merge Coverage,
    werden die Einzelsegmente genommen.

    - Voll-Loop-Merge: immer exakt geprüft (Ringschluss überstreicht
      teils weniger als die Einzelsegmente)
    - Teilbogen-Merge: nur mit ``check_partial`` geprüft (wie der Lehrer,
      reproduziert dessen Runs; kostet einen Singleton-Aufbau je Gruppe)
    - alle Runs frisch angeheftet
    """
    order_by_loop = loop_segment_order(contour)
    runs: list[CutRun] = []
    rid = 1
    for loop_id, group in group_contiguous(contour, set(selected)):
        merged = make_group_run(contour, loop_id, list(group), run_id=rid)
        attach_at_speed(merged, material, phys, v, kerf)
        feasible = merged.is_feasible and merged.swept_polygon is not None
        is_full = len(group) == len(order_by_loop.get(loop_id, []))

        if feasible and (is_full or check_partial):
            singles, target = _singletons_for(
                contour, group, material, phys, v, kerf, coords)
            merged_mask = shapely.contains_xy(
                merged.swept_polygon, coords[:, 0], coords[:, 1])
            if not bool(np.all(target <= merged_mask)):
                for r in singles:  # Merge verliert Coverage
                    if r.is_feasible and r.swept_polygon is not None:
                        r.run_id = rid
                        runs.append(r)
                        rid += 1
                continue

        if feasible:
            runs.append(merged)
            rid += 1
        else:
            for s in group:  # Merge selbst infeasible -> Einzelsegmente
                seg = contour.segments[s]
                r = make_group_run(contour, seg.loop_id, [s], run_id=rid)
                attach_at_speed(r, material, phys, v, kerf)
                if r.is_feasible and r.swept_polygon is not None:
                    runs.append(r)
                    rid += 1
    return runs


# ---------------------------------------------------------------------------
# Travel-Erreichbarkeit (vollständig umschlossene Löcher ausschließen)
# ---------------------------------------------------------------------------

def linkable_segments(
    contour: SegmentedContour,
    feasible,
    material,
    phys: PhysParams,
    cutter: Cutter,
    kerf: float,
) -> set[int]:
    """Segmente, deren Loop per Eilgang von der Außenkontur erreichbar ist.

    Vollständig umschlossene Löcher fallen raus, damit das
    Coverage-Ziel des Lehrers erreichbar bleibt.
    """
    feas = set(feasible)
    order_by_loop = loop_segment_order(contour)
    outer_ids = [l.loop_id for l in contour.loops if l.kind == "outer"]

    def full_run(loop_id, rid):
        order = [s for s in order_by_loop.get(loop_id, []) if s in feas]
        if not order:
            return None
        r = make_group_run(contour, loop_id, order, run_id=rid)
        return attach_at_speed(r, material, phys, phys.v_cut, kerf)

    link = LinkPlanner(material, clearance=phys.gap)
    seq = Sequencer(cutter, contour, link)
    linkable: set[int] = set()
    outer_run = full_run(outer_ids[0], 1) if outer_ids else None

    for l in contour.loops:
        if l.loop_id not in order_by_loop:
            continue
        if l.kind == "outer" or outer_run is None or not outer_run.is_feasible:
            linkable.add(l.loop_id)
            continue
        hole_run = full_run(l.loop_id, 2)
        if hole_run is None or not hole_run.is_feasible:
            linkable.add(l.loop_id)  # nicht testbar -> behalten
            continue
        try:
            seq.order_runs([outer_run, hole_run])
            linkable.add(l.loop_id)
        except LinkInfeasibleError:
            pass  # vollständig umschlossen -> ausschließen
    return {s for s in feas if contour.segments[s].loop_id in linkable}


# ---------------------------------------------------------------------------
# Coverage-erhaltende Geschwindigkeits-Anhebung
# ---------------------------------------------------------------------------

def fastest_safe_speed(
    run: CutRun,
    coords: np.ndarray,
    material,
    phys: PhysParams,
    kerf: float,
    v_lo: float,
    v_hi: float,
) -> float:
    """Schnellste Geschwindigkeit, die die Basis-Abdeckung des Runs hält.

    - Run muss bei ``v_lo`` angeheftet sein
    - analytisch statt Binärsuche: d_r = größter Abstand eines
      abgedeckten Punkts zur Run-Kontur, v_r = speed_for_depth(d_r)
    - einmal bei v_r anheften und exakt prüfen, sonst zurück auf v_lo
      -> Plan wird nur schneller, Coverage bleibt

    Run bleibt bei der gewählten Geschwindigkeit angeheftet.
    """
    if run.swept_polygon is None or v_hi <= v_lo + 1e-9:
        return v_lo
    base_mask = shapely.contains_xy(run.swept_polygon, coords[:, 0], coords[:, 1])
    if not base_mask.any():
        return v_lo

    # Benötigte Tiefe d_r = tiefster abgedeckter Punkt (Abstand zur Kontur)
    covered = coords[base_mask]
    tree = cKDTree(np.asarray(run.polyline, dtype=float))
    d_r = float(tree.query(covered, k=1)[0].max())
    v = phys.speed_for_depth(d_r)
    if v <= v_lo + 1e-9:
        return v_lo  # keine Beschleunigung möglich

    attach_at_speed(run, material, phys, v, kerf)
    if (run.swept_polygon is not None and run.is_feasible):
        m = shapely.contains_xy(run.swept_polygon, coords[:, 0], coords[:, 1])
        if bool(np.all(base_mask <= m)):
            return v
    # Approximationsfehler -> zurück auf Basisgeschwindigkeit
    attach_at_speed(run, material, phys, v_lo, kerf)
    return v_lo


def speed_up_runs(
    runs: list[CutRun],
    coords: np.ndarray,
    material,
    phys: PhysParams,
    kerf: float,
) -> dict[int, float]:
    """Hebt jeden Run coverage-erhaltend auf seine schnellste
    Geschwindigkeit (Runs bei v_cut angeheftet). Rückgabe run_id -> v.
    """
    speeds: dict[int, float] = {}
    for run in runs:
        if not run.is_feasible or run.swept_polygon is None:
            speeds[run.run_id] = phys.v_cut
            continue
        v = fastest_safe_speed(run, coords, material, phys, kerf,
                               v_lo=phys.v_cut, v_hi=phys.v_max)
        speeds[run.run_id] = v
    return speeds


# ---------------------------------------------------------------------------
# DP-Split: zeitminimale Kettenzerlegung (Geschwindigkeitsregel + t_switch)
# ---------------------------------------------------------------------------

def clamp_rule_speed(phys: PhysParams, depth: float) -> float:
    """Regelgeschwindigkeit für eine Tiefe, geklemmt auf [v_cut, v_max]
    (nie langsamer als v_cut).
    """
    v_hi = max(phys.v_max, phys.v_cut)
    return float(min(max(phys.speed_for_depth(float(depth)), phys.v_cut),
                     v_hi))


def run_required_depth(run: CutRun, coords: np.ndarray,
                       mask: np.ndarray | None = None) -> float:
    """Benötigte Schnitttiefe [mm]: Abstand des konturfernsten abgedeckten
    Punkts zur Run-Kontur. Ohne Attach; ``mask`` = schon berechnete
    Punktmaske.
    """
    if mask is None:
        if run.swept_polygon is None:
            return 0.0
        mask = shapely.contains_xy(run.swept_polygon,
                                   coords[:, 0], coords[:, 1])
    if not mask.any():
        return 0.0
    tree = cKDTree(np.asarray(run.polyline, dtype=float))
    return float(tree.query(coords[mask], k=1)[0].max())


def split_group_for_speed(
    lengths,
    depths,
    phys: PhysParams,
    t_switch: float | None = None,
) -> tuple[list[tuple[int, int]], list[float], float]:
    """Zeitminimale Zerlegung einer Segmentkette in Blöcke mit je einer
    Geschwindigkeit (DP, O(m^2)).

    - statt die ganze Kette mit der Geschwindigkeit der tiefsten Stelle
      zu fahren, dürfen Blöcke schneller fahren
    - Aufschlag ``t_switch`` je Blockübergang (kein Abheben, kein Pierce)
    - Vollkreis-Ketten: lineare Kette ab Gruppenanfang (für alle Planer
      gleich)

    lengths  : Schnittlänge je Kettensegment [mm]
    depths   : benötigte Tiefe je Kettensegment [mm]
    t_switch : Aufschlag [s]; None -> ``phys.t_switch``; inf -> ein Block

    Rückgabe (blocks, speeds, total_time):
      blocks     : (start, ende_exklusiv) in Kettenindizes
      speeds     : Geschwindigkeit je Block [mm/s]
      total_time : Schnitt- + Wechselzeit der Kette [s]

    Grenzfälle (tests/test_speed_split.py): t_switch = inf -> 1 Block;
    t_switch = 0 -> sum(len_i / v_i).
    """
    m = len(lengths)
    if t_switch is None:
        t_switch = float(phys.t_switch)
    if m == 0:
        return [], [], 0.0
    lengths = [float(x) for x in lengths]
    depths = [float(x) for x in depths]
    if not math.isfinite(t_switch):
        v = clamp_rule_speed(phys, max(depths))
        return [(0, m)], [v], sum(lengths) / max(v, 1e-9)

    # dp_cost[i] = minimale Zeit der ersten i Segmente; Tie-Break: weniger
    # Blöcke (vermeidet nutzlose Splits bei t_switch = 0).
    INF = math.inf
    dp_cost = [INF] * (m + 1)
    dp_blocks = [0] * (m + 1)
    parent = [-1] * (m + 1)
    dp_cost[0] = 0.0
    for i in range(1, m + 1):
        d_max = 0.0
        length = 0.0
        for j in range(i - 1, -1, -1):      # Block = Kettensegmente j..i-1
            d_max = max(d_max, depths[j])
            length += lengths[j]
            v = clamp_rule_speed(phys, d_max)
            c = dp_cost[j] + length / max(v, 1e-9) + (t_switch if j > 0
                                                      else 0.0)
            b = dp_blocks[j] + 1
            if (c < dp_cost[i] - 1e-12
                    or (abs(c - dp_cost[i]) <= 1e-12 and b < dp_blocks[i])):
                dp_cost[i], dp_blocks[i], parent[i] = c, b, j

    blocks: list[tuple[int, int]] = []
    i = m
    while i > 0:
        j = parent[i]
        blocks.append((j, i))
        i = j
    blocks.reverse()
    speeds = [clamp_rule_speed(phys, max(depths[a:b])) for a, b in blocks]
    return blocks, speeds, float(dp_cost[m])


# ---------------------------------------------------------------------------
# ChainedRun: Kette als EIN Sequencer-Makro-Knoten + Sub-Run-Rollout
# ---------------------------------------------------------------------------

@dataclass
class ChainedRun:
    """Zusammenhängende Segmentkette als ein Sequencer-Knoten.

    - Held-Karp sieht nur ``macro`` (Kettenenden, Richtung frei)
    - danach werden die Sub-Runs (DP-Blöcke) vom gewählten Ende aus
      ausgerollt; Übergänge nahtlos (kein Link, kein Pierce, nur t_switch)

    macro    : leichter Run nur fürs Sequencing (ohne Swept Area)
    sub_runs : angeheftete Sub-Runs in Kettenreihenfolge
    speeds   : Geschwindigkeit je Sub-Run [mm/s]
    cut_time / switch_time / n_switches : vorgerechnete Zeiten für
               Ketten ohne Sub-Runs
    """
    macro: CutRun
    sub_runs: list[CutRun] = field(default_factory=list)
    speeds: list[float] = field(default_factory=list)
    cut_time: float = 0.0
    switch_time: float = 0.0
    n_switches: int = 0

    def rolled_out(self, ordered_macro: CutRun) -> tuple[list[CutRun],
                                                         list[float]]:
        """Sub-Runs in der vom Sequencer gewählten Richtung ausrollen."""
        if ordered_macro.direction == self.macro.direction:
            return list(self.sub_runs), list(self.speeds)
        return ([r.reversed() for r in reversed(self.sub_runs)],
                list(reversed(self.speeds)))


def _chain_macro(contour, loop_id: int, group: list[int], run_id: int,
                 tcp_start: np.ndarray, tcp_end: np.ndarray,
                 tcp_length: float) -> CutRun:
    """Leichter Makro-Run (nur Endpunkte, kein Attach). Endpunkte aus den
    Sub-Runs, damit Held-Karp exakt die Übergänge des Rollouts bewertet.
    """
    loop = contour.loop_by_id(loop_id)
    first = contour.segments[group[0]]
    last = contour.segments[group[-1]]
    full_loop = first.start_pos == last.end_pos
    return CutRun(
        run_id=run_id, loop_id=loop_id,
        start_pos=first.start_pos,
        end_pos=first.start_pos if full_loop else last.end_pos,
        direction=+1, positions=[],
        polyline=np.asarray([loop.points[first.start_pos],
                             loop.points[first.start_pos if full_loop
                                         else last.end_pos]], dtype=float),
        length=float(sum(contour.segments[s].length for s in group)),
        tcp_polyline=np.asarray([tcp_start, tcp_end], dtype=float),
        tcp_length=float(tcp_length),
    )


def _covers(run: CutRun, coords: np.ndarray, target: np.ndarray) -> bool:
    """Deckt der (angeheftete) Run alle ``target``-Punkte exakt ab?"""
    if not target.any():
        return run.is_feasible and run.swept_polygon is not None
    if not run.is_feasible or run.swept_polygon is None:
        return False
    hit = shapely.contains_xy(run.swept_polygon, coords[target, 0],
                              coords[target, 1])
    return bool(np.all(hit))


def build_speed_chains(
    contour: SegmentedContour,
    selected,
    material,
    phys: PhysParams,
    kerf: float,
    coords: np.ndarray,
    seg_run: dict[int, CutRun] | None = None,
    seg_mask: dict[int, np.ndarray] | None = None,
    t_switch: float | None = None,
) -> list[ChainedRun]:
    """Gemeinsame, exakt geprüfte DP-Split-Stufe aller Planer (gleicher
    Entscheidungsraum -> der Lehrer dominiert).

    Je zusammenhängender Gruppe:
      1. Tiefe/Länge je Segment aus den Singleton-Runs (v_cut) -> DP-Split
      2. jeden Block bei seiner Geschwindigkeit anheften, gegen die
         Singleton-Masken prüfen; verfehlt -> v_cut; immer noch ->
         Einzelsegmente mit ``fastest_safe_speed``
      3. Makro-Run je Kette für den Sequencer

    Zugesagte Abdeckung = Vereinigung der Singleton-Masken. Beim Prüfen
    können Geschwindigkeiten nur fallen -> DP-Zeit ist Untergrenze.
    """
    if t_switch is None:
        t_switch = float(phys.t_switch)
    v_cut = phys.v_cut
    v_hi = max(phys.v_max, v_cut)
    sel = sorted(set(selected))
    if seg_run is None or seg_mask is None:
        seg_run, seg_mask = build_singletons(
            contour, sel, material, phys, v_cut, kerf, coords)

    chains: list[ChainedRun] = []
    rid = 1
    for loop_id, group in group_contiguous(contour, set(sel)):
        depths: list[float] = []
        lengths: list[float] = []
        for s in group:
            r = seg_run.get(s)
            m = seg_mask.get(s)
            if r is None or not r.is_feasible or m is None or not m.any():
                depths.append(0.0)
                lengths.append(float(contour.segments[s].length))
            else:
                depths.append(run_required_depth(r, coords, m))
                lengths.append(float(r.tcp_length if r.tcp_length > 0
                                     else r.length))
        blocks, speeds, _ = split_group_for_speed(lengths, depths, phys,
                                                  t_switch)

        sub_runs: list[CutRun] = []
        sub_speeds: list[float] = []
        for (a, b), v_blk in zip(blocks, speeds):
            block = list(group[a:b])
            target = np.zeros(len(coords), dtype=bool)
            for s in block:
                if s in seg_mask:
                    target |= seg_mask[s]
            run = make_group_run(contour, loop_id, block, run_id=0)
            attach_at_speed(run, material, phys, v_blk, kerf)
            if _covers(run, coords, target):
                sub_runs.append(run)
                sub_speeds.append(v_blk)
                continue
            if abs(v_blk - v_cut) > 1e-9:
                attach_at_speed(run, material, phys, v_cut, kerf)
                if _covers(run, coords, target):
                    sub_runs.append(run)
                    sub_speeds.append(v_cut)
                    continue
            # auch bei v_cut verfehlt -> Einzelsegmente, je einzeln geprüft
            for s in block:
                single = make_group_run(contour, loop_id, [s], run_id=0)
                attach_at_speed(single, material, phys, v_cut, kerf)
                if not single.is_feasible or single.swept_polygon is None:
                    continue
                v_s = fastest_safe_speed(single, coords, material, phys,
                                         kerf, v_lo=v_cut, v_hi=v_hi)
                sub_runs.append(single)
                sub_speeds.append(v_s)
        if not sub_runs:
            continue
        for r in sub_runs:
            r.run_id = rid
            rid += 1
        macro = _chain_macro(
            contour, loop_id, list(group), run_id=rid,
            tcp_start=sub_runs[0].tcp_start, tcp_end=sub_runs[-1].tcp_end,
            tcp_length=float(sum(r.tcp_length for r in sub_runs)))
        rid += 1
        chains.append(ChainedRun(macro=macro, sub_runs=sub_runs,
                                 speeds=sub_speeds))
    return chains


# ---------------------------------------------------------------------------
# Robuster Planbau mit variablen Geschwindigkeiten
# ---------------------------------------------------------------------------

@dataclass
class SpeedPlan:
    """Ergebnis von ``build_plan_with_speeds``; ``switch_time``/``n_switches``
    = Geschwindigkeitswechsel im laufenden Schnitt (wie ``pierce_time``
    separat ausgewiesen).
    """
    ordered_runs: list[CutRun] = field(default_factory=list)
    run_speeds: list[float] = field(default_factory=list)
    total_time: float = 0.0
    cut_time: float = 0.0
    travel_time: float = 0.0
    pierce_time: float = 0.0
    switch_time: float = 0.0
    n_pierces: int = 0
    n_switches: int = 0
    dropped_run_ids: list[int] = field(default_factory=list)
    is_optimal: bool = False


def _pairwise_linkable(seq: Sequencer, runs: list[CutRun]) -> dict[int, int]:
    """Zahl der verbindbaren Partner je Run (in beliebiger Richtung)."""
    variants = [(r, r.reversed()) for r in runs]
    conn = {r.run_id: 0 for r in runs}
    for i, (fi, ri) in enumerate(variants):
        for j, (fj, rj) in enumerate(variants):
            if i == j:
                continue
            ok = any(math.isfinite(seq._trans_cost(a.tcp_end, b.tcp_start))
                     for a in (fi, ri) for b in (fj, rj))
            if ok:
                conn[runs[i].run_id] += 1
    return conn


def build_plan_with_speeds(
    sequencer: Sequencer,
    runs: list,
    run_speed: dict[int, float],
    cutter: Cutter,
    drop_unlinkable: bool = True,
    t_switch: float = 0.0,
) -> SpeedPlan:
    """Zeitminimal geordneter Plan mit variablen Schnittgeschwindigkeiten.

    runs darf CutRun und ChainedRun mischen (run_ids eindeutig):
      CutRun     : Geschwindigkeit aus ``run_speed``
      ChainedRun : ein Held-Karp-Knoten je Kette, danach Sub-Runs
                   ausrollen; ohne Sub-Runs gehen die vorgerechneten
                   Zeiten ein

    - nahtloser Übergang (< CHAIN_TOL): kein Link, kein Pierce, t_switch
      bei Geschwindigkeitswechsel; sonst Eilgang + Pierce
    - ``LinkInfeasibleError``: nicht verbindbare Runs werden verworfen,
      ihre Punkte bleiben ungeschnitten
    """
    plan = SpeedPlan()
    chains: dict[int, ChainedRun] = {}
    macros: list[CutRun] = []
    for r in runs:
        if isinstance(r, ChainedRun):
            chains[r.macro.run_id] = r
            macros.append(r.macro)
        elif r.is_feasible and r.swept_polygon is not None:
            macros.append(r)
    if not macros:
        return plan

    # nicht verbindbar -> schlechtesten Knoten verwerfen, neu versuchen
    while macros:
        try:
            ordered, is_opt = sequencer.order_runs(macros)
            break
        except LinkInfeasibleError:
            if not drop_unlinkable or len(macros) <= 1:
                # ein einzelner Knoten geht immer (keine Übergänge)
                if len(macros) == 1:
                    ordered, is_opt = [macros[0]], True
                    break
                return plan
            conn = _pairwise_linkable(sequencer, macros)
            # wenigste Partner fällt; Gleichstand -> der kürzere Knoten
            # (weniger Coverage-Verlust; bei test_lochjson bleibt so die
            # Außenkontur statt des umschlossenen Lochs)
            worst = min(macros, key=lambda r: (
                conn[r.run_id],
                r.tcp_length if r.tcp_length > 0 else r.length))
            plan.dropped_run_ids.append(worst.run_id)
            macros = [r for r in macros if r is not worst]
    else:
        return plan

    plan.is_optimal = is_opt
    pierce = cutter.pierce_time()

    prev_end = None
    prev_v: float | None = None
    for macro in ordered:
        chain = chains.get(macro.run_id)
        if chain is not None and not chain.sub_runs:
            # Kette ohne Sub-Runs: Zeiten vorgerechnet
            plan.cut_time += chain.cut_time
            plan.switch_time += chain.switch_time
            plan.n_switches += chain.n_switches
            items: list[tuple[CutRun, float | None]] = [(macro, None)]
        elif chain is not None:
            items = list(zip(*chain.rolled_out(macro)))
        else:
            items = [(macro, run_speed.get(macro.run_id,
                                           cutter.cutting_speed))]

        for run, v in items:
            if v is not None:
                cut_len = run.tcp_length if run.tcp_length > 0 else run.length
                plan.cut_time += cut_len / max(v, 1e-9)
                plan.run_speeds.append(float(v))
            else:
                plan.run_speeds.append(float("nan"))
            plan.ordered_runs.append(run)

            needs_pierce = True
            if prev_end is not None:
                gap = float(np.linalg.norm(run.tcp_start - prev_end))
                if gap < CHAIN_TOL:
                    # nahtlos: Brenner bleibt an, nur t_switch bei v-Wechsel
                    needs_pierce = False
                    if (v is not None and prev_v is not None
                            and abs(v - prev_v) > 1e-9):
                        plan.switch_time += t_switch
                        plan.n_switches += 1
                else:
                    link = sequencer.link_between(prev_end, run.tcp_start)
                    if link is not None:
                        plan.travel_time += link.length / cutter.rapid_speed
            if needs_pierce:
                plan.pierce_time += pierce
                plan.n_pierces += 1
            prev_end = run.tcp_end
            prev_v = v

    plan.total_time = (plan.cut_time + plan.travel_time
                       + plan.pierce_time + plan.switch_time)
    return plan

"""Brute-Force-Lehrer: Aufzählung aller Segment-Teilmengen.

1. Singleton-Runs: jedes Segment einmal bei v_cut -> exakte Punktmaske
2. alle Teilmengen der verbindbaren Segmente aufzählen, nur vollständige
   Abdeckungen behalten
3. jede vollständige Abdeckung exakt bewerten (DP-Split, Held-Karp,
   LinkPlanner), die schnellste gewinnt

- Ergebnis: exaktes Optimum über Auswahl x Reihenfolge x Richtung x
  Geschwindigkeitsblöcke bei fester Segmentierung
- Aufwand 2^k: mehr als ``k_max`` verbindbare Segmente -> ``TeacherSkipped``

Verwendet für die Trainingslabels (``dataset``), das Optimum im Benchmark
und Taste B im Simulator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import shapely

try:
    from ..planning import LinkPlanner
    from ..segments import SegmentedContour, compute_grid_coverage
    from .features import phys_from_cutter
    from .instances import default_cutter
    from .params import KERF, LABEL_VERSION
    from .runutils import (
        ExactSequencer, attach_at_speed, build_plan_with_speeds,
        build_speed_chains, linkable_segments, make_group_run,
    )
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.planning import LinkPlanner
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, compute_grid_coverage,
    )
    from plasma_cutter.segment_simulation.surrogate.features import (
        phys_from_cutter,
    )
    from plasma_cutter.segment_simulation.surrogate.instances import (
        default_cutter,
    )
    from plasma_cutter.segment_simulation.surrogate.params import (
        KERF, LABEL_VERSION,
    )
    from plasma_cutter.segment_simulation.surrogate.runutils import (
        ExactSequencer, attach_at_speed, build_plan_with_speeds,
        build_speed_chains, linkable_segments, make_group_run,
    )

# LABEL_VERSION, KERF: aus ``params`` re-exportiert
__all__ = ["LABEL_VERSION", "KERF", "K_MAX_DEFAULT", "TeacherSkipped",
           "ExhaustiveResult", "iter_complete_covers", "exhaustive_plan"]
# praktische Obergrenze, die Rechenzeit wächst exponentiell mit k
K_MAX_DEFAULT = 18
_LOW_BITS = 16


class TeacherSkipped(RuntimeError):
    """Der Brute-Force-Lehrer wurde übersprungen (zu viele Segmente)."""


@dataclass
class ExhaustiveResult:
    """Ergebnis des Aufzählungs-Lehrers."""
    selected: list[int] = field(default_factory=list)   # optimale seg_ids
    total_time: float = 0.0        # exakte T des Siegerplans [s]
    coverage: float = 0.0          # exakte Coverage des Siegerplans
    n_segments: int = 0            # Segmente der Kontur
    n_feasible: int = 0            # verbindbare Segmente (Aufzählbasis)
    n_subsets: int = 0             # 2^n_feasible - 1
    n_covers: int = 0              # vollständige Abdeckungen (bewertet)
    n_unlinkable: int = 0          # verworfen: nicht verbindbar
    plan_time: float = 0.0         # Wall-Clock gesamt [s]
    plan: object = None            # SpeedPlan des Siegers
    # (seg_ids, T) je bewerteter Abdeckung, nur mit ``keep_covers``
    covers: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def _pack_mask(mask: np.ndarray) -> int:
    """Boolesche Punktmaske als Python-Integer (schnelle Mengen-OR)."""
    return int.from_bytes(np.packbits(mask).tobytes(), "big")


def _cover_table(mask_int: list[int]) -> list[int]:
    """Vereinigungsmaske jeder Teilmenge (Index = Bitmaske), DP über das
    niedrigste gesetzte Bit.
    """
    n = len(mask_int)
    cov = [0] * (1 << n)
    for b in range(1, 1 << n):
        low = b & -b
        cov[b] = cov[b ^ low] | mask_int[low.bit_length() - 1]
    return cov


def iter_complete_covers(mask_int: list[int], full_int: int):
    """Bitmasken aller Teilmengen mit vollständiger Abdeckung, aufsteigend.

    Untere ``_LOW_BITS`` Segmente als Tabelle, obere Bits in der äußeren
    Schleife (2^16 gespeicherte Masken statt 2^k).
    """
    n = len(mask_int)
    nl = min(n, _LOW_BITS)
    nh = n - nl
    low = _cover_table(mask_int[:nl])
    high = mask_int[nl:]
    for hb in range(1 << nh):
        acc = 0
        for j in range(nh):
            if hb >> j & 1:
                acc |= high[j]
        base = hb << nl
        for lb, cl in enumerate(low):
            if (acc | cl) == full_int and (base | lb):
                yield base | lb


def exhaustive_plan(
    grid,
    cutter=None,
    kerf: float = KERF,
    k_max: int = K_MAX_DEFAULT,
    contour: SegmentedContour | None = None,
    keep_covers: bool = True,
    speed_rule: bool = True,
) -> ExhaustiveResult:
    """Exaktes Optimum durch vollständige Aufzählung der Segment-Teilmengen.

    contour     : Segmentierung der seg_ids (Simulator: dessen Kontur),
                  sonst ``SegmentedContour.from_grid``
    speed_rule  : False -> v_max = v_cut, nur Basisgeschwindigkeit
    keep_covers : alle bewerteten Abdeckungen in ``covers`` behalten
    """
    t0 = time.perf_counter()
    if cutter is None:
        cutter = default_cutter()
    if contour is None:
        contour = SegmentedContour.from_grid(grid)
    n_seg = len(contour.segments)
    res = ExhaustiveResult(n_segments=n_seg)
    if n_seg == 0:
        res.plan_time = time.perf_counter() - t0
        return res

    material = contour.material_polygon()
    phys = phys_from_cutter(cutter)
    if not speed_rule:
        phys = replace(phys, v_max=phys.v_cut)
    t_switch = float(phys.t_switch)
    coords = np.asarray(grid.coords, dtype=float)
    v_cut = phys.v_cut
    timings = {"singletons": 0.0, "enumerate": 0.0, "exact": 0.0,
               "final": 0.0}

    # 1. Singleton-Runs: exakte Maske je Segment
    t_st = time.perf_counter()
    seg_run, seg_mask = {}, {}
    feasible: list[int] = []
    for seg in contour.segments:
        s = seg.seg_id
        run = make_group_run(contour, seg.loop_id, [s], run_id=s + 1)
        attach_at_speed(run, material, phys, v_cut, kerf)
        if not run.is_feasible or run.swept_polygon is None:
            continue
        col = shapely.contains_xy(run.swept_polygon, coords[:, 0], coords[:, 1])
        seg_run[s], seg_mask[s] = run, col
        feasible.append(s)
    linkable = linkable_segments(contour, feasible, material, phys, cutter, kerf)
    feasible = [s for s in feasible if s in linkable]
    timings["singletons"] = time.perf_counter() - t_st
    nf = len(feasible)
    res.n_feasible = nf
    if nf > k_max:
        raise TeacherSkipped(f"{nf} verbindbare Segmente > k_max={k_max}")
    if nf == 0:
        res.plan_time = time.perf_counter() - t0
        return res

    # 2. Alle Teilmengen, vollständige Abdeckungen behalten
    t_st = time.perf_counter()
    full_mask = np.zeros(len(coords), dtype=bool)
    for s in feasible:
        full_mask |= seg_mask[s]
    full_int = _pack_mask(full_mask)
    mask_int = [_pack_mask(seg_mask[s]) for s in feasible]
    covers = list(iter_complete_covers(mask_int, full_int))
    res.n_subsets = (1 << nf) - 1
    timings["enumerate"] = time.perf_counter() - t_st

    # 3. Jede vollständige Abdeckung exakt bewerten, Minimum behalten
    sequencer = ExactSequencer(cutter, contour,
                               LinkPlanner(material, clearance=phys.gap))

    def sel_of(b: int) -> set[int]:
        return {feasible[i] for i in range(nf) if b >> i & 1}

    def price_exact(selected: set[int]):
        chains = build_speed_chains(
            contour, selected, material, phys, kerf, coords,
            seg_run=seg_run, seg_mask=seg_mask, t_switch=t_switch)
        plan = build_plan_with_speeds(sequencer, chains, {}, cutter,
                                      drop_unlinkable=True, t_switch=t_switch)
        if plan.dropped_run_ids or not plan.ordered_runs:
            return None, None
        return float(plan.total_time), plan

    best_T, best_sel, best_plan = np.inf, None, None
    t_st = time.perf_counter()
    for b in covers:
        selected = sel_of(b)
        T, plan = price_exact(selected)
        if T is None:
            res.n_unlinkable += 1
            continue
        res.n_covers += 1
        if keep_covers:
            res.covers.append((sorted(selected), T))
        if T < best_T:
            best_T, best_sel, best_plan = T, sorted(selected), plan
    timings["exact"] = time.perf_counter() - t_st

    # 4. Ergebnis
    t_st = time.perf_counter()
    if best_sel is not None and best_plan is not None:
        res.selected = list(best_sel)
        res.plan = best_plan
        res.total_time = float(best_plan.total_time)
        res.coverage = compute_grid_coverage(grid, best_plan.ordered_runs).fraction
    timings["final"] = time.perf_counter() - t_st
    res.timings = timings
    res.plan_time = time.perf_counter() - t0
    return res

"""Surrogat-Planer und Greedy+ (BA Kap. 5).

``surrogate_plan(grid, model)``:

    Features -> Modell p(s) -> Greedy Set Cover in p(s)-Reihenfolge auf
    EXAKTEN Singleton-Masken (lazy, nur beruehrte Segmente) -> optional
    Pruning redundanter Segmente -> EIN Planbau (DP-Split + Held-Karp)
    -> EIN exakter Verify -> klassischer Fallback (Greedy-Auswahl).

Das Modell bestimmt nur die REIHENFOLGE. Masken, Planbau, Verify und
Fallback sind exakt bzw. klassisch, die Coverage-Garantie haengt nie am
Modell: selbst ein Zufallsmodell liefert am Ende einen gueltigen Plan.

``greedy_plus_plan(grid)``: die klassische Greedy-Set-Cover-Auswahl des
``AutoPlanner`` (Kap. 4), aber mit DERSELBEN Geschwindigkeitsstufe wie
Surrogat und Lehrer (DP-Split je Kette) -- die faire Vergleichsbasis.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

try:
    from ...geometry.point_grid import PointGrid
    from ..segments import SegmentedContour, compute_grid_coverage
    from ..planning import LinkPlanner, RunKinematics
    from ..autoplan import AutoPlanner
    from .features import segment_features, phys_from_cutter
    from .instances import default_cutter
    from .params import KERF
    from .runutils import (
        ExactSequencer, build_plan_with_speeds, SpeedPlan,
        build_speed_chains, build_singletons,
    )
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.geometry.point_grid import PointGrid
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, compute_grid_coverage,
    )
    from plasma_cutter.segment_simulation.planning import (
        LinkPlanner, RunKinematics,
    )
    from plasma_cutter.segment_simulation.autoplan import AutoPlanner
    from plasma_cutter.segment_simulation.surrogate.features import (
        segment_features, phys_from_cutter,
    )
    from plasma_cutter.segment_simulation.surrogate.instances import (
        default_cutter,
    )
    from plasma_cutter.segment_simulation.surrogate.params import KERF
    from plasma_cutter.segment_simulation.surrogate.runutils import (
        ExactSequencer, build_plan_with_speeds, SpeedPlan,
        build_speed_chains, build_singletons,
    )


# ---------------------------------------------------------------------------
# Ergebnis
# ---------------------------------------------------------------------------

@dataclass
class SurrogatePlanResult:
    """Ergebnis von ``surrogate_plan``.

    Attributes
    ----------
    runs          : geordnete CutRuns des finalen Plans
    plan          : SpeedPlan (Zeitbilanz mit variablen Schnittgeschw.)
    selected      : gewaehlte seg_ids (nach Pruning bzw. des Fallbacks)
    coverage      : exakte Querschnitts-Coverage [0..1]
    T             : Ausfuehrungszeit des Plans [s]
    t_plan        : Wall-Clock der GESAMTEN Pipeline [s]
    n_pruned      : Anzahl exakt weggekuerzter (redundanter) Segmente
    used_fallback : True, wenn auf die Greedy-Auswahl zurueckgefallen wurde
    n_unreachable : ehrlich unerreichbare Gitterpunkte (auch fuer Greedy)
    n_candidates  : Anzahl Primitiv-Segmente
    n_selected    : Anzahl gewaehlter Segmente
    timings       : Stufen-Timings [s]
    """
    runs: list = field(default_factory=list)
    plan: SpeedPlan = field(default_factory=SpeedPlan)
    selected: list[int] = field(default_factory=list)
    coverage: float = 0.0
    T: float = 0.0
    t_plan: float = 0.0
    n_pruned: int = 0
    used_fallback: bool = False
    n_unreachable: int = 0
    n_candidates: int = 0
    n_selected: int = 0
    timings: dict = field(default_factory=dict)

    def summary(self) -> str:       #Übersicht alle Lautfzeiten in ms
        tg = self.timings
        stages = " ".join(f"{k}={tg.get(k, 0) * 1e3:.1f}ms"
                          for k in ("features", "predict", "masks", "prune",
                                    "speedup", "sequence", "verify", "repair"))
        return (
            f"Surrogat: {self.n_selected}/{self.n_candidates} Segmente -> "
            f"{len(self.runs)} Schnitt(e), Coverage {self.coverage:.1%}"
            + (f" ({self.n_unreachable} unerreichbar)" if self.n_unreachable else "")
            + f", T={self.T:.1f}s, geplant in {self.t_plan * 1e3:.1f}ms"
            + (" [FALLBACK]" if self.used_fallback else "")
            + f", gepruned={self.n_pruned}\n  Stufen: {stages}"
        )


# ---------------------------------------------------------------------------
# Robuster klassischer Fallback (crash-frei gegen LinkInfeasibleError)
# ---------------------------------------------------------------------------

def _fallback_plan(grid: PointGrid, contour, material, cutter, phys, kerf):
    """Klassische Greedy-Auswahl (``AutoPlanner``: Greedy + Pruning +
    Merge), robust gegen nicht verbindbare Touren, sequenziert bei
    Basisgeschwindigkeit. Liefert (runs, mask, SpeedPlan, selected).
    Dies ist zugleich die Referenz fuer die ehrliche Erreichbarkeit.
    """
    blade = cutter.blade_length(cutter.cutting_speed)
    kin = RunKinematics(material, clearance=phys.gap,
                        blade_length=blade, kerf=kerf)
    link = LinkPlanner(material, clearance=phys.gap)
    seq = ExactSequencer(cutter, contour, link)
    selected: list[int] = []
    try:
        ap = AutoPlanner(grid, contour, cutter, kin, seq)
        selected = sorted(ap._prune(ap._greedy()))
        runs = [r for r in ap._merge_to_runs(selected)
                if r.is_feasible and r.swept_polygon is not None]
        run_speed = {r.run_id: cutter.cutting_speed for r in runs} #  Weist Jedem Schnitt die Standard-Geschwindigkeit zu
        plan = build_plan_with_speeds(seq, runs, run_speed, cutter,
                                      drop_unlinkable=True)
    except Exception:
        # Auch die klassische Auswahl kann an ungueltiger Geometrie
        # scheitern -> leere Referenz (kein Crash).
        plan = SpeedPlan()
    kept = plan.ordered_runs  #Welche runs geschnitten werden können
    mask = compute_grid_coverage(grid, kept).mask #Welche Punkte geschnitten werden können
    return kept, mask, plan, selected


# ---------------------------------------------------------------------------
# Greedy+ : klassische Auswahl MIT Geschwindigkeitsregel (fairer Vergleich)
# ---------------------------------------------------------------------------

def greedy_plus_plan(grid: PointGrid, cutter=None, kerf: float = KERF,
                     contour: SegmentedContour | None = None):
    """Greedy-Set-Cover-Auswahl des ``AutoPlanner``, aber mit gleicher
    Geschwindigkeitsstufe wie Surrogat/Lehrer: Dient als Baseline.

    Returns
    -------
    dict mit T (Ausfuehrungszeit), coverage, n_runs, t_plan (Wall-Clock
    der gesamten Planung [s]) und timings (Stufen-Timings [s]:
    candidates = Kandidaten/Swept-Areas des AutoPlanner, select =
    Greedy-Set-Cover + Pruning, sequence = Held-Karp + LinkPlanner,
    speedup = DP-Split der Regel 4.5, verify = exakte Coverage). Bei
    Crash der Auswahl: T=None (nicht in den Vergleich aufnehmen).
    ``contour``: vorgegebene Segmentierung (Benchmark mit feiner
    Segmentierung); sonst ``SegmentedContour.from_grid``.
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {"candidates": 0.0, "select": 0.0,
                                 "sequence": 0.0, "speedup": 0.0,
                                 "verify": 0.0}

    def _fail():
        return {"T": None, "coverage": 0.0, "n_runs": 0,
                "t_plan": time.perf_counter() - t_start, "timings": timings}

    if cutter is None:
        cutter = default_cutter()
    phys = phys_from_cutter(cutter)
    try:
        if contour is None:
            contour = SegmentedContour.from_grid(grid)
        material = contour.material_polygon()
    except Exception:
        return _fail()
    if material is None:
        return _fail()

    coords = np.asarray(grid.coords, dtype=float) #NumPy-Array

    # Klassische Greedy-Auswahl + Basisplan als Referenz
    blade = cutter.blade_length(cutter.cutting_speed)
    kin = RunKinematics(material, clearance=phys.gap,
                        blade_length=blade, kerf=kerf)
    seq = ExactSequencer(cutter, contour,
                         LinkPlanner(material, clearance=phys.gap))
    try:
        t0 = time.perf_counter()
        ap = AutoPlanner(grid, contour, cutter, kin, seq)
        timings["candidates"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        selected = ap._prune(ap._greedy())
        timings["select"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        base_runs = [r for r in ap._merge_to_runs(selected)
                     if r.is_feasible and r.swept_polygon is not None]
        base_speed = {r.run_id: cutter.cutting_speed for r in base_runs}
        base_plan = build_plan_with_speeds(seq, base_runs, base_speed,
                                           cutter, drop_unlinkable=True)
        timings["sequence"] += time.perf_counter() - t0
    except Exception:
        return _fail()
    if not base_plan.ordered_runs:
        return _fail()
    t0 = time.perf_counter()
    base_mask = compute_grid_coverage(grid, base_plan.ordered_runs).mask #Pro Punkt Boolean
    timings["verify"] += time.perf_counter() - t0

    # Gemeinsame DP-Split-Stufe (identisch zu Lehrer und Surrogat)
    try:
        t0 = time.perf_counter()
        chains = build_speed_chains(contour, selected, material, phys,
                                    kerf, coords)
        timings["speedup"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        plan = build_plan_with_speeds(seq, chains, {}, cutter,
                                      drop_unlinkable=True,
                                      t_switch=phys.t_switch)
        timings["sequence"] += time.perf_counter() - t0
        if plan.ordered_runs:
            t0 = time.perf_counter()
            mask = compute_grid_coverage(grid, plan.ordered_runs).mask
            timings["verify"] += time.perf_counter() - t0
            if (bool(np.all(base_mask <= mask))
                    and plan.total_time <= base_plan.total_time + 1e-9): #Check ob Punkte verloren + Zeit schlechter
                return {"T": plan.total_time, "coverage": float(mask.mean()),
                        "n_runs": len(plan.ordered_runs),
                        "t_plan": time.perf_counter() - t_start,
                        "timings": timings}
    except Exception:
        pass
    return {"T": base_plan.total_time, "coverage": float(base_mask.mean()),
            "n_runs": len(base_plan.ordered_runs),
            "t_plan": time.perf_counter() - t_start, "timings": timings}


# ---------------------------------------------------------------------------
# Kern: surrogate_plan
# ---------------------------------------------------------------------------

def surrogate_plan(
    grid: PointGrid,
    model,
    cutter=None,
    kerf: float = KERF,
    prune: bool = True,
    proba_override=None,
    contour: SegmentedContour | None = None,
    speed_rule: bool = True,
) -> SurrogatePlanResult:
    """Surrogat-Planer

    ``prune=False`` laesst die Auswahl des Greedy Set Cover unveraendert;
    ``proba_override`` ersetzt die Modellausgabe (Ablation: Zufall/
    Heuristik). ``contour`` ist die Segmentierung, auf die sich die
    seg_ids beziehen (Simulator: dessen Kontur). ``speed_rule=False``
    klemmt v_max auf v_cut fuer Planbau und Fallback (Simulator mit
    Regel-4.5-Schalter AUS); die Merkmale werden weiterhin mit den
    Originalparametern berechnet, mit denen das Modell trainiert wurde.
    """
    t_start = time.perf_counter()
    timings = {"features": 0.0, "predict": 0.0, "masks": 0.0, "prune": 0.0,
               "speedup": 0.0, "sequence": 0.0, "verify": 0.0, "repair": 0.0}
    if cutter is None:
        cutter = default_cutter()
    phys = phys_from_cutter(cutter)
    phys_plan = phys if speed_rule else replace(phys, v_max=phys.v_cut)
    if contour is None:
        contour = SegmentedContour.from_grid(grid)
    material = contour.material_polygon()
    n_seg = len(contour.segments)
    coords = np.asarray(grid.coords, dtype=float)
    n_pts = len(coords)
    result = SurrogatePlanResult(n_candidates=n_seg, timings=timings)
    
    if n_seg == 0 or material is None:
        result.t_plan = time.perf_counter() - t_start
        return result

    # A: Features + Modell-Rangfolge
    t0 = time.perf_counter()
    X = segment_features(grid, contour, cutter, phys=phys)
    timings["features"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = (np.asarray(proba_override, dtype=float) if proba_override is not None
             else np.asarray(model.predict_proba(X), dtype=float))
    order = np.argsort(-proba)
    timings["predict"] = time.perf_counter() - t0

    # B: Greedy Set Cover
    t0 = time.perf_counter()
    seg_run, seg_mask = {}, {}
    covered = np.zeros(n_pts, dtype=bool)
    selected: set[int] = set() #IDs der Segmente die gewählt werden
    for s in order:
        if covered.all():
            break
        s = int(s)
        r1, m1 = build_singletons(contour, [s], material, phys_plan,
                                  phys_plan.v_cut, kerf, coords)
        seg_run.update(r1)
        seg_mask.update(m1)
        if bool((seg_mask[s] & ~covered).any()): #bringt das neue segmente Punkte
            selected.add(s)
            covered |= seg_mask[s]
    timings["masks"] = time.perf_counter() - t0

    # C: Pruning (optional): Segment raus, wenn jeder seiner Punkte noch
    # von einem anderen gewaehlten Segment gedeckt ist; unsicherste zuerst
    t0 = time.perf_counter()
    n_pruned = 0
    if prune and len(selected) > 1:
        cover_count = np.zeros(n_pts, dtype=int)
        for s in selected:
            cover_count += seg_mask[s]
        for s in sorted(selected, key=lambda i: float(proba[i])):
            col = seg_mask[s]
            if bool(col.any()) and bool(np.all(cover_count[col] >= 2)):
                selected.discard(s)
                cover_count[col] -= 1
                n_pruned += 1
    timings["prune"] = time.perf_counter() - t0
    result.n_pruned = n_pruned

    # D: EIN Planbau (Singleton-Reuse) + EIN Sequencing
    seq = ExactSequencer(cutter, contour,
                         LinkPlanner(material, clearance=phys_plan.gap))
    t0 = time.perf_counter()
    chains = build_speed_chains(contour, selected, material, phys_plan, kerf,
                                coords, seg_run=seg_run, seg_mask=seg_mask,
                                t_switch=phys_plan.t_switch)
    timings["speedup"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    plan = build_plan_with_speeds(seq, chains, {}, cutter,
                                  drop_unlinkable=True,
                                  t_switch=phys_plan.t_switch)
    timings["sequence"] = time.perf_counter() - t0

    # E: EIN exakter Verify
    t0 = time.perf_counter()
    mask = compute_grid_coverage(grid, plan.ordered_runs).mask
    timings["verify"] = time.perf_counter() - t0

    # F:  Fallback, wenn  erreichbare Punkte fehlen
    used_fallback = False
    n_unreachable = 0
    if not mask.all():
        t0 = time.perf_counter()
        fb_runs, fb_mask, fb_plan, fb_sel = _fallback_plan(
            grid, contour, material, cutter, phys_plan, kerf)
        if int(((~mask) & fb_mask).sum()) > 0:
            used_fallback = True
            plan, mask, selected = fb_plan, fb_mask, set(fb_sel)
        n_unreachable = int(np.count_nonzero(~mask))
        timings["repair"] = time.perf_counter() - t0

    result.runs = plan.ordered_runs
    result.plan = plan
    result.selected = sorted(int(s) for s in selected)
    result.coverage = float(mask.mean()) if n_pts else 0.0
    result.T = plan.total_time
    result.used_fallback = used_fallback
    result.n_unreachable = n_unreachable
    result.n_selected = len(selected)
    result.t_plan = time.perf_counter() - t_start
    return result

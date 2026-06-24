from __future__ import annotations

"""Internes Benchmark fuer den Auto-Planer (autoplan.AutoPlanner).

Ziel
----
Die Parameter durchspielen, die die Laufzeit WIRKLICH bewegen, und
jeweils Laufzeit / Coverage / Punktezahl vergleichen:

  A) Punktabstand   (grid_spacing des Innenrasters)  -> treibt P
  B) Segmentlaenge   (target_segment_length)          -> treibt K
  C) Held-Karp N     (Sequencer.EXACT_MAX_RUNS)        -> Sequencer-Kosten

A und B werden end-to-end am echten Auto-Planer gemessen, inkl.
Phasen-Aufschluesselung (Kandidaten / Greedy / Pruning / Merge /
Sequencer). C ist ein isolierter Sequencer-Mikrobenchmark, weil der
Auto-Planer nach dem Verschmelzen typischerweise nur sehr wenige Runs
erzeugt (~Anzahl Konturen) und die Held-Karp-Schwelle dort gar nicht
greift -- der Effekt von N wird erst bei vielen Runs sichtbar.

Die Punktabstand-Sweep haelt contour_spacing FEST und variiert nur das
Innenraster, damit wirklich nur P (Flaechenfuellung) variiert und nicht
zusaetzlich die Konturaufloesung (= K).

Aufruf
------
    python segment_simulation/benchmark_autoplan.py
    python segment_simulation/benchmark_autoplan.py --repeats 7
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # kein GUI-Backend noetig

import numpy as np

# --- Paket-Importpfad (Skriptstart ohne Paket-Kontext) --------------------
_ROOT = Path(__file__).resolve().parent.parent        # .../plasma_cutter
import sys
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from plasma_cutter.geometry.point_grid import PointGrid
from plasma_cutter.geometry.geometry_processor import (
    load_raw_contour, densify_contour, generate_inner_points,
)
from plasma_cutter.segment_simulation.segments import (
    SegmentedContour, compute_grid_coverage,
)
from plasma_cutter.segment_simulation.planning import (
    LinkPlanner, Sequencer, RunKinematics, compute_score, LinkInfeasibleError,
)
from plasma_cutter.segment_simulation.autoplan import AutoPlanner
from plasma_cutter.segment_simulation.simulation import make_default_cutter


RAW_DIR = _ROOT / "geometry" / "Geometrie_Konturen_ungeprüft"

# Basiswerte (ein Faktor wird pro Sweep variiert, die anderen bleiben hier)
BASE_CONTOUR_SPACING = 5.0
BASE_GRID_SPACING = 2.5
BASE_SEGMENT_LENGTH: float | None = None   # None = automatisch
BASE_EXACT_MAX_RUNS = 11
KERF = 3.0


# ---------------------------------------------------------------------------
# Raster im Speicher bauen (beliebiger Punktabstand)
# ---------------------------------------------------------------------------

def build_grid(
    raw_path: Path,
    contour_spacing: float = BASE_CONTOUR_SPACING,
    grid_spacing: float = BASE_GRID_SPACING,
) -> PointGrid:
    """Erzeugt ein PointGrid aus einer Roh-Kontur bei waehlbarem Abstand.

    Nutzt dieselben Bausteine wie der GeometryProcessor (Verdichten +
    Innenraster), nur in-memory statt ueber JSON. Reihenfolge der Punkte:
    erst Aussenkontur, dann Lochrand, dann Innenpunkte -- exakt wie von
    PointGrid.from_json erwartet (outer/hole/inner nach Index sortiert).
    """
    outer_raw, hole_raw = load_raw_contour(raw_path)
    dense_outer, _ = densify_contour(outer_raw, contour_spacing)
    dense_hole = (densify_contour(hole_raw, contour_spacing)[0]
                  if hole_raw else None)
    inner = generate_inner_points(dense_outer, dense_hole, grid_spacing)

    coords: list[list[float]] = []
    status: list[int] = []
    coords.extend(dense_outer)
    status.extend([PointGrid._STATUS_OUTER] * len(dense_outer))
    if dense_hole:
        coords.extend(dense_hole)
        status.extend([PointGrid._STATUS_HOLE] * len(dense_hole))
    coords.extend(inner)
    status.extend([PointGrid._STATUS_INNER] * len(inner))

    return PointGrid(
        coords=np.asarray(coords, dtype=np.float64),
        status=np.asarray(status, dtype=np.int8),
        point_spacing=grid_spacing,
        contour_spacing=contour_spacing,
    )


# ---------------------------------------------------------------------------
# Ein instrumentierter Auto-Planer-Lauf (kalt, mit Phasen-Zeiten)
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    # Diagnose (deterministisch)
    n_points: int = 0
    n_outer: int = 0
    n_hole: int = 0
    n_inner: int = 0
    n_candidates: int = 0     # K (Primitiv-Segmente)
    n_selected: int = 0       # nach Pruning
    n_runs: int = 0           # nach Verschmelzen
    n_feasible_runs: int = 0
    n_pierces: int = 0
    n_unreachable: int = 0
    is_optimal: bool = False
    coverage: float = 0.0
    score: float = 0.0
    # Zeiten [s]
    t_build: float = 0.0      # Kandidaten + Abdeckbarkeits-Matrix
    t_greedy: float = 0.0
    t_prune: float = 0.0
    t_merge: float = 0.0
    t_seq: float = 0.0        # Sequencer (Reihenfolge + Links)
    t_coverage: float = 0.0
    t_total: float = 0.0


def plan_instrumented(
    grid: PointGrid,
    segment_length: float | None,
    exact_max_runs: int,
    kerf: float = KERF,
) -> RunMetrics:
    """Fuehrt einen vollstaendigen Auto-Planer-Lauf KALT aus und misst
    jede Phase einzeln. Baut alle Objekte frisch auf (kein Cache-Leak)."""
    m = RunMetrics()
    cutter = make_default_cutter()

    t0 = time.perf_counter()

    contour = SegmentedContour.from_grid(grid, target_segment_length=segment_length)
    material = contour.material_polygon()
    blade_length = cutter.blade_length(cutter.cutting_speed)
    kinematics = RunKinematics(material, clearance=cutter.minimum_gap,
                               blade_length=blade_length, kerf=kerf)
    link_planner = LinkPlanner(material, clearance=cutter.minimum_gap)
    sequencer = Sequencer(cutter, contour, link_planner)
    sequencer.EXACT_MAX_RUNS = exact_max_runs

    # --- Phase 1: Kandidaten + Abdeckbarkeits-Matrix (AutoPlanner.__init__)
    tb = time.perf_counter()
    planner = AutoPlanner(grid, contour, cutter, kinematics, sequencer)
    m.t_build = time.perf_counter() - tb

    # --- Phase 2: Greedy
    tg = time.perf_counter()
    selected = planner._greedy()
    m.t_greedy = time.perf_counter() - tg

    # --- Phase 3: Pruning
    tp = time.perf_counter()
    selected = planner._prune(selected)
    m.t_prune = time.perf_counter() - tp

    # --- Phase 4: Verschmelzen
    tm = time.perf_counter()
    runs = planner._merge_to_runs(selected)
    m.t_merge = time.perf_counter() - tm

    # --- Phase 5: Sequencer
    feasible = [r for r in runs if r.is_feasible]
    ts = time.perf_counter()
    plan = sequencer.build_plan(feasible)
    m.t_seq = time.perf_counter() - ts

    # --- Coverage + Score
    tc = time.perf_counter()
    report = compute_grid_coverage(grid, runs)
    m.t_coverage = time.perf_counter() - tc
    score = compute_score(plan.total_time, report.fraction)

    m.t_total = time.perf_counter() - t0

    # Diagnose
    m.n_points = grid.total_points
    m.n_outer = int(np.sum(grid._status == PointGrid._STATUS_OUTER))
    m.n_hole = int(np.sum(grid._status == PointGrid._STATUS_HOLE))
    m.n_inner = int(np.sum(grid._status == PointGrid._STATUS_INNER))
    m.n_candidates = len(contour.segments)
    m.n_selected = len(selected)
    m.n_runs = len(runs)
    m.n_feasible_runs = len(feasible)
    m.n_pierces = plan.n_pierces
    m.is_optimal = plan.is_optimal
    m.coverage = report.fraction
    m.score = score
    m.n_unreachable = int(np.count_nonzero(~planner._A.any(axis=1)))
    return m


def measure(
    grid: PointGrid,
    segment_length: float | None,
    exact_max_runs: int,
    repeats: int,
) -> RunMetrics:
    """Wiederholt den kalten Lauf und nimmt pro Zeitfeld das Minimum
    (robust gegen Jitter); Diagnosefelder sind deterministisch."""
    samples = [plan_instrumented(grid, segment_length, exact_max_runs)
               for _ in range(repeats)]
    best = samples[0]
    for key in ("t_build", "t_greedy", "t_prune", "t_merge", "t_seq",
                "t_coverage", "t_total"):
        setattr(best, key, min(getattr(s, key) for s in samples))
    return best


# ---------------------------------------------------------------------------
# Tabellen-Ausgabe
# ---------------------------------------------------------------------------

def _fmt_ms(s: float) -> str:
    return f"{s * 1e3:7.1f}"


def print_sweep_table(title: str, var_label: str, rows: list[tuple[str, RunMetrics]]) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    header = (f"{var_label:>12} | {'P':>6} {'K':>4} {'sel':>4} {'runs':>4} "
              f"{'pierce':>6} | {'build':>7} {'greedy':>7} {'prune':>7} "
              f"{'merge':>7} {'seq':>7} {'cov':>7} {'TOTAL':>8} | "
              f"{'Cov%':>6} {'Score':>7} {'opt':>4}")
    print(header)
    print("-" * len(header))
    for label, m in rows:
        print(f"{label:>12} | {m.n_points:>6} {m.n_candidates:>4} "
              f"{m.n_selected:>4} {m.n_runs:>4} {m.n_pierces:>6} | "
              f"{_fmt_ms(m.t_build)} {_fmt_ms(m.t_greedy)} {_fmt_ms(m.t_prune)} "
              f"{_fmt_ms(m.t_merge)} {_fmt_ms(m.t_seq)} {_fmt_ms(m.t_coverage)} "
              f"{_fmt_ms(m.t_total)} | {m.coverage * 100:>5.1f} "
              f"{m.score:>7.0f} {('ja' if m.is_optimal else 'nein'):>4}")
    print("(Zeiten in ms, Minimum ueber alle Wiederholungen)")


# ---------------------------------------------------------------------------
# Sweep A: Punktabstand
# ---------------------------------------------------------------------------

def sweep_point_spacing(raw_path: Path, repeats: int) -> list[tuple[float, RunMetrics]]:
    spacings = [6.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0]
    data: list[tuple[float, RunMetrics]] = []
    for h in spacings:
        grid = build_grid(raw_path, contour_spacing=BASE_CONTOUR_SPACING,
                          grid_spacing=h)
        m = measure(grid, BASE_SEGMENT_LENGTH, BASE_EXACT_MAX_RUNS, repeats)
        data.append((h, m))
    rows = [(f"{h:.1f} mm", m) for h, m in data]
    print_sweep_table(
        f"SWEEP A  |  Punktabstand (Innenraster)  |  {raw_path.stem}  |  "
        f"contour_spacing={BASE_CONTOUR_SPACING} mm fest",
        "grid_spacing", rows)
    # P/Laufzeit-Verhaeltnis: zeigt ~lineare Abhaengigkeit von P (= ~1/h^2)
    print("\n  Skalierungs-Check (erwartet: P ~ 1/h^2, t_total ~ P):")
    base = data[0][1]
    for label, m in rows:
        rp = m.n_points / base.n_points
        rt = m.t_total / base.t_total if base.t_total > 0 else float("nan")
        print(f"    {label:>8}:  P x{rp:4.1f}   t_total x{rt:4.1f}")
    return data


# ---------------------------------------------------------------------------
# Sweep B: Segmentlaenge
# ---------------------------------------------------------------------------

def sweep_segment_length(raw_path: Path, repeats: int) -> list[tuple[str, RunMetrics]]:
    grid = build_grid(raw_path, BASE_CONTOUR_SPACING, BASE_GRID_SPACING)
    # Umfang grob schaetzen, um sinnvolle Laengen abzuleiten
    contour = SegmentedContour.from_grid(grid)
    perim = sum(loop.length for loop in contour.loops)
    lengths: list[float | None] = [None, perim / 4, perim / 8,
                                   perim / 16, perim / 32]
    rows: list[tuple[str, RunMetrics]] = []
    for L in lengths:
        m = measure(grid, L, BASE_EXACT_MAX_RUNS, repeats)
        label = "auto" if L is None else f"{L:.0f} mm"
        rows.append((label, m))
    print_sweep_table(
        f"SWEEP B  |  Ziel-Segmentlaenge  |  {raw_path.stem}  |  "
        f"grid_spacing={BASE_GRID_SPACING} mm, P fest",
        "seg_length", rows)
    print(f"\n  (Umfang ~ {perim:.0f} mm; kuerzere Segmente -> mehr Kandidaten K "
          f"-> Greedy O(K^2 * P) waechst)")
    return rows


# ---------------------------------------------------------------------------
# Sweep C: Held-Karp-Schwelle (isolierter Sequencer-Mikrobenchmark)
# ---------------------------------------------------------------------------

def sweep_held_karp(raw_path: Path, repeats: int) -> list[dict]:
    """Misst NUR den Sequencer mit n synthetischen Runs, exakt vs.
    heuristisch. Verwendet die feasiblen Kandidaten-Runs der Aussenkontur
    als Run-Pool (ohne Verschmelzen), um die Held-Karp-Schwelle gezielt
    zu ueber-/unterschreiten."""
    grid = build_grid(raw_path, BASE_CONTOUR_SPACING, BASE_GRID_SPACING)
    cutter = make_default_cutter()
    # Viele Kandidaten erzwingen (kurze Segmente)
    contour = SegmentedContour.from_grid(grid, target_segment_length=8.0)
    material = contour.material_polygon()
    blade_length = cutter.blade_length(cutter.cutting_speed)
    kin = RunKinematics(material, clearance=cutter.minimum_gap,
                        blade_length=blade_length, kerf=KERF)
    link_planner = LinkPlanner(material, clearance=cutter.minimum_gap)
    sequencer = Sequencer(cutter, contour, link_planner)
    planner = AutoPlanner(grid, contour, cutter, kin, sequencer)

    # Pool feasibler Runs der Aussenkontur (loop_id 0), Kollisionen selten
    full_pool = [r for r in planner._cand_runs
                 if r is not None and r.is_feasible and r.loop_id == 0]
    if len(full_pool) < 4:
        full_pool = [r for r in planner._cand_runs
                     if r is not None and r.is_feasible]

    # Begrenzte Wiederholungen: ein exakter Lauf bei n=14 dauert ~1 s.
    reps = min(repeats, 3)

    print(f"\n{'=' * 100}\nSWEEP C  |  Sequencer: Held-Karp (exakt) vs. NN+2-opt "
          f"(heuristisch)  |  {raw_path.stem}\n{'=' * 100}")
    print(f"Run-Pool: {len(full_pool)} feasible Kandidaten-Runs (Aussenkontur), "
          f"je n verteilt gesampelt (nicht-benachbart -> echte Uebergaenge)")
    header = (f"{'n_runs':>6} | {'exact[ms]':>10} {'heur[ms]':>10} "
              f"{'exact_cost':>11} {'heur_cost':>10} {'gap%':>6} | "
              f"{'Speedup':>8}")
    print(header)
    print("-" * len(header))

    ns = [4, 6, 8, 9, 10, 11, 12, 13, 14]
    out: list[dict] = []
    for n in ns:
        if n > len(full_pool):
            break
        # Verteilt samplen statt benachbarter Boegen -> Uebergaenge != 0
        idx = np.unique(np.linspace(0, len(full_pool) - 1, n).round().astype(int))
        runs = [full_pool[i] for i in idx]
        n = len(runs)

        # Cache vorwaermen (Linkplanung aus der Algorithmen-Zeit nehmen)
        try:
            sequencer.EXACT_MAX_RUNS = n
            sequencer.order_runs(runs)
        except LinkInfeasibleError:
            print(f"{n:>6} | (infeasible: kein kollisionsfreier Pfad)")
            continue

        # Exakt (Held-Karp): EXACT_MAX_RUNS >= n
        t_exact, exact_order = _time_order(sequencer, runs, n, reps)
        # Heuristik: EXACT_MAX_RUNS < n erzwingt NN + 2-opt
        t_heur, heur_order = _time_order(sequencer, runs, n - 1, reps)

        c_exact = sequencer._sequence_cost(exact_order)
        c_heur = sequencer._sequence_cost(heur_order)
        gap = (100.0 * (c_heur - c_exact) / c_exact) if c_exact > 0 else 0.0
        speed = t_heur / t_exact if t_exact > 0 else float("nan")
        print(f"{n:>6} | {t_exact * 1e3:>10.2f} {t_heur * 1e3:>10.2f} "
              f"{c_exact:>11.2f} {c_heur:>10.2f} {gap:>6.1f} | "
              f"{speed:>7.2f}x")
        out.append(dict(n=n, t_exact=t_exact, t_heur=t_heur,
                        c_exact=c_exact, c_heur=c_heur, gap=gap))
    print("(exact_cost/heur_cost = Summe der Uebergangszeiten [s]; "
          "gap% = Mehrkosten der Heuristik; Links vorab gecacht)")
    return out


def _time_order(sequencer: Sequencer, runs, exact_max_runs: int, repeats: int):
    sequencer.EXACT_MAX_RUNS = exact_max_runs
    best = math.inf
    order = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        order, _ = sequencer.order_runs(runs)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, order


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_PHASES = [
    ("t_build", "Kandidaten/Swept-Area", "#1f77b4"),
    ("t_greedy", "Greedy", "#ff7f0e"),
    ("t_prune", "Pruning", "#2ca02c"),
    ("t_merge", "Verschmelzen", "#d62728"),
    ("t_seq", "Sequencer", "#9467bd"),
    ("t_coverage", "Coverage", "#8c564b"),
]
_GEOM_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]


def make_plots(
    a_data: dict[str, list[tuple[float, RunMetrics]]],
    b_data: dict[str, list[tuple[str, RunMetrics]]],
    c_data: dict[str, list[dict]],
    outdir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    geoms = list(a_data.keys())
    saved: list[Path] = []

    # --- Plot 0: Phasen-Aufschluesselung (Basis-Konfig, grid=2.5, seg=auto) --
    fig, ax = plt.subplots(figsize=(9, 0.9 * len(geoms) + 2.2))
    y = np.arange(len(geoms))
    left = np.zeros(len(geoms))
    for key, name, color in _PHASES:
        vals = []
        for g in geoms:
            base = next((m for h, m in a_data[g] if abs(h - BASE_GRID_SPACING) < 1e-9),
                        a_data[g][0][1])
            vals.append(getattr(base, key) * 1e3)
        vals = np.array(vals)
        ax.barh(y, vals, left=left, color=color, label=name, edgecolor="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(geoms)
    ax.invert_yaxis()
    ax.set_xlabel("Laufzeit [ms]")
    ax.set_title("Phasen-Aufschluesselung des Auto-Planers (Basis: grid=2.5 mm, "
                 "seg=auto, N=11)\nDie Swept-Area-Kinematik dominiert -- "
                 "der Greedy-Kern ist verschwindend")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    p = outdir / "bench_phasen.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    saved.append(p)

    # --- Plot A: Punktabstand ------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, g in enumerate(geoms):
        c = _GEOM_COLORS[i % len(_GEOM_COLORS)]
        P = [m.n_points for _, m in a_data[g]]
        tot = [m.t_total * 1e3 for _, m in a_data[g]]
        cov = [m.coverage * 100 for _, m in a_data[g]]
        order = np.argsort(P)
        P = np.array(P)[order]; tot = np.array(tot)[order]; cov = np.array(cov)[order]
        ax1.plot(P, tot, "o-", color=c, label=g)
        ax2.plot(P, cov, "o-", color=c, label=g)
    ax1.set_xlabel("Anzahl Gitterpunkte P  (feineres Raster ->)")
    ax1.set_ylabel("Gesamtlaufzeit [ms]")
    ax1.set_title("Laufzeit vs. Punktzahl\n(bleibt nahezu flach -- P ist NICHT der Treiber)")
    ax1.set_ylim(bottom=0)
    ax1.grid(linestyle="--", alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel("Anzahl Gitterpunkte P  (feineres Raster ->)")
    ax2.set_ylabel("Coverage [%]")
    ax2.set_title("Coverage vs. Punktzahl\n(steigt -- Metrik-Effekt, nicht besserer Plan)")
    ax2.grid(linestyle="--", alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle("SWEEP A | Punktabstand (contour_spacing fest)", fontweight="bold")
    fig.tight_layout()
    p = outdir / "bench_A_punktabstand.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    saved.append(p)

    # --- Plot B: Segmentlaenge ----------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, g in enumerate(geoms):
        c = _GEOM_COLORS[i % len(_GEOM_COLORS)]
        K = [m.n_candidates for _, m in b_data[g]]
        tot = [m.t_total * 1e3 for _, m in b_data[g]]
        seq = [m.t_seq * 1e3 for _, m in b_data[g]]
        score = [m.score for _, m in b_data[g]]
        order = np.argsort(K)
        K = np.array(K)[order]
        tot = np.array(tot)[order]; seq = np.array(seq)[order]
        score = np.array(score)[order]
        ax1.plot(K, tot, "o-", color=c, label=f"{g} (gesamt)")
        ax1.plot(K, seq, "s--", color=c, alpha=0.6, label=f"{g} (Sequencer)")
        ax2.plot(K, score, "o-", color=c, label=g)
    ax1.set_xlabel("Anzahl Kandidaten-Segmente K  (kuerzere Segmente ->)")
    ax1.set_ylabel("Laufzeit [ms]")
    ax1.set_title("Laufzeit vs. Segmentanzahl\n(Sequencer treibt die Zeit, nicht der Greedy)")
    ax1.set_ylim(bottom=0)
    ax1.grid(linestyle="--", alpha=0.3); ax1.legend(fontsize=7)
    ax2.set_xlabel("Anzahl Kandidaten-Segmente K")
    ax2.set_ylabel("Punktezahl (Score)")
    ax2.set_title("Score vs. Segmentanzahl")
    ax2.grid(linestyle="--", alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle("SWEEP B | Ziel-Segmentlaenge (Raster fest)", fontweight="bold")
    fig.tight_layout()
    p = outdir / "bench_B_segmentlaenge.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    saved.append(p)

    # --- Plot C: Held-Karp ---------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, g in enumerate(geoms):
        c = _GEOM_COLORS[i % len(_GEOM_COLORS)]
        rows = c_data.get(g, [])
        if not rows:
            continue
        n = [r["n"] for r in rows]
        te = [r["t_exact"] * 1e3 for r in rows]
        th = [r["t_heur"] * 1e3 for r in rows]
        ax1.plot(n, te, "o-", color=c, label=f"{g} exakt")
        ax1.plot(n, th, "s--", color=c, alpha=0.6, label=f"{g} heuristisch")
        gap = [r["gap"] for r in rows]
        ax2.plot(n, gap, "o-", color=c, label=g)
    ax1.axvline(BASE_EXACT_MAX_RUNS, color="gray", linestyle=":",
                label=f"Schwelle N={BASE_EXACT_MAX_RUNS}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Anzahl Runs n")
    ax1.set_ylabel("Sequencer-Zeit [ms] (log)")
    ax1.set_title("Held-Karp (exakt) vs. NN+2-opt\nexakt explodiert mit 2^n")
    ax1.grid(linestyle="--", alpha=0.3, which="both"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Anzahl Runs n")
    ax2.set_ylabel("Mehrkosten der Heuristik [%]")
    ax2.set_title("Qualitaetsverlust der Heuristik\n(~0 % -> Heuristik praktisch optimal)")
    ax2.set_ylim(-0.5, 5)
    ax2.grid(linestyle="--", alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle("SWEEP C | Sequencer: Held-Karp-Schwelle N", fontweight="bold")
    fig.tight_layout()
    p = outdir / "bench_C_heldkarp.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    saved.append(p)

    return saved


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-Planer Benchmark")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Wiederholungen pro Messpunkt (min wird genommen)")
    parser.add_argument("--geometry", type=str, default=None,
                        help="Roh-Kontur-Name (ohne .json) aus "
                             "Geometrie_Konturen_ungeprüft")
    parser.add_argument("--no-plot", action="store_true",
                        help="Keine PNG-Plots erzeugen")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Zielordner fuer die Plots")
    args = parser.parse_args()

    if args.geometry:
        geoms = [RAW_DIR / f"{args.geometry}.json"]
    else:
        geoms = [RAW_DIR / "kontur.json",
                 RAW_DIR / "test_mit_loch.json",
                 RAW_DIR / "T-Träger Test.json"]

    print(f"Auto-Planer Benchmark  |  repeats={args.repeats}  |  "
          f"Basis: contour={BASE_CONTOUR_SPACING}mm grid={BASE_GRID_SPACING}mm "
          f"seg=auto N={BASE_EXACT_MAX_RUNS}")

    a_data: dict[str, list[tuple[float, RunMetrics]]] = {}
    b_data: dict[str, list[tuple[str, RunMetrics]]] = {}
    c_data: dict[str, list[dict]] = {}

    for raw_path in geoms:
        if not raw_path.exists():
            print(f"\n[uebersprungen] {raw_path} nicht gefunden")
            continue
        g = raw_path.stem
        print(f"\n\n########## GEOMETRIE: {g} ##########")
        a_data[g] = sweep_point_spacing(raw_path, args.repeats)
        b_data[g] = sweep_segment_length(raw_path, args.repeats)
        c_data[g] = sweep_held_karp(raw_path, args.repeats)

    if not args.no_plot and a_data:
        outdir = Path(args.outdir) if args.outdir else (_ROOT / "segment_simulation"
                                                        / "benchmark_results")
        saved = make_plots(a_data, b_data, c_data, outdir)
        print(f"\n\nPlots gespeichert in {outdir}:")
        for p in saved:
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()
from __future__ import annotations

"""Anschauliche Visualisierung, WIE der Auto-Planer arbeitet.

Rendert die Pipeline aus autoplan.AutoPlanner in drei Stufen
nebeneinander und speichert sie als PNG:

  1. Kandidaten   -- alle Primitiv-Segmente + Knoten (moegliche Schnitte)
  2. Auswahl      -- nach Greedy Set-Cover + Pruning gewaehlte Segmente
  3. Fertiger Plan-- verschmolzene Runs mit Swept Areas, zeitminimaler
                     Reihenfolge (O1 -> O2 -> ...), Verfahrwegen und
                     Querschnitts-Coverage (gruen = abgedeckt, rot = fehlt)

Aufruf
------
    python segment_simulation/visualize_autoplan.py
    python segment_simulation/visualize_autoplan.py --geometry test_mit_loch
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import sys
_ROOT = Path(__file__).resolve().parent.parent          # .../plasma_cutter
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from plasma_cutter.geometry.point_grid import PointGrid
from plasma_cutter.segment_simulation.segments import (
    SegmentedContour, compute_grid_coverage,
)
from plasma_cutter.segment_simulation.planning import (
    LinkPlanner, Sequencer, RunKinematics, compute_score,
)
from plasma_cutter.segment_simulation.autoplan import AutoPlanner
from plasma_cutter.segment_simulation.simulation import make_default_cutter


GEPRUEFT = _ROOT / "geometry" / "Geometrie_Konturen_geprüft"

C = dict(
    inner="#C8C8C8", covered="#22A84E", missing="#E02020",
    seg_a="#1E6FBF", seg_b="#7FB2E5", node="#FFFFFF", node_edge="#1A2840",
    sel="#FF8C00", swept="#FF6644", tcp="#B8860B", link="#555555",
    pierce="#CC0000",
)
RUN_COLORS = ["#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E",
              "#17BECF", "#8C564B", "#E377C2"]


def _draw_swept(ax, poly, color, alpha=0.13):
    if poly is None or poly.is_empty:
        return
    geoms = poly.geoms if hasattr(poly, "geoms") else [poly]
    for g in geoms:
        try:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, color=color, alpha=alpha, zorder=2)
        except AttributeError:
            pass


def greedy_trace(planner: AutoPlanner):
    """Rekonstruiert die Greedy-Entscheidungstabelle Runde fuer Runde.

    Spiegelt AutoPlanner._greedy exakt, protokolliert aber pro Runde fuer
    jedes noch waehlbare Segment: neue Punkte, Zusatzzeit, Verhaeltnis.

    Returns
    -------
    rounds : Liste von dict  seg_id -> (new, cost, ratio)
    chosen : Liste der pro Runde gewaehlten seg_ids (= Greedy-Ergebnis)
    """
    A = planner._A
    K = A.shape[1]
    feasible = planner._feasible
    covered = np.zeros(A.shape[0], dtype=bool)
    sel_set: set[int] = set()
    rounds: list[dict[int, tuple[int, float, float]]] = []
    chosen: list[int] = []

    while True:
        row: dict[int, tuple[int, float, float]] = {}
        best_s, best_ratio, best_cost = -1, 0.0, np.inf
        for s in range(K):
            if s in sel_set or not feasible[s]:
                continue
            new = int(np.count_nonzero(A[:, s] & ~covered))
            if new == 0:
                continue
            cost = planner._extra_time(s, sel_set)
            ratio = new / cost
            row[s] = (new, cost, ratio)
            if (ratio > best_ratio + 1e-12
                    or (abs(ratio - best_ratio) <= 1e-12 and cost < best_cost)):
                best_s, best_ratio, best_cost = s, ratio, cost
        if best_s < 0:
            break
        rounds.append(row)
        chosen.append(best_s)
        sel_set.add(best_s)
        covered |= A[:, best_s]
    return rounds, chosen


def _draw_decision_table(ax, rounds, chosen) -> None:
    """Zeichnet die Greedy-Entscheidungstabelle als farbcodiertes Raster.

    Zeilen = Kandidaten-Segmente, Spalten = Greedy-Runden.
    Zellwert/Farbe = Verhaeltnis (neue Punkte / Zusatzzeit). Pro Runde
    wird die Zelle mit dem groessten Verhaeltnis rot umrandet (= gewaehlt).
    Zellen, die 0 neue Punkte bringen oder deren Segment schon gewaehlt
    wurde, bleiben leer.
    """
    n_rounds = len(rounds)
    # Zeilen: alle Segmente, die je bewertet wurden (sonst leere Zeilen)
    seg_ids = sorted({s for r in rounds for s in r})
    row_of = {s: i for i, s in enumerate(seg_ids)}

    M = np.full((len(seg_ids), n_rounds), np.nan)
    for j, r in enumerate(rounds):
        for s, (_new, _cost, ratio) in r.items():
            M[row_of[s], j] = ratio

    masked = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#F2F2F2")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, origin="upper")

    ax.set_xticks(range(n_rounds))
    ax.set_xticklabels([f"Runde {j + 1}" for j in range(n_rounds)], fontsize=8)
    ax.set_yticks(range(len(seg_ids)))
    ax.set_yticklabels([f"S{s}" for s in seg_ids], fontsize=7)
    ax.set_xlabel("Greedy-Runde  (pro Runde wird das beste Verhaeltnis gewaehlt ->)",
                  fontsize=8)
    ax.set_ylabel("Kandidaten-Segment", fontsize=8)

    # Zelltext: Verhaeltnis; bei gewaehlter Zelle zusaetzlich new/cost
    for j, r in enumerate(rounds):
        for s, (new, cost, ratio) in r.items():
            i = row_of[s]
            picked = (s == chosen[j])
            txt = f"{ratio:.1f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, fontweight="bold" if picked else "normal",
                    color="black")
        # gewaehlte Zelle markieren
        ci = row_of[chosen[j]]
        ax.add_patch(mpatches.Rectangle(
            (j - 0.5, ci - 0.5), 1, 1, fill=False,
            edgecolor="#0033CC", linewidth=2.5, zorder=5))

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Verhaeltnis = neue Punkte / Zusatzzeit", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax.set_title(
        "Entscheidungstabelle des Greedy Set-Cover  |  Zahl = neue Punkte / "
        "Zusatzzeit  |  blaue Umrandung = pro Runde gewaehlt  |  grau = 0 neue "
        "Punkte / schon gewaehlt", fontsize=9)


def _set_limits(ax, contour, gap):
    pts = np.vstack([loop.points for loop in contour.loops])
    span = max(float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
    m = span * 0.10 + 9.0 + gap
    ax.set_xlim(pts[:, 0].min() - m, pts[:, 0].max() + m)
    ax.set_ylim(pts[:, 1].min() - m, pts[:, 1].max() + m)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.tick_params(labelsize=7)


def visualize(grid: PointGrid, name: str, outdir: Path) -> Path:
    cutter = make_default_cutter()
    contour = SegmentedContour.from_grid(grid)
    material = contour.material_polygon()
    blade = cutter.blade_length(cutter.cutting_speed)
    kin = RunKinematics(material, clearance=cutter.minimum_gap,
                        blade_length=blade, kerf=3.0)
    link_planner = LinkPlanner(material, clearance=cutter.minimum_gap)
    sequencer = Sequencer(cutter, contour, link_planner)

    # --- Pipeline-Stufen einzeln ausfuehren -------------------------------
    planner = AutoPlanner(grid, contour, cutter, kin, sequencer)
    greedy_sel = planner._greedy()
    pruned_sel = planner._prune(list(greedy_sel))
    runs = planner._merge_to_runs(pruned_sel)
    feasible = [r for r in runs if r.is_feasible]
    plan = sequencer.build_plan(feasible)
    report = compute_grid_coverage(grid, runs)
    score = compute_score(plan.total_time, report.fraction)

    rounds, chosen = greedy_trace(planner)

    coords = grid.coords
    gap = cutter.minimum_gap

    # Layout: oben 3 raeumliche Panels, unten die Entscheidungstabelle
    n_seg_rows = len({s for r in rounds for s in r})
    table_h = max(3.0, 0.32 * n_seg_rows + 1.5)
    fig = plt.figure(figsize=(19, 7 + table_h))
    gs = fig.add_gridspec(2, 3, height_ratios=[7, table_h], hspace=0.32,
                          wspace=0.12, left=0.05, right=0.97,
                          top=0.92, bottom=0.06)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    ax_table = fig.add_subplot(gs[1, :])

    # === Panel 1: Kandidaten ==============================================
    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], s=7, color=C["inner"],
               alpha=0.5, zorder=1)
    label_off = max(6.0, grid.contour_spacing * 1.2)
    for seg in contour.segments:
        loop = contour.loop_by_id(seg.loop_id)
        col = C["seg_a"] if seg.seg_id % 2 == 0 else C["seg_b"]
        pl = loop.polyline(seg.positions)
        ax.plot(pl[:, 0], pl[:, 1], color=col, linewidth=2.2, alpha=0.9,
                zorder=4, solid_capstyle="round")
        # Segment-ID nach aussen versetzt beschriften (Zuordnung zur Tabelle)
        mid_pos = seg.positions[len(seg.positions) // 2]
        nrm = loop.outward_normal(mid_pos, material)
        lp = loop.points[mid_pos] + nrm * label_off
        ax.annotate(f"S{seg.seg_id}", xy=tuple(lp), ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="#10325A", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec=col, lw=0.8, alpha=0.9))
    for loop in contour.loops:
        nodes = contour.nodes.get(loop.loop_id, [])
        if nodes:
            npts = loop.points[nodes]
            ax.scatter(npts[:, 0], npts[:, 1], s=55, color=C["node"],
                       edgecolors=C["node_edge"], linewidths=1.2, zorder=6)
    _set_limits(ax, contour, gap)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.set_title(f"1. Kandidaten (S0..S{len(contour.segments) - 1})\n"
                 f"{len(contour.segments)} Primitiv-Segmente "
                 f"-- IDs = Zeilen der Tabelle unten", fontsize=10)

    # === Panel 2: Greedy + Pruning ========================================
    ax = axes[1]
    ax.scatter(coords[:, 0], coords[:, 1], s=7, color=C["inner"],
               alpha=0.4, zorder=1)
    sel_set = set(pruned_sel)
    greedy_only = set(greedy_sel) - sel_set
    for seg in contour.segments:
        loop = contour.loop_by_id(seg.loop_id)
        pl = loop.polyline(seg.positions)
        if seg.seg_id in sel_set:
            ax.plot(pl[:, 0], pl[:, 1], color=C["sel"], linewidth=4.0,
                    alpha=0.95, zorder=5, solid_capstyle="round")
        elif seg.seg_id in greedy_only:
            # vom Greedy gewaehlt, aber vom Pruning wieder verworfen
            ax.plot(pl[:, 0], pl[:, 1], color=C["sel"], linewidth=3.0,
                    alpha=0.35, linestyle=":", zorder=4)
        else:
            ax.plot(pl[:, 0], pl[:, 1], color="#BBBBBB", linewidth=1.4,
                    alpha=0.7, zorder=3)
    _set_limits(ax, contour, gap)
    pruned_note = (f"  (Pruning entfernt {len(greedy_only)})"
                   if greedy_only else "")
    ax.set_title(f"2. Greedy Set-Cover + Pruning\n{len(sel_set)} Segmente "
                 f"gewaehlt{pruned_note}", fontsize=10)

    # === Panel 3: Fertiger Plan ===========================================
    ax = axes[2]
    cov_idx = np.where(report.mask)[0]
    mis_idx = np.where(~report.mask)[0]
    ax.scatter(coords[cov_idx, 0], coords[cov_idx, 1], s=14,
               color=C["covered"], marker="P", alpha=0.8, zorder=3)
    if len(mis_idx):
        ax.scatter(coords[mis_idx, 0], coords[mis_idx, 1], s=16,
                   color=C["missing"], marker="x", alpha=0.85, zorder=3)

    ordered = plan.runs_in_order
    for i, run in enumerate(ordered):
        col = RUN_COLORS[i % len(RUN_COLORS)]
        _draw_swept(ax, run.swept_polygon, C["swept"])
        if run.tcp_polyline is not None:
            ax.plot(run.tcp_polyline[:, 0], run.tcp_polyline[:, 1], "--",
                    color=C["tcp"], linewidth=1.2, alpha=0.7, zorder=7)
        ax.plot(run.polyline[:, 0], run.polyline[:, 1], color=col,
                linewidth=4.0, alpha=0.8, zorder=8, solid_capstyle="round")
        mid = run.polyline[len(run.polyline) // 2]
        ax.annotate(f"O{i + 1}", xy=tuple(mid), fontsize=11, fontweight="bold",
                    color=col, zorder=12, xytext=(5, 5),
                    textcoords="offset points",
                    bbox=dict(boxstyle="circle,pad=0.18", fc="white",
                              ec=col, lw=1.4, alpha=0.9))
        # Pierce-Start markieren
        ax.scatter([run.tcp_start[0]], [run.tcp_start[1]], s=60,
                   marker="*", color=C["pierce"], edgecolors="white",
                   linewidths=0.6, zorder=11)

    # Verfahrwege (Eilgang) zwischen den Runs
    for step in plan.steps:
        if step.kind == "link" and step.link is not None:
            pl = step.link.points
            ax.plot(pl[:, 0], pl[:, 1], ":", color=C["link"], linewidth=1.8,
                    alpha=0.9, zorder=9)
    _set_limits(ax, contour, gap)
    opt = "exakt" if plan.is_optimal else "heuristisch"
    ax.set_title(f"3. Fertiger Plan ({opt})\n{len(ordered)} Schnitt(e), "
                 f"Reihenfolge O1->...->O{len(ordered)}", fontsize=10)

    # --- Legende fuer Panel 3 ---
    ax.legend(handles=[
        Line2D([0], [0], color=RUN_COLORS[0], lw=4, label="Schnitt (Kontur)"),
        Line2D([0], [0], color=C["tcp"], lw=1.4, ls="--", label="TCP-Pfad"),
        Line2D([0], [0], color=C["link"], lw=1.8, ls=":", label="Verfahrweg"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C["pierce"],
               markersize=10, label="Zuendung"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor=C["covered"],
               markersize=9, label="abgedeckt"),
        Line2D([0], [0], marker="x", color=C["missing"], ls="",
               markersize=8, label="fehlt"),
    ], fontsize=7.5, loc="upper right", framealpha=0.9, ncol=1)

    # === Entscheidungstabelle (untere Reihe) ==============================
    _draw_decision_table(ax_table, rounds, chosen)

    # --- Gesamttitel + Kennzahlen ---
    miss = "" if report.is_complete else f"  |  fehlt {report.missing_fraction:.1%}"
    fig.suptitle(
        f"Auto-Planer: {name}   |   Coverage {report.fraction:.1%}{miss}   |   "
        f"Zuendungen {plan.n_pierces}   |   "
        f"Gesamtzeit {plan.total_time:.1f} s   |   Punktezahl {score:.0f}",
        fontsize=13, fontweight="bold", y=0.975)

    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"autoplan_{name}.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualisiert den Auto-Planer-Ablauf als PNG")
    parser.add_argument("--geometry", type=str, default=None,
                        help="Name (ohne .json) aus Geometrie_Konturen_geprüft")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    if args.geometry:
        files = [GEPRUEFT / f"{args.geometry}.json"]
    else:
        files = [GEPRUEFT / "kontur.json",
                 GEPRUEFT / "test_mit_loch.json",
                 GEPRUEFT / "T-Träger Test.json"]

    outdir = (Path(args.outdir) if args.outdir
              else _ROOT / "segment_simulation" / "benchmark_results")

    for f in files:
        if not f.exists():
            print(f"[uebersprungen] {f} nicht gefunden")
            continue
        grid = PointGrid.from_json(f)
        p = visualize(grid, f.stem, outdir)
        print(f"  gespeichert: {p}")


if __name__ == "__main__":
    main()

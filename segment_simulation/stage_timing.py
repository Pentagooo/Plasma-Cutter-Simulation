from __future__ import annotations

"""Zeitaufteilung der drei Pipeline-Stufen.

  1. Auto-Planer   -- WAS wird geschnitten (Swept-Area-Kandidaten +
                      Greedy Set-Cover + Pruning + Verschmelzen)
  2. Sequencer     -- in WELCHER Reihenfolge (Held-Karp / NN+2-opt)
  3. Motion Planner-- WIE faehrt der Brenner kollisionsfrei dazwischen
                      (LinkPlanner: Sichtbarkeitsgraph + Dijkstra)

Wichtig: Der Motion Planner laeuft nicht als eigener Block, sondern wird
VOM Sequencer aufgerufen (jeder bewertete Uebergang -> ein gecachter
Link-Query). Daher wird hier die Zeit INNERHALB des LinkPlanners
(Konstruktion des Sichtbarkeitsgraphen + alle plan()-Aufrufe) separat
mitgezaehlt und aus der Sequencer-Zeit herausgerechnet.

Aufruf
------
    python segment_simulation/stage_timing.py
    python segment_simulation/stage_timing.py --repeats 7
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from plasma_cutter.geometry.point_grid import PointGrid
from plasma_cutter.segment_simulation.segments import (
    SegmentedContour, compute_grid_coverage,
)
from plasma_cutter.segment_simulation import planning as _planning
from plasma_cutter.segment_simulation.planning import (
    LinkPlanner, Sequencer, RunKinematics, compute_score,
)
from plasma_cutter.segment_simulation.autoplan import AutoPlanner
from plasma_cutter.segment_simulation.simulation import make_default_cutter


GEPRUEFT = _ROOT / "geometry" / "Geometrie_Konturen_geprüft"


class TimedLinkPlanner(LinkPlanner):
    """LinkPlanner, der seine Konstruktions- und Query-Zeit mitzaehlt."""

    def __init__(self, *args, **kwargs) -> None:
        t0 = time.perf_counter()
        super().__init__(*args, **kwargs)
        self.init_time = time.perf_counter() - t0
        self.query_time = 0.0
        self.n_queries = 0

    def plan(self, start, goal):
        t0 = time.perf_counter()
        out = super().plan(start, goal)
        self.query_time += time.perf_counter() - t0
        self.n_queries += 1
        return out


@dataclass
class StageTimes:
    # Auto-Planer (Auswahl)
    t_kin_build: float = 0.0   # Swept-Area-Kinematik je Kandidat + Matrix
    t_greedy: float = 0.0
    t_prune: float = 0.0
    t_merge: float = 0.0       # enthaelt erneute Kinematik der Merge-Runs
    # Sequencer (Reihenfolge) -- ohne Motion-Planner-Anteil
    t_seq_pure: float = 0.0
    # Motion Planner (Verfahrwege)
    t_motion_init: float = 0.0
    t_motion_query: float = 0.0
    # Setup (Geometrie/Kollisionsmodell-Konstruktion)
    t_setup: float = 0.0
    # Coverage/Score-Auswertung
    t_eval: float = 0.0
    # Diagnose
    n_runs: int = 0
    n_pierces: int = 0
    n_link_queries: int = 0

    @property
    def autoplanner(self) -> float:
        return self.t_kin_build + self.t_greedy + self.t_prune + self.t_merge

    @property
    def sequencer(self) -> float:
        return self.t_seq_pure

    @property
    def motion(self) -> float:
        return self.t_motion_init + self.t_motion_query

    @property
    def total(self) -> float:
        return (self.autoplanner + self.sequencer + self.motion
                + self.t_setup + self.t_eval)


def time_stages(grid: PointGrid) -> StageTimes:
    st = StageTimes()
    cutter = make_default_cutter()

    # --- Setup: Geometrie + Kollisionsmodelle -----------------------------
    t = time.perf_counter()
    contour = SegmentedContour.from_grid(grid)
    material = contour.material_polygon()
    blade = cutter.blade_length(cutter.cutting_speed)
    kin = RunKinematics(material, clearance=cutter.minimum_gap,
                        blade_length=blade, kerf=3.0)
    st.t_setup = time.perf_counter() - t

    # --- Motion-Planner-Konstruktion (Sichtbarkeitsgraph) -----------------
    link_planner = TimedLinkPlanner(material, clearance=cutter.minimum_gap)
    st.t_motion_init = link_planner.init_time
    sequencer = Sequencer(cutter, contour, link_planner)

    # --- Auto-Planer: Kandidaten (Swept Areas) + Matrix -------------------
    t = time.perf_counter()
    planner = AutoPlanner(grid, contour, cutter, kin, sequencer)
    st.t_kin_build = time.perf_counter() - t

    # --- Auto-Planer: Greedy / Pruning / Merge ----------------------------
    t = time.perf_counter(); selected = planner._greedy()
    st.t_greedy = time.perf_counter() - t
    t = time.perf_counter(); selected = planner._prune(selected)
    st.t_prune = time.perf_counter() - t
    t = time.perf_counter(); runs = planner._merge_to_runs(selected)
    st.t_merge = time.perf_counter() - t
    feasible = [r for r in runs if r.is_feasible]

    # --- Sequencer (ruft intern den Motion Planner) -----------------------
    link_planner.query_time = 0.0
    link_planner.n_queries = 0
    t = time.perf_counter()
    plan = sequencer.build_plan(feasible)
    t_seq_wall = time.perf_counter() - t
    st.t_motion_query = link_planner.query_time
    st.n_link_queries = link_planner.n_queries
    # Reine Sequencer-Zeit = Wandzeit minus die Zeit im Motion Planner
    st.t_seq_pure = max(0.0, t_seq_wall - link_planner.query_time)

    # --- Auswertung -------------------------------------------------------
    t = time.perf_counter()
    report = compute_grid_coverage(grid, runs)
    _ = compute_score(plan.total_time, report.fraction)
    st.t_eval = time.perf_counter() - t

    st.n_runs = len(feasible)
    st.n_pierces = plan.n_pierces
    return st


def measure(grid: PointGrid, repeats: int) -> StageTimes:
    samples = [time_stages(grid) for _ in range(repeats)]
    best = samples[0]
    for f in ("t_kin_build", "t_greedy", "t_prune", "t_merge", "t_seq_pure",
              "t_motion_init", "t_motion_query", "t_setup", "t_eval"):
        setattr(best, f, min(getattr(s, f) for s in samples))
    return best


def _ms(x: float) -> str:
    return f"{x * 1e3:7.1f}"


def print_table(rows: list[tuple[str, StageTimes]]) -> None:
    print(f"\n{'=' * 104}")
    print("ZEITAUFTEILUNG DER DREI STUFEN  (ms, Minimum ueber Wiederholungen)")
    print('=' * 104)
    hdr = (f"{'Geometrie':>16} | {'runs':>4} {'pierce':>6} {'links':>5} | "
           f"{'AUTO-PLAN':>10} {'SEQUENCER':>10} {'MOTION':>9} {'setup':>7} "
           f"{'eval':>6} | {'TOTAL':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in rows:
        print(f"{name:>16} | {s.n_runs:>4} {s.n_pierces:>6} {s.n_link_queries:>5} | "
              f"{_ms(s.autoplanner)} {_ms(s.sequencer)} {_ms(s.motion)} "
              f"{_ms(s.t_setup)} {_ms(s.t_eval)} | {_ms(s.total)}")
    print("-" * len(hdr))
    # Aufschluesselung Auto-Planer
    print("\nAuto-Planer im Detail (ms):")
    sub = (f"{'Geometrie':>16} | {'Swept-Area+Matrix':>18} {'Greedy':>8} "
           f"{'Pruning':>8} {'Merge':>8}")
    print(sub)
    print("-" * len(sub))
    for name, s in rows:
        print(f"{name:>16} | {_ms(s.t_kin_build):>18} {_ms(s.t_greedy)} "
              f"{_ms(s.t_prune)} {_ms(s.t_merge)}")
    # Aufschluesselung Motion Planner
    print("\nMotion Planner im Detail (ms):")
    sub = (f"{'Geometrie':>16} | {'Graph-Aufbau (init)':>20} "
           f"{'Querys (Dijkstra)':>18} {'#Querys':>8}")
    print(sub)
    print("-" * len(sub))
    for name, s in rows:
        print(f"{name:>16} | {_ms(s.t_motion_init):>20} "
              f"{_ms(s.t_motion_query):>18} {s.n_link_queries:>8}")


def make_plot(rows: list[tuple[str, StageTimes]], outdir: Path) -> Path:
    import matplotlib.pyplot as plt
    names = [n for n, _ in rows]
    y = np.arange(len(names))
    parts = [
        ("Auto-Planer", [s.autoplanner * 1e3 for _, s in rows], "#0072B2"),
        ("Sequencer", [s.sequencer * 1e3 for _, s in rows], "#D55E00"),
        ("Motion Planner", [s.motion * 1e3 for _, s in rows], "#009E73"),
        ("Setup", [s.t_setup * 1e3 for _, s in rows], "#999999"),
        ("Eval", [s.t_eval * 1e3 for _, s in rows], "#CCCCCC"),
    ]
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(names) + 2.2))
    left = np.zeros(len(names))
    for label, vals, color in parts:
        vals = np.array(vals)
        ax.barh(y, vals, left=left, color=color, label=label, edgecolor="white")
        left += vals
    for i, (_, s) in enumerate(rows):
        ax.text(s.total * 1e3 + 1, i, f"{s.total * 1e3:.0f} ms",
                va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Laufzeit [ms]")
    ax.set_title("Zeitaufteilung: Auto-Planer vs. Sequencer vs. Motion Planner\n"
                 "(Motion-Planner-Zeit aus dem Sequencer herausgerechnet)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "bench_stufen.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Stufen-Zeitaufteilung")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    files = [GEPRUEFT / "kontur.json",
             GEPRUEFT / "test_mit_loch.json",
             GEPRUEFT / "T-Träger Test.json"]

    rows: list[tuple[str, StageTimes]] = []
    for f in files:
        if not f.exists():
            print(f"[uebersprungen] {f} nicht gefunden")
            continue
        grid = PointGrid.from_json(f)
        rows.append((f.stem, measure(grid, args.repeats)))

    print_table(rows)
    if not args.no_plot and rows:
        p = make_plot(rows, _ROOT / "segment_simulation" / "benchmark_results")
        print(f"\nPlot gespeichert: {p}")


if __name__ == "__main__":
    main()

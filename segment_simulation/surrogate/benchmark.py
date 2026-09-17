"""Benchmark: Greedy+ / Surrogat (mit und ohne Pruning) / Brute Force.

Ungesehene Instanzen (seed 7: n Katalog + reale Testgeometrien). Je
Instanz: Ausfuehrungszeit T (deterministisch aus dem Zeitmodell),
Coverage, Fallback, Planzeit (Best-of-``reps`` nach Warm-up). Das Optimum
kommt vom Brute-Force-Lehrer -- entweder direkt (sequentiell, bis
``k_max``) oder aus einem vorher parallel gelabelten Ordner
(``--opt-labels DIR``, erzeugt mit ``dataset --seed 7 --out DIR``).

Alle drei Planer bekommen dieselbe Segmentierung (``--seg-divisor``,
``--seg-min-spacings``) und denselben Kerf aus ``params``.

Ausgabe: ``<out>/benchmark.csv`` und ``benchmark.md``.

CLI (aus dem Elternordner von plasma_cutter):
    python -m plasma_cutter.segment_simulation.surrogate.benchmark \\
        --n 30 --seed 7 --reps 3 [--model PFAD] [--out DIR] [--opt-labels DIR]
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

try:
    from .dataset import ARTIFACTS, cache_key, spec_of_grid
    from .instances import generate_instances
    from .model import load_model
    from .params import (
        SEG_DIVISOR_DEFAULT, SEG_MIN_SPACINGS_DEFAULT, label_params,
        make_contour, params_hash, phys_hash,
    )
    from .planner import greedy_plus_plan, surrogate_plan
    from .teacher import TeacherSkipped, exhaustive_plan
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.surrogate.dataset import (
        ARTIFACTS, cache_key, spec_of_grid,
    )
    from plasma_cutter.segment_simulation.surrogate.instances import (
        generate_instances,
    )
    from plasma_cutter.segment_simulation.surrogate.model import load_model
    from plasma_cutter.segment_simulation.surrogate.params import (
        SEG_DIVISOR_DEFAULT, SEG_MIN_SPACINGS_DEFAULT, label_params,
        make_contour, params_hash, phys_hash,
    )
    from plasma_cutter.segment_simulation.surrogate.planner import (
        greedy_plus_plan, surrogate_plan,
    )
    from plasma_cutter.segment_simulation.surrogate.teacher import (
        TeacherSkipped, exhaustive_plan,
    )


def _best_time(fn, reps: int) -> float:
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _optimum_from_labels(grid, seed: int, p, opt_labels_dir: Path):
    """T_opt usw. aus einem Label-Ordner (None, wenn nicht gelabelt)."""
    path = Path(opt_labels_dir) / f"{cache_key(spec_of_grid(grid, seed), p)}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    if str(d["phys_hash"]) != phys_hash(p):
        raise SystemExit(f"FEHLER: {path.name} hat andere Physik "
                         f"({d['phys_hash']} != {phys_hash(p)}).")
    return dict(T_opt=float(d["T"]), t_opt=float(d["plan_time"]),
                n_subsets=int(d["n_subsets"]), n_covers=int(d["n_covers"]))


def run_benchmark(model, n_instances: int = 30, seed: int = 7, reps: int = 3,
                  k_max: int = 18, out_dir: Path | None = None,
                  verbose: bool = True, with_teacher: bool = True,
                  stem: str = "benchmark",
                  seg_divisor: float = SEG_DIVISOR_DEFAULT,
                  seg_min_spacings: float = SEG_MIN_SPACINGS_DEFAULT,
                  opt_labels_dir: Path | None = None) -> list[dict]:
    """``with_teacher=False`` laesst das Optimum aus (Lernkurve: T_opt wird
    dann aus einem gecachten Lauf uebernommen); ``opt_labels_dir`` nimmt
    das Optimum aus einem Label-Ordner statt den Lehrer zu rufen; ``stem``
    ist der Dateiname der CSV/MD-Ausgabe."""
    out_dir = Path(out_dir) if out_dir else ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)
    p = label_params(seg_divisor, seg_min_spacings)
    instances = generate_instances(n_instances, seed)
    # Warm-up (Caches, Modul-Imports)
    c0 = make_contour(instances[0], p)
    surrogate_plan(instances[0], model, kerf=p.kerf, contour=c0)
    greedy_plus_plan(instances[0], kerf=p.kerf, contour=c0)

    rows: list[dict] = []
    for grid in instances:
        name = getattr(grid, "_instance_name", "?")
        contour = make_contour(grid, p)
        n_seg = len(contour.segments)
        gp = greedy_plus_plan(grid, kerf=p.kerf, contour=contour)
        t_gp = _best_time(lambda: greedy_plus_plan(grid, kerf=p.kerf,
                                                   contour=contour), reps)
        sp = surrogate_plan(grid, model, kerf=p.kerf, contour=contour, prune=True)
        t_sp = _best_time(lambda: surrogate_plan(grid, model, kerf=p.kerf,
                                                 contour=contour, prune=True), reps)
        sn = surrogate_plan(grid, model, kerf=p.kerf, contour=contour, prune=False)
        t_sn = _best_time(lambda: surrogate_plan(grid, model, kerf=p.kerf,
                                                 contour=contour, prune=False), reps)
        row = dict(name=name, family=getattr(grid, "_family_name", "?"),
                   n_seg=n_seg,
                   T_gplus=gp["T"], cov_gplus=gp["coverage"], t_gplus=t_gp,
                   T_sur=sp.T, cov_sur=sp.coverage, fb_sur=int(sp.used_fallback),
                   n_sel_sur=sp.n_selected, n_pruned=sp.n_pruned, t_sur=t_sp,
                   T_sur_noprune=sn.T, cov_sur_noprune=sn.coverage,
                   fb_sur_noprune=int(sn.used_fallback), t_sur_noprune=t_sn,
                   T_opt=math.nan, t_opt=math.nan, n_subsets=0, n_covers=0)
        if with_teacher and opt_labels_dir is not None:
            opt = _optimum_from_labels(grid, seed, p, opt_labels_dir)
            if opt:
                row.update(opt)
        elif with_teacher:
            try:
                tr = exhaustive_plan(grid, kerf=p.kerf, k_max=k_max,
                                     contour=contour, keep_covers=False)
                row.update(T_opt=tr.total_time, t_opt=tr.plan_time,
                           n_subsets=tr.n_subsets, n_covers=tr.n_covers)
            except TeacherSkipped:
                pass
        rows.append(row)
        if verbose:
            print(f"{name:16s} k={n_seg:2d}  T G+ {row['T_gplus']:6.2f}  "
                  f"sur {row['T_sur']:6.2f} (no-prune {row['T_sur_noprune']:6.2f})"
                  f"  opt {row['T_opt']:6.2f}   plan ms G+ {t_gp*1e3:5.1f} "
                  f"sur {t_sp*1e3:5.1f}  opt {row['t_opt']:6.2f} s", flush=True)

    keys = list(rows[0].keys())
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if with_teacher:
        _write_markdown(rows, out_dir / f"{stem}.md", n_instances, seed, reps, p)
    return rows


def _write_markdown(rows, path: Path, n_instances, seed, reps, p) -> None:
    has_opt = [r for r in rows if not math.isnan(r["T_opt"])]

    def mean(key, subset):
        v = [r[key] for r in subset]
        return float(np.mean(v)) if v else math.nan

    lines = [
        f"# Benchmark (seed {seed}, n={n_instances} + reale, Best-of-{reps})",
        "",
        f"Instanzen: {len(rows)}, davon mit Optimum: {len(has_opt)}. "
        f"Stempel L{p.label_version} phys {phys_hash(p)} params {params_hash(p)}, "
        f"Segmentierung {p.seg_divisor:g}/{p.seg_min_spacings:g}, "
        f"t_switch {p.t_switch:g} s, v_cut {p.v_cut:g}, v_max {p.v_max:g} mm/s.",
        "",
        "| Verfahren | T Mittel [s] (n mit Optimum) | Luecke zum Optimum | "
        "Planzeit Median [ms] | Planzeit Max [ms] | Coverage >= G+ | Fallback |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    T_opt = mean("T_opt", has_opt)
    for label, tk, ck, fk, pk in (
            ("Greedy+", "T_gplus", "cov_gplus", None, "t_gplus"),
            ("Surrogat (Pruning)", "T_sur", "cov_sur", "fb_sur", "t_sur"),
            ("Surrogat (ohne Pruning)", "T_sur_noprune", "cov_sur_noprune",
             "fb_sur_noprune", "t_sur_noprune"),
            ("Brute Force (Optimum)", "T_opt", None, None, "t_opt")):
        Tm = mean(tk, has_opt)
        gap = Tm / T_opt if T_opt else math.nan
        cov_ok = (sum(1 for r in rows if r[ck] >= r["cov_gplus"] - 1e-12)
                  if ck else len(rows))
        fb = sum(r[fk] for r in rows) if fk else 0
        pv = [r[pk] for r in rows if not math.isnan(r[pk])]
        med_ms = np.median(pv) * 1e3 if pv else math.nan
        max_ms = max(pv) * 1e3 if pv else math.nan
        lines.append(f"| {label} | {Tm:.2f} | {gap:.3f}x | {med_ms:.1f} | "
                     f"{max_ms:.0f} | {cov_ok}/{len(rows)} | {fb}/{len(rows)} |")
    lines += ["", "| Instanz | k | T G+ | T sur | T sur o. Pruning | T opt | "
              "gepruned | Teilmengen | Abdeckungen | t opt [s] |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['n_seg']} | {r['T_gplus']:.2f} | "
                     f"{r['T_sur']:.2f} | {r['T_sur_noprune']:.2f} | "
                     f"{r['T_opt']:.2f} | {r['n_pruned']} | {r['n_subsets']} | "
                     f"{r['n_covers']} | {r['t_opt']:.1f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark Greedy+ / Surrogat / Brute Force.")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=18,
                    help="Optimum nur bis zu dieser Segmentzahl (Brute Force)")
    ap.add_argument("--model", type=str, default=None,
                    help="Modellpfad (Default: artifacts/surrogate_model.joblib)")
    ap.add_argument("--out", type=str, default=None,
                    help="Ausgabeordner (Default: artifacts/)")
    ap.add_argument("--stem", type=str, default="benchmark")
    ap.add_argument("--seg-divisor", type=float, default=SEG_DIVISOR_DEFAULT)
    ap.add_argument("--seg-min-spacings", type=float,
                    default=SEG_MIN_SPACINGS_DEFAULT)
    ap.add_argument("--opt-labels", type=str, default=None,
                    help="Label-Ordner mit dem Optimum (statt Lehrer-Aufruf)")
    ap.add_argument("--no-teacher", action="store_true",
                    help="kein Optimum (nur Greedy+ vs. Surrogat)")
    args = ap.parse_args()
    model = load_model(Path(args.model) if args.model else None)
    run_benchmark(model, n_instances=args.n, seed=args.seed, reps=args.reps,
                  k_max=args.k_max, out_dir=Path(args.out) if args.out else None,
                  with_teacher=not args.no_teacher, stem=args.stem,
                  seg_divisor=args.seg_divisor,
                  seg_min_spacings=args.seg_min_spacings,
                  opt_labels_dir=Path(args.opt_labels) if args.opt_labels else None)


if __name__ == "__main__":
    main()

"""Lernkurve: Was bringt wie viel Trainingsdaten?

Trainiert Modelle auf den ersten N Instanzen des Datensatzes
(Katalogreihenfolge, Familien gleichmaessig gemischt) und misst jedes auf
einem festen Testsatz ungesehener Instanzen (seed 7: n Katalog + reale).
Das Optimum des Testsatzes kommt aus einem Label-Ordner (``--opt-labels``,
parallel erzeugt mit ``dataset --seed 7``) oder wird EINMAL sequentiell mit
dem Brute-Force-Lehrer gerechnet und gecacht
(``optimum_seed<s>_n<n>_P<hash>.csv``); Instanzen ueber ``k_max``
Segmenten bekommen kein Optimum und zaehlen nicht in die Luecke.

Ausgabe: ``<out>/learning_curve.csv`` / ``.md`` (+ je Modell eine
Benchmark-CSV unter ``<out>/learning_curve_runs/``).

CLI (aus dem Elternordner von plasma_cutter):
    python -m plasma_cutter.segment_simulation.surrogate.learning_curve \\
        --dataset .../runs/main --n-list 100,250,500,1000,2000,4000 --n-eval 60 \\
        [--opt-labels .../runs/main_eval/labels] [--seg-divisor 24 --seg-min-spacings 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

# Training nicht auf allen Kernen (die Skripte setzen den Wert explizit)
os.environ.setdefault("OMP_NUM_THREADS", "6")

import joblib
import numpy as np

try:
    from .benchmark import run_benchmark
    from .dataset import ARTIFACTS, cache_key, lower_priority, spec_of_grid
    from .instances import generate_instances
    from .model import SurrogateModel
    from .params import (
        SEG_DIVISOR_DEFAULT, SEG_MIN_SPACINGS_DEFAULT, label_params,
        make_contour, params_hash,
    )
    from .teacher import TeacherSkipped, exhaustive_plan
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.surrogate.benchmark import run_benchmark
    from plasma_cutter.segment_simulation.surrogate.dataset import (
        ARTIFACTS, cache_key, lower_priority, spec_of_grid,
    )
    from plasma_cutter.segment_simulation.surrogate.instances import (
        generate_instances,
    )
    from plasma_cutter.segment_simulation.surrogate.model import SurrogateModel
    from plasma_cutter.segment_simulation.surrogate.params import (
        SEG_DIVISOR_DEFAULT, SEG_MIN_SPACINGS_DEFAULT, label_params,
        make_contour, params_hash,
    )
    from plasma_cutter.segment_simulation.surrogate.teacher import (
        TeacherSkipped, exhaustive_plan,
    )


def _rows_of_first_instances(meta: dict, n: int) -> int:
    """Zeilenzahl der ersten n Instanzen (Zeilen liegen in Meta-Reihenfolge)."""
    return int(sum(i["n_segments"] for i in meta["instances"][:n]))


def optimum_table(n_eval: int, seed: int, out_dir: Path, p, k_max: int = 18,
                  opt_labels_dir: Path | None = None,
                  verbose: bool = True) -> dict:
    """T_opt je Testinstanz: aus ``opt_labels_dir`` oder gecacht als CSV
    (sequentieller Lehrer; Instanzen ueber ``k_max`` bekommen kein Optimum)."""
    if opt_labels_dir is not None:
        t_opt = {}
        for grid in generate_instances(n_eval, seed):
            f = Path(opt_labels_dir) / f"{cache_key(spec_of_grid(grid, seed), p)}.npz"
            if f.exists():
                d = np.load(f, allow_pickle=True)
                t_opt[getattr(grid, "_instance_name", "?")] = float(d["T"])
        return t_opt
    path = out_dir / f"optimum_seed{seed}_n{n_eval}_P{params_hash(p)}.csv"
    if path.exists():
        return {r["name"]: float(r["T_opt"])
                for r in csv.DictReader(open(path, encoding="utf-8"))}
    t_opt = {}
    rows = []
    for grid in generate_instances(n_eval, seed):
        name = getattr(grid, "_instance_name", "?")
        contour = make_contour(grid, p)
        try:
            tr = exhaustive_plan(grid, kerf=p.kerf, k_max=k_max, contour=contour,
                                 keep_covers=False)
            t_opt[name] = tr.total_time
            rows.append(dict(name=name, n_seg=tr.n_segments, T_opt=tr.total_time,
                             t_teacher=tr.plan_time, n_covers=tr.n_covers))
            if verbose:
                print(f"  Optimum {name:16s} k={tr.n_segments:2d} T={tr.total_time:6.2f} "
                      f"({tr.plan_time:.1f} s)", flush=True)
        except TeacherSkipped as exc:
            if verbose:
                print(f"  Optimum {name:16s} uebersprungen: {exc}", flush=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return t_opt


def _summarize(rows: list[dict], t_opt: dict) -> dict:
    rows = [r for r in rows if r["name"] in t_opt]
    T_s = np.array([r["T_sur"] for r in rows])
    T_g = np.array([r["T_gplus"] for r in rows])
    T_o = np.array([t_opt[r["name"]] for r in rows])
    gap = T_s / T_o - 1
    return dict(
        n_eval=len(rows),
        T_sur=float(T_s.mean()), T_gplus=float(T_g.mean()), T_opt=float(T_o.mean()),
        gap_mean=float(T_s.mean() / T_o.mean()), gap_median=float(np.median(gap)),
        gap_gplus=float(T_g.mean() / T_o.mean()),
        at_opt=int((gap < 1e-6).sum()), better_gplus=int((T_s < T_g - 1e-6).sum()),
        worse_gplus=int((T_s > T_g + 1e-6).sum()),
        t_plan_med_ms=float(np.median([r["t_sur"] for r in rows]) * 1e3),
        t_plan_gplus_med_ms=float(np.median([r["t_gplus"] for r in rows]) * 1e3),
        fallback=int(sum(r["fb_sur"] for r in rows)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Lernkurve des Surrogats.")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Ordner mit dataset.npz/dataset_meta.json (Default artifacts/)")
    ap.add_argument("--out", type=str, default=None,
                    help="Ausgabeordner (Default = --dataset)")
    ap.add_argument("--n-list", type=str,
                    default="100,250,500,1000,1500,2000,4000")
    ap.add_argument("--n-eval", type=int, default=30,
                    help="Katalog-Testinstanzen (plus reale)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--k-max", type=int, default=18)
    ap.add_argument("--seg-divisor", type=float, default=None,
                    help="Default: Wert des Datensatzes")
    ap.add_argument("--seg-min-spacings", type=float, default=None)
    ap.add_argument("--opt-labels", type=str, default=None,
                    help="Label-Ordner mit dem Optimum des Testsatzes")
    ap.add_argument("--extra-model", action="append", default=[],
                    help="weitere fertige Modelle (label=pfad.joblib), mehrfach")
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()

    lower_priority()
    ds = Path(args.dataset) if args.dataset else ARTIFACTS
    out = Path(args.out) if args.out else ds
    scratch = out / "learning_curve_runs"
    scratch.mkdir(parents=True, exist_ok=True)

    meta = json.loads((ds / "dataset_meta.json").read_text(encoding="utf-8"))
    seg_div = (args.seg_divisor if args.seg_divisor is not None
               else float(meta.get("seg_divisor", SEG_DIVISOR_DEFAULT)))
    seg_min = (args.seg_min_spacings if args.seg_min_spacings is not None
               else float(meta.get("seg_min_spacings", SEG_MIN_SPACINGS_DEFAULT)))
    p = label_params(seg_div, seg_min)
    opt_dir = Path(args.opt_labels) if args.opt_labels else None
    t_opt = optimum_table(args.n_eval, args.seed, out, p, k_max=args.k_max,
                          opt_labels_dir=opt_dir)
    data = np.load(ds / "dataset.npz", allow_pickle=True)
    X, y, g = data["X"], data["y"], data["groups"]
    n_avail = len(meta["instances"])

    results = []

    def run(label, model, n_inst, n_rows, t_train, t_label=math.nan):
        t0 = time.perf_counter()
        rows = run_benchmark(model, n_instances=args.n_eval, seed=args.seed,
                             reps=args.reps, out_dir=scratch, verbose=False,
                             with_teacher=False, stem=f"bench_{label}",
                             seg_divisor=seg_div, seg_min_spacings=seg_min)
        s = _summarize(rows, t_opt)
        s.update(label=label, n_instances=n_inst, n_rows=n_rows,
                 t_label_cpu_s=t_label, t_train_s=t_train,
                 t_bench_s=time.perf_counter() - t0)
        results.append(s)
        print(f"{label:16s} N={n_inst:5d} rows={n_rows:6d}  T_sur {s['T_sur']:.2f} "
              f"({s['gap_mean']:.3f}x)  at-opt {s['at_opt']}/{s['n_eval']}  "
              f"better/worse G+ {s['better_gplus']}/{s['worse_gplus']}  "
              f"plan {s['t_plan_med_ms']:.0f} ms  label {t_label / 3600:.2f} CPU-h  "
              f"train {t_train:.0f} s", flush=True)

    n_list = sorted({min(int(x), n_avail) for x in args.n_list.split(",") if x.strip()})
    if n_avail not in n_list:
        n_list.append(n_avail)
    for n in n_list:
        k = _rows_of_first_instances(meta, n)
        # kumulierte Lehrerzeit (CPU) fuer die ersten n Labels
        t_label = float(sum(i.get("plan_time", 0.0) for i in meta["instances"][:n]))
        m = SurrogateModel()
        t0 = time.perf_counter()
        m.train(X[:k], y[:k], g[:k], out_dir=scratch, write_importances=False)
        run(f"N{n}", m, n, k, time.perf_counter() - t0, t_label)

    for spec in args.extra_model:
        label, path = spec.split("=", 1)
        d = joblib.load(path)
        run(label, SurrogateModel(estimator=d["estimator"], tau=d["tau"],
                                  feature_names=d["feature_names"]),
            int(d.get("n_instances", 0)), int(d.get("n_rows", 0)), math.nan)

    keys = list(results[0].keys())
    with open(out / "learning_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    r0 = results[0]
    lines = [f"# Lernkurve -- Datensatz {ds.name}, Testsatz seed {args.seed}: "
             f"{r0['n_eval']} Instanzen mit Optimum; Segmentierung "
             f"{seg_div:g}/{seg_min:g}, params {params_hash(p)}",
             "",
             "| Modell | Instanzen | Zeilen | Lehrerzeit [CPU-h] | Training [s] | "
             "T Surrogat [s] | Luecke | am Optimum | besser/schlechter als G+ | "
             "Planzeit Median [ms] |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s in results:
        tl = s["t_label_cpu_s"]
        tl_txt = "" if math.isnan(tl) else f"{tl / 3600:.2f}"
        tt = s["t_train_s"]
        tt_txt = "" if math.isnan(tt) else f"{tt:.0f}"
        lines.append(f"| {s['label']} | {s['n_instances']} | {s['n_rows']} | "
                     f"{tl_txt} | {tt_txt} | "
                     f"{s['T_sur']:.2f} | {s['gap_mean']:.3f}x | {s['at_opt']}/{s['n_eval']} | "
                     f"{s['better_gplus']}/{s['worse_gplus']} | {s['t_plan_med_ms']:.0f} |")
    lines.append("")
    lines.append(f"Greedy+: {r0['T_gplus']:.2f} s ({r0['gap_gplus']:.3f}x), Planzeit Median "
                 f"{r0['t_plan_gplus_med_ms']:.0f} ms; Optimum {r0['T_opt']:.2f} s")
    (out / "learning_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Label-Pipeline: Trainingsdaten mit dem Brute-Force-Lehrer (BA Kap. 5).

Instanzen aus ``instances`` (Katalog, stabil ueber (seed, idx)), Labels vom
Aufzaehlungs-Lehrer ``teacher.exhaustive_plan``: je Segment 1, wenn es in
der zeitoptimalen Auswahl liegt. Jede Instanz wird EINZELN als .npz in
``<out>/labels/`` gecacht; der Dateiname traegt ``LABEL_VERSION`` und den
Parameter-Hash (``params.params_hash``), so dass Labels mit anderer Physik
oder anderer Segmentierung nie verwechselt werden. Ein Neustart labelt nur
die fehlenden. Das Labeln ist CPU-gebunden -> Prozess-Pool ueber Instanzen
mit Gleitfenster: bei Zeitbudget oder Stop-Datei werden keine neuen
Instanzen mehr begonnen, laufende rechnen zu Ende (nichts geht verloren).

Die realen Testgeometrien sind standardmaessig NICHT im Trainingsdatensatz
(sie sind Testinstanzen des Benchmarks); ``--include-real`` nimmt sie auf.

CLI (aus dem Elternordner von plasma_cutter):
    python -m plasma_cutter.segment_simulation.surrogate.dataset \\
        --n 2000 --seed 42 --n-jobs 8 [--k-max 18] [--out .../runs/main]
        [--seg-divisor 18 --seg-min-spacings 3]      # eine feste Segmentierung
        [--seg-mix "12/4:0.5,16/3:0.3,20/3:0.2"]     # Mischung je Instanz (gewichtet)
        [--max-minutes 1080 --stop-file STOP]         # Budget / sanfter Stopp
        [--extra-labels DIR ...]                      # fremde Labels mitnehmen
        [--dry-run]                                   # nur Segmentzahlen
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

try:
    from ...geometry.point_grid import PointGrid
    from .features import FEATURE_NAMES, segment_features
    from .instances import (
        FAMILIES, TEST_GEOMETRY_DIR, default_cutter, make_catalog_instance,
        tag_instance,
    )
    from .params import (
        SEG_DIVISOR_DEFAULT, SEG_MIN_SPACINGS_DEFAULT, LabelParams,
        label_params, make_contour, params_hash, phys_hash, stamp,
    )
    from .teacher import K_MAX_DEFAULT, TeacherSkipped, exhaustive_plan
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.geometry.point_grid import PointGrid
    from plasma_cutter.segment_simulation.surrogate.features import (
        FEATURE_NAMES, segment_features,
    )
    from plasma_cutter.segment_simulation.surrogate.instances import (
        FAMILIES, TEST_GEOMETRY_DIR, default_cutter, make_catalog_instance,
        tag_instance,
    )
    from plasma_cutter.segment_simulation.surrogate.params import (
        SEG_DIVISOR_DEFAULT, SEG_MIN_SPACINGS_DEFAULT, LabelParams,
        label_params, make_contour, params_hash, phys_hash, stamp,
    )
    from plasma_cutter.segment_simulation.surrogate.teacher import (
        K_MAX_DEFAULT, TeacherSkipped, exhaustive_plan,
    )

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


# ---------------------------------------------------------------------------
# Cache-Schluessel
# ---------------------------------------------------------------------------

def cache_key(spec: tuple, p: LabelParams) -> str:
    """Dateiname (ohne .npz) eines Labels im Cache: Instanz + LABEL_VERSION +
    Parameter-Hash. Das Suffix ``_exact`` benennt die Bewertungsart des
    Lehrers (jede Abdeckung exakt gebaut)."""
    tag = f"L{p.label_version}_P{params_hash(p)}_exact"
    if spec[0] == "cat":
        return f"cat_s{spec[2]}_i{spec[1]}_{tag}"
    return f"real_{Path(spec[1]).stem}_{tag}"


def spec_of_grid(grid, seed: int) -> tuple:
    """Spec einer Instanz aus ``generate_instances`` (fuer ``cache_key``)."""
    if getattr(grid, "_family_name", "") == "real":
        return ("real", str(getattr(grid, "_instance_name", "?")))
    return ("cat", int(getattr(grid, "_catalog_idx")), int(seed))


def parse_seg_mix(text: str) -> list[tuple[float, float, float]]:
    """``"12/4:0.5,16/3:0.3,20/3:0.2"`` -> [(divisor, min_spacings, weight), ...].
    Gewicht optional (Default 1)."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        seg, _, w = part.partition(":")
        d, _, m = seg.partition("/")
        out.append((float(d), float(m) if m else SEG_MIN_SPACINGS_DEFAULT,
                    float(w) if w else 1.0))
    if not out:
        raise ValueError("leere --seg-mix")
    return out


def seg_for(spec: tuple, mix: list, seed: int) -> tuple[float, float]:
    """Segmentierung (divisor, min_spacings) einer Instanz: bei einer
    Mischung deterministisch aus (seed, idx) gezogen, gewichtet."""
    if len(mix) == 1:
        return float(mix[0][0]), float(mix[0][1])
    idx = int(spec[1]) if spec[0] == "cat" else (hash(Path(spec[1]).stem) & 0xFFFF)
    rng = np.random.default_rng((int(seed) & 0xFFFFFFFF) * 1_000_003 + idx * 977 + 17)
    w = np.asarray([m[2] for m in mix], dtype=float)
    k = int(rng.choice(len(mix), p=w / w.sum()))
    return float(mix[k][0]), float(mix[k][1])


def _load_grid(spec: tuple):
    if spec[0] == "cat":
        return make_catalog_instance(spec[1], spec[2])
    grid = PointGrid.from_json(spec[1])
    return tag_instance(grid, "real", Path(spec[1]).stem)


# ---------------------------------------------------------------------------
# Betriebssystem-Hilfen
# ---------------------------------------------------------------------------

def lower_priority() -> None:
    """Setzt den aktuellen Prozess auf niedrige CPU-Prioritaet (Windows:
    BELOW_NORMAL, sonst nice +10), damit der Rechner waehrend eines
    Label-Laufs benutzbar bleibt."""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
    except Exception:
        try:
            os.nice(10)
        except Exception:
            pass


def keep_awake(on: bool = True) -> bool:
    """Haelt Windows waehrend des Laufs wach (Linux: ohne Wirkung, False)."""
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetThreadExecutionState.restype = ctypes.c_uint
        k32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        return bool(k32.SetThreadExecutionState(flags))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Eine Instanz labeln
# ---------------------------------------------------------------------------

def _label_one(args: tuple):
    """Labelt EINE Instanz und cacht sie als .npz (picklebar fuer den
    Prozess-Pool). Rueckgabe (key, status) mit status in
    {'cached', 'new', 'skip', 'too_big', 'error'}."""
    spec, k_max, cache_dir, low_prio, seg_divisor, seg_min_spacings = args
    if low_prio:
        lower_priority()
    p = label_params(seg_divisor, seg_min_spacings)
    key = cache_key(spec, p)
    path = Path(cache_dir) / f"{key}.npz"
    if path.exists():
        return key, "cached"
    try:
        grid = _load_grid(spec)
        if grid is None:
            return key, "skip"
        contour = make_contour(grid, p)
        n_seg = len(contour.segments)
        if n_seg == 0:
            return key, "skip"
        cutter = default_cutter()
        try:
            tr = exhaustive_plan(grid, cutter=cutter, kerf=p.kerf, k_max=k_max,
                                 contour=contour, keep_covers=False)
        except TeacherSkipped:
            return key, "too_big"
        if not tr.selected:
            return key, "skip"
        X = segment_features(grid, contour, cutter)
        y = np.zeros(n_seg, dtype=int)
        for s in tr.selected:
            if 0 <= s < n_seg:
                y[s] = 1
        fam = getattr(grid, "_family", FAMILIES["real"])
        np.savez(path, X=X, y=y, family=int(fam),
                 T=float(tr.total_time), coverage=float(tr.coverage),
                 name=getattr(grid, "_instance_name", "?"),
                 plan_time=float(tr.plan_time),
                 n_subsets=int(tr.n_subsets), n_covers=int(tr.n_covers),
                 n_feasible=int(tr.n_feasible),
                 seg_divisor=float(seg_divisor),
                 seg_min_spacings=float(seg_min_spacings),
                 label_version=int(p.label_version),
                 phys_hash=phys_hash(p), params_hash=params_hash(p))
        return key, "new"
    except Exception:
        return key, "error"


# ---------------------------------------------------------------------------
# Datensatz bauen
# ---------------------------------------------------------------------------

def _specs(n_instances: int, seed: int, include_real: bool) -> list[tuple]:
    specs = [("cat", i, seed) for i in range(n_instances)]
    if include_real:
        specs += [("real", str(f))
                  for f in sorted(TEST_GEOMETRY_DIR.glob("*.json"))]
    return specs


def dry_run(n_instances: int, seed: int, k_max: int, mix: list,
            include_real: bool = False) -> dict:
    """Nur Konturen bauen: Histogramm der Segmentzahlen je Familie und
    Anteil ueber ``k_max`` (zum Justieren von Segmentierung und
    Massbereichen). Kein Lehrer, Sekunden statt Stunden. ``mix`` wie in
    ``build_dataset`` (eine oder mehrere Segmentierungen, gewichtet)."""
    # k = Segmente der AUSSENkontur: Innenloop-Segmente (Loch) sind ohne
    # Z-Hub nicht verbindbar und zaehlen beim Lehrer nicht (linkable_segments).
    hist: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_total_seg: dict[str, list] = collections.defaultdict(list)
    k_all: collections.Counter = collections.Counter()
    for spec in _specs(n_instances, seed, include_real):
        grid = _load_grid(spec)
        if grid is None:
            continue
        p = label_params(*seg_for(spec, mix, seed))
        contour = make_contour(grid, p)
        outer_ids = {lp.loop_id for lp in contour.loops if lp.kind == "outer"}
        k = sum(1 for sg in contour.segments if sg.loop_id in outer_ids)
        fam = getattr(grid, "_family_name", "?")
        hist[fam][k] += 1
        k_all[k] += 1
        n_total_seg[fam].append(len(contour.segments))
    total = sum(sum(c.values()) for c in hist.values())
    over = sum(v for c in hist.values() for k, v in c.items() if k > k_max)
    mix_txt = ",".join(f"{d:g}/{m:g}:{w:g}" for d, m, w in mix)
    print(f"Dry run: {total} Instanzen, seg-mix {mix_txt}, k_max={k_max}: "
          f"{over} ({over / max(1, total):.1%}) ueber k_max "
          f"(k = verbindbare Aussensegmente)")
    print("  gesamt   " + " ".join(f"{k}:{k_all[k]}" for k in sorted(k_all)))
    for fam in sorted(hist):
        c = hist[fam]
        ks = sorted(c)
        n = sum(c.values())
        med = int(np.median([k for k, v in c.items() for _ in range(v)]))
        tot = int(np.median(n_total_seg[fam]))
        print(f"  {fam:10s} n={n:5d} median k={med:2d} (alle Segmente {tot:2d})  "
              + " ".join(f"{k}:{c[k]}" for k in ks))
    return {fam: dict(c) for fam, c in hist.items()} | {"_all": dict(k_all)}


def build_dataset(
    n_instances: int,
    seed: int = 42,
    k_max: int = K_MAX_DEFAULT,
    out_dir: Path | None = None,
    n_jobs: int = 1,
    resume: bool = True,
    verbose: bool = True,
    max_minutes: float | None = None,
    low_priority: bool = False,
    stay_awake: bool = False,
    seg_divisor: float = SEG_DIVISOR_DEFAULT,
    seg_min_spacings: float = SEG_MIN_SPACINGS_DEFAULT,
    include_real: bool = False,
    extra_label_dirs: tuple = (),
    stop_file: Path | None = None,
    seg_mix: list | None = None,
) -> dict:
    """Baut ``dataset.npz`` + ``dataset_meta.json`` in ``out_dir``
    (Default ``artifacts/``), resumebar und parallel.

    ``max_minutes``: nach Ablauf werden keine neuen Instanzen begonnen,
    laufende rechnen zu Ende; ``stop_file``: dasselbe, sobald die Datei
    existiert. Danach wird der Datensatz aus allen gecachten Labels
    gebaut (auch aus ``extra_label_dirs``, sofern deren Labels dieselbe
    LABEL_VERSION und Physik tragen). Ein spaeterer Aufruf mit demselben
    Kommando labelt nur die fehlenden nach.

    ``seg_mix``: Liste (divisor, min_spacings, weight); jede Instanz bekommt
    deterministisch (seed, idx) EINE dieser Segmentierungen -> ein Datensatz
    mit Segmentzahlen ueber die ganze Spanne. Ohne ``seg_mix`` gilt
    ``seg_divisor``/``seg_min_spacings`` fuer alle. Der Stempel des Datensatzes
    (und damit des Modells) ist der des ERSTEN Mischungseintrags."""
    mix = list(seg_mix) if seg_mix else [(float(seg_divisor), float(seg_min_spacings), 1.0)]
    p = label_params(mix[0][0], mix[0][1])          # Basis-Stempel
    allowed = {params_hash(label_params(d, m)) for d, m, _ in mix}
    out_dir = Path(out_dir) if out_dir else ARTIFACTS
    label_dir = out_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(stop_file) if stop_file else None

    specs = _specs(n_instances, seed, include_real)
    seg_of = {spec: seg_for(spec, mix, seed) for spec in specs}
    p_of = {spec: label_params(*seg_of[spec]) for spec in specs}
    if not resume:
        for spec in specs:
            f = label_dir / f"{cache_key(spec, p_of[spec])}.npz"
            if f.exists():
                f.unlink()
    todo = [(spec, k_max, str(label_dir), bool(low_priority),
             seg_of[spec][0], seg_of[spec][1])
            for spec in specs
            if not (label_dir / f"{cache_key(spec, p_of[spec])}.npz").exists()]
    if low_priority:
        lower_priority()
    if stay_awake:
        ok = keep_awake(True)
        if verbose:
            print(f"  Standby-Sperre aktiv: {ok}", flush=True)
    t0 = time.perf_counter()
    status_count: dict[str, int] = {}
    if verbose:
        mix_txt = ",".join(f"{d:g}/{m:g}:{w:g}" for d, m, w in mix)
        print(f"Labeling (exakter Lehrer, k_max={k_max}, seg-mix {mix_txt}, "
              f"L{p.label_version} phys {phys_hash(p)}): "
              f"{len(specs)} Instanzen, {len(todo)} offen, n_jobs={n_jobs}",
              flush=True)

    budget = (max_minutes * 60.0) if max_minutes else None
    chatty = len(todo) < 1000

    def _stop_requested() -> bool:
        if budget and time.perf_counter() - t0 > budget:
            return True
        return bool(stop_file and stop_file.exists())

    def _record(key, status, done, total):
        status_count[status] = status_count.get(status, 0) + 1
        if verbose and (chatty or done % 100 == 0 or done == total):
            el = time.perf_counter() - t0
            rate = done / el if el > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("inf")
            print(f"  [{done}/{total}] {el:.0f}s, ETA {eta:.0f}s  {status_count}"
                  + (f"  {key} {status}" if chatty else ""), flush=True)

    done = 0
    stopped_early = False
    if todo and n_jobs > 1:
        # Gleitfenster: n_jobs Futures in Flug; bei Stopp nichts mehr
        # einreichen, aber laufende zu Ende rechnen lassen.
        pending = list(todo)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            running = set()
            while pending and len(running) < n_jobs:
                running.add(ex.submit(_label_one, pending.pop(0)))
            while running:
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for fut in finished:
                    key, status = fut.result()
                    done += 1
                    _record(key, status, done, len(todo))
                if pending and not stopped_early and _stop_requested():
                    stopped_early = True
                    if verbose:
                        print(f"  Stopp: keine neuen Instanzen mehr, "
                              f"{len(running)} laufen zu Ende, {len(pending)} "
                              f"bleiben offen (resume).", flush=True)
                while pending and not stopped_early and len(running) < n_jobs:
                    running.add(ex.submit(_label_one, pending.pop(0)))
    else:
        for i, args in enumerate(todo):
            if _stop_requested():
                stopped_early = True
                if verbose:
                    print(f"  Stopp: {len(todo) - i} Instanzen bleiben offen "
                          f"(resume).", flush=True)
                break
            key, status = _label_one(args)
            done += 1
            _record(key, status, done, len(todo))

    # Assembly aus dem Cache (hart scheitern statt stale Daten schreiben)
    X_rows, y_rows, groups, inst_meta = [], [], [], []
    t_teacher_total = 0.0
    n_missing = 0
    seen: set[str] = set()

    def _take(path: Path, strict: bool) -> bool:
        try:
            d = np.load(path, allow_pickle=True)
        except Exception:
            if strict:
                raise
            return False            # z.B. gerade im Schreiben (Zwischenstand)
        lv = int(d["label_version"]) if "label_version" in d.files else -1
        ph = str(d["phys_hash"]) if "phys_hash" in d.files else None
        pa = str(d["params_hash"]) if "params_hash" in d.files else None
        if lv != p.label_version or ph != phys_hash(p):
            if strict:
                raise SystemExit(
                    f"FEHLER: {path.name} traegt label_version={lv}, "
                    f"phys_hash={ph}; dieser Lauf hat L{p.label_version} "
                    f"phys {phys_hash(p)}. Fremde Labels im Cache?")
            return False
        if strict and pa not in allowed:
            raise SystemExit(f"FEHLER: {path.name}: params_hash {pa} gehoert "
                             f"nicht zu diesem Lauf (andere Segmentierung/"
                             f"Katalog?).")
        name = str(d["name"])
        if name in seen or d["X"].shape[0] == 0:
            return False
        seen.add(name)
        X_rows.append(d["X"])
        y_rows.append(np.asarray(d["y"], dtype=int))
        fam = int(d["family"])
        groups.extend([fam] * len(d["y"]))
        nonlocal t_teacher_total
        t_teacher_total += float(d["plan_time"])
        inst_meta.append({
            "name": name, "family": fam, "source": str(path.parent),
            "n_segments": int(len(d["y"])), "n_selected": int(np.sum(d["y"])),
            "T": float(d["T"]), "coverage": float(d["coverage"]),
            "n_subsets": int(d["n_subsets"]), "n_covers": int(d["n_covers"]),
            "plan_time": float(d["plan_time"]), "params_hash": pa,
            "seg_divisor": (float(d["seg_divisor"]) if "seg_divisor" in d.files else None),
            "seg_min_spacings": (float(d["seg_min_spacings"])
                                 if "seg_min_spacings" in d.files else None),
        })
        return True

    for spec in specs:
        path = label_dir / f"{cache_key(spec, p_of[spec])}.npz"
        if not path.exists():
            n_missing += 1
            continue
        _take(path, strict=True)
    n_extra = 0
    for d in extra_label_dirs:
        for path in sorted(Path(d).glob("*.npz")):
            if _take(path, strict=False):
                n_extra += 1
    if not X_rows:
        raise SystemExit("FEHLER: keine gelabelten Instanzen -- dataset.npz "
                         "wurde NICHT geschrieben.")
    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    g = np.asarray(groups, dtype=int)
    np.savez_compressed(out_dir / "dataset.npz", X=X, y=y, groups=g,
                        feature_names=np.array(FEATURE_NAMES))
    meta = {
        "label_version": p.label_version, "stamp": stamp(p),
        "pricing": "exact", "k_max": int(k_max),
        "teacher": "exhaustive enumeration (all subsets, exact pricing)",
        "seg_divisor": float(mix[0][0]), "seg_min_spacings": float(mix[0][1]),
        "seg_mix": [[float(d), float(m), float(w)] for d, m, w in mix],
        "include_real": bool(include_real),
        "n_instances_requested": int(n_instances),
        "n_instances_used": len(inst_meta),
        "n_instances_missing": int(n_missing),
        "n_instances_extra": int(n_extra),
        "extra_label_dirs": [str(d) for d in extra_label_dirs],
        "seed": int(seed), "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]), "feature_names": FEATURE_NAMES,
        "positive_fraction": float(y.mean()),
        "teacher_time_total_s": t_teacher_total,
        "label_run_status": status_count, "families": FAMILIES,
        "max_minutes": max_minutes, "stopped_early": bool(stopped_early),
        "instances": inst_meta,
    }
    with open(out_dir / "dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"Dataset: X={X.shape}, y+={int(y.sum())}/{len(y)} "
              f"({meta['positive_fraction']:.1%}), Instanzen {len(inst_meta)} "
              f"(fehlend {n_missing}, extra {n_extra}), Lehrer gesamt "
              f"{t_teacher_total:.0f} s -> {out_dir / 'dataset.npz'}", flush=True)
    if stay_awake:
        keep_awake(False)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Datensatz mit dem Brute-Force-Lehrer labeln.")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-max", type=int, default=K_MAX_DEFAULT)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--out", type=str, default=None,
                    help="Ausgabeordner (Default: artifacts/)")
    ap.add_argument("--seg-divisor", type=float, default=SEG_DIVISOR_DEFAULT,
                    help="Ziel-Segmentlaenge = Umfang / Divisor (Default 12)")
    ap.add_argument("--seg-min-spacings", type=float,
                    default=SEG_MIN_SPACINGS_DEFAULT,
                    help="Untergrenze der Segmentlaenge in Punktabstaenden (Default 4)")
    ap.add_argument("--seg-mix", type=str, default=None,
                    help='Mischung je Instanz, z.B. "12/4:0.5,16/3:0.3,20/3:0.2" '
                         '(divisor/min_spacings:gewicht); ueberstimmt --seg-divisor')
    ap.add_argument("--include-real", action="store_true",
                    help="reale Testgeometrien mit labeln (sonst nur Katalog)")
    ap.add_argument("--extra-labels", action="append", default=[],
                    help="weiterer Label-Ordner fuer die Assembly (mehrfach)")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="Zeitbudget; danach keine neuen Instanzen, Assembly")
    ap.add_argument("--stop-file", type=str, default=None,
                    help="existiert diese Datei, werden keine neuen Instanzen begonnen")
    ap.add_argument("--dry-run", action="store_true",
                    help="nur Segmentzahlen je Familie ausgeben (kein Lehrer)")
    ap.add_argument("--low-priority", action="store_true",
                    help="Worker mit niedriger CPU-Prioritaet")
    ap.add_argument("--keep-awake", action="store_true",
                    help="Windows-Standby waehrend des Laufs unterdruecken")
    args = ap.parse_args()
    mix = (parse_seg_mix(args.seg_mix) if args.seg_mix
           else [(args.seg_divisor, args.seg_min_spacings, 1.0)])
    if args.dry_run:
        dry_run(args.n, args.seed, args.k_max, mix, include_real=args.include_real)
        return
    build_dataset(n_instances=args.n, seed=args.seed, k_max=args.k_max,
                  n_jobs=args.n_jobs, resume=not args.no_resume,
                  out_dir=Path(args.out) if args.out else None,
                  max_minutes=args.max_minutes,
                  low_priority=args.low_priority,
                  stay_awake=args.keep_awake,
                  seg_divisor=args.seg_divisor,
                  seg_min_spacings=args.seg_min_spacings,
                  include_real=args.include_real,
                  extra_label_dirs=tuple(args.extra_labels),
                  stop_file=args.stop_file, seg_mix=mix)


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.freeze_support()
    main()

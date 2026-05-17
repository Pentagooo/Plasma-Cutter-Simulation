"""Phase 1 -- Heuristik-Baseline: Aussenkontur folgen.

Diese Datei enthaelt die deterministische Vergleichs-Strategie, gegen die
das spaetere RL-Training antreten muss. Die Strategie ist bewusst SO
EINFACH WIE MOEGLICH gewaehlt:

    Fahre einmal die Aussenkontur des Werkstuecks ab.
    Der TCP haelt dabei `minimum_gap` Sicherheitsabstand nach aussen,
    die Lichtschwert-Klinge zeigt senkrecht nach innen ins Material und
    "wischt" einen Streifen von `blade_length` Tiefe x `kerf_width` Breite
    vom Rand frei.

Warum genau diese Baseline?
---------------------------
1. **Trivial implementierbar:** keine Optimierung, kein Tuning -- die
   Wahl der Waypoints folgt direkt aus der Geometrie. Damit ist
   ausgeschlossen, dass die Baseline durch heimliches Tuning besser
   wird als sie sollte.
2. **Garantiert ausfuehrbar:** der TCP ist per Konstruktion ausserhalb
   des Materials, der Planner muss nichts korrigieren.
3. **Realistische Untergrenze:** sie schneidet nur den Randbereich frei.
   Bei massiven Werkstuecken erreicht sie KEINE 99% Coverage -- und
   genau das ist der Punkt: das ist die Latte, die RL ueberspringen muss.
4. **Liefert nebenbei Demonstrationen:** die erzeugten Waypoint-Listen
   koennen spaeter als Behavior-Cloning-Pretraining-Daten verwendet
   werden (Phase 7).

Was diese Datei NICHT tut
-------------------------
- Sie fuehrt KEINE Spiral- oder Offset-Bahnen nach innen aus. Der User
  hat explizit "Aussenkontur folgen" als Baseline gewaehlt. Wenn sich
  spaeter herausstellt, dass die Baseline zu schwach ist, wird das HIER
  erweitert -- nicht versteckt im Trainingscode.
- Sie kuemmert sich nicht um Loecher (innere Konturen). Die aktuelle
  Implementierung von `outer_points_ordered` liefert die Aussenpunkte;
  Innenkonturen werden bewusst ignoriert. Wenn das spaeter ein Problem
  wird, bitte hier ergaenzen.
"""

from __future__ import annotations

import json
import time
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

# Direkter Import statt try/except: dieses Modul wird nur aus dem
# `plasma_cutter`-Package heraus aufgerufen, nie als Standalone.
from ..geometry.point_grid import PointGrid
from ..cutter.cutter import Cutter
from ..cutter.continuous_path import (
    ContinuousPlanner, ContinuousPathResult, calculate_reward,
)

from .config import RLConfig, default_config


# ---------------------------------------------------------------------------
# Datenklasse fuer Baseline-Resultate pro Geometrie
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """Ein einzelnes Baseline-Ergebnis fuer eine Geometrie.

    Diese Struktur wird pro Geometrie erzeugt und am Ende als JSON
    serialisiert. Spaetere Phasen (insbesondere die Erfolg-Auswertung
    in Phase 8) lesen die `total_time`-Werte hier raus, um das
    `time_ratio_vs_baseline`-Kriterium pro Geometrie zu berechnen.
    """

    geometry_file:     str    # Dateiname (ohne Pfad), Schluessel fuer Lookups
    total_points:      int    # Anzahl Gitterpunkte (= Coverage-Nenner)
    total_area:        float  # mm^2 -- Querschnittsflaeche der Geometrie
    coverage:          float  # [0..1] -- erreichte Coverage
    total_time:        float  # s -- Schnitt + Verfahr
    total_cutting_time: float # s -- nur Schnitt (Plasma an)
    total_travel_time: float  # s -- nur Verfahr (Plasma aus)
    n_pierces:         int    # Anzahl Zuendungen
    path_length:       float  # mm -- Gesamtlaenge des TCP-Pfads
    is_feasible:       bool   # True wenn alle Cuts ohne Constraint-Verletzung
    reward:            float  # Reward-Funktion-Wert (zur Plausibilitaetspruefung)
    success:           bool   # coverage >= cfg.success.coverage_target ?


# ---------------------------------------------------------------------------
# Strategie: Aussenkontur folgen
# ---------------------------------------------------------------------------


def build_outer_contour_waypoints(
    grid: PointGrid,
    minimum_gap: float,
    safety_margin: float = 1.05,
) -> list[np.ndarray]:
    """Baut den TCP-Pfad fuer "einmal Aussenkontur abfahren".

    Idee
    ----
    Wir bauen aus den Aussenpunkten des Grids ein Polygon und benutzen
    Shapelys `polygon.buffer(+d)`, um eine NACH AUSSEN versetzte
    Hueltkurve zu erzeugen. Deren Aussenrand (`exterior`) ist
    per Konstruktion ueberall genau `d = minimum_gap * safety_margin`
    vom Material entfernt -- das sind unsere TCP-Waypoints.

    Warum nicht punktweise Normalen-Versatz?
    ----------------------------------------
    Die naive Variante "fuer jeden Konturpunkt: addiere eine
    Normale * Offset" funktioniert NUR fuer konvexe Konturen. Sobald
    die Kontur konkav ist (z.B. T-Traeger, Lochkonturen), zeigen die
    lokalen Normalen in der Konkavitaet nach INNEN ins Material und
    der TCP-Pfad geht durchs Material. Genau dieser Fehler hat den
    ersten Smoke-Test der Baseline auf allen Geometrien als
    `is_feasible=False` markiert.

    Shapelys Buffer-Operation loest das Problem korrekt:
      - bei konvexen Bereichen: glatte Aussen-Versetzung
      - bei Konkavitaeten:      automatische Eckenrundung/-trimmung
      - garantiert geschlossene, einfache Polylinie

    Warum `safety_margin` > 1?
    --------------------------
    Numerische Toleranz: der `_validate_grip_path` im Planner verlangt
    `dist >= minimum_gap`. Wenn unser Buffer EXAKT `minimum_gap` waere,
    koennten Floating-Point-Rundungsfehler einzelne Waypoints knapp
    darunter rutschen lassen. Ein paar Prozent Puffer machen die
    Strategie robust, ohne das Ergebnis spuerbar zu verschlechtern.

    Was ist mit Loechern (innere Konturen)?
    ---------------------------------------
    Werden in der Baseline IGNORIERT. Wir nehmen ausschliesslich den
    `exterior` des Buffer-Polygons. Innenkonturen muesste man als
    eigene Schnitte modellieren -- explizites Phase-1-Nicht-Ziel.
    """
    outer = grid.outer_points_ordered
    if len(outer) < 3:
        # Fallback: zu wenig Punkte fuer eine Kontur -- leerer Pfad,
        # der Caller muss das als degeneriert behandeln.
        return []

    # Schritt 1: Polygon aus den Aussenpunkten bauen.
    # Die Punkte sind per `outer_points_ordered` in Konturreihenfolge.
    coords = [(float(p.x), float(p.y)) for p in outer]
    coords.append(coords[0])  # explizit schliessen, sonst Polygon-Warnung
    poly = Polygon(coords)
    if not poly.is_valid:
        # `buffer(0)` repariert selbstschneidende Polygone.
        poly = poly.buffer(0)
    if poly.is_empty:
        return []

    # Schritt 2: Polygon nach aussen versetzen.
    # `join_style=2` (mitre) statt round, weil wir scharfe Ecken erhalten
    # wollen -- runde Ecken wuerden den TCP an konvexen Spitzen unnoetig
    # Bogen fahren lassen, was Zeit kostet.
    # `mitre_limit=2.0` deckelt extreme Spitzen, damit sie nicht ins
    # Unendliche herausragen.
    offset_distance = minimum_gap * safety_margin
    offset_poly = poly.buffer(
        offset_distance,
        join_style=2,    # 2 = mitre (scharfe Ecken)
        mitre_limit=2.0,
    )

    # Bei sehr verzweigten Geometrien kann der Buffer in mehrere
    # Teilpolygone zerfallen. Wir nehmen den FLAECHENGROESSTEN Teil --
    # er enthaelt das eigentliche Werkstueck.
    if isinstance(offset_poly, MultiPolygon):
        offset_poly = max(offset_poly.geoms, key=lambda g: g.area)
    if offset_poly.is_empty:
        return []

    # Schritt 3: Aussenrand als geordnete Waypoint-Liste extrahieren.
    # `exterior.coords` liefert die Punkte in CCW-Reihenfolge inkl.
    # Schliessungspunkt am Ende -- exakt das, was `plan_custom`
    # erwartet.
    waypoints = [np.array(c, dtype=float) for c in offset_poly.exterior.coords]
    return waypoints


def run_outer_contour_baseline(
    grid: PointGrid,
    planner: ContinuousPlanner,
    cfg: RLConfig,
) -> ContinuousPathResult:
    """Fuehrt die Baseline-Strategie auf einem (bereits ge-resetten) Grid aus.

    Wichtig: das `grid` wird im Zuge der Ausfuehrung modifiziert
    (Cut-Punkte werden gesetzt). Der Caller muss `grid.reset()` selber
    aufrufen, falls er das Grid danach erneut verwenden will.
    """
    waypoints = build_outer_contour_waypoints(
        grid=grid,
        minimum_gap=cfg.cutter.minimum_gap,
    )

    if len(waypoints) < 2:
        # Degenerierte Geometrie -> leeres Ergebnis. Wir geben das vom
        # Planner gelieferte "empty result" zurueck, damit Downstream-Code
        # sich auf eine konsistente Struktur verlassen kann.
        return planner.plan_custom([], apply=True)

    # Ein einziges Schnitt-Segment ueber alle Waypoints. is_cutting=True,
    # damit das Plasma an ist und die Klinge tatsaechlich Material abtraegt.
    return planner.plan_custom([(waypoints, True)], apply=True)


# ---------------------------------------------------------------------------
# Runner: Baseline auf allen Geometrien des Scopes
# ---------------------------------------------------------------------------


def _build_planner_from_config(cfg: RLConfig) -> tuple[Cutter, float]:
    """Erzeugt einen `Cutter` mit den Werten aus `CutterConfig`.

    Wir geben den `Cutter` UND die `kerf_width` getrennt zurueck, weil der
    `ContinuousPlanner` beide separat braucht. Das spiegelt die aktuelle
    API von `cutter/continuous_path.py` -- aenderbar, falls sich die API
    spaeter konsolidiert.
    """
    cutter = Cutter(
        max_depth     = cfg.cutter.max_depth,
        cutting_speed = cfg.cutter.cutting_speed,
        moving_speed  = cfg.cutter.moving_speed,
        rapid_speed   = cfg.cutter.rapid_speed,
        minimum_gap   = cfg.cutter.minimum_gap,
    )
    return cutter, cfg.cutter.kerf_width


def list_geometries(cfg: RLConfig) -> list[Path]:
    """Listet alle .json-Geometrien im konfigurierten Verzeichnis auf.

    Sortierung alphabetisch -- damit ist der Train/Test-Split bei gleichem
    `cfg.seed` deterministisch reproduzierbar.
    """
    geom_dir = cfg.geometry_dir_path()
    if not geom_dir.exists():
        raise FileNotFoundError(
            f"Geometrie-Verzeichnis nicht gefunden: {geom_dir}\n"
            f"Pruefe `cfg.scope.geometry_dir`."
        )
    files = sorted(geom_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"Keine .json-Dateien in {geom_dir}."
        )
    return files


def split_train_test(
    files: list[Path],
    cfg: RLConfig,
) -> tuple[list[Path], list[Path]]:
    """Teilt die Geometrieliste deterministisch in Train/Test.

    Wir mischen mit einem von `cfg.seed` abgeleiteten Generator, damit
    die Aufteilung reproduzierbar ist UND nicht von der globalen
    `numpy.random`-Sequenz abhaengt.

    Spaeter relevant fuer Phase 6 (Generalisierung) und Phase 8 (Eval):
    der Test-Split darf NIE ins Training fliessen.
    """
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)

    n_test = max(1, int(round(len(files) * cfg.scope.test_split)))
    test_idx = set(indices[:n_test].tolist())

    train_files = [f for i, f in enumerate(files) if i not in test_idx]
    test_files  = [f for i, f in enumerate(files) if i in test_idx]
    return train_files, test_files


def _evaluate_geometry(
    geom_file: Path,
    cfg: RLConfig,
) -> BaselineResult:
    """Laedt eine Geometrie und fuehrt die Baseline genau einmal aus."""
    grid = PointGrid.from_json(geom_file)
    grid.reset()  # sicherheitshalber, falls das Grid Cut-State mitbringt

    cutter, kerf = _build_planner_from_config(cfg)
    planner = ContinuousPlanner(grid=grid, cutter=cutter, kerf_width=kerf)

    result = run_outer_contour_baseline(grid=grid, planner=planner, cfg=cfg)

    # is_feasible aggregieren: ein Cut zaehlt als infeasible, wenn
    # mindestens eines seiner Segmente vom Planner als infeasible markiert
    # wurde.
    all_feasible = all(c.is_feasible for c in result.continuous_cuts) \
                   if result.continuous_cuts else False

    # Reward zur Plausibilitaetspruefung -- der Wert sollte nicht negativ
    # sein, wenn die Baseline auch nur halbwegs etwas schneidet. Wenn doch:
    # Hinweis auf Bug in Reward-Gewichten oder in der Pfadkonstruktion.
    reward = calculate_reward(result)

    return BaselineResult(
        geometry_file      = geom_file.name,
        total_points       = result.total_points,
        total_area         = result.total_area,
        coverage           = result.coverage,
        total_time         = result.total_time,
        total_cutting_time = result.total_cutting_time,
        total_travel_time  = result.total_travel_time,
        n_pierces          = result.n_pierces,
        path_length        = result.total_path_length,
        is_feasible        = all_feasible,
        reward             = reward,
        success            = (result.coverage >= cfg.success.coverage_target),
    )


def run_baseline(
    cfg: RLConfig | None = None,
    output_path: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    """Fuehrt die Baseline auf ALLEN Geometrien des Scopes aus.

    Parameters
    ----------
    cfg         : RL-Config. Default = `default_config()`.
    output_path : Wohin die JSON-Ergebnisse geschrieben werden. Default
                  = `<rl-package>/results/baseline_v<version>.json`.
                  Die Versionsnummer im Dateinamen verhindert versehent-
                  liches Ueberschreiben aelterer Resultate.
    verbose     : Bei True wird pro Geometrie eine Zeile ausgegeben.

    Returns
    -------
    dict mit Schluesseln:
        - "config"   : komplette Config (zur Reproduzierbarkeit)
        - "train"    : Liste von BaselineResult-dicts (Trainingsgeometrien)
        - "test"     : Liste von BaselineResult-dicts (Test-Set)
        - "summary"  : aggregierte Statistiken
    """
    cfg = cfg or default_config()
    files = list_geometries(cfg)
    train_files, test_files = split_train_test(files, cfg)

    if verbose:
        print(f"[Phase 1] Baseline 'Aussenkontur folgen' -- v{cfg.version}")
        print(f"[Phase 1] Geometrien gesamt: {len(files)} "
              f"(train={len(train_files)}, test={len(test_files)})")
        print()

    def _eval_split(split_files: list[Path], split_name: str) -> list[BaselineResult]:
        results: list[BaselineResult] = []
        for f in split_files:
            t0 = time.perf_counter()
            try:
                r = _evaluate_geometry(f, cfg)
            except Exception as e:
                # Eine kaputte Geometrie soll den gesamten Run nicht killen.
                # Wir loggen den Fehler und machen weiter -- spaeter kann
                # man die fehlerhaften Files manuell pruefen.
                if verbose:
                    print(f"  [{split_name}] {f.name:40s}  FEHLER: {e}")
                continue
            elapsed = time.perf_counter() - t0
            results.append(r)
            if verbose:
                flag = "OK" if r.success else "  "
                feas = "" if r.is_feasible else " [infeasible]"
                print(
                    f"  [{split_name}] {f.name:40s}  "
                    f"cov={r.coverage*100:5.1f}%  "
                    f"t={r.total_time:6.1f}s  "
                    f"pierces={r.n_pierces:2d}  "
                    f"reward={r.reward:7.2f}  "
                    f"({elapsed*1000:.0f} ms)  {flag}{feas}"
                )
        return results

    train_results = _eval_split(train_files, "TRAIN")
    test_results  = _eval_split(test_files,  "TEST ")

    summary = _summarize(train_results, test_results)

    if verbose:
        print()
        print("[Phase 1] Zusammenfassung:")
        for k, v in summary.items():
            print(f"  {k:30s} {v}")

    output = {
        "config":  cfg.to_dict(),
        "train":   [asdict(r) for r in train_results],
        "test":    [asdict(r) for r in test_results],
        "summary": summary,
    }

    # Default-Output: rl/results/baseline_v<version>.json -- die Version im
    # Dateinamen verhindert, dass eine Reward-/Scope-Aenderung stillschweigend
    # alte Resultate ueberschreibt.
    if output_path is None:
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"baseline_v{cfg.version}.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n[Phase 1] Ergebnisse geschrieben nach: {output_path}")

    return output


def _summarize(
    train: list[BaselineResult],
    test:  list[BaselineResult],
) -> dict:
    """Erzeugt eine kompakte Statistik-Zusammenfassung.

    Wird sowohl in der Konsolen-Ausgabe als auch in der JSON-Datei
    abgelegt. Spaeter (Phase 8) wird das Erfolgskriterium daraus
    berechnet -- z.B. "auf 95% der Test-Konturen >= 99% Coverage".
    """
    def stats(results: list[BaselineResult]) -> dict:
        if not results:
            return {"n": 0}
        cov  = [r.coverage   for r in results]
        time_ = [r.total_time for r in results]
        return {
            "n":                len(results),
            "coverage_mean":    float(np.mean(cov)),
            "coverage_min":     float(np.min(cov)),
            "coverage_max":     float(np.max(cov)),
            "time_mean":        float(np.mean(time_)),
            "time_min":         float(np.min(time_)),
            "time_max":         float(np.max(time_)),
            "n_success":        int(sum(r.success for r in results)),
            "n_infeasible":     int(sum(not r.is_feasible for r in results)),
            "success_rate":     float(np.mean([r.success for r in results])),
        }

    return {
        "train": stats(train),
        "test":  stats(test),
    }


# ---------------------------------------------------------------------------
# CLI-Einstiegspunkt
# ---------------------------------------------------------------------------
#
# Aufruf: `python -m plasma_cutter.rl.baseline`
# Spaeter (Phase 8) wird die JSON-Ausgabe von hier durch den Eval-Code
# eingelesen und mit den RL-Resultaten verglichen.
# ---------------------------------------------------------------------------

# Backwards-compat alias (frueher unterstrich-privat). Neue Aufrufer
# sollen `list_geometries` verwenden.
_list_geometries = list_geometries


if __name__ == "__main__":
    run_baseline()

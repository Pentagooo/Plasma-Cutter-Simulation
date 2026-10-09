# Segment-Simulation — Architektur

## 1. Physikalische Größen → Umsetzung

| Größe | Bedeutung | Umsetzung |
|---|---|---|
| **Klingenlänge L(v)** | Wie tief der Plasmastrahl ins Material reicht; schneller = kürzer. | `cutter/assumptions.py` (`BladeLengthModel`, linear), abgefragt über `Cutter.blade_length(v)`. |
| **Effektive Tiefe** | `L(v) − clearance` (Mindestabstand TCP–Material). Punkte tiefer als das sind physikalisch unerreichbar. | `planning.RunKinematics.effective_depth` |
| **Swept Area** | Vom Strahl überstrichene Fläche entlang eines Schnitts. | `RunKinematics.attach` baut je `CutRun` ein Shapely-Polygon. |
| **Coverage** | Anteil der Gitterpunkte in einer Swept Area = Grad der Durchtrennung. | `segments.compute_grid_coverage` |
| **Pierce-Zeit** | Pauschale je Zündung; entfällt bei nahtlos anschließenden Schnitten. | `Cutter.pierce_time()` / `PierceTimeModel` |
| **Geschwindigkeitsregel** | Schnellste coverage-erhaltende Geschwindigkeit je Block; `t_switch` je Wechsel. | `surrogate/runutils.py` (`split_group_for_speed`, `build_speed_chains`) |
| **T** | Ausführungszeit: Schnitt + Eilgang + Pierce + Geschwindigkeitswechsel. | `runutils.build_plan_with_speeds` → `SpeedPlan.total_time` |

## 2. Pipeline und die vier Einstiege

```
PointGrid ─► SegmentedContour.from_grid ─► [Auswahl] ─► merge / build_speed_chains
          ─► ExactSequencer (Held-Karp) + LinkPlanner ─► compute_grid_coverage ─► SpeedPlan (T)
```

| Auswahl | Funktion | Bemerkung |
|---|---|---|
| Manuell | `simulation.SegmentCutSimulation` (Klick) | Runs enden an beliebigen Konturknoten |
| Automatic Planner | `autoplan.AutoPlanner._greedy` + `_prune`; headless `surrogate.planner.greedy_plus_plan` | Kandidaten-Swept-Areas für ALLE Segmente (teuerste Stufe) |
| Surrogat | `surrogate.planner.surrogate_plan` | Modell-Rangfolge → Greedy Set Cover auf exakten Masken (lazy) → Pruning → ein Planbau → Verify → Fallback |
| Brute Force | `surrogate.teacher.exhaustive_plan` | alle 2^k Teilmengen, jede vollständige Abdeckung exakt bewertet |

Alle drei Planer benutzen dieselbe Geschwindigkeits-/Gruppierungsstufe
(`build_speed_chains`), deshalb gilt `T_BruteForce ≤ T_AutomaticPlanner` bei
gleicher Coverage (Test `test_teacher_dominates.py`). Im Simulator merkt
`_chain_sel` die Planer-Auswahl; Enter mit Geschwindigkeitsregel AN baut daraus die
DP-Split-Ketten und reproduziert das gemeldete `T`.

## 3. Datei → Rolle

**`segment_simulation/`**

| Datei | Rolle |
|---|---|
| `segments.py` | `ContourLoop`, Segmentierung (`Segment`, `SegmentedContour`), `CutRun`, Coverage |
| `planning.py` | `RunKinematics`, `LinkPlanner` (Sichtbarkeitsgraph + Dijkstra), `Sequencer` (Held-Karp), `CutPlan`, `compute_score` |
| `autoplan.py` | `AutoPlanner` (Greedy Set Cover + Pruning + Merge), `auto_plan(grid)` |
| `simulation.py` | interaktiver Simulator, `make_default_cutter`, `check_segments` |

**`segment_simulation/surrogate/`**

| Datei | Rolle |
|---|---|
| `features.py` | 37 Merkmale je Segment aus Kontur + Distanztransformation (keine Swept Areas); Modell nutzt 20 (`DEFAULT_MODEL_FEATURES`: Kantenkontext, Ersetzbarkeit/Überlappung, Tiefe/Reichweite, Geometrie, Kontext); `dataset --refeaturize` rechnet X der Labels neu |
| `instances.py` | Katalog (`SHAPE_VERSION` 7): sechs Normprofile (Flach, Winkel, T, U, Holland, Doppel-T), `attached` (angeschweißte Anbauteile), `assembly` (Schweißbaugruppe mit Bohrung/Langloch); stabil über (seed, idx); Testgeometrien (`real`), `default_cutter` |
| `teacher.py` | `exhaustive_plan`, `TeacherSkipped`, `LABEL_VERSION`, `K_MAX_DEFAULT = 18` |
| `planner.py` | `greedy_plus_plan`, `surrogate_plan`, `SurrogatePlanResult`, Fallback |
| `runutils.py` | Merge, Singleton-Masken, DP-Split, Ketten, `build_plan_with_speeds`, `speed_up_runs` |
| `model.py` | `SurrogateModel` (HistGradientBoosting, GroupKFold nach Familie, tau), `train_model`, `load_model` |
| `dataset.py` | Label-Pipeline: Cache `artifacts/labels/<key>.npz`, multiprocessing, Zeitbudget, `dataset.npz` |
| `benchmark.py`, `learning_curve.py` | Auswertung auf ungesehenen Instanzen (Testsätze seed 7 und seed 11) |
| `tests/` | Merkmale, DP-Split, Coverage-Garantie (Zufalls-/Null-/Eins-Modell), Fallback, Lehrer, Dominanz (`slow`), Modell-Guard |
| `artifacts/` | im Repo nur das produktive Modell (`surrogate_model.joblib`); Läufe unter `runs/` bleiben lokal |

**außerhalb**: `geometry/point_grid.py` (`PointGrid`, JSON), `geometry/geometry_processor.py` (CLI Rohkontur → geprüfte Geometrie), `cutter/cutter.py` (`Cutter`), `cutter/assumptions.py` (`BladeLengthModel`, `PierceTimeModel`, `CuttingAssumptions`).

## 4. Referenzlauf (Regressionskontrolle)

Mit `kontur.json` und `Test2.json` und dem Standard-Cutter (Physik im
Konstantenblock von `simulation.py`) im Simulator Automatic Planner (P), Surrogat
(S) und Brute Force (B) nacheinander auswählen und mit Enter planen.
Erwartet: vollständige Coverage bei allen drei Verfahren, Brute Force
nie langsamer als Automatic Planner, und Enter reproduziert das in der Statuszeile
gemeldete `T`. `test_lochjson.json`: die vollständig
umschlossene Lochkontur ist ohne Z-Hub nicht erreichbar; Surrogat und
Greedy melden ehrlich unerreichbare Punkte statt abzubrechen; Brute Force
wirft `TeacherSkipped` (mehr Segmente als `K_MAX_DEFAULT`).

# Plasmaschneider — Bahnplaner & Segment-Simulation

Bahnplaner für einen robotergeführten Plasmaschneider. Aus einer
Bauteilgeometrie (Außenkontur + Löcher, als Punktgitter) berechnet das
System ein **vollständiges, zeitminimales und kollisionsfreies
Schneidprimitiv**: Welche Stücke der Kontur werden geschnitten, in welcher
Reihenfolge und Richtung, mit welcher Geschwindigkeit?

## Modell

- TCP hält während des Schneidens einen konstanten Mindestabstand `d_min`
  zum Material
- Länge des Plasmastrahls sinkt linear mit der Geschwindigkeit:
  `L(v) = L0 − slope · v`
- **Coverage** = Anteil der Gitterpunkte, die der Strahl erfasst
- **Ziel**: volle Coverage in minimaler Ausführungszeit `T`, innerhalb des
  Planungsbudgets

| Größe | Bedeutung | Code |
|---|---|---|
| Klingenlänge `L(v)` | Reichweite des Strahls; schneller = kürzer | `cutter/assumptions.py` (`BladeLengthModel`), `Cutter.blade_length(v)` |
| Effektive Tiefe | `L(v) − d_min` | `planning.RunKinematics.effective_depth` |
| Swept Area | vom Strahl überstrichene Fläche eines Schnitts | `RunKinematics.attach` (Shapely-Polygon je `CutRun`) |
| Coverage | Anteil der Gitterpunkte in den Swept Areas | `segments.compute_grid_coverage` |
| Pierce-Zeit | Pauschale je Zündung; entfällt, wenn direkt davor das benachbarte Segment geschnitten wurde | `Cutter.pierce_time()`, `PierceTimeModel` |
| Geschwindigkeitsregel | jeder Abschnitt eines Schnitts so schnell, wie der Strahl die tiefste zu schneidende Stelle noch erreicht; jeder Geschwindigkeitswechsel kostet `t_switch` | `surrogate/runutils.py` (`split_group_for_speed`, `build_speed_chains`) |
| `T` | `T_cut + T_rapid + T_pierce + T_switch` | `runutils.build_plan_with_speeds` → `SpeedPlan.total_time` |

## Pipeline

Alle Verfahren laufen durch dieselbe Pipeline und unterscheiden sich nur in
der Auswahl:

```
PointGrid
  -> SegmentedContour.from_grid   Kontur in Primitivsegmente zerlegen
  -> Auswahl                      manuell / Automatic Planner / Surrogat / Brute Force
  -> build_speed_chains           Nachbarsegmente zu Ketten, DP-Split in
                                  Geschwindigkeitsblöcke, jeden Block exakt prüfen
  -> build_plan_with_speeds       Sequencer: LinkPlanner (kollisionsfreie Übergänge,
                                  t_link + Pierce) -> Held-Karp (Reihenfolge, Richtung),
                                  danach Zeitbilanz T
  -> compute_grid_coverage        exakte Coverage
```

| Verfahren | Auswahl | Einstieg |
|---|---|---|
| **Manuell** | Start- und Endknoten jedes Schnitts per Mausklick wählen | Simulator (Klick) |
| **Automatic Planner** | Greedy Set Cover + Pruning des `AutoPlanner` | Taste P, `surrogate.planner.greedy_plus_plan` |
| **Surrogat** | gelerntes Modell ordnet die Segmente nach p(s) „liegt im Optimum“; Greedy Set Cover in dieser Reihenfolge auf exakten Masken, Pruning, Verify, Fallback auf Greedy; die Coverage hängt nie am Modell | Taste S, `surrogate.planner.surrogate_plan` |
| **Brute Force** | alle 2^k Segment-Teilmengen, jede vollständige Abdeckung exakt bewertet = Optimum (bis `K_MAX_DEFAULT` Segmente); erzeugt auch die Trainingslabels | Taste B, `surrogate.teacher.exhaustive_plan` |

## Schnellstart

- Python ≥ 3.10 (getestet mit 3.14)
- Ordner muss `plasma_cutter` heißen, Befehle aus dem Elternordner starten
- Modell mit scikit-learn 1.9 gespeichert; die `params_hash`-Warnung beim
  Laden ist erwartet (Modell stammt von einem älteren Katalogstand)

```bash
git clone https://igm-git.igm.rwth-aachen.de/sherec/auto_cutting_primitives.git plasma_cutter
cd plasma_cutter
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
cd ..
```

### Interaktiver Simulator

```bash
python -m plasma_cutter.segment_simulation.simulation \
    --geometry "plasma_cutter/geometry/Geometrie_Konturen_geprüft/kontur.json"
# ohne --geometry: Dateidialog; Parameter: --v-cut --v-max --blade-length --blade-slope --clearance --segment-length
```

| Taste / Button | Wirkung |
|---|---|
| Klick | manuelle Auswahl: Start-, dann Endknoten eines Schnitts |
| `A` | alle restlichen Konturen komplett auswählen |
| `P` / Automatic planner | Auswahl des Automatic Planner |
| `S` / Surrogate | Surrogat-Auswahl (lädt `surrogate/artifacts/surrogate_model.joblib`) |
| `B` / Brute force | exaktes Optimum (Rechenzeit wächst exponentiell mit der Segmentzahl) |
| `V` / Speed rule | Geschwindigkeitsregel an/aus |
| `Enter` | planen (Reihenfolge, Verbindungen, Geschwindigkeiten) + animieren |
| `U` / Rechtsklick, `R`, `Esc`, `+`/`-` | Undo, Reset, Abbruch, Animationstempo |

Die Statuszeile meldet je Verfahren Schnitte, `T`, Coverage und Planzeit;
Enter reproduziert das gemeldete `T` exakt.

### Headless

```python
from plasma_cutter.geometry.point_grid import PointGrid
from plasma_cutter.segment_simulation.surrogate import planner, teacher, model

grid = PointGrid.from_json("plasma_cutter/geometry/Geometrie_Konturen_geprüft/kontur.json")
gp   = planner.greedy_plus_plan(grid)                 # dict: T, coverage, n_runs, t_plan
sur  = planner.surrogate_plan(grid, model.load_model())
print(sur.summary(), sur.selected)
opt  = teacher.exhaustive_plan(grid, k_max=18)        # TeacherSkipped bei > k_max Segmenten
```

## Training, Benchmark, Tests

```bash
# Parameterstempel anzeigen (alle label-relevanten Werte + Hashes)
python -m plasma_cutter.segment_simulation.surrogate.params

# Segmentzahlen je Familie prüfen (ohne Lehrer)
python -m plasma_cutter.segment_simulation.surrogate.dataset --dry-run --n 400 [--seg-divisor 18 --seg-min-spacings 3]

# Labels mit dem Brute-Force-Lehrer (resumebar, je Instanz gecacht in <out>/labels/);
# --seg-mix gibt jeder Instanz eine Segmentierung aus einer gewichteten Mischung
python -m plasma_cutter.segment_simulation.surrogate.dataset --n 4200 --seed 42 --n-jobs 8 --k-max 20 \
    --seg-mix "12/4:0.50,14/4:0.25,16/3:0.20,18/3:0.05" \
    --out plasma_cutter/segment_simulation/surrogate/artifacts/runs/main [--max-minutes 600 --stop-file STOP]

# Modell trainieren  ->  <out>/surrogate_model.joblib + model_meta.json
# Bei geänderter Merkmalsliste: Labels behalten, nur X neu rechnen (kein Relabel)
python -m plasma_cutter.segment_simulation.surrogate.dataset --refeaturize .../runs/main/labels
python -m plasma_cutter.segment_simulation.surrogate.model --train --out .../runs/main

# Optimum des Testsatzes parallel labeln, dann Benchmark Automatic Planner / Surrogat / Brute Force
python -m plasma_cutter.segment_simulation.surrogate.dataset --n 60 --seed 7 --include-real --n-jobs 8 --out .../runs/main_eval
python -m plasma_cutter.segment_simulation.surrogate.benchmark --n 60 --seed 7 --reps 3 \
    --model .../runs/main/surrogate_model.joblib --out .../runs/main --opt-labels .../runs/main_eval/labels

# Lernkurve (Modelle auf den ersten N Instanzen, fester Testsatz seed 7)
python -m plasma_cutter.segment_simulation.surrogate.learning_curve --dataset .../runs/main \
    --n-list 100,500,2000,4000 --n-eval 60 --opt-labels .../runs/main_eval/labels

# Tests (erste Zeile schnell; zweite zusätzlich mit den langsamen Dominanz-Tests des Lehrers)
python -m pytest plasma_cutter/segment_simulation/surrogate/tests -q -m "not slow"
python -m pytest plasma_cutter/segment_simulation/surrogate/tests -q
```

## Projektstruktur

```
plasma_cutter/
├── segment_simulation/
│   ├── segments.py            Kontur in Segmente zerlegen, Schnitte beschreiben, Coverage zählen
│   ├── planning.py            Strahl und überstrichene Fläche (Swept Area) je Schnitt,
│   │                          kollisionsfreie Verfahrwege, beste Reihenfolge der Schnitte
│   ├── autoplan.py            Automatic Planner: wählt Segmente, bis alles abgedeckt ist
│   ├── simulation.py          interaktiver Simulator mit Animation; Physik-Konstanten des Schneiders
│   └── surrogate/
│       ├── planner.py         Surrogat-Planer und Automatic Planner mit Geschwindigkeitsregel
│       ├── teacher.py         Brute Force: probiert alle Segmentkombinationen, liefert das Optimum
│       ├── model.py           gelerntes Modell trainieren und laden
│       ├── features.py        Kennzahlen je Segment, aus denen das Modell lernt
│       ├── dataset.py         Trainingsdaten erzeugen (Brute Force auf vielen Bauteilen)
│       ├── instances.py       Bauteilkatalog fürs Training: Normprofile, Anbauteile, Schweißbaugruppen
│       ├── params.py          Stempel: erkennt, ob Modell, Daten und Physik zusammenpassen
│       ├── runutils.py        gemeinsame Schritte aller Planer: Segmente zu Schnitten verbinden,
│       │                      Geschwindigkeiten, Reihenfolge, Gesamtzeit
│       ├── benchmark.py       Planer auf neuen Bauteilen vergleichen
│       ├── learning_curve.py  Modellgüte über der Menge an Trainingsdaten
│       ├── tests/             automatische Tests
│       └── artifacts/         trainiertes Modell (im Repo); Trainingsläufe nur lokal
├── cutter/                    Physikmodell des Schneiders: Strahllänge L(v), Zündzeit
├── geometry/                  Punktgitter laden, Rohkonturen aufbereiten (geometry_processor.py)
│   ├── Geometrie_Konturen_ungeprüft/   Rohkonturen aus dem Kontur-Editor
│   └── Geometrie_Konturen_geprüft/     aufbereitete Testgeometrien
├── requirements.txt           benötigte Python-Pakete
└── README.md
```

## Parameter und Stempel

Werte, die Trainingsdaten und Modell bestimmen:

- Physik des Schneiders (`v_cut`, `v_max`, `L0`, `blade_slope`,
  `minimum_gap` = `d_min`, `rapid_speed`, `t_switch`, Pierce):
  Konstantenblock in `segment_simulation/simulation.py`
- Kerf und Segmentierung: `surrogate/params.py`
- Abtastung des TCP-Pfads: `planning.TCP_SAMPLE_STEP`
- Punktdichte und Bauteilkatalog: `surrogate/instances.py`

Der Stempel (`python -m plasma_cutter.segment_simulation.surrogate.params`)
steckt in jedem Label und im Modell und erkennt Änderungen:

- andere Physik (`phys_hash`): Training und Laden des Modells brechen ab →
  neu labeln und trainieren
- andere Segmentierung oder anderer Katalog (`params_hash`): nur eine Warnung

Neue Bauteilfamilien in `instances.py` nur anhängen und `SHAPE_VERSION`
erhöhen.

## Geometrien

`geometry/geometry_processor.py` wandelt eine Rohkontur aus dem Kontur-Editor
(Beispiele in `Geometrie_Konturen_ungeprüft/`) in eine geprüfte Geometrie
(verdichtete Außen-/Lochkontur + Innenraster, Ausgabe nach
`Geometrie_Konturen_geprüft/`):

```bash
python plasma_cutter/geometry/geometry_processor.py "pfad/zur/kontur.json" --no-plot
```

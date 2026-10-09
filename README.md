# Plasmaschneider — Bahnplaner & Segment-Simulation

Bahnplaner für einen robotergeführten Plasmaschneider. Aus einer
Bauteilgeometrie (Außenkontur + Löcher, als Punktgitter) berechnet das
System einen **vollständigen, zeitminimalen und kollisionsfreien
Schneidplan**: Welche Stücke der Kontur werden geschnitten, in welcher
Reihenfolge und Richtung, mit welcher Geschwindigkeit?

## Modell

- TCP fährt auf einem Offset-Pfad mit konstantem Mindestabstand zum Material
- Plasmastrahl ragt Richtung Material, seine Länge sinkt linear mit der
  Geschwindigkeit: `L(v) = L0 − slope · v`
- **Coverage** = überstrichene Querschnittsfläche (alle vom Strahl erfassten
  Gitterpunkte), nicht nur die Kontur
- **Ziel**: volle Coverage in minimaler Ausführungszeit `T`

| Größe | Bedeutung | Code |
|---|---|---|
| Klingenlänge `L(v)` | Reichweite des Strahls; schneller = kürzer | `cutter/assumptions.py` (`BladeLengthModel`), `Cutter.blade_length(v)` |
| Effektive Tiefe | `L(v) − clearance`; tiefere Punkte sind unerreichbar | `planning.RunKinematics.effective_depth` |
| Swept Area | vom Strahl überstrichene Fläche eines Schnitts | `RunKinematics.attach` (Shapely-Polygon je `CutRun`) |
| Coverage | Anteil der Gitterpunkte in den Swept Areas | `segments.compute_grid_coverage` |
| Pierce-Zeit | Pauschale je Zündung, entfällt bei nahtlosem Anschluss | `Cutter.pierce_time()`, `PierceTimeModel` |
| Geschwindigkeitsregel | schnellste coverage-erhaltende Geschwindigkeit je Block, `t_switch` je Wechsel | `surrogate/runutils.py` (`split_group_for_speed`, `build_speed_chains`) |
| `T` | Schnitt + Eilgang + Pierce + Geschwindigkeitswechsel | `runutils.build_plan_with_speeds` → `SpeedPlan.total_time` |

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
| **Manuell** | Bediener klickt Start-/Endknoten in die Zeichnung | Simulator (Klick) |
| **Automatic Planner** | Greedy Set Cover + Pruning des `AutoPlanner` | Taste P, `surrogate.planner.greedy_plus_plan` |
| **Surrogat** | gelerntes Modell ordnet die Segmente nach p(s) „liegt im Optimum“; Greedy Set Cover in dieser Reihenfolge auf exakten Masken, Pruning, Verify, Fallback auf Greedy; die Coverage hängt nie am Modell | Taste S, `surrogate.planner.surrogate_plan` |
| **Brute Force** | alle 2^k Segment-Teilmengen, jede vollständige Abdeckung exakt bewertet = Optimum (bis `K_MAX_DEFAULT` Segmente); erzeugt auch die Trainingslabels | Taste B, `surrogate.teacher.exhaustive_plan` |

- gleiche Pipeline ab der Auswahl → `T_BruteForce ≤ T_AutomaticPlanner` bei
  gleicher Coverage (`tests/test_teacher_dominates.py`)
- Schalter **Geschwindigkeitsregel** (Taste V): AN = DP-Split je Kette bei
  allen drei Planern, AUS = alles bei Basisgeschwindigkeit
- fairer Vergleich manuell vs. Planer: Geschwindigkeitsregel AUS (manuelle
  Runs bekommen keine Geschwindigkeitsblöcke)

## Schnellstart

- Python ≥ 3.10 (getestet mit 3.14)
- der Ordner **muss** `plasma_cutter` heißen (interne Importe
  `plasma_cutter.…`); Kommandos laufen aus dem **Elternordner**
- das mitgelieferte Modell ist mit scikit-learn 1.9 gespeichert; mit einer
  anderen Version warnt scikit-learn beim Laden

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

**Kurzcheck im Simulator** nach Änderungen an Physik oder Planung:

- `kontur.json` und `Test2.json`: P, S und B nacheinander auswählen, je mit
  Enter planen
- erwartet: volle Coverage bei allen drei, Brute Force nie langsamer als der
  Automatic Planner, Enter reproduziert das gemeldete `T`
- `test_lochjson.json`: die vollständig umschlossene Lochkontur ist ohne
  Z-Hub nicht erreichbar; Surrogat und Brute Force decken den erreichbaren
  Rest ab, statt abzubrechen

## Projektstruktur

```
plasma_cutter/
├── segment_simulation/
│   ├── segments.py            ContourLoop, Segmentierung (Segment, SegmentedContour), CutRun, Coverage
│   ├── planning.py            RunKinematics (Swept Area), LinkPlanner (Sichtbarkeitsgraph + Dijkstra),
│   │                          Sequencer (Held-Karp), CutPlan, compute_score
│   ├── autoplan.py            AutoPlanner: Greedy Set Cover + Pruning + Merge, auto_plan(grid)
│   ├── simulation.py          interaktiver Simulator (manuell / P / S / B, Animation), make_default_cutter
│   └── surrogate/
│       ├── features.py        Merkmale je Segment (nur Kontur + Distanzen); Modell nutzt DEFAULT_MODEL_FEATURES
│       ├── params.py          Parameterstempel (LABEL_VERSION, phys_hash/params_hash)
│       ├── instances.py       Katalog (SHAPE_VERSION): Normprofile (Flach, Winkel, T, U, HP, H),
│       │                      Anbauteile (attached), Schweißbaugruppe (assembly); Testgeometrien
│       ├── teacher.py         Brute-Force-Lehrer (exhaustive_plan, K_MAX_DEFAULT)
│       ├── planner.py         greedy_plus_plan, surrogate_plan
│       ├── runutils.py        gemeinsame Planungsstufen (Merge, DP-Split, Ketten, Planbau)
│       ├── model.py           SurrogateModel (HistGradientBoosting, GroupKFold nach Familie), train_model, load_model
│       ├── dataset.py         Label-Pipeline (Cache je Instanz, parallel, resumebar)
│       ├── benchmark.py       Automatic Planner / Surrogat / Brute Force auf ungesehenen Instanzen
│       ├── learning_curve.py  Lernkurve
│       ├── tests/             Merkmale, Stempel, DP-Split, Merge, Coverage-Garantie, Lehrer,
│       │                      Dominanz (slow), Modell-Guard
│       └── artifacts/         Produktivmodell (im Repo); runs/<name>/ je Label-/Trainingslauf (nur lokal)
├── cutter/                    Physikmodell: Cutter, BladeLengthModel L(v), PierceTimeModel, CuttingAssumptions
├── geometry/                  PointGrid (JSON), geometry_processor.py (Kontur-Editor-JSON -> geprüfte Geometrie)
│   ├── Geometrie_Konturen_ungeprüft/   Rohkonturen aus dem Kontur-Editor (Eingabe)
│   └── Geometrie_Konturen_geprüft/     Testgeometrien (Ausgabe, von Simulator und Katalog genutzt)
├── requirements.txt
└── README.md
```

## Parameter und Stempel

Alle label-relevanten Werte stehen an einer Stelle:

- Physik des Schneiders: Konstantenblock in `segment_simulation/simulation.py`
  (`v_cut`, `v_max`, `L0`, `blade_slope`, `minimum_gap`, `rapid_speed`,
  `t_switch`, Pierce)
- Kerf und Segmentierungsregel: `surrogate/params.py`
- Abtastung: `planning.TCP_SAMPLE_STEP`
- Punktdichte und Katalog: `surrogate/instances.py`

`params.label_params()` sammelt sie; zwei Hashes davon stehen in jedem
Label-Dateinamen, in `dataset_meta.json` und im Modell:

- `phys_hash` (Physik, Kerf, Abtastung, Eckwinkel) muss zwischen Labels,
  Modell und Code übereinstimmen, sonst brechen `train_model`/`load_model` ab
- `params_hash` (zusätzlich Segmentierung, Punktdichte, `SHAPE_VERSION`)
  unterscheidet Datensätze; beim Laden eines Modells nur eine Warnung

**Jede Änderung eines dieser Werte macht alle Labels und das Modell
ungültig** (neu labeln). Neue Katalogfamilien in `instances.py` nur
anhängen (`SHAPE_ORDER`, `FAMILIES`) und `SHAPE_VERSION` erhöhen.

## Artefakte

`segment_simulation/surrogate/artifacts/`: im Repo liegt nur das produktive
Modell, alles andere bleibt per `.gitignore` lokal.

| Ort | Inhalt | im Repo |
|---|---|---|
| `surrogate_model.joblib` | produktives Modell (Taste S), Stempel in der Datei; trainiert auf einem älteren Katalogstand (`SHAPE_VERSION` inzwischen erhöht), daher beim Laden nur die `params_hash`-Warnung | ja |
| `runs/<name>/` | ein Label-/Trainingslauf: `labels/` (je Instanz ein `.npz` mit Merkmalen, Label, `T`, Lehrerzeit, Stempel), `dataset.npz`, `dataset_meta.json`, Modell, `benchmark*.{csv,md}`, `learning_curve*.{csv,md}` | nein |

## Geometrien

`geometry/geometry_processor.py` wandelt eine Rohkontur aus dem Kontur-Editor
(Beispiele in `Geometrie_Konturen_ungeprüft/`) in eine geprüfte Geometrie
(verdichtete Außen-/Lochkontur + Innenraster, Ausgabe nach
`Geometrie_Konturen_geprüft/`):

```bash
python plasma_cutter/geometry/geometry_processor.py "pfad/zur/kontur.json" --no-plot
```

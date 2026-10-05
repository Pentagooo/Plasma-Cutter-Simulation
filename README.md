# Plasmaschneider — Bahnplaner & Segment-Simulation

Bahnplaner für einen robotergeführten Plasmaschneider. Aus einer
Bauteilgeometrie (Außenkontur + Löcher, als Punktgitter) berechnet das
System einen **vollständigen, zeitminimalen und kollisionsfreien
Schneidplan**: Welche Stücke der Kontur werden geschnitten, in welcher
Reihenfolge und Richtung, mit welcher Geschwindigkeit?

**Schneidmodell.** Der TCP fährt auf einem Offset-Pfad mit konstantem
Mindestabstand zum Material. Der Plasmastrahl ragt Richtung Material mit
geschwindigkeitsabhängiger Länge `L(v) = L0 − slope · v`. **Coverage** ist
die überstrichene Querschnittsfläche (alle vom Strahl erfassten
Gitterpunkte), nicht nur die Kontur. **Ziel** ist 100 % Coverage in
minimaler Ausführungszeit `T` (Schnitt + Eilgang + Zündungen +
Geschwindigkeitswechsel).

## Vier Auswahlverfahren, eine Pipeline

Die Kontur wird in Primitivsegmente zerlegt. Jedes Verfahren
wählt eine Teilmenge dieser Segmente; danach läuft für alle **dieselbe
Pipeline**: coverage-erhaltendes Verschmelzen zu Runs, DP-Split der
Geschwindigkeitsblöcke je Kette (Geschwindigkeitsregel + `t_switch`), Reihenfolge und
Richtung per Held-Karp, kollisionsfreie Verbindungen per Sichtbarkeitsgraph,
exakte Coverage-Prüfung.

| Verfahren | Auswahl | Wo |
|---|---|---|
| **Manuell** | Bediener klickt Start-/Endknoten in die Zeichnung | Simulator |
| **Automatic Planner** | Greedy Set Cover + Pruning des `AutoPlanner`, mit der gemeinsamen Geschwindigkeitsstufe | Simulator (P), `surrogate.planner.greedy_plus_plan` |
| **Surrogat** | Gelerntes Modell (Gradient Boosting, 20 Konturmerkmale aus 37 berechneten) ordnet die Segmente nach p(s) "liegt im Optimum"; Greedy Set Cover in dieser Reihenfolge auf **exakten** Masken, Pruning, ein Planbau, ein exakter Verify, Fallback auf Greedy. Die Garantie hängt nie am Modell. | Simulator (S), `surrogate.planner.surrogate_plan` |
| **Brute Force** | Alle 2^k Segment-Teilmengen, unvollständige Abdeckungen verwerfen, jede vollständige exakt bewerten, Minimum behalten = **Optimum** (bis 18 Segmente). Erzeugt auch die Trainingslabels. | Simulator (B), `surrogate.teacher.exhaustive_plan` |

Der Schalter **Geschwindigkeitsregel** (Taste V) entscheidet, ob die
Geschwindigkeitszuweisung Teil der Zielfunktion ist (AN: DP-Split je Kette
bei allen drei Planern; AUS: alles bei Basisgeschwindigkeit). Für einen
fairen Vergleich manuell vs. Planer: Geschwindigkeitsregel AUS (manuelle Runs bekommen
keine Geschwindigkeitsblöcke).

Benchmark und Lernkurve (Automatic Planner / Surrogat / Brute Force auf ungesehenen
Testinstanzen) lassen sich mit den Befehlen unter
[Benchmark, Label-Pipeline, Training, Lernkurve, Tests](#benchmark-label-pipeline-training-lernkurve-tests)
selbst erzeugen.

## Schnellstart

Voraussetzung: Python ≥ 3.10 (getestet mit 3.14; unter Linux 3.12 empfohlen). Der Ordner **muss**
`plasma_cutter` heißen (interne Importe `plasma_cutter.…`); Kommandos
laufen aus dem **Elternordner**. Das mitgelieferte Modell ist mit
scikit-learn 1.9 gespeichert (siehe `requirements.txt`).

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
| `B` / Brute force | exaktes Optimum (bis 18 Segmente; Rechenzeit wächst exponentiell mit der Segmentzahl) |
| `V` / Speed rule | Geschwindigkeitszuweisung an/aus |
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

### Benchmark, Label-Pipeline, Training, Lernkurve, Tests

```bash
# Parameterstempel anzeigen (alle label-relevanten Werte + Hashes)
python -m plasma_cutter.segment_simulation.surrogate.params

# Segmentzahlen je Familie prüfen (Sekunden, kein Lehrer)
python -m plasma_cutter.segment_simulation.surrogate.dataset --dry-run --n 400 [--seg-divisor 18 --seg-min-spacings 3]

# Labels mit dem Brute-Force-Lehrer (resumebar, je Instanz gecacht in <out>/labels/);
# --seg-mix gibt jeder Instanz eine Segmentierung aus einer gewichteten Mischung (k 5..22)
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
│   ├── segments.py            Konturen, Segmentierung, CutRun, Coverage
│   ├── planning.py            Kinematik (Swept Area), LinkPlanner, Sequencer (Held-Karp), Score
│   ├── autoplan.py            AutoPlanner: Greedy Set Cover + Pruning + Merge
│   ├── simulation.py          interaktiver Simulator (manuell / P / S / B, Animation)
│   ├── ARCHITEKTUR.md         Datei -> Rolle, Pipeline, Referenzlauf
│   ├── surrogate/
│   │   ├── features.py        37 Merkmale je Segment (nur Kontur + Distanzen), Modell nutzt 20 (DEFAULT_MODEL_FEATURES)
│   │   ├── params.py          Parameterstempel (LABEL_VERSION, phys_hash/params_hash)
│   │   ├── instances.py       Instanzkatalog (SHAPE_VERSION 7): 6 Normprofile (Flach, Winkel, T, U, HP, H), Anbauteile (attached), Schweißbaugruppe (assembly), Testgeometrien
│   │   ├── teacher.py         Brute-Force-Lehrer (exhaustive_plan)
│   │   ├── planner.py         greedy_plus_plan, surrogate_plan
│   │   ├── runutils.py        gemeinsame Planungsstufen (Merge, DP-Split, Ketten, Verify)
│   │   ├── model.py           SurrogateModel, train_model, load_model
│   │   ├── dataset.py         Label-Pipeline (Cache je Instanz, parallel, resumebar)
│   │   ├── benchmark.py       Automatic Planner / Surrogat / Brute Force
│   │   ├── learning_curve.py  Lernkurve
│   │   ├── tests/             Merkmale, DP-Split, Garantie, Fallback, Lehrer, Modell-Guard
│   │   └── artifacts/         Produktivmodell (im Repo); runs/<name>/ je Label-/Trainingslauf (nur lokal)
├── cutter/                    Physikmodell: Cutter, BladeLengthModel L(v), PierceTimeModel
├── geometry/                  PointGrid (JSON), geometry_processor.py (Kontur-Editor-JSON -> geprüfte Geometrie)
│   ├── Geometrie_Konturen_ungeprüft/   Rohkonturen aus dem Kontur-Editor (Eingabe)
│   └── Geometrie_Konturen_geprüft/     sechs Testgeometrien (Ausgabe, von Simulator und Katalog genutzt)
├── tools/                     Linux-Lauf: setup_ubuntu.sh, run_final_training.sh, collect_results.sh, ANLEITUNG_LAUF.md
├── requirements.txt
└── README.md
```

## Parameter und Stempel

Alle label-relevanten Werte stehen an einer Stelle: die Physik des
Schneiders im Konstantenblock von `segment_simulation/simulation.py`
(`v_cut`, `v_max`, `L0`, `blade_slope`, `minimum_gap`, `rapid_speed`,
`speed_switch_time`, Pierce), Kerf und Segmentierungsregel in
`surrogate/params.py`, Abtastung in `planning.TCP_SAMPLE_STEP`, Punktdichte
und Katalog in `surrogate/instances.py`. `params.label_params()` sammelt sie;
zwei Hashes davon stehen in jedem Label-Dateinamen, in `dataset_meta.json`
und im Modell:

- `phys_hash` (Physik, Kerf, Abtastung, Eckwinkel) muss zwischen Labels,
  Modell und Code übereinstimmen, sonst brechen `train_model`/`load_model` ab.
- `params_hash` (zusätzlich Segmentierung, Punktdichte, `SHAPE_VERSION`)
  unterscheidet Datensätze; beim Laden eines Modells nur eine Warnung.

**Jede Änderung eines dieser Werte macht alle Labels und das Modell
ungültig** (neu labeln). Neue Katalogfamilien in `instances.py` nur
anhängen (`SHAPE_ORDER`, `FAMILIES`) und `SHAPE_VERSION` erhöhen.

## Artefakte

`segment_simulation/surrogate/artifacts/`. Im Repo liegt nur das
produktive Modell; alles andere ist groß und per `.gitignore` lokal.

| Ort | Inhalt | im Repo |
|---|---|---|
| `surrogate_model.joblib`, `model_meta.json` | das produktive Modell (Taste S im Simulator), Kopie aus `runs/main_f2/`; Stempel und Trainingsdaten stehen in `model_meta.json`. Trainiert auf einem älteren Katalogstand (`SHAPE_VERSION` im Code inzwischen erhöht); beim Laden kommt deshalb nur die `params_hash`-Warnung | ja |
| `runs/<name>/` | ein Label-/Trainingslauf: `labels/` (je Instanz ein `.npz` mit Merkmalen, Label, `T`, Lehrerzeit, Stempel), `dataset.npz`, `dataset_meta.json`, Modell, `benchmark*.{csv,md}`, `learning_curve*.{csv,md}` | nein |
| `archive_L1/` | Stand vor dem Parameterstempel (alte `LABEL_VERSION`; lädt nicht mehr) | nein |

## Abschlusslauf auf einem Linux-Rechner (`tools/`)

```bash
# im Elternordner des Klons; der Ordner muss "plasma_cutter" heißen
plasma_cutter/tools/setup_ubuntu.sh                       # venv, Tests, Smoke-Label
nohup plasma_cutter/tools/run_final_training.sh phase1 > logs/phase1.out 2>&1 &   # Testsätze, Hauptlauf (10 000 Instanzen, k<=21, Budget 45 h), Training, Benchmarks
nohup plasma_cutter/tools/run_final_training.sh phase2 > logs/phase2.out 2>&1 &   # optional: Hauptlauf fortsetzen (10 h), neu trainieren
plasma_cutter/tools/run_final_training.sh phase3                                  # nur Benchmarks + Lernkurve
plasma_cutter/tools/collect_results.sh                                             # tar.gz zurück
```

`touch STOP` beendet einen Label-Lauf sanft (laufende Instanzen rechnen zu
Ende). Kerne: `nproc`; die GPU wird nicht genutzt. Schritt-für-Schritt-Anleitung
für den Ubuntu-Rechner: `tools/ANLEITUNG_LAUF.md`.

## Geometrien

`geometry/geometry_processor.py` wandelt eine Rohkontur aus dem Kontur-Editor
(Beispiele in `Geometrie_Konturen_ungeprüft/`) in eine geprüfte Geometrie
(verdichtete Außen-/Lochkontur + Innenraster, Ausgabe nach
`Geometrie_Konturen_geprüft/`):

```bash
python plasma_cutter/geometry/geometry_processor.py "pfad/zur/kontur.json" --no-plot
```

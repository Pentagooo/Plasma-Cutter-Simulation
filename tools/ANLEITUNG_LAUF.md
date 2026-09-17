# Anleitung: Abschluss-Trainingslauf auf dem Ubuntu-Rechner

Rechner: AMD Threadripper PRO 5965WX (24 Kerne / 48 Threads). Ziel: Phase 1
in etwa 18 Stunden. GPU wird nicht gebraucht.

Alle Befehle im Programm **Terminal** (Strg+Alt+T) eingeben, jeweils eine
Zeile, dann Enter. Zeilen, die mit `#` beginnen, sind Erklärungen.

## 1. Einmalig: Werkzeuge und Code holen (5 Minuten)

```bash
# Git und Python 3.12 installieren (fragt nach dem Passwort)
sudo apt update
sudo apt install -y git python3.12 python3.12-venv

# Arbeitsordner anlegen und Code holen (kein Login nötig, Repo ist öffentlich)
mkdir -p ~/work && cd ~/work
git clone -b InstanceTraining https://github.com/Pentagooo/Plasma-Cutter-Simulation.git plasma_cutter
```

Wichtig: Der Ordner muss `plasma_cutter` heißen, und alle weiteren Befehle
laufen aus `~/work`, nicht aus dem Ordner selbst.

## 2. Einmalig: Umgebung einrichten und prüfen (5 bis 10 Minuten)

```bash
cd ~/work
bash plasma_cutter/tools/setup_ubuntu.sh
```

Das Skript legt eine virtuelle Python-Umgebung an, installiert die
Abhängigkeiten, prüft alle Dateien, lässt die Tests laufen, druckt den
Parameterstempel und labelt drei Instanzen zur Probe.

Kontrolle: In der Ausgabe muss stehen

```
"phys_hash": "9a095e7d"
```

und weiter unten `passed` bei den Tests. Steht ein anderer Hash, stimmen die
Physikparameter nicht mit Windows überein: dann abbrechen und melden.

## 3. Lauf starten (läuft dann etwa 18 Stunden allein)

```bash
cd ~/work
mkdir -p logs
nohup plasma_cutter/tools/run_final_training.sh phase1 > logs/phase1.out 2>&1 &
```

`nohup` und `&` sorgen dafür, dass der Lauf weiterläuft, wenn du das
Terminal schließt oder dich abmeldest. Der Rechner darf nicht in den
Ruhezustand gehen: in den Ubuntu-Einstellungen unter Energie den
automatischen Standby ausschalten.

Was Phase 1 macht, in dieser Reihenfolge:

1. Testsätze labeln (seed 7, Standard- und feine Segmentierung), etwa 1 h.
2. Hauptlauf: 4200 Katalog-Instanzen mit gemischter Segmentierung
   (Segmentzahl 5 bis 20), Zeitbudget 15 h. Danach werden keine neuen
   Instanzen begonnen, laufende rechnen zu Ende.
3. Modell trainieren (Minuten).
4. Benchmark auf beiden Testsätzen und Lernkurve (unter einer Stunde).

## 4. Fortschritt ansehen

```bash
cd ~/work
tail -f logs/phase1.out            # live; Strg+C beendet nur die Anzeige
ls plasma_cutter/segment_simulation/surrogate/artifacts/runs/main/labels | wc -l   # fertige Labels
```

Im Log erscheint alle 100 Instanzen eine Zeile wie
`[1200/4200] 7200s, ETA 18000s {'new': 1190, 'too_big': 10}`.
Am Ende steht `== phase1 fertig ==`.

**Zwischenergebnis mit Modell** (jederzeit während des Laufs, dauert wenige
Minuten, stört das Labeln nicht):

```bash
cd ~/work
plasma_cutter/tools/run_final_training.sh snapshot
```

Sammelt die bis dahin fertigen Labels ein, trainiert ein Modell und
benchmarkt es gegen Greedy+ und das Optimum auf dem Standard-Testsatz.
Ergebnis in `plasma_cutter/segment_simulation/surrogate/artifacts/runs/snapshot_<Uhrzeit>/benchmark.md`
(Tabelle oben: T, Lücke zum Optimum, Planzeit). Der Testsatz ist ab etwa einer
Stunde nach dem Start vorhanden.

## 5. Falls etwas dazwischenkommt

```bash
cd ~/work
touch STOP          # Labeln sanft beenden: laufende Instanzen rechnen zu Ende
rm STOP             # vor einem Neustart wieder löschen
nohup plasma_cutter/tools/run_final_training.sh phase1 > logs/phase1b.out 2>&1 &   # macht weiter, wo der Cache steht
```

Jede Instanz liegt einzeln im Cache. Ein Neustart labelt nur die fehlenden.

Optional, wenn nach Phase 1 noch Zeit ist (10 weitere Stunden Labeln, dann
neues Training und neue Benchmarks):

```bash
nohup plasma_cutter/tools/run_final_training.sh phase2 > logs/phase2.out 2>&1 &
```

## 6. Ergebnisse einsammeln

```bash
cd ~/work
bash plasma_cutter/tools/collect_results.sh
ls -lh results_final_*.tar.gz
```

Die tar.gz-Datei auf USB oder OneDrive kopieren. Sie enthält alle Labels,
Datensätze, Modelle, Benchmark- und Lernkurven-Dateien und die Logs.

## 7. Zurück auf Windows

1. tar.gz nach `plasma_cutter\segment_simulation\surrogate\artifacts\` entpacken
   (es entstehen `runs\...` und `logs\...`).
2. Aus `artifacts\runs\main\` die Dateien `surrogate_model.joblib` und
   `model_meta.json` nach `artifacts\` kopieren.
3. Simulator starten, Taste S drücken: das Modell wird geladen und geprüft.

Ergebnisse zum Lesen: `runs\main\benchmark.md` (Standard-Segmentierung),
`runs\main\benchmark_fine.md` (feine Segmentierung), `runs\main\learning_curve.md`.

## Stellschrauben (vor dem Start als Umgebungsvariable setzen)

```bash
N_MAIN=3000 nohup plasma_cutter/tools/run_final_training.sh phase1 > logs/phase1.out 2>&1 &   # weniger Instanzen
BUDGET1_MIN=600 ...    # Label-Budget in Minuten (Voreinstellung 900 = 15 h)
KMAX_MAIN=22 ...       # Segmentzahl bis 22 zulassen (deutlich teurer)
NJOBS=40 ...           # Anzahl paralleler Worker (Voreinstellung: Threads minus 4)
```

Nichts an den Physikwerten oder am Katalog ändern, sobald der Lauf gestartet
ist: jede Änderung macht die bis dahin erzeugten Labels ungültig.

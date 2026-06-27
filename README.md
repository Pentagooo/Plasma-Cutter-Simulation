# Plasmaschneider — Bahnplaner & Segment-Simulation

Bahnplaner für einen roboter geführten Plasmaschneider. Aus einer gegebenen
Bauteil geometrie (Außenkontur + Löcher) berechnet das System automatisch
einen **vollständigen, zeitoptimalen und kollisionsfreien Schneidplan** —
Ziel: deutlich unter einer Sekunde Planzeit.



## Worum geht es?

Die zentrale Frage ist ein **Überdeckungs- und Reihenfolge-Problem**:

> *Welche Stücke der Kontur muss ich schneiden, in welcher Reihenfolge, um
> mit minimaler Zeit 100 % Materialtrennung zu erreichen?*

**Schneidmodell** Der TCP fährt auf einem Offset-Pfad mit
konstantem Mindestabstand `MINIMUM_GAP = 3 mm` zum Material. Die Plasmaklinge
ragt Richtung Objekt mit geschwindigkeits abhängiger Länge
`L(v) = blade_length − blade_slope · v` (linear, Parameter einstellbar).
**Coverage = Querschnittsfläche** (alle abgedeckten Gitterpunkte via
Swept Areas), nicht nur die Kontur. **Bewertung:**
`score = coverage · max(0, 1000 − Zeit)` 

---

## Status

| Baustein | Status | Datei |
|---|---|---|
| Segmentierung + Klick-Auswahl (Start-/Endknoten) | ✅ | `segment_simulation/segments.py` |
| Coverage über Querschnittsfläche (Swept Areas) | ✅ | `segment_simulation/segments.py` |
| Kinematik (Offset-Ring, Klingen-Fächer) | ✅ | `segment_simulation/planning.py` |
| LinkPlanner (Sichtbarkeitsgraph + Dijkstra, kollisionsfrei) | ✅ | `segment_simulation/planning.py` |
| Sequencer zeitminimal (Held-Karp) | ✅ | `segment_simulation/planning.py` |
| **Auto-Planer** (Greedy Set-Cover + Pruning + Merge) | ✅ | `segment_simulation/autoplan.py` |
| Interaktive UI mit Animation | ✅ | `segment_simulation/simulation.py` |
| Headless-Einstieg (`auto_plan`, `check_segments`) | ✅ | `segment_simulation/` |

Alle 6 Testgeometrien laufen durch, jeweils **< 0,15 s** (A3 erfüllt).
4 Geometrien erreichen 100 % Coverage
Gegenüber der „alles schneiden"-Baseline: **40–50 %
Zeitersparnis** bei gleicher Coverage.

Offene Punkte stehen in der [Roadmap](#roadmap).

---

## Schnellstart

**Voraussetzung:** Python ≥ 3.10 (getestet mit 3.14). Abhängigkeiten:
numpy, shapely, matplotlib.

```bash
# 1. Repository klonen und hineinwechseln (der Ordner MUSS "plasma_cutter" heißen,
#    da intern absolut als "plasma_cutter.…" importiert wird)
git clone <repo-url> plasma_cutter
cd plasma_cutter

# 2. Virtuelle Umgebung + Abhängigkeiten
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
python -m pip install -r requirements.txt
```

### Interaktive Simulation

```bash
# Vom ELTERN-Verzeichnis von plasma_cutter starten (damit das Paket importierbar ist):
python -m plasma_cutter.segment_simulation.simulation

# mit Parametern
python -m plasma_cutter.segment_simulation.simulation \
    --v-cut 5 --v-max 10 --blade-length 27.5 --blade-slope 1.5 --clearance 3
```

**Tastenbelegung:** Klick = Start-/Endknoten setzen · `A` = alles ·
`P` = Auto-Planer · `Enter` = planen + animieren · `U` / Rechtsklick = Undo ·
`R` / Button = Reset.

### Headless (Skript / Demo / Test)

```python
from plasma_cutter.geometry.point_grid import PointGrid
from plasma_cutter.segment_simulation.autoplan import auto_plan

grid   = PointGrid.from_json("geometry/Geometrie_Konturen_geprüft/kontur.json")
result = auto_plan(grid)
print(result.summary())   # Coverage, Score, Anzahl Schnitte, Laufzeit
```

---

## Projektstruktur

```
plasma_cutter/
├── segment_simulation/        ← (Bahnplaner + Simulation)
│   ├── segments.py            Segmentierung, CutRun, Coverage
│   ├── planning.py            Kinematik, LinkPlanner, Sequencer, Score
│   ├── autoplan.py            Auto-Planer (Greedy Set-Cover + Pruning + Merge)
│   ├── simulation.py          interaktive UI + check_segments()
│   ├── benchmark_autoplan.py  ┐
│   ├── stage_timing.py        ├ Analyse-Werkzeuge
│   ├── visualize_autoplan.py  ┘
│   ├── benchmark_results/     erzeugte Plots
│   └── *.md                   Architektur- & Konzept-Dokumentation (s. u.)
├── cutter/                    Physikmodell
│   ├── cutter.py              Geschwindigkeiten, L(v), Tiefen, Zeiten
│   └── assumptions.py         BladeLengthModel, PierceTimeModel, TorchTiltDistribution …
├── geometry/                  Geometrie-Ein-/Aufbereitung
│   ├── point_grid.py          PointGrid (Punktgitter, JSON-Laden)
│   ├── geometry_processor.py  Kontur einlesen/verdichten/innere Punkte
│   └── Geometrie_Konturen_*/  Testgeometrien (JSON)
├── requirements.txt
└── README.md                  ← diese Datei
```

## Dokumentation

Die Quelldateien sind durchgängig deutsch und zweischichtig kommentiert
(erst *was passiert*, dann *wie im Code umgesetzt*)

## Roadmap

1. **v(Segment)-Regel** 

2. **Robustheit**  — TCP-Störmodell des Roboters 
3. **BA-Schreiben**
4. **Supervised Learning**

---


---

*Kontakt: Max Meyer · IGMR, RWTH Aachen*
# Plasma-Cutter — Konzeptnotizen: Segment-Simulation & Bachelorarbeit

Stand: 2026-06-10. Gedächtnisnotiz für Max & Claude — fasst zusammen, was
gebaut wurde, welche Modellannahmen gelten, wie der automatische Planer
entstehen soll und wie das Ganze als **Maschinenbau-Bachelorarbeit**
(IGMR, RWTH) aufgebaut wird.

---

## 1. Was bisher gebaut wurde (`segment_simulation/`)

Neuentwicklung der Simulation (alter Lichtschwert-Stack in `cutter/` und
`simulation/` blieb unverändert). Drei Module:

| Modul | Inhalt |
|---|---|
| `segments.py` | Konturen als Loops, automatische Segmentierung (Ecken + Ziellänge), Klick-Snapping auf Knoten, `CutRun`, Kontur- und **Querschnitts-Coverage** (`compute_grid_coverage`) |
| `planning.py` | `RunKinematics` (TCP-Offset-Pfad, Klinge, Swept Area), `LinkPlanner` (kollisionsfreie Verfahrwege), `Sequencer` (zeitminimale Reihenfolge), `compute_score` |
| `simulation.py` | Interaktive Matplotlib-UI, Animation, Headless-API `check_segments()`, CLI |

Start: `python -m plasma_cutter.segment_simulation.simulation`
(Optionen: `--geometry`, `--v-cut`, `--v-max`, `--blade-length`,
`--blade-slope`, `--clearance`, `--kerf-width`, `--segment-length`)

### Physikalisches Modell (nach Max' Korrekturen)

- **Konstanter Mindestabstand** `MINIMUM_GAP = 3.0 mm`: Der TCP fährt NIE
  auf der Kontur, sondern auf dem Offset-Pfad (Rand von
  `material.buffer(gap)`, inkl. abgerundeter Ecken).
- **Klinge linear geschwindigkeitsabhängig**:
  `L(v) = blade_length − blade_slope · v` (Defaults: 27.5 mm, 1.5 mm/(mm/s)
  → L(5) = 20 mm). Einstellbar; `Cutter` hat neuen optionalen Parameter
  `max_cutting_speed` (v_cut > v_max wirft Fehler).
  Effektive Schnitttiefe ab Oberfläche = L(v) − gap.
- **Ziel = Querschnittsfläche**: Coverage zählt ALLE Gitterpunkte
  (innen + außen), abgedeckt über die Swept Areas der Klinge
  (Vereinigung der Klingen-Vierecke pro Pfadkante + Kerf-Puffer).
  Nicht erreichbare Tiefen werden ehrlich als fehlend gemeldet
  (z. B. test_lochjson: innerste Punkte 55 mm tief, Klinge 17 mm → 37 %).
- **Punktezahl**: `score = coverage · max(0, 1000 − 1.0 · Gesamtzeit)`
  (`SCORE_BASE`, `SCORE_TIME_WEIGHT` in planning.py) — besteht vor allem
  aus der Zeit, weniger ist besser; Coverage-Faktor verhindert
  „nichts schneiden = gut“.

### Planner (Sequencer) — zeitminimal

- Reihenfolge UND Schnittrichtung jedes Segments frei (Start/Ende
  vertauschbar, z. B. E1→S1→S2→E2 …).
- Bis `EXACT_MAX_RUNS = 11` Segmente **exakt** per Held-Karp-DP über
  (besuchte Menge, letztes Segment, Richtung); gegen Brute-Force
  verifiziert. Darüber Heuristik (NN + 2-opt), gleiche Kostenfunktion.
- Übergangskosten = ECHTE LinkPlanner-Pfade (Sichtbarkeitsgraph +
  Dijkstra um das gepufferte Material, Überflug-Fallback mit
  2 s-Pauschale bei umschlossenen Löchern) + Pierce-Zeit. Gecacht
  (symmetrisch). Nahtloser Anschluss = 0 Kosten → Verkettung ohne
  Neuzündung entsteht von selbst.
- `CutPlan.is_optimal` zeigt exakt/heuristisch (UI-Zeile „Optimalität“).

---

## 2. Ziel & Anforderungen an den Planer

**Endziel:** Ein vollwertiger Planer — Geometrie rein, ausführbarer
Schnittplan raus — der in **< 1 Sekunde** einen optimalen oder
fast-optimalen Pfad findet.

| # | Anforderung | Bedeutung | Status |
|---|---|---|---|
| A1 | **Vollständigkeit** | 100 % Durchtrennung des Querschnitts (Coverage) — oder ehrliche Meldung, was physikalisch unerreichbar ist | Coverage-Prüfung fertig; Auto-Wahl offen |
| A2 | **Optimalität** | zeitminimal oder beweisbar nah dran (Optimalitätsgap angeben) | Sequencing exakt (Held-Karp ≤ 11); Segment**wahl** offen |
| A3 | **Echtzeitfähigkeit** | Geometrie → Plan in < 1 s | Sequencing ~0,1 s; Auto-Wahl muss Budget einhalten |
| A4 | **Robustheit** | Plan erfüllt A1 auch unter realen Prozessabweichungen (Brenner, Roboter, Geometrie) | offen — Störmodell Brenner existiert schon (`assumptions.py`) |

Etappen (Max' ursprüngliche Formulierung):

1. **Minimalanforderung**: gegebene Segmente auf Feasibility prüfen und
   in optimale Reihenfolge bringen → **ERLEDIGT** (`check_segments()` ist
   der Headless-Einstiegspunkt, auch für späteres Lernen).
2. **Wunschziel**: individueller Primitivplaner für beliebige Objekte —
   Geometrie rein, optimale Segmentwahl raus → technischer Plan in Kap. 4.

---

## 3. Bachelorarbeit (Maschinenbau): Roter Faden

**Einordnung:** Maschinenbau-Arbeit (IGMR), keine reine Informatik.
Der wissenschaftliche Kern ist die **automatisierte Prozessplanung für
robotergeführtes Plasmaschneiden** — Modellbildung, Prozessverständnis,
Parametereinflüsse, fertigungstechnische Bewertung. Algorithmen/ML sind
Werkzeuge, deren Auswahl begründet wird, nicht Forschungsgegenstand.

**Roter Faden in einem Satz:** *Ein automatischer Schnittplaner, der
nachweislich vollständig, zeitoptimal und echtzeitfähig plant — und
dessen Pläne durch physikalisch begründete Sicherheitsmargen robust
gegen die in der Praxis unvermeidbaren Prozess- und
Geometrieabweichungen sind.*

**Robustheit ist die vierte Anforderung (A4), kein Nebenthema** — sie
verbindet die State-of-the-Art-Kapitel (Abweichungen) mit dem Planer:
Das Abweichungskapitel liefert die Störmodelle, das
Vollständigkeits-/Optimalitätskapitel die Bewertungsbegriffe. Nichts
hängt in der Luft.

### Kapitelstruktur

1. **Einleitung** — Robotergeführtes Plasmaschneiden von Profilen;
   manuelle Schnittplanung als Engpass; Ziel: automatischer Planer mit
   Anforderungen A1–A4.
2. **Stand der Technik**
   - 2.1 Plasmaschneidprozess: Eigenschaften, Winkel-/Prozessabweichungen
     (ISO 9013, ±2°-Toleranz) → *bereits geschrieben*.
   - 2.2 Industrieroboter: Absolut- vs. Wiederholgenauigkeit (ISO 9283),
     Bahnabweichungen bei kontinuierlichen Bahnen,
     Geschwindigkeitsschwankungen; Abgrenzung: Prozesskräfte beim
     Plasmaschneiden klein → Steifigkeit unkritisch (anders als Fräsen)
     → *zu ergänzen*.
   - 2.3 Pfadplanung: Vollständigkeit & Optimalität (*bereits
     geschrieben*), Coverage-Planung, kombinatorische
     Reihenfolgeplanung.
3. **Modellbildung** — 2D-Querschnittsmodell mit Annahmen (L(v) linear,
   Pierce, Kerf, Mindestabstand; Literaturbezug aus `assumptions.py`)
   **plus Abweichungsmodelle**. Scharnier der Arbeit: Tabelle
   „reale Abweichung (Kap. 2) → Modellgröße (Kap. 3)“, siehe Kap. 5
   unten.
4. **Planungssystem** (eigener Beitrag; Stufe 1 ist implementiert)
   - 4.1 Segmentierung & Feasibility-Prüfung
   - 4.2 Reihenfolge + Richtung: Held-Karp exakt (→ erfüllt A2, knüpft
     an Kap.-2.3-Begriffe an), Heuristik-Fallback mit Gap-Angabe
   - 4.3 Verfahrwegplanung: Sichtbarkeitsgraph (vollständig in 2D →
     A1-Argument), Überflug-Fallback
   - 4.4 **Automatische Segmentwahl** (das Neue, Plan in Kap. 4 unten)
   - 4.5 Geschwindigkeitswahl je Segment aus benötigter Schnitttiefe
     (Distanztransformation) — erzeugt den Margen-Hebel für Kap. 5
5. **Robustheitsanalyse** (zweiter eigener Beitrag,
   Alleinstellungsmerkmal) — Plan in Kap. 5 unten.
6. **Ergebnisse** — Laufzeitnachweis (< 1 s über Geometriekatalog),
   Optimalitätsgap (gegen Held-Karp/ILP auf kleinen Instanzen),
   Parameterstudien, Robustheitsstudien.
7. **Diskussion & Ausblick** — gelernter Planer, 3D-Profile, reale
   Schneidversuche.

### Änderungsvorschläge zum bisherigen Schreibstand

1. **Robustheit zu Anforderung A4 befördern** — sonst bleibt das
   Abweichungskapitel Referat ohne Verwendung.
2. **Vollständigkeits-/Optimalitätskapitel rückkoppeln:** Am Ende von
   Kap. 4 jede Eigenschaft explizit einlösen (Held-Karp = optimal,
   Sichtbarkeitsgraph = vollständig, Greedy = ln(n)-Approximation,
   Repair = Coverage-Garantie). Verwandelt Theorie in den
   Bewertungsmaßstab des eigenen Systems.
3. **„Perfekte Geometrie“-Problem als Untersuchungsfrage formulieren:**
   „Wie empfindlich ist ein auf Nominalgeometrie geplanter Schnittplan
   gegenüber realen Abweichungen, und wie viel Marge macht ihn robust?“
   Gestörte Geometrien = Verrauschen der Konturpunkte in der
   bestehenden Simulation.
4. **ML herabstufen** auf optionalen Vergleich/Ausblick — für < 1 s
   reicht Greedy + Pruning (+ SA mit Zeitbudget) bei K ≈ 20 locker;
   die Robustheitsanalyse trägt in einer Maschinenbau-Arbeit mehr.
5. **Vorhandenes nutzen:** `TorchTiltDistribution` (OU-Prozess,
   ISO-9013-begründet) ist das fertige Brenner-Störmodell; ein analoges
   TCP-Störmodell für den Roboter ist eine kleine Ergänzung.

### Physikalischer Clou (Kopplung A2 ↔ A4)

**Robustheit kostet Zeit:** Mehr Sicherheitsmarge auf die Schnitttiefe
heißt langsamer fahren (L(v)!), mehr Marge beim Mindestabstand heißt
längere Wege. → Pareto-Front „Fertigungszeit vs. tolerierbare
Abweichung“ ist *das* Ergebnisdiagramm der Arbeit.

---

## 4. Technischer Plan: Automatische Segmentwahl (BA-Kap. 4.4)

### Problemformulierung (wichtigste Erkenntnis)

- Es reicht eine **binäre Entscheidung pro Primitiv-Segment:
  schneiden / überspringen** (Bitvektor, K ≈ 13–25). Zusammenhängende
  „schneiden“-Segmente verschmelzen automatisch zu Runs (Verkettung).
  Strukturell ein Covering-Tour-Problem (Set Cover + TSP), NP-hart —
  in der BA ein Absatz Einordnung, kein Theoriekapitel.
- **Distanztransformation / Medialachse** als geometrisches Werkzeug:
  Punkt p ist von Segment s abdeckbar ⇔ dist(p, s) ≤ eff. Tiefe.
  Einmal vorberechnen (Abdeckbarkeits-Matrix A[p, s]) → Coverage-Check
  wird reine Mengenoperation, kein Shapely im Optimierungs-Loop.
  Schlüssel für das < 1-s-Budget.

### Lösungsweg für < 1 s (Kern der Arbeit)

- **Greedy Set-Cover**: wiederholt Segment mit bestem Verhältnis
  „neue Punkte / Zusatzzeit“ wählen bis 100 %, danach **Pruning**
  (redundante Segmente entfernen — z. B. dünner Steg muss nur von
  EINER Seite geschnitten werden). Approximationsgüte ln(n) → A2-Bezug.
- **+ Simulated Annealing mit Zeitbudget** über den Bitvektor; Fitness =
  echter Score (Coverage + Held-Karp, ~0,1 s/Auswertung). Budget so
  wählen, dass Gesamtlaufzeit < 1 s bleibt.
- **ILP / Brute-Force offline** (OR-Tools) nur als Referenz, um den
  Optimalitätsgap des schnellen Planers auf kleinen Instanzen zu
  beziffern → Ergebniskapitel.

### Geschwindigkeit je Segment (BA-Kap. 4.5)

Deterministische Regel statt Lernen: benötigte Tiefe je Segment aus der
Distanztransformation (max. Distanz der NUR von diesem Segment
abdeckbaren Punkte) →
`v = (blade_length − gap − d_benötigt − Marge) / blade_slope`.
Dick = langsam, dünn = schnell. Geschlossen herleitbar; die **Marge**
ist der Stellhebel für Kap. 5.

### Gelernte Segmentwahl (herabgestuft: Ausblick / optionaler Vergleich)

Falls Zeit übrig oder als Ausblick-Kapitel — Kurzfassung des früheren
Brainstormings:

- Pipeline „learning to propose, optimizer to verify“: Zufallsgeometrien
  + Optimierer-Labels → Modell schlägt Bitvektor vor → **Repair-Loop**
  (fehlende Punkte greedy nachdecken, Redundanz entfernen) → exakter
  Sequencer. Coverage-Garantie kommt IMMER vom Repair, nie vom Modell.
- Wenn ML, dann **ein** erklärbares Verfahren: Gradient Boosting auf
  physikalisch interpretierbaren Merkmalen (Länge, Krümmung, lokale
  Materialdicke, Anteil exklusiv erreichbarer Punkte) —
  Feature-Importance lässt sich fertigungstechnisch deuten.
  (GNN/U-Net/Transformer nur als Ausblick erwähnen.)
- Stolperstein **mehrdeutige Labels** (Symmetrie: Steg links ODER
  rechts): mehrere Optima sammeln → Soft-Labels, oder marginalen
  Score-Beitrag als Regressionsziel.
- RL: instabiler & datenhungriger als Supervised bei billigen
  Optimierer-Labels → höchstens Ausblick.

### Ehrliche Einordnung

Bei K ≈ 20 Segmenten ist der klassische Optimierer schnell genug für
alle Anforderungen. Lernen begründet sich nur über feinere
Segmentierung, Freiformgeometrie oder als wissenschaftlicher Vergleich —
fürs BA-Ziel (< 1 s, fast-optimal) ist es NICHT nötig.

---

## 5. Robustheitsanalyse (BA-Kap. 5)

### Abweichungs → Modellgrößen-Tabelle (Scharnier zwischen Kap. 2 und 3)

| Reale Abweichung (Kap. 2) | Modellgröße in der Simulation | Quelle/Norm | Status |
|---|---|---|---|
| Brennerwinkel-Streuung | `TorchTiltDistribution` (abgeschnittene Normalverteilung ±2°, OU-Prozess entlang des Pfads) → effektive Tiefe/Swept Area variiert | ISO 9013, Hypertherm | **existiert** in `assumptions.py` |
| Roboter-Bahnfehler (TCP) | additives Rauschen auf TCP-Pfad (z. B. OU, Amplitude 0,1–0,5 mm); Wiederhol- vs. Absolutgenauigkeit | ISO 9283 | zu ergänzen (analog zum Tilt-Modell) |
| Geometrieabweichung Werkstück | Verrauschen/Skalieren der Konturpunkte; Lageversatz | Walztoleranzen (z. B. EN 10279) | trivial in Simulation |
| Blechdicken-/Materialstreuung | Streuung von L(v)-Parametern bzw. Pierce-Zeit | Cut-Charts | Parameter vorhanden |

### Vorgehen

1. **Monte-Carlo-Bewertung:** Nominalen Plan unter N gestörten
   Realisierungen simulieren → Verteilung der Coverage; „ab welcher
   Störamplitude verliert der Plan Punkte?“
2. **Robustheitskennzahl definieren:** z. B. *Robustheitsradius* =
   minimale Störamplitude, bei der Coverage < 100 % fällt; oder
   probabilistisch P(Coverage = 100 %) ≥ 99 %.
3. **Robuste Planung:** Sicherheitsmarge auf effektive Tiefe und
   Mindestabstand als Entwurfsparameter; Planer mit Marge erneut laufen
   lassen.
4. **Trade-off quantifizieren:** Pareto-Front Fertigungszeit vs.
   Robustheitsradius (Kopplung über L(v), siehe Kap. 3).

Die Robustheits-Pipeline ist größtenteils **Auswertung, nicht Neubau** —
Störmodelle injizieren und die vorhandene Coverage-/Zeit-Rechnung
wiederverwenden.

---

## 6. Anhang: Planner-Pseudocode (Reihenfolge + Schnittrichtung)

Wie der Sequencer (`planning.py`) die zeitminimale Reihenfolge und
Richtung findet. Grundidee: Die Schnittzeiten der Segmente sind
konstant (unabhängig von Reihenfolge/Richtung) — zeitminimal heißt
also: **minimiere die Summe der Übergänge** (Eilgang + Zündungen).
Jedes Segment hat 2 Varianten: vorwärts (Start→Ende) oder rückwärts
(Ende→Start).

### 6.1 Übergangskosten (Kostenfunktion)

```text
FUNKTION trans_cost(ende_A, start_B):                    # Zeit [s] von Run A zu Run B
    WENN |start_B − ende_A| < TOLERANZ:
        RETURN 0                                         # nahtloser Anschluss:
                                                         # Brenner bleibt an → kein
                                                         # Eilgang, keine Zündung
    link = link_between(ende_A, start_B)                 # ECHTER kollisionsfreier Pfad
    RETURN zeit(link) + pierce_zeit                      # Eilgang + neue Zündung

FUNKTION link_between(a, b):                             # mit Cache
    WENN (a,b) oder (b,a) im Cache: RETURN (ggf. umgedreht)
    WENN Strecke a→b frei (schneidet gepuffertes Material nicht):
        pfad = [a, b]                                    # Direktverbindung
    SONST WENN Dijkstra über Sichtbarkeitsgraph findet Weg:
        pfad = kürzester Weg um das Material herum
    SONST:
        pfad = [a, b], ÜBERFLUG                          # Z-Hub, +2 s Pauschale
    Cache speichern; RETURN pfad

zeit(link) = länge / v_eilgang   (+ 2 s falls Überflug)
```

### 6.2 Hauptverzweigung

```text
FUNKTION order_runs(runs):
    WENN n ≤ 11:  RETURN exakt_held_karp(runs),  optimal = WAHR
    SONST:        RETURN heuristik_nn_2opt(runs), optimal = FALSCH
```

### 6.3 Exakt: Held-Karp-DP über (Menge, letztes Segment, Richtung)

```text
FUNKTION exakt_held_karp(runs):
    # Varianten: variant[i][0] = Run i vorwärts, variant[i][1] = rückwärts
    # (rückwärts = Start-/Endpunkt vertauscht, Pfad gespiegelt)

    # Schritt 1: Kostenmatrix für ALLE Übergänge vorberechnen
    FÜR i, oi, j, oj  (i ≠ j):
        cost[i][oi][j][oj] = trans_cost( ENDE von variant[i][oi],
                                         START von variant[j][oj] )

    # Schritt 2: Dynamische Programmierung
    # Zustand: (mask = Menge bereits geschnittener Runs,
    #           j    = zuletzt geschnittener Run,
    #           oj   = dessen Richtung)
    # dp[mask][j][oj] = minimale Übergangszeit, um genau die Runs in
    #                   mask zu schneiden und bei (j, oj) zu enden

    dp[·][·][·] = ∞
    FÜR jeden Run i, jede Richtung o:
        dp[{i}][i][o] = 0                                # jeder darf Erster sein

    FÜR mask = kleine → große Mengen:
        FÜR jeden Zustand (mask, j, oj) mit dp < ∞:
            FÜR jeden Run k ∉ mask, jede Richtung ok:
                neu = dp[mask][j][oj] + cost[j][oj][k][ok]
                WENN neu < dp[mask ∪ {k}][k][ok]:
                    dp[mask ∪ {k}][k][ok] = neu
                    parent[mask ∪ {k}, k, ok] = (mask, j, oj)   # für Rekonstruktion

    # Schritt 3: bestes Ende suchen, Pfad rückwärts ablaufen
    (j*, oj*) = argmin dp[VOLLE_MENGE][·][·]
    folge = parent-Kette von (VOLLE_MENGE, j*, oj*) zurück bis zum Start, umdrehen
    RETURN [variant[j][oj] für (j, oj) in folge]
```

Warum exakt: Jede mögliche Tour entspricht genau einer Folge von
DP-Übergängen, und der Zustand merkt sich alles, was für die Zukunft
relevant ist (welche Runs fehlen + wo der Brenner steht).
Aufwand `2ⁿ · n · 2` Zustände — bei n = 11 ca. 0,1 s, statt
`n! · 2ⁿ` beim Durchprobieren.

### 6.4 Heuristik (Fallback ab 12 Segmenten)

```text
FUNKTION heuristik_nn_2opt(runs):
    beste_folge = NICHTS
    FÜR jeden Run i, jede Richtung o:                    # jeder Startkandidat
        folge = [variant[i][o]]
        SOLANGE Runs übrig:
            hänge den Run+Richtung mit den geringsten
            trans_cost(aktuelles_ende, kandidat_start) an   # Nearest Neighbour
        merke folge, wenn Gesamtkosten besser

    # Lokale Verbesserung bis nichts mehr hilft:
    WIEDERHOLE:
        FÜR jeden Run in beste_folge:
            Richtung umdrehen → besser? behalten
        FÜR jedes Teilstück [i..j]:
            Teilstück umkehren UND alle Richtungen darin flippen   # 2-opt
            → besser? behalten
    RETURN beste_folge
```

### 6.5 Plan zusammenbauen

```text
FUNKTION build_plan(runs):
    geordnet, optimal = order_runs(runs)
    FÜR jeden Run in geordnet:
        WENN Vorgänger-Ende == Run-Start:                # verkettete Segmente
            Schnitt OHNE neue Zündung anhängen
        SONST:
            Link-Schritt (aus Cache) + Schnitt MIT Zündung anhängen
    Zeiten summieren (Schnitt / Eilgang / Pierce), is_optimal = optimal
```

**Der entscheidende Trick** (Beispiel „E1→S1→S2→E2→…"): Die
Richtungsfreiheit steckt komplett in den 2 Varianten pro Segment —
der DP-Zustand `(Menge, letzter Run, Richtung)` probiert alle
Kombinationen implizit durch, ohne sie aufzuzählen. Und weil
verkettete Übergänge 0 kosten, „findet" der Optimierer das
Zusammenlegen benachbarter Segmente zu einem Schnitt mit nur einer
Zündung von ganz allein.

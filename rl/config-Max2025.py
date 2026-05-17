"""Phase 0 -- Ziel- und Erfolgsdefinition fuer das RL-Training.

Diese Datei ist die ZENTRALE Konfigurationsstelle fuer das gesamte
RL-Projekt. Alle nachfolgenden Phasen (Baseline, Env, Reward, Training,
Evaluation) lesen ihre Parameter aus `RLConfig`. Wenn sich eine Design-
entscheidung aendert, wird sie HIER veraendert -- nicht verstreut im
restlichen Code.

Designprinzipien dieser Config
------------------------------
1. **Eine Wahrheit:** Jeder Hyperparameter / jede Schwellwertentscheidung
   existiert genau einmal. Module wie env.py, reward.py, train.py
   importieren `RLConfig`, statt eigene Konstanten zu definieren.

2. **Versionierbar:** `RLConfig.version` markiert jede Reward-/Scope-
   Aenderung. Trainings mit unterschiedlicher Version sind NICHT
   vergleichbar -- die Versionierung zwingt uns, das bewusst zu sehen.

3. **Reproduzierbarkeit:** `RLConfig.seed` wird durch alle stochastischen
   Komponenten propagiert (Geometrie-Sampling, Env-Reset, Policy-Init).

4. **Erweiterbar ohne Bruch:** Neue Optionen kommen als zusaetzliche
   Felder mit Default-Wert hinzu. Bestehende Trainings bleiben gueltig.

Phase-0-Entscheidungen (uebernommene Empfehlungen)
--------------------------------------------------
- **Scope:**         Familie aus `Geometrie_Konturen_geprueft`,
                     mit zufaelliger Skalierung/Rotation.
- **Objective:**     Coverage als HARTER Constraint (>= 99%).
                     Darunter Zeit MINIMIEREN. Pierces sekundaer.
- **Success:**       Auf Test-Konturen >= 99% Coverage UND
                     <= 90% der Zeit der Heuristik-Baseline.
- **Constraints:**   `is_feasible=False` -> Episode-Ende + Strafe.
                     Klingenlaenge fix. Keine Dynamik-Constraints.

Primaeres Ziel des Users (explizit):
    "Zeit minimieren bei >= 99% Coverage."
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Sub-Configs
# ---------------------------------------------------------------------------
#
# Wir gliedern die Gesamt-Config in vier thematische Bloecke. Das hat zwei
# Vorteile:
#   (a) Jeder Block entspricht einer der vier Phase-0-Fragen, dadurch ist
#       die Zuordnung "Designentscheidung -> Konfigfeld" sofort sichtbar.
#   (b) Sub-Configs lassen sich einzeln austauschen (z.B. eine andere
#       ScopeConfig fuer Generalisierungstests, ohne den Rest anzufassen).
# ---------------------------------------------------------------------------


@dataclass
class ScopeConfig:
    """**Auf welcher Klasse von Geometrien soll die Policy funktionieren?**

    Phase-0-Entscheidung: Variante (B) -- Familie aus dem Verzeichnis
    `Geometrie_Konturen_geprueft`, mit zufaelliger Skalierung und
    Rotation als Domain Randomization.

    Begruendung: Eine einzige Kontur (A) waere zu eng und wuerde nicht
    beweisen, dass RL die Heuristik schlagen kann. Volle Allgemeinheit
    (C/D) wuerde zwingend Bild-State + CNN erfordern und das Training
    in die Wochen-Dimension treiben. (B) ist der pragmatische Mittelweg:
    realistisch genug fuer interessante Resultate, klein genug fuer
    schnelle Iterationen.
    """

    # Verzeichnis mit den Trainingsgeometrien (relativ zum Projekt-Root).
    # Spaeter kann hier z.B. auf "Geometrie_Konturen_ungepruefte" oder
    # einen synthetischen Generator umgeschaltet werden.
    geometry_dir: str = "geometry/Geometrie_Konturen_geprüft"

    # Anteil der Geometrien, der als Test-Set zurueckgehalten wird
    # (NIE im Training gesehen). Wird fuer Phase-8-Evaluation gebraucht.
    test_split: float = 0.2

    # Domain Randomization: Bei jedem reset() wird die geladene Kontur
    # zufaellig skaliert/rotiert, damit die Policy nicht auf absolute
    # Koordinaten overfittet. Bereiche bewusst klein gewaehlt, damit
    # die Phase-1-Baseline weiter realistisch bleibt.
    randomize_scale: bool = True
    scale_range: tuple[float, float] = (0.8, 1.2)   # multiplikativ

    randomize_rotation: bool = True
    rotation_range_deg: tuple[float, float] = (-180.0, 180.0)

    # Optionaler Override: nur eine einzige Geometrie verwenden
    # (nuetzlich fuer Phase 5 -- Sanity-Check "Lernbarkeit auf einem Fall").
    # None = volles Verzeichnis verwenden.
    single_geometry_file: str | None = None


@dataclass
class ObjectiveConfig:
    """**Was soll die Policy primaer optimieren?**

    Phase-0-Entscheidung:
        Coverage ist HARTER Constraint (>= `coverage_threshold`).
        Darunter wird Zeit MINIMIERT. Pierces sind sekundaer.

    Begruendung: Der User hat das primaere Ziel explizit so formuliert.
    Coverage als Constraint statt als gewichtetes Reward-Ziel zu modellieren
    verhindert das klassische Failure Mode "Agent opfert Coverage fuer
    Zeit". Die Umsetzung im Reward ist:

        - Pro Step: + Δcoverage  (positives Signal fuer jeden Schnitt)
                    -  ε * Δzeit  (kontinuierlicher Zeitdruck)
        - Bei Constraint-Verletzung am Episodenende: grosse Strafe
        - Bei Erfolg (Coverage >= Threshold): grosser Bonus, je
          schneller desto hoeher.

    Die exakten Gewichte sind Phase-2-Thema und stehen daher in
    `RewardConfig` (folgt spaeter). Hier definieren wir nur das
    konzeptionelle Ziel und die Schwellwerte.
    """

    # Harter Coverage-Constraint. Unterhalb dieser Schwelle gilt eine
    # Episode als FAILURE, egal wie schnell sie war.
    coverage_threshold: float = 0.99

    # Welche Groesse ist die eigentliche Optimierungsvariable?
    # 'time'         -- minimiere total_time (UNSER FALL)
    # 'pierces'      -- minimiere Anzahl Zuendungen
    # 'path_length' -- minimiere Pfadlaenge
    # 'weighted'     -- gewichtete Summe (siehe weights unten)
    primary_objective: Literal["time", "pierces", "path_length", "weighted"] = "time"

    # Sekundaere Gewichte. Werden NUR verwendet, wenn primary_objective
    # auf 'weighted' steht ODER um in der Reward-Funktion sekundaere
    # Terme zu addieren (kleine Gewichte!). Bewusst klein gegenueber
    # dem Zeitterm, damit Zeit dominant bleibt.
    weight_time:        float = 1.0   # mm/s -> dimensionslos via Normalisierung in Phase 2
    weight_pierces:     float = 0.05  # pro zusaetzlicher Zuendung
    weight_path_length: float = 0.0   # default: irrelevant


@dataclass
class SuccessConfig:
    """**Wann gilt das Training als erfolgreich abgeschlossen?**

    Phase-0-Entscheidung:
        Auf den zurueckgehaltenen Test-Konturen muss die Policy
          (1) >= `coverage_target` Coverage erreichen UND
          (2) <= `time_ratio_vs_baseline` * Heuristik-Zeit benoetigen
        und das auf mindestens `min_success_rate` der Test-Konturen.

    Begruendung: Eine harte, ueberpruefbare Definition ist Pflicht --
    sonst trainiert man unendlich. Der Vergleich gegen die Heuristik
    (Phase 1) statt gegen Absolutzahlen ist robust gegenueber
    Geometrie-Variationen: 30 Sekunden sind fuer ein kleines Werkstueck
    viel, fuer ein grosses wenig.
    """

    # Coverage, die JEDE Test-Episode erreichen soll, damit sie als
    # "erfolgreich" zaehlt. Bewusst leicht ueber dem Reward-Constraint
    # (0.99), damit das Erfolgskriterium strenger als die Reward-Schwelle
    # ist und keine Knapp-Erfolge durchrutschen.
    coverage_target: float = 0.99

    # Wie schnell muss die Policy gegenueber der Heuristik-Baseline
    # sein? 0.9 = mindestens 10% schneller.
    time_ratio_vs_baseline: float = 0.9

    # Anteil der Test-Konturen, auf denen beide Bedingungen erfuellt
    # sein muessen, damit das gesamte Training als "fertig" gilt.
    min_success_rate: float = 0.95


@dataclass
class ConstraintsConfig:
    """**Welche harten physikalischen Bedingungen darf die Policy nie verletzen?**

    Phase-0-Entscheidung:
        - `is_feasible=False` (TCP zu nah/im Material): Episode endet
          sofort mit grosser terminaler Strafe.
        - Klingenlaenge ist FIX (kein Action-Freiheitsgrad).
        - Keine Dynamik-Constraints (Beschleunigung/Geschwindigkeit
          werden im aktuellen Modell nicht abgebildet -- `Cutter` ist
          rein laengenbasiert).

    Begruendung: Constraints, die nicht hart erzwungen werden, lernt
    der Agent zu ignorieren ("Reward Hacking"). Der `is_feasible`-
    Check ist bereits in `ContinuousPlanner._validate_grip_path`
    implementiert -- wir muessen nur dafuer sorgen, dass die Env das
    Signal in eine Termination + Strafe uebersetzt.
    """

    # Wenn der Planner einen Pfad als nicht ausfuehrbar markiert
    # (TCP zu nah am Material o.ae.), wird die Episode sofort beendet.
    terminate_on_infeasible: bool = True

    # Skalare Strafe, die in diesem Fall am Episodenende vergeben wird.
    # Muss deutlich groesser sein als jeder positive Reward, den der
    # Agent durch "Schummeln" erreichen koennte. Mit reduzierter
    # Reward-Skala (w_coverage_delta=50) ist -10 immer noch das
    # Doppelte des max. positiven Returns (~5).
    infeasible_penalty: float = -10.0    # war: -100.0

    # Maximale Episode-Laenge in Steps. Verhindert unendliche Episoden
    # (Trainings-Killer). 400 Steps erlauben komplexere Trajektorien
    # und verhindern, dass der Agent vor Erreichen der Coverage-Schwelle
    # per truncated abgebrochen wird.
    max_steps_per_episode: int = 400

    # Maximales Zeitbudget pro Episode in Sekunden Schnittzeit.
    # Wenn die Policy bis hierhin die Coverage nicht erreicht hat,
    # gilt die Episode als FAILURE (nicht als technischer Abbruch).
    # 'None' = kein Limit.
    max_cutting_time: float | None = None


# ---------------------------------------------------------------------------
# Phase 2 -- MDP-Design (Action / Observation / Reward)
# ---------------------------------------------------------------------------
#
# Die folgenden drei Configs kapseln die Phase-2-Entscheidungen. Sie sind
# bewusst von der Gym-Env-Implementierung (Phase 3) getrennt, damit man
# Hyperparameter des MDP aendern kann, ohne die Env-Wrapping-Logik
# anzufassen.
#
# Designwahl (User-Vorgabe):
#   - Action-Space   = waypoint-basiert (Agent setzt naechsten Wegpunkt)
#   - Observation    = Multi-Channel-Bild (fuer spaetere Uebertragung
#                      auf Punktwolken)
#   - Ziel           = Zeiteffizienz bei ~100% Coverage
# ---------------------------------------------------------------------------


@dataclass
class ActionConfig:
    """**Wie waehlt der Agent seine Aktionen?**

    Der Action-Space ist WAYPOINT-BASIERT:
        Pro Step setzt der Agent den naechsten TCP-Waypoint
        (relativ zur aktuellen Position) und entscheidet, ob auf
        dem Weg dorthin das Plasma ein ("cut") oder aus ("rapid") ist.

    Numerisch ist die Aktion ein Box-Vector im Intervall [-1, +1]:

        action[0] = Δx   -- normalisiert auf [-1, 1]
        action[1] = Δy   -- normalisiert auf [-1, 1]
        action[2] = cut  -- > 0 bedeutet schneiden, <= 0 bedeutet Eilgang
                           (nur vorhanden wenn include_cut_flag=True)

    Die Δ-Werte werden beim Decodieren in Weltkoordinaten (mm)
    skaliert, und zwar proportional zur GEOMETRIE-GROESSE. Das macht
    die Policy skalierungs-invariant: ein Agent, der auf einem
    100-mm-Werkstueck gelernt hat, verhaelt sich auf einem 200-mm-
    Werkstueck automatisch analog.
    """

    # Max. Schrittweite als Bruchteil der Bounding-Box-Diagonale der
    # aktuellen Geometrie. 0.15 = ein Step ueberbrueckt max. 15% der
    # Geometrie-Diagonale. Bei einer 200-mm-Diagonale sind das 30 mm --
    # etwa 1.5x die Klingenlaenge, was sich gut mit dem Blade-Sweep
    # ueberlappt.
    max_step_fraction: float = 0.15

    # Optionales hartes Minimum (in mm), damit auf sehr kleinen Geometrien
    # der Agent nicht in Sub-mm-Schritten herumkriecht.
    min_step_absolute_mm: float = 2.0

    # Optionales hartes Maximum (in mm). None = nur durch Fraction
    # begrenzt. Nuetzlich wenn man den Agent auf realistische
    # Roboter-Beschleunigungen einschraenken will.
    max_step_absolute_mm: Optional[float] = None

    # Soll der Agent ueberhaupt ueber cut/rapid entscheiden?
    # False = jede Bewegung ist ein Schnitt (einfacherer Action-Space,
    #         aber Agent kann keine Verfahrwege durch Luft nutzen).
    # True  = zusaetzliche Dimension im Action-Vector.
    include_cut_flag: bool = True

    # Schwellwert fuer die kontinuierliche cut-Dimension. Werte > threshold
    # werden als "schneiden" interpretiert. Default 0.0 ist symmetrisch.
    cut_threshold: float = 0.0


@dataclass
class ObservationConfig:
    """**Was sieht der Agent?**

    Die Observation ist ein Multi-Channel-Bild mit fester Aufloesung,
    das die aktuelle Geometrie, den Schnittfortschritt und die TCP-
    Position encodiert. Ein CNN verarbeitet das spaeter in der Policy.

    Channels (in Reihenfolge):
        0: geometry_mask  -- 1 wo Material existiert, 0 sonst.
                             Pro Episode STATISCH.
        1: coverage_mask  -- 1 wo bereits geschnitten wurde, 0 sonst.
                             Waechst monoton im Laufe der Episode.
        2: tcp_channel    -- Gauss-Blob am aktuellen TCP-Standort.
                             Bewegt sich mit jedem Step.

    Warum Bild und nicht Vektor?
    ----------------------------
    Der User moechte das Projekt spaeter auf PUNKTWOLKEN aus echten
    Scans uebertragen. Punktwolken lassen sich trivial zu einem Bild
    rasterisieren -- eine Bild-Policy kann damit ohne Architektur-
    aenderung weiterverwendet werden. Eine Vektor-Policy (Kontur-
    Features) muesste neu designt werden.

    Frame-Wahl
    ----------
    Das Bild wird in einer pro-Episode berechneten Bounding-Box der
    Geometrie gerendert, mit etwas Padding. So sieht der Agent
    immer ein "voll ausgefuelltes Canvas" unabhaengig von der
    absoluten Geometriegroesse -- das ist die Grundlage der
    Skalierungsinvarianz.
    """

    # Bildaufloesung. 64 ist ein guter Start (schnell, CNN-freundlich),
    # 96 oder 128 bei Bedarf spaeter erhoehen.
    resolution: int = 64

    # Padding um die Geometrie-Bbox (Bruchteil der Bbox-Diagonale).
    # 0.15 = 15% Rand links/rechts/oben/unten. Wichtig, weil der TCP
    # den Sicherheitsabstand nach aussen haelt und sonst am Bildrand
    # abgeschnitten werden koennte.
    bbox_padding_fraction: float = 0.15

    # Channel-Toggles. Ermoeglicht Ablation-Studies ("was passiert wenn
    # ich den TCP-Channel weglasse?") ohne Code-Aenderung.
    use_geometry_channel: bool = True
    use_coverage_channel: bool = True
    use_tcp_channel:      bool = True

    # Breite des Gauss-Blobs am TCP-Standort, in Pixeln. Groessere Werte
    # = sanftere Gradienten durch den CNN, kleinere = praezisere
    # Lokalisierung. 2.0 ist ein guter Kompromiss bei 64x64.
    tcp_blob_sigma_px: float = 2.0

    # Maximalwert im Bild. Standard 1.0 (alle Kanaele sind Masken oder
    # normalisierte Blobs). Aenderung nur noetig, wenn man spaeter
    # Float-Werte (z.B. Distanzfelder) als Kanal einbaut.
    clip_max: float = 1.0


@dataclass
class RewardConfig:
    """**Wie wird der Agent belohnt?**

    Reward-Design (Zerlegung der Phase-1-Reward-Formel in Per-Step- und
    Terminal-Terme):

        Pro Step (lineares Potential, seit Phase 5b):
            r_t = w_coverage_delta * (cov_t - cov_{t-1})
                - w_time_delta     * Δzeit_t

        Am Episodenende (zusaetzlich):
            + w_completion  wenn cov >= completion_threshold
            - w_pierce      * max(0, n_pierces - 1)
            + w_failure_penalty  * (1 - cov)   wenn Timeout ohne Erfolg
            = infeasible_penalty               wenn `is_feasible=False`
              (UEBERSCHREIBT die obigen Terme, grosse negative Zahl)

    Warum diese Zerlegung?
    ----------------------
    Die Summe aller Per-Step- + Terminal-Rewards ueber eine Episode
    ergibt EXAKT die Phase-1-`calculate_reward()`-Formel (bis auf
    Failure-/Infeasible-Sonderfaelle). Das hat zwei Vorteile:

    (1) Vergleichbarkeit: ein RL-Return kann direkt mit dem
        Baseline-Reward aus Phase 1 verglichen werden.
    (2) Potential-basiertes Shaping: die dichte Coverage-Belohnung
        pro Step AENDERT die optimale Policy NICHT (Ng et al., 1999),
        beschleunigt aber das Lernen erheblich, weil der Agent
        bereits in der Mitte einer Episode ein positives Signal sieht.

    Warum quadratisch (cov²)?
    -------------------------
    Die letzten Prozente Coverage sind am schwersten und am wichtigsten.
    cov² sorgt dafuer, dass der Sprung von 90% auf 99% mehr belohnt
    wird als der von 0% auf 9%.
    """

    # Dense-Terme (werden pro Step vergeben).
    # Skala um Faktor 10 reduziert: die Value-Function kann bei groesseren
    # Return-Ranges (hunderte) das MSE-Target nicht mehr tracken und der
    # value_loss explodiert -> Advantages werden Muell -> std steigt.
    w_coverage_delta: float = 50.0    # war: 500.0
    w_time_delta:     float = 0.05    # war: 0.5

    # Terminal-Terme (werden nur am Episodenende vergeben).
    w_completion: float = 5.0         # war: 50.0
    w_pierce:     float = 0.2         # war: 2.0

    # Schwellwert fuer den Completion-Bonus. Leicht UNTER dem harten
    # `objective.coverage_threshold` (0.99), damit knappe Erfolge
    # noch den Bonus bekommen und der Reward-Gradient zur Schwelle hin
    # glatt ist.
    completion_threshold: float = 0.99

    # Strafe bei Timeout ohne Erfolg: proportional zur fehlenden Coverage.
    # 5.0 * (1 - cov): max. 5 bei 0%, 0 bei 100%. Gleiche Skala wie die
    # anderen Terminal-Terme.
    w_failure_penalty: float = 5.0    # war: 50.0


# ---------------------------------------------------------------------------
# Cutter-/Grid-Defaults (Physik-Parameter)
# ---------------------------------------------------------------------------
#
# Diese Werte werden vom `Cutter`-Konstruktor und vom `ContinuousPlanner`
# konsumiert. Sie haben nichts mit RL zu tun, gehoeren aber in die zentrale
# Config, weil sie Trainingslaeufe vergleichbar machen muessen: ein
# Geschwindigkeitswechsel macht alle vorherigen Ergebnisse hinfaellig.
# ---------------------------------------------------------------------------


@dataclass
class CutterConfig:
    """Physik des Plasma-Brenners. Spiegelt die Defaults aus `cutter.Cutter`,
    aber explizit, damit sie versionierbar sind."""

    max_depth:     float = 20.0   # mm  -- Klingenlaenge
    cutting_speed: float = 5.0    # mm/s -- Plasma an
    moving_speed:  float = 20.0   # mm/s -- Plasma aus, nahe Material
    rapid_speed:   float = 50.0   # mm/s -- Eilgang
    minimum_gap:   float = 3.0    # mm  -- Sicherheitsabstand TCP <-> Material

    # Schnittspaltbreite, wird vom `ContinuousPlanner` benoetigt.
    kerf_width:    float = 3.0    # mm


# ---------------------------------------------------------------------------
# Top-Level RLConfig
# ---------------------------------------------------------------------------


@dataclass
class RLConfig:
    """Gesamt-Konfiguration fuer das RL-Projekt.

    Aufrufmuster
    ------------
    Statt direkt `RLConfig()` zu instanziieren, sollte in der Regel
    `default_config()` verwendet werden. Dadurch sind kleine Anpassungen
    pro Experiment moeglich, ohne die Defaults zu beruehren::

        cfg = default_config()
        cfg.objective.coverage_threshold = 0.995  # strenger fuer Eval
        cfg.scope.single_geometry_file = "T-Träger Test.json"  # Sanity
    """

    # --- Sub-Configs ------------------------------------------------------
    scope:       ScopeConfig        = field(default_factory=ScopeConfig)
    objective:   ObjectiveConfig    = field(default_factory=ObjectiveConfig)
    success:     SuccessConfig      = field(default_factory=SuccessConfig)
    constraints: ConstraintsConfig  = field(default_factory=ConstraintsConfig)
    cutter:      CutterConfig       = field(default_factory=CutterConfig)

    # Phase-2-Sub-Configs (MDP-Design).
    action:      ActionConfig       = field(default_factory=ActionConfig)
    observation: ObservationConfig  = field(default_factory=ObservationConfig)
    reward:      RewardConfig       = field(default_factory=RewardConfig)

    # --- Reproduzierbarkeit ----------------------------------------------
    # Globaler Seed. Wird durch alle stochastischen Komponenten propagiert
    # (Geometrie-Sampling im Env-Reset, numpy, torch, ...).
    seed: int = 42

    # --- Versionierung ----------------------------------------------------
    # Wird bei jeder semantischen Aenderung (Reward-Formel, Scope, ...)
    # erhoeht. Beim Vergleich von Trainingslaeufen MUSS die Version
    # uebereinstimmen, sonst sind Resultate nicht vergleichbar.
    # Format: "phaseX.minor" -- z.B. "0.1" = erste Phase-0-Definition,
    # "0.2" = Phase-2-Erweiterung um Action/Observation/Reward.
    version: str = "0.3"

    # Menschlich lesbarer Beschreibungstext, taucht in Trainings-Logs auf.
    # Beim Aendern der Config bitte hier kurz dokumentieren, WAS sich
    # geaendert hat -- macht Tensorboard-Vergleiche viel angenehmer.
    description: str = (
        "Phase 0 Default: Coverage >= 99% als harter Constraint, "
        "primaeres Ziel = Zeit minimieren. Scope = geprueft-Konturen "
        "mit Skalierungs-/Rotations-Randomisierung."
    )

    # ---------------------------------------------------------------------
    # Hilfsmethoden
    # ---------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialisiert die Config flach fuer Logging (Tensorboard, W&B).

        Wichtig fuer Phase 5+: jede Trainings-Run sollte die komplette
        Config in seinen Logs ablegen, damit Ergebnisse spaeter
        reproduziert werden koennen.
        """
        return asdict(self)

    def project_root(self) -> Path:
        """Pfad zum `plasma_cutter`-Package (zwei Ebenen ueber dieser Datei).

        Wird gebraucht, um relative Pfade aus `ScopeConfig.geometry_dir`
        in absolute Pfade aufzuloesen, ohne dass der Aufrufer das
        Working-Directory kennen muss.
        """
        return Path(__file__).resolve().parent.parent

    def geometry_dir_path(self) -> Path:
        """Absoluter Pfad zum konfigurierten Geometrie-Verzeichnis."""
        return self.project_root() / self.scope.geometry_dir


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def default_config() -> RLConfig:
    """Liefert die Phase-0-Default-Config.

    Diese Funktion ist der EINZIGE oeffentliche Einstiegspunkt fuer alle
    nachfolgenden Phasen. Wenn ein Experiment Abweichungen braucht,
    erzeugt es eine Default-Config und mutiert nur die relevanten
    Felder -- so bleibt klar, was sich gegenueber dem Standard
    geaendert hat.
    """
    return RLConfig()

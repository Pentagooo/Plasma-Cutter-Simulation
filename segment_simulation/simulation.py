from __future__ import annotations

"""Interaktive Segment-Simulation fuer den Plasma-Schnitt (Lichtschwert).

Modell
------
  - Der TCP (Brennergriff) haelt den konstanten Mindestabstand
    ``MINIMUM_GAP`` zum Material und faehrt auf dem Offset-Pfad um die
    Kontur.
  - Die Klinge (Plasmastrahl) ragt vom TCP in Richtung Objekt. Ihre
    Laenge ist LINEAR geschwindigkeitsabhaengig:
        L(v) = blade_length - blade_slope * v
    Sowohl die Grundlaenge als auch die maximale Geschwindigkeit sind
    einstellbar (CLI: --blade-length, --blade-slope, --v-max, --v-cut).
  - Ziel ist es, ALLE Punkte der Querschnittsflaeche (Innen- UND
    Aussenpunkte) zu ueberstreichen -- nicht nur die Kontur. Die
    Coverage wird ueber die Swept Areas der Klinge berechnet.
  - Punktezahl (Score): besteht vor allem aus der Zeit -- weniger Zeit
    ist besser (siehe planning.compute_score).

Ablauf
------
1. Die Kontur wird automatisch in Segmente unterteilt; die Segment-
   Knoten (weisse Kreise) sind die moeglichen Start-/Endpunkte.
2. Linksklick #1: Startpunkt waehlen (snappt auf den naechsten Knoten).
   Linksklick #2: Endpunkt -> der Konturbogen dazwischen wird als
   Schnitt-Segment uebernommen. Zweimal derselbe Knoten = ganzer Loop.
3. Das Panel zeigt live die Querschnitts-Coverage; fehlende Punkte
   werden rot markiert.
4. Enter: Sequencer ordnet die Segmente optimal, der LinkPlanner
   verbindet sie kollisionsfrei, die Ausfuehrung wird animiert und am
   Ende gibt es die Punktezahl.

Bedienung
---------
  Linksklick  : Start-/Endpunkt waehlen
  Rechtsklick / U : letztes Segment entfernen
  A           : alle restlichen Konturen komplett auswaehlen
  P / Button  : Greedy+ (Greedy Set Cover + Pruning des AutoPlanner;
                mit Regel 4.5 AN dieselbe DP-Split-Geschwindigkeitsstufe
                wie Surrogat und Brute Force, AUS = Basisgeschwindigkeit)
  S / Button  : Surrogat (gelerntes Modell ordnet die Segmente, Greedy
                Set Cover auf exakten Masken, ein Planbau, exakter Verify,
                Fallback auf die Greedy-Auswahl)
  B / Button  : Brute Force (exakte, zeitminimale Segmentauswahl;
                vollstaendige Aufzaehlung aller Teilmengen -- derselbe
                Lehrer, mit dem die Trainingslabels entstehen)
  V / Button  : Regel 4.5 an/aus (Geschwindigkeitszuweisung je Run,
                Kap. 4.5; wirkt auf Planung, Animation und alle Planer)
  Enter       : Planen + Simulation starten
  R           : alles zuruecksetzen
  Esc         : Auswahl/Animation abbrechen
  +/- / Slider: Animations-Geschwindigkeit
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

# --- Interaktives Backend beim Direktstart erzwingen -------------------
# Was passiert: Beim direkten Ausführen dieses Skripts wird ein echtes,
# interaktives Fenster erzwungen, damit Auswahl per Maus/Tasten und die
# Animation funktionieren.
# Warum: Manche IDEs (z.B. PyCharm "Show plots in tool window"/SciView
# oder "Run with Python Console") setzen ein NICHT-interaktives Backend.
# Dann zeigt plt.show() nur ein statisches Bild -- es "läuft nichts".
# Wie umgesetzt: ``matplotlib.use(...)`` muss VOR dem Import von pyplot
# erfolgen. Nur beim Direktstart (``__name__ == "__main__"``) schalten wir
# auf TkAgg um; als importierte Bibliothek bleibt das Backend unangetastet
# (Headless-Betrieb/Tests funktionieren weiter). Fällt TkAgg aus (kein
# tkinter), behalten wir das bestehende Backend und warnen.
import matplotlib
if __name__ == "__main__":
    try:
        matplotlib.use("TkAgg")
    except Exception as _exc:
        print(f"WARNING: interactive backend (TkAgg) not available "
              f"({_exc}). Please run from a terminal instead of the IDE.")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from shapely.geometry import Polygon

try:
    from ..geometry.point_grid import PointGrid
    from ..cutter.cutter import Cutter
    from ..cutter.assumptions import (
        BladeLengthModel, CuttingAssumptions, PierceTimeModel,
    )
    from .segments import (
        SegmentedContour, CutRun, compute_coverage, covered_positions,
        compute_grid_coverage, GridCoverageReport
    )
    from .planning import (
        LinkPlanner, Sequencer, CutPlan, PlannedStep, RunKinematics,
        compute_score, CHAIN_TOL, LinkInfeasibleError,
    )
    from .autoplan import AutoPlanner
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.geometry.point_grid import PointGrid
    from plasma_cutter.cutter.cutter import Cutter
    from plasma_cutter.cutter.assumptions import (
        BladeLengthModel, CuttingAssumptions, PierceTimeModel,
    )
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, CutRun, compute_coverage, covered_positions,
        compute_grid_coverage, GridCoverageReport,
    )
    from plasma_cutter.segment_simulation.planning import (
        LinkPlanner, Sequencer, CutPlan, PlannedStep, RunKinematics,
        compute_score, CHAIN_TOL, LinkInfeasibleError,
    )
    from plasma_cutter.segment_simulation.autoplan import AutoPlanner


# ---------------------------------------------------------------------------
# Konstanten / Defaults
# ---------------------------------------------------------------------------

# Konstanter Mindestabstand TCP <-> Materialoberfläche [mm].
# Wird dem Cutter als minimum_gap mitgegeben (CLI: --clearance).
MINIMUM_GAP = 3.6

# Lineares Klingenmodell  L(v) = BLADE_LENGTH - BLADE_SLOPE * v
BLADE_LENGTH = 29.9   # Grundlänge der Schneide bei v = 0 [mm]
BLADE_SLOPE = 0.327   # Verkürzung pro mm/s [mm / (mm/s)]

# Geschwindigkeiten [mm/s]
DEFAULT_CUTTING_SPEED = 19.4      # minimale Schnittgeschwindigkeit v_cut
DEFAULT_MAX_CUTTING_SPEED = 34.7  # maximale Schnittgeschwindigkeit v_max
RAPID_SPEED = 100.0               # Eilgang zwischen Schnitten

# Zeitaufschlag je Geschwindigkeitswechsel IM laufenden Schnitt [s]
# (Roboterrampe + Lichtbogen-Transient; Kap. 4.5, t_switch).
SPEED_SWITCH_TIME = 0.0

# Zuendung (Pierce): t_pierce = PIERCE_T0 + PIERCE_K * SHEET_THICKNESS
PIERCE_T0 = 1.0               # [s]
PIERCE_K = 0.0                # [s/mm] (Pauschale, dickenunabhaengig)
SHEET_THICKNESS = 15.0        # Nennblechdicke [mm]

# ALLE Werte dieses Blocks sind label-relevant: eine Aenderung macht die
# Trainingslabels und das Surrogat-Modell ungueltig. ``surrogate.params``
# stempelt sie (phys_hash) in Labels, Datensatz und Modell; ``load_model``
# und ``train_model`` brechen bei Abweichung ab.


def make_default_cutter(
    cutting_speed: float = DEFAULT_CUTTING_SPEED,
    max_cutting_speed: float = DEFAULT_MAX_CUTTING_SPEED,
    blade_length: float = BLADE_LENGTH,
    blade_slope: float = BLADE_SLOPE,
    minimum_gap: float = MINIMUM_GAP,
    rapid_speed: float = RAPID_SPEED,
    speed_switch_time: float = SPEED_SWITCH_TIME,
    pierce_t0: float = PIERCE_T0,
    pierce_k: float = PIERCE_K,
    sheet_thickness: float = SHEET_THICKNESS,
) -> Cutter:
    """Cutter mit linearem L(v)-Klingenmodell für die Segment-Simulation.

    Was passiert
    ------------
    Baut einen fertig konfigurierten ``Cutter`` (Brenner/Werkzeug)
    zusammen, der die Lichtschwert-Annahme verwendet: Die Klingenlänge
    hängt linear von der Schnittgeschwindigkeit ab. Wird genutzt, wenn
    dem Konstruktor kein eigener Cutter übergeben wird.

    Wie umgesetzt
    -------------
    Es wird zuerst ein ``BladeLengthModel`` im Modus "linear" erzeugt:
    L(v) = L_ref + slope * v. Da die Klinge mit steigender
    Geschwindigkeit KÜRZER wird, geht ``blade_slope`` als NEGATIVE
    Steigung (``slope=-blade_slope``) ein; ``L_max``/``L_min`` begrenzen
    das Ergebnis. Dieses Modell landet in ``CuttingAssumptions`` (Flag
    ``use_velocity_dependent_blade=True``) und schließlich im
    zurückgegebenen ``Cutter`` zusammen mit den Geschwindigkeiten und
    dem Mindestabstand.
    """
    blade = BladeLengthModel(
        mode="linear",
        L_ref=blade_length, v_ref=0.0, slope=-blade_slope,
        L_max=blade_length, L_min=0.0,
    )
    assumptions = CuttingAssumptions(
        blade=blade, use_velocity_dependent_blade=True,
        pierce=PierceTimeModel(t0=pierce_t0, k=pierce_k),
        sheet_thickness=sheet_thickness)
    return Cutter(
        cutting_speed=cutting_speed,
        max_cutting_speed=max_cutting_speed,
        rapid_speed=rapid_speed,
        minimum_gap=minimum_gap,
        speed_switch_time=speed_switch_time,
        assumptions=assumptions,
    )


def _surrogate_tools():
    """Lazy-Import der Planer-Helfer (Regel 4.5, Greedy+/Surrogat/Brute Force).

    Erst beim ersten Gebrauch importieren: haelt den Simulationsstart
    schlank (scipy/sklearn werden nur bei Bedarf geladen) und vermeidet
    einen Import-Zyklus (das surrogate-Paket importiert seinerseits
    ``make_default_cutter`` aus diesem Modul lazy).
    """
    from types import SimpleNamespace
    try:
        from .surrogate.features import phys_from_cutter
        from .surrogate.runutils import (
            merge_covering_runs, speed_up_runs, build_speed_chains,
        )
        from .surrogate.teacher import TeacherSkipped, exhaustive_plan
        from .surrogate.planner import surrogate_plan
        from .surrogate.model import load_model
    except ImportError:
        from plasma_cutter.segment_simulation.surrogate.features import (
            phys_from_cutter,
        )
        from plasma_cutter.segment_simulation.surrogate.runutils import (
            merge_covering_runs, speed_up_runs, build_speed_chains,
        )
        from plasma_cutter.segment_simulation.surrogate.teacher import (
            TeacherSkipped, exhaustive_plan,
        )
        from plasma_cutter.segment_simulation.surrogate.planner import (
            surrogate_plan,
        )
        from plasma_cutter.segment_simulation.surrogate.model import load_model
    return SimpleNamespace(
        phys_from_cutter=phys_from_cutter, speed_up_runs=speed_up_runs,
        merge_covering_runs=merge_covering_runs,
        build_speed_chains=build_speed_chains,
        exhaustive_plan=exhaustive_plan, TeacherSkipped=TeacherSkipped,
        surrogate_plan=surrogate_plan, load_model=load_model)


# ---------------------------------------------------------------------------
# Farben
# ---------------------------------------------------------------------------

_C = dict(
    inner        = "#BBBBBB",
    seg_colors   = ["#1E6FBF", "#6FA8DC"],     # alternierende Segmentfarben
    seg_hole     = ["#C0504D", "#E6A09E"],     # alternierend für Lochkontur
    node         = "#FFFFFF",
    node_edge    = "#1A2840",
    pending      = "#FFD700",
    covered      = "#22A84E",
    missing      = "#E02020",
    torch        = "#FFD700",
    torch_edge   = "#B8860B",
    blade        = "#FF2222",
    blade_glow   = "#FF6644",
    link         = "#666666",
    tcp_trail    = "#FF8C00",
    stats_bg     = "#EEF4FF",
    grid_bg      = "#F9FAFB",
    infeasible   = "#CC0000",
    run_colors   = [
        "#FF4444", "#FF8C00", "#DDAA00", "#33AA55",
        "#2299AA", "#3366CC", "#6644BB", "#AA33AA",
    ],
)


class _State(Enum):
    """Zustände der UI-Zustandsmaschine (was die Klicks gerade tun).

    Was passiert
    ------------
    Steuert, wie der nächste Linksklick interpretiert wird und ob
    Eingaben überhaupt erlaubt sind:
      - IDLE      : nichts ausgewählt, der nächste Klick setzt einen
                    Startpunkt.
      - PICK_END  : Startpunkt steht, der nächste Klick setzt den
                    Endpunkt und legt damit das Schnitt-Segment an.
      - ANIMATING : die Simulation läuft gerade, Klicks/Tasten zur
                    Auswahl sind gesperrt.

    Wie umgesetzt
    -------------
    Reines ``Enum`` mit ``auto()``-Werten; das aktuelle Mitglied liegt
    in ``SegmentCutSimulation._state`` und wird in den Event-Handlern
    (_on_click/_on_key) abgefragt und umgeschaltet.
    """
    IDLE      = auto()   # wartet auf Startpunkt-Klick
    PICK_END  = auto()   # Startpunkt gesetzt, wartet auf Endpunkt
    ANIMATING = auto()


@dataclass
class _Frame:
    """Ein Animations-Frame: Zeitpunkt, TCP, Klingenspitze, Modus.

    Was passiert
    ------------
    Ein einzelnes "Standbild" der Bewegung. Die Animation ist nichts
    weiter als eine lange Liste solcher Frames, die der Reihe nach
    abgespielt wird.

    Wie umgesetzt
    -------------
    Schlankes ``dataclass``-Datenpaket:
      - ``time`` : simulierte Uhrzeit dieses Bildes [s].
      - ``pos``  : Position des TCP (Brennergriff) als (x, y).
      - ``tip``  : Position der Klingenspitze, oder ``None`` solange der
                   Brenner nur verfährt (kein Strahl).
      - ``mode`` : "pierce" (zünden), "cut" (schneiden) oder "link"
                   (verfahren) -- steuert Farben und Coverage-Stempel.
      - ``run_id``: zu welchem Schnitt-Run der Frame gehört (-1 = Link).
    """
    time: float
    pos: np.ndarray              # TCP-Position
    tip: np.ndarray | None       # Klingenspitze (None bei link)
    mode: str                    # "pierce" | "cut" | "link"
    run_id: int = -1


# ---------------------------------------------------------------------------
# SegmentCutSimulation
# ---------------------------------------------------------------------------

class SegmentCutSimulation:
    """Interaktive Simulation: Segmente wählen, prüfen, ausführen.

    Parameters
    ----------
    grid                   : PointGrid der Geometrie
    cutter                 : Cutter; minimum_gap = konstanter Mindestabstand,
                             blade_length(v) = lineare Klingenlänge.
                             None -> make_default_cutter()
    kerf_width             : Schnittspaltbreite [mm]
    target_segment_length  : Ziel-Segmentlänge [mm] (None = automatisch)
    fps                    : Animations-Framerate
    """

    def __init__(
        self,
        grid: PointGrid,
        cutter: Cutter | None = None,
        kerf_width: float = 3.0,
        target_segment_length: float | None = None,
        fps: int = 30,
    ) -> None:
        # Was passiert: Hier wird der gesamte Simulationszustand einmalig
        # aufgebaut -- Geometrie einlesen, Kinematik/Planer-Objekte
        # erzeugen und alle Zustandsfelder (Auswahl, Plan, Animation,
        # matplotlib-Handles) auf ihre Startwerte setzen.
        # Wie umgesetzt: reine Feldzuweisungen; die UI selbst (Figure,
        # Achsen, Widgets) entsteht erst später in run().
        self.grid = grid
        self.cutter = cutter or make_default_cutter()
        self.kerf_width = kerf_width
        self.fps = fps

        # Klingenlänge bei der eingestellten Schnittgeschwindigkeit
        self.blade_length = self.cutter.blade_length(self.cutter.cutting_speed)

        # Kontur in Segmente zerlegen und das Materialpolygon ableiten:
        # Grundlage für Knoten (Klickziele), Kollisionsprüfung und
        # Coverage. material = die zu schneidende Querschnittsfläche.
        self.contour = SegmentedContour.from_grid(
            grid, target_segment_length=target_segment_length)
        self.material = self.contour.material_polygon()
        self.kinematics = RunKinematics(
            self.material,
            clearance=self.cutter.minimum_gap,
            blade_length=self.blade_length,
            kerf=kerf_width,
        )
        # LinkPlanner verbindet zwei Schnitte kollisionsfrei (Eilgang um
        # das Material herum); Sequencer bestimmt die günstigste
        # Reihenfolge der Schnitte und ruft dazu den LinkPlanner auf.
        self.link_planner = LinkPlanner(
            self.material, clearance=self.cutter.minimum_gap)
        self.sequencer = Sequencer(
            self.cutter, self.contour, self.link_planner)

        # Warnung, falls die Klinge bei dieser Geschwindigkeit kürzer
        # ist als der Mindestabstand -- dann erreicht der Strahl das
        # Material gar nicht und es kann nicht geschnitten werden.
        if self.kinematics.effective_depth <= 0:
            print(f"WARNING: plasma arc too short! "
                  f"L(v={self.cutter.cutting_speed}) "
                  f"= {self.blade_length:.1f} mm <= standoff "
                  f"{self.cutter.minimum_gap:.1f} mm -> no cut possible. "
                  f"Reduce the speed or increase the arc length.")

        # Auswahl-/Planungszustand:
        #   _runs    : vom Nutzer gewählte Schnitt-Segmente (CutRun).
        #   _plan    : geordneter+verbundener Plan (None = noch ungeplant).
        #   _score   : zeitbasierte Punktezahl (None = noch nicht bewertet).
        #   _pending : (loop_id, pos) des bereits geklickten Startpunkts
        #              während auf den Endpunkt gewartet wird.
        #   _snap_radius : Fangradius in mm für das Knoten-Snapping; vier
        #              Konturpunkt-Abstände gelten als "nah genug".
        self._runs: list[CutRun] = []
        self._plan: CutPlan | None = None
        self._score: float | None = None
        self._state = _State.IDLE
        self._pending: tuple[int, int] | None = None  # (loop_id, pos)
        self._snap_radius = grid.contour_spacing * 4.0

        # Animations-Zustand:
        #   _anim      : laufende FuncAnimation (None = keine).
        #   _speed     : Abspielfaktor (Slider, 0.25x .. 16x).
        #   _anim_mask : bool-Array über alle Gitterpunkte; True = von der
        #                Klinge während der Animation bereits überstrichen.
        #   _status_msg: aktuelle Statuszeile für das Info-Panel.
        self._anim: FuncAnimation | None = None
        self._speed = 1.0
        self._anim_mask: np.ndarray | None = None
        self._status_msg = ""

        # Regel 4.5 (Kap. 4.5): Geschwindigkeitszuweisung je Run.
        #   _use_rule45 : Schalter (Button/Taste V). AUS = bisheriges
        #                 Verhalten, alle Runs mit cutter.cutting_speed.
        #   _run_speeds : run_id -> zugewiesene Geschwindigkeit [mm/s],
        #                 bei der letzten Planung vergeben (leer = Basis).
        #   _chain_sel  : Segment-Auswahl (seg_ids) hinter der aktuellen
        #                 Run-Liste, wenn sie von einem Planer stammt
        #                 (Taste P/S/B). Nur dann kann Enter die
        #                 DP-Split-Kettenausfuehrung nutzen (Sub-Runs mit
        #                 eigener Geschwindigkeit, nahtlos ohne Pierce)
        #                 -- dieselbe Semantik wie Lehrer/Greedy+/Surrogat.
        #                 None = manuelle Auswahl -> Regel 4.5 je Run.
        self._use_rule45 = False
        self._run_speeds: dict[int, float] = {}
        self._chain_sel: list[int] | None = None
        #   _model      : Surrogat-Modell, beim ersten Druck auf S lazy
        #                 geladen (artifacts/surrogate_model.joblib)
        self._model = None

        # matplotlib-Handles, erst in run() befüllt (vorher None):
        # Hauptachse (Zeichnung), Statistik-Panel rechts sowie die
        # Bedien-Widgets (Speed-Slider, Reset-, Greedy+-, Surrogat-,
        # Brute-Force- und Regel-4.5-Button).
        self._fig: plt.Figure | None = None
        self._ax_main: plt.Axes | None = None
        self._ax_stats: plt.Axes | None = None
        self._speed_slider: Slider | None = None
        self._reset_button: Button | None = None
        self._gp_button: Button | None = None
        self._sur_button: Button | None = None
        self._bf_button: Button | None = None
        self._rule_button: Button | None = None

    # ------------------------------------------------------------------

    @classmethod
    def run_with_dialog(cls, **kwargs) -> SegmentCutSimulation | None:
        """Geometrie per Datei-Dialog wählen und Simulation starten.

        Was passiert
        ------------
        Komfort-Einstieg ohne festen Geometriepfad: Es öffnet sich ein
        Datei-Auswahldialog; nach der Auswahl wird die Simulation gebaut
        und sofort gestartet.

        Wie umgesetzt
        -------------
        ``initial_dir`` wird aus ``kwargs`` herausgelöst und an
        ``PointGrid.from_json_dialog`` weitergereicht. Bricht der Nutzer
        den Dialog ab (Rückgabe ``None``), endet die Methode mit
        ``None``. Sonst werden die restlichen ``kwargs`` an den
        Konstruktor durchgereicht und ``run()`` aufgerufen.
        """
        initial_dir = kwargs.pop("initial_dir", None)
        grid = PointGrid.from_json_dialog(initial_dir=initial_dir)
        if grid is None:
            return None
        sim = cls(grid=grid, **kwargs)
        sim.run()
        return sim

    @property
    def runs(self) -> list[CutRun]:
        """Kopie der aktuell gewählten Schnitt-Runs (lesender Zugriff).

        Gibt bewusst eine flache Kopie zurück, damit Aufrufer die
        interne Auswahl ``_runs`` nicht versehentlich verändern.
        """
        return list(self._runs)

    @property
    def plan(self) -> CutPlan | None:
        """Der zuletzt erstellte Plan, oder ``None`` falls noch keiner.

        Wird nach jeder Auswahl-Änderung intern auf ``None`` gesetzt
        und erst durch Enter (build_plan) wieder befüllt.
        """
        return self._plan

    def grid_coverage(self) -> GridCoverageReport:
        """Aktuelle Querschnitts-Coverage der gewählten Segmente.

        Was passiert: prüft, welcher Anteil ALLER Gitterpunkte des
        Querschnitts von den Swept Areas der gewählten Runs überdeckt
        wird (Ziel ist 100 %).
        Wie umgesetzt: delegiert an ``compute_grid_coverage`` (segments.py)
        mit dem aktuellen Gitter und der Run-Auswahl; liefert einen
        ``GridCoverageReport`` mit Maske, Anteil und Restpunkten.
        """
        return compute_grid_coverage(self.grid, self._runs)

    def run(self) -> None:
        """Baut das Fenster auf und startet die interaktive Schleife.

        Was passiert
        ------------
        Erzeugt das gesamte UI: links die große Zeichenfläche, rechts
        das Statistik-Panel, unten den Speed-Slider und den Reset-Button.
        Danach wird gezeichnet und die matplotlib-Ereignisschleife läuft
        bis zum Schließen des Fensters.

        Wie umgesetzt
        -------------
        Eine ``Figure`` mit ``GridSpec`` (Verhältnis 4 : 1.15) liefert
        die beiden Hauptachsen. Zwei eigene Achsen tragen die Widgets
        (``Slider``, ``Button``). Klick- und Tasten-Events werden über
        ``mpl_connect`` an ``_on_click``/``_on_key`` gebunden. Ein
        ``_redraw()`` zeichnet den Startzustand, ``plt.show()`` blockiert
        bis zum Fensterschluss.
        """
        # Matplotlib-Standardtasten freimachen: 's' waere "Speichern",
        # 'p' waere "Pan" -- hier sind es Surrogat und Greedy+.
        plt.rcParams["keymap.save"] = ["ctrl+s"]
        plt.rcParams["keymap.pan"] = []

        self._fig = plt.figure(figsize=(15, 9), facecolor="white")
        self._fig.suptitle(self._title(), fontsize=11, fontweight="bold",
                           y=0.98, color="#1A2840")

        gs = self._fig.add_gridspec(
            1, 2, width_ratios=[4, 1.15],
            left=0.05, right=0.98, top=0.88, bottom=0.12, wspace=0.03)
        self._ax_main = self._fig.add_subplot(gs[0])
        self._ax_stats = self._fig.add_subplot(gs[1])

        ax_speed = self._fig.add_axes([0.10, 0.04, 0.27, 0.03])
        self._speed_slider = Slider(
            ax=ax_speed, label="Sim speed ",
            valmin=0.25, valmax=16.0, valinit=1.0,
            valstep=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            color="#0066CC")
        self._speed_slider.on_changed(self._on_speed)

        ax_reset = self._fig.add_axes([0.415, 0.035, 0.075, 0.045])
        self._reset_button = Button(
            ax_reset, "Reset (R)", color="#F0D0D0", hovercolor="#E0A0A0")
        self._reset_button.label.set_fontsize(9)
        self._reset_button.on_clicked(lambda _evt: self._full_reset())

        # Die drei automatischen Auswahlverfahren nebeneinander (die
        # manuelle Auswahl ist der Klick in die Zeichnung).
        ax_gp = self._fig.add_axes([0.497, 0.035, 0.105, 0.045])
        self._gp_button = Button(
            ax_gp, "Greedy+ (P)", color="#D0DEF0", hovercolor="#A8C4E4")
        self._gp_button.label.set_fontsize(9)
        self._gp_button.on_clicked(lambda _evt: self._greedy_select())

        ax_sur = self._fig.add_axes([0.609, 0.035, 0.105, 0.045])
        self._sur_button = Button(
            ax_sur, "Surrogate (S)", color="#D0DEF0", hovercolor="#A8C4E4")
        self._sur_button.label.set_fontsize(9)
        self._sur_button.on_clicked(lambda _evt: self._surrogate_select())

        ax_bf = self._fig.add_axes([0.721, 0.035, 0.115, 0.045])
        self._bf_button = Button(
            ax_bf, "Brute force (B)", color="#D0DEF0", hovercolor="#A8C4E4")
        self._bf_button.label.set_fontsize(9)
        self._bf_button.on_clicked(lambda _evt: self._brute_force_select())

        ax_rule = self._fig.add_axes([0.843, 0.035, 0.137, 0.045])
        self._rule_button = Button(ax_rule, "", color="#E0E0E0",
                                   hovercolor="#C8C8C8")
        self._rule_button.label.set_fontsize(9)
        self._rule_button.on_clicked(lambda _evt: self._toggle_rule45())
        self._style_rule_button()

        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._redraw()

        # Sicherheitshinweis: Läuft trotz allem noch ein nicht-interaktives
        # Backend (z.B. PyCharm-Werkzeugfenster), erscheint nur ein Standbild
        # ohne Animation. Statt stiller Verwirrung eine klare Meldung geben.
        _backend = matplotlib.get_backend().lower()
        if _backend == "agg" or "inline" in _backend or "interagg" in _backend:
            print(
                f"WARNING: non-interactive matplotlib backend "
                f"('{matplotlib.get_backend()}') -- only a static image "
                f"is shown, without animation.\n"
                f"  -> Run from a terminal, OR in PyCharm: disable Settings > "
                f"Tools > Python Scientific > 'Show plots in tool window' and "
                f"disable 'Run with Python Console' in the run configuration.")

        plt.show()

    def _title(self) -> str:
        """Baut die Titelzeile mit den Eckdaten des Lichtschwerts.

        Was passiert: erzeugt den Fenstertitel mit Dateiname (falls
        bekannt), Klingenlänge bei Schnittgeschwindigkeit, effektiver
        Schnitttiefe und konstantem Mindestabstand.
        Wie umgesetzt: reine f-String-Formatierung; der Dateiname wird
        nur angehängt, wenn ``grid._source_path`` existiert.
        """
        name = (f"  -  {self.grid._source_path.stem}"
                if hasattr(self.grid, "_source_path") else "")
        return (f"Segment Simulation{name}  |  "
                f"L(v={self.cutter.cutting_speed:.0f}) = "
                f"{self.blade_length:.1f} mm  |  "
                f"Eff. depth: {self.kinematics.effective_depth:.1f} mm  |  "
                f"Standoff: {self.cutter.minimum_gap:.0f} mm (const.)")

    # ------------------------------------------------------------------
    # Auswahl
    # ------------------------------------------------------------------

    def _add_run(self, loop_id: int, start_pos: int, end_pos: int) -> CutRun:
        """Erzeugt einen CutRun inkl. Lichtschwert-Kinematik.

        Was passiert
        ------------
        Legt aus zwei Knotenpositionen (Start/Ende) auf derselben Kontur
        ein neues Schnitt-Segment an, berechnet dessen Bewegung samt
        Klinge und fügt es der Auswahl hinzu.

        Wie umgesetzt
        -------------
        ``covered_positions`` ermittelt zuerst, welche Punkte dieser
        Kontur schon von früheren Runs abgedeckt sind (damit der neue
        Run sich nahtlos anschließen kann). ``contour.make_run`` baut
        den Bogen, ``kinematics.attach`` ergänzt TCP-Pfad,
        Klingenspitzen und Swept Area. Der Run wird angehängt; ein evtl.
        bestehender Plan/Score wird verworfen (auf ``None``), da die
        Auswahl sich geändert hat.
        """
        covered = covered_positions(self.contour, self._runs, loop_id)
        run = self.contour.make_run(
            run_id=len(self._runs) + 1,
            loop_id=loop_id, start_pos=start_pos, end_pos=end_pos,
            covered=covered)
        self.kinematics.attach(run)
        self._runs.append(run)
        self._plan = None
        self._score = None
        self._chain_sel = None   # manuelle Aenderung: keine Planer-Auswahl mehr
        return run

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_speed(self, val: float) -> None:
        """Reagiert auf den Speed-Slider (Abspielgeschwindigkeit).

        Was passiert
        ------------
        Stellt ein, wie schnell die Animation läuft. Läuft gerade eine
        Animation, wird ihr Timer sofort angepasst, sonst wird der Wert
        nur gemerkt und beim nächsten Start verwendet.

        Wie umgesetzt
        -------------
        Der neue Faktor landet in ``_speed``. Bis Faktor 2x wird die
        echte Framerate (Timer-Intervall) erhöht; darüber bleibt das
        Intervall konstant und der ``frame_gen`` überspringt stattdessen
        Frames (siehe _run_animation). ``_draw_stats`` aktualisiert die
        Anzeige des Faktors.
        """
        self._speed = val
        if self._anim is not None and self._anim.event_source is not None:
            # Bis 2x über echte FPS beschleunigen; darüber würde der
            # Timer zu eng -- dann skippt der Generator stattdessen Frames.
            effective_fps = self.fps * min(self._speed, 2.0)
            self._anim.event_source.interval = max(1, int(1000 / effective_fps))
        self._draw_stats()

    def _on_click(self, event) -> None:
        """Maus-Handler: Start-/Endpunkt setzen oder Undo.

        Was passiert
        ------------
        Auf der Hauptachse wählt Linksklick nacheinander Start- und
        Endpunkt eines Schnitts (jeweils auf den nächsten Knoten
        gefangen). Zwei Klicks ergeben ein Segment; liegen sie auf
        verschiedenen Konturen, wird abgelehnt. Rechtsklick macht den
        letzten Schritt rückgängig.

        Wie umgesetzt
        -------------
        Klicks außerhalb ``_ax_main`` oder während der Animation werden
        ignoriert. Maustaste 3 ruft ``_undo_last``, nur Taste 1 zählt
        weiter. Die Klickposition wird per ``contour.snap_node`` auf den
        nächsten Segmentknoten innerhalb ``_snap_radius`` gerundet. Die
        Zustandsmaschine (``_state``) entscheidet, ob der Knoten Start
        (IDLE -> PICK_END) oder Ende (PICK_END -> _add_run -> IDLE) ist.
        """
        if event.inaxes is not self._ax_main:
            return
        if self._state == _State.ANIMATING:
            return

        if event.button == 3:  # Rechtsklick = Undo
            self._undo_last()
            return
        if event.button != 1:
            return

        # Klickkoordinate auf den nächstgelegenen Knoten "einfangen"
        # (snap); ohne Treffer im Fangradius passiert nichts.
        xy = np.array([event.xdata, event.ydata])
        snapped = self.contour.snap_node(xy, max_dist=self._snap_radius)
        if snapped is None:
            self._status_msg = "No segment node nearby."
            self._draw_stats()
            return

        if self._state == _State.IDLE:
            # Erster Klick: Startknoten merken, auf Endpunkt warten.
            self._pending = snapped
            self._state = _State.PICK_END
            self._status_msg = "Select end point ..."
        elif self._state == _State.PICK_END:
            # Zweiter Klick: Endknoten. Muss auf derselben Kontur liegen.
            loop_id, start_pos = self._pending
            end_loop, end_pos = snapped
            if end_loop != loop_id:
                self._status_msg = ("Start and end must be on the same "
                                    "contour!")
                self._draw_stats()
                return
            run = self._add_run(loop_id, start_pos, end_pos)
            self._pending = None
            self._state = _State.IDLE
            if run.is_feasible:
                self._status_msg = f"Segment R{run.run_id} added."
            else:
                self._status_msg = (f"R{run.run_id} NOT feasible: "
                                    f"{run.reason}")
        self._redraw()

    def _on_key(self, event) -> None:
        """Tastatur-Handler: Geschwindigkeit, Auswahl, Planung, Reset.

        Was passiert
        ------------
        Wertet alle Tastaturbefehle aus:
          +/-  : Animation schneller/langsamer
          R    : alles zurücksetzen
          Esc  : Animation stoppen bzw. laufende Auswahl abbrechen
          U    : letztes Segment entfernen
          A    : alle restlichen Konturen automatisch wählen
          P    : Greedy+ (Greedy Set Cover + Pruning)
          S    : Surrogat (gelerntes Modell + exakte Nachrechnung)
          B    : Brute Force (vollständige Aufzählung aller Teilmengen)
          V    : Regel 4.5 an/aus (Geschwindigkeitszuweisung je Run)
          Enter: planen und Simulation starten

        Wie umgesetzt
        -------------
        Früh-Returns prüfen die Tasten der Reihe nach. ``+``/``-``
        schieben den Speed-Slider (verdoppeln/halbieren mit Begrenzung).
        Auswahl-ändernde Tasten (U/A/P/S/B/V/Enter) sind während der
        Animation gesperrt -- nur ``escape`` und ``r`` wirken dann noch.
        """
        if event.key in ("+", "="):
            self._speed_slider.set_val(min(self._speed * 2.0, 16.0))
            return
        if event.key in ("-", "_"):
            self._speed_slider.set_val(max(self._speed / 2.0, 0.25))
            return

        if event.key == "r":
            self._full_reset()
            return

        if event.key == "escape":
            if self._state == _State.ANIMATING:
                self._stop_animation()
            elif self._state == _State.PICK_END:
                self._pending = None
                self._state = _State.IDLE
                self._status_msg = "Selection cancelled."
                self._redraw()
            return

        # Ab hier: nur erlaubt, wenn gerade KEINE Animation läuft.
        if self._state == _State.ANIMATING:
            return

        if event.key == "u":
            self._undo_last()
            return

        if event.key == "a":
            self._select_all_remaining()
            return

        if event.key == "p":
            self._greedy_select()
            return

        if event.key == "s":
            self._surrogate_select()
            return

        if event.key == "b":
            self._brute_force_select()
            return

        if event.key == "v":
            self._toggle_rule45()
            return

        if event.key == "enter":
            if self._runs:
                self._plan_and_animate()
            else:
                self._status_msg = "No segments selected."
                self._draw_stats()
            return

    def _undo_last(self) -> None:
        """Macht den letzten Auswahl-Schritt rückgängig.

        Was passiert: Wartet die Auswahl gerade auf den Endpunkt, wird
        nur dieser halbe Schritt verworfen. Sonst wird das zuletzt
        hinzugefügte Segment wieder entfernt.
        Wie umgesetzt: im Zustand PICK_END nur ``_pending`` löschen und
        nach IDLE zurück; andernfalls ``_runs.pop()`` und Plan/Score
        invalidieren. Abschließend neu zeichnen.
        """
        if self._state == _State.PICK_END:
            self._pending = None
            self._state = _State.IDLE
        elif self._runs:
            removed = self._runs.pop()
            self._plan = None
            self._score = None
            self._chain_sel = None   # Auswahl manuell veraendert
            self._status_msg = f"Segment R{removed.run_id} removed."
        self._redraw()

    def _select_all_remaining(self) -> None:
        """Wählt für jede Kontur die noch fehlenden Bögen aus.

        Was passiert
        ------------
        Taste A: Vervollständigt die Auswahl, so dass am Ende JEDE
        Kontur (Außen- und Lochkonturen) komplett abgedeckt ist. Schon
        gewählte Bereiche bleiben erhalten, nur die Lücken werden
        ergänzt.

        Wie umgesetzt
        -------------
        Pro Loop wird über ``covered_positions`` die bereits abgedeckte
        Punktmenge bestimmt. Ist nichts abgedeckt, wird ein voller Loop
        (Start = Ende am Ankerknoten) angelegt. Sonst liefert
        ``_missing_arcs`` die unabgedeckten Bögen, die einzeln per
        ``_add_run`` ergänzt werden; ``covered`` wird dabei mitgeführt,
        damit sich folgende Bögen korrekt anschließen.
        """
        added = 0
        for loop in self.contour.loops:
            covered = covered_positions(self.contour, self._runs, loop.loop_id)
            if len(covered) >= loop.n:
                continue
            nodes = self.contour.nodes[loop.loop_id]
            anchor = nodes[0] if nodes else 0
            if not covered:
                # Kontur noch völlig frei -> kompletter Loop in einem Run.
                self._add_run(loop.loop_id, anchor, anchor)
                added += 1
            else:
                # Fehlende zusammenhängende Bögen einzeln hinzufügen
                for a, b in self._missing_arcs(loop.n, covered):
                    run = self._add_run(loop.loop_id, a, b)
                    covered.update(run.positions)
                    added += 1
        self._status_msg = (f"{added} segment(s) added."
                            if added else "Contour already fully selected.")
        self._redraw()

    @staticmethod
    def _missing_arcs(n: int, covered: set[int]) -> list[tuple[int, int]]:
        """Zusammenhängende unabgedeckte Bereiche als (start, ende)-Paare.

        Start/Ende sind die angrenzenden ABGEDECKTEN Punkte, damit der
        neue Schnitt nahtlos an Bestehendes anschließt.

        Was passiert
        ------------
        Auf einem RINGförmig nummerierten Loop (Positionen 0..n-1) wird
        bestimmt, welche zusammenhängenden Stücke noch fehlen, und
        jeweils als (Startknoten, Endknoten) zurückgegeben.

        Wie umgesetzt
        -------------
        1. Fehlende Positionen aufsteigend sammeln.
        2. In Gruppen aufeinanderfolgender Positionen bündeln.
        3. Ringübergang behandeln: läuft die erste Gruppe bei 0 los und
           die letzte bei n-1, sind beide über die Naht verbunden und
           werden verschmolzen.
        4. Pro Lücke je einen Nachbarn nach außen erweitern
           ((g[0]-1)%n bzw. (g[-1]+1)%n), damit der neue Schnitt einen
           bereits abgedeckten Punkt überlappt und nahtlos anschließt.
        """
        missing = sorted(p for p in range(n) if p not in covered)
        if not missing:
            return []
        # Schritt 2: laufend in Gruppen direkt benachbarter Indizes teilen.
        groups: list[list[int]] = [[missing[0]]]
        for p in missing[1:]:
            if p == groups[-1][-1] + 1:
                groups[-1].append(p)
            else:
                groups.append([p])
        # Schritt 3: Naht 0<->n-1 schließen (erste+letzte Gruppe mergen).
        if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        # Schritt 4: jede Lücke um einen abgedeckten Nachbarn aufweiten.
        return [((g[0] - 1) % n, (g[-1] + 1) % n) for g in groups]

    def _greedy_select(self) -> None:
        """Greedy+ (Taste P / Button): Greedy Set Cover + Pruning des
        ``AutoPlanner`` (siehe autoplan.py) ersetzt die aktuelle Auswahl.

        Die Segment-Auswahl wird in ``_chain_sel`` gemerkt: mit Regel 4.5
        AN nutzt Enter dann die DP-Split-Kettenausführung -- exakt die
        Geschwindigkeitsstufe von ``surrogate.planner.greedy_plus_plan``,
        Surrogat und Brute Force (fairer Vergleich). Mit Regel 4.5 AUS
        fahren alle Runs Basisgeschwindigkeit (klassischer Greedy-Planer).
        Enter startet danach wie gewohnt die Planung + Animation.
        """
        self._status_msg = "Greedy+ running ..."
        self._draw_stats()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        # Statusmeldung sofort sichtbar machen (draw+flush), bevor der
        # ggf. mehrere Sekunden dauernde Auto-Planer das UI blockiert.
        planner = AutoPlanner(
            self.grid, self.contour, self.cutter,
            self.kinematics, self.sequencer)
        result = planner.plan()

        # Auswahl komplett durch das Greedy-Ergebnis ersetzen; Plan/Score
        # und ein evtl. hängender Pending-Punkt werden zurückgesetzt.
        self._runs = list(result.runs)
        self._plan = None
        self._score = None
        self._pending = None
        self._run_speeds = {}
        # Segment-Auswahl merken -> Enter kann die DP-Split-Ketten-
        # ausfuehrung nutzen (gleiche Semantik wie greedy_plus_plan).
        self._chain_sel = list(result.selected_segments) or None
        self._state = _State.IDLE

        print()
        print(result.summary())
        cov = result.report.fraction if result.report else 0.0
        self._status_msg = (
            f"Greedy+: {len(result.runs)} cut(s), "
            f"coverage {cov:.1%}, {result.elapsed:.2f} s "
            f"-- press Enter to start.")
        self._redraw()

    def _brute_force_select(self) -> None:
        """Brute-Force-Löser (Taste B / Button): ersetzt die Auswahl durch
        die EXAKT zeitminimale Segmentauswahl des Aufzählungs-Lehrers v2
        (``surrogate.teacher.exhaustive_plan``).

        Was passiert
        ------------
        Alle Segment-Teilmengen werden aufgezählt, jede vollständige
        Abdeckung wird EXAKT gebaut und bewertet -- derselbe Lehrer, mit
        dem die Trainingslabels entstehen (bei 14 Segmenten ~1 min, bei
        16 einige Minuten). Der Lehrer ist exakt im Raum Auswahl x
        Reihenfolge x Richtung bei DIESER Segmentierung. Steht der
        Regel-4.5-Schalter auf AN, ist die Geschwindigkeitszuweisung
        (DP-Split je Kette) Teil der Zielfunktion; sonst wird die reine
        Basis-Zeit minimiert. Enter startet danach wie gewohnt die
        Planung + Animation.

        Wie umgesetzt
        -------------
        Der Lehrer bekommt die Kontur DIESER Simulation (gleiche
        Segmentierung wie in der Anzeige). Die optimalen seg_ids werden
        über ``merge_covering_runs`` coverage-erhaltend zu CutRuns
        verschmolzen und bei Basisgeschwindigkeit angeheftet; die
        Regel-4.5-Anhebung passiert erst beim Planen (Enter, über
        ``_chain_sel`` als DP-Split-Ketten -- reproduziert das gemeldete
        Lehrer-T exakt). Bei zu vielen Segmenten (``TeacherSkipped``)
        bleibt die Auswahl unverändert.
        """
        if self._state == _State.ANIMATING:
            return  # Button ist auch während der Animation klickbar
        tools = _surrogate_tools()
        phys_from_cutter = tools.phys_from_cutter
        merge_covering_runs = tools.merge_covering_runs
        exhaustive_plan = tools.exhaustive_plan
        TeacherSkipped = tools.TeacherSkipped
        label = "Brute force"

        n_seg = len(self.contour.segments)
        rule_txt = "with" if self._use_rule45 else "without"
        self._status_msg = (f"{label} running ({n_seg} segments, "
                            f"{rule_txt} rule 4.5) ...")
        self._draw_stats()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        try:
            # rein exakt: jede vollstaendige Abdeckung wird gebaut --
            # bei 14 Segmenten ~1 min, bei 16 einige Minuten
            result = exhaustive_plan(
                self.grid, cutter=self.cutter, kerf=self.kerf_width,
                contour=self.contour, speed_rule=self._use_rule45,
                keep_covers=False)
            detail = (f"{result.n_subsets} subsets, {result.n_covers} "
                      f"complete covers, all built exactly")
        except TeacherSkipped as exc:
            self._status_msg = f"{label} skipped: {exc}"
            self._draw_stats()
            return

        phys = phys_from_cutter(self.cutter)
        # check_partial=True: exakt dieselbe Merge-Entscheidung wie der
        # Lehrer (jede Gruppe wird gegen ihre Einzelsegmente geprueft),
        # damit die Runs dessen T und Coverage reproduzieren.
        runs = merge_covering_runs(
            self.contour, result.selected, self.material, phys,
            self.cutter.cutting_speed, self.kerf_width,
            np.asarray(self.grid.coords, dtype=float),
            check_partial=True)

        self._runs = list(runs)
        self._plan = None
        self._score = None
        self._pending = None
        self._run_speeds = {}
        # Segment-Auswahl merken: Enter kann damit die DP-Split-Ketten-
        # ausfuehrung nutzen (reproduziert das gemeldete Lehrer-T exakt).
        self._chain_sel = list(result.selected)
        self._state = _State.IDLE

        print()
        print(f"{label} ({rule_txt} rule 4.5): "
              f"{len(result.selected)}/{result.n_segments} segments "
              f"-> {len(runs)} cut(s), T = {result.total_time:.1f} s, "
              f"coverage {result.coverage:.1%}, {detail} in "
              f"{result.plan_time:.2f} s")
        if not runs:
            self._status_msg = f"{label}: no feasible segments."
        else:
            self._status_msg = (
                f"{label} ({rule_txt} rule 4.5): {len(runs)} "
                f"cut(s), T={result.total_time:.1f} s, coverage "
                f"{result.coverage:.1%}, {result.plan_time:.1f} s "
                f"-- press Enter to start.")
        self._redraw()

    def _surrogate_model(self):
        """Surrogat-Modell einmal lazy laden und auf der Instanz cachen."""
        if self._model is None:
            self._model = _surrogate_tools().load_model()
        return self._model

    def _surrogate_select(self) -> None:
        """Surrogat (Taste S / Button): ersetzt die Auswahl durch die des
        gelernten Surrogat-Planers (``surrogate.planner.surrogate_plan``).

        Was passiert
        ------------
        Merkmale nur aus Kontur + Distanzen -> Modell p(s) -> Greedy Set
        Cover in p(s)-Reihenfolge auf EXAKTEN Singleton-Masken -> Pruning
        -> ein Planbau (DP-Split + Held-Karp) -> ein exakter Verify ->
        Fallback auf die Greedy-Auswahl, falls erreichbare Punkte fehlen.
        Das Modell bestimmt nur die Reihenfolge; die Coverage-Garantie
        hängt nie am Modell. Regel-4.5-Schalter wie bei P und B.

        Wie umgesetzt
        -------------
        Wie bei B: die gewählten seg_ids werden über
        ``merge_covering_runs(check_partial=True)`` bei Basisgeschwindigkeit
        materialisiert, ``_chain_sel`` merkt die Auswahl, Enter reproduziert
        das gemeldete T über die DP-Split-Ketten.
        """
        if self._state == _State.ANIMATING:
            return
        tools = _surrogate_tools()
        label = "Surrogate"
        n_seg = len(self.contour.segments)
        rule_txt = "with" if self._use_rule45 else "without"
        self._status_msg = (f"{label} running ({n_seg} segments, "
                            f"{rule_txt} rule 4.5) ...")
        self._draw_stats()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        try:
            model = self._surrogate_model()
        except (FileNotFoundError, SystemExit) as exc:
            self._status_msg = (
                f"{label}: no model ({exc}). Train it with: python -m "
                f"plasma_cutter.segment_simulation.surrogate.model --train")
            self._draw_stats()
            return

        result = tools.surrogate_plan(
            self.grid, model, cutter=self.cutter, kerf=self.kerf_width,
            contour=self.contour, speed_rule=self._use_rule45)

        phys = tools.phys_from_cutter(self.cutter)
        runs = tools.merge_covering_runs(
            self.contour, result.selected, self.material, phys,
            self.cutter.cutting_speed, self.kerf_width,
            np.asarray(self.grid.coords, dtype=float),
            check_partial=True)

        self._runs = list(runs)
        self._plan = None
        self._score = None
        self._pending = None
        self._run_speeds = {}
        self._chain_sel = list(result.selected) or None
        self._state = _State.IDLE

        print()
        print(f"{label} ({rule_txt} rule 4.5): " + result.summary())
        if not runs:
            self._status_msg = f"{label}: no feasible segments."
        else:
            fb = " [fallback to Greedy]" if result.used_fallback else ""
            self._status_msg = (
                f"{label} ({rule_txt} rule 4.5): {len(runs)} cut(s), "
                f"T={result.T:.1f} s, coverage {result.coverage:.1%}, "
                f"{result.t_plan * 1e3:.0f} ms{fb} -- press Enter to start.")
        self._redraw()

    # ------------------------------------------------------------------
    # Regel 4.5 (Geschwindigkeitszuweisung je Run)
    # ------------------------------------------------------------------

    def _toggle_rule45(self) -> None:
        """Regel 4.5 an-/ausschalten (Taste V / Button).

        AN  : Beim Planen (Enter) bekommt jeder Run die schnellste
              coverage-erhaltende Schnittgeschwindigkeit (Kap. 4.5);
              die Klinge L(v) wird entsprechend kürzer. Auch der
              Brute-Force-Löser optimiert dann mit dieser Regel.
        AUS : Alle Runs fahren mit der Basis-Schnittgeschwindigkeit
              (bisheriges Verhalten).

        Ein bestehender Plan wird verworfen (die Zeiten würden sich
        ändern); beim Ausschalten werden bereits angehobene Runs sofort
        wieder bei Basisgeschwindigkeit angeheftet, damit Anzeige und
        Coverage konsistent bleiben.
        """
        if self._state == _State.ANIMATING:
            return
        self._use_rule45 = not self._use_rule45
        if not self._use_rule45:
            self._reset_run_speeds()
        self._plan = None
        self._score = None
        self._style_rule_button()
        self._status_msg = (
            "Rule 4.5 ON: v per run assigned during planning."
            if self._use_rule45 else
            "Rule 4.5 OFF: all runs at base speed.")
        self._redraw()

    def _style_rule_button(self) -> None:
        """Beschriftung + Farbe des Regel-4.5-Buttons an den Zustand
        anpassen (grün = AN, grau = AUS)."""
        if self._rule_button is None:
            return
        on = self._use_rule45
        self._rule_button.label.set_text(
            f"Rule 4.5: {'ON' if on else 'OFF'} (V)")
        self._rule_button.color = "#C8E6C9" if on else "#E0E0E0"
        self._rule_button.hovercolor = "#A5D6A7" if on else "#C8C8C8"
        self._rule_button.ax.set_facecolor(self._rule_button.color)

    def _reset_run_speeds(self) -> None:
        """Heftet alle Runs wieder bei Basisgeschwindigkeit an.

        Nötig, nachdem Regel 4.5 einzelnen Runs höhere Geschwindig-
        keiten (und damit kürzere Klingen / kleinere Swept Areas)
        zugewiesen hat -- sonst zeigt die Coverage-Anzeige die
        angehobenen Sweeps, obwohl die Regel abgeschaltet wurde.
        """
        if not self._run_speeds:
            return
        for run in self._runs:
            self.kinematics.attach(run)
        self._run_speeds = {}

    def _assign_rule45_speeds(self) -> None:
        """Regel 4.5 (Kap. 4.5): schnellste coverage-erhaltende
        Geschwindigkeit je Run zuweisen.

        Wie umgesetzt
        -------------
        Zuerst werden alle Runs bei Basisgeschwindigkeit angeheftet
        (Vorbedingung von ``speed_up_runs``). ``speed_up_runs``
        bestimmt dann je Run analytisch die Zielgeschwindigkeit aus der
        benötigten Schnitttiefe, heftet EINMAL bei v_r an und prüft
        exakt, dass die Basis-Abdeckung erhalten bleibt (sonst Rückfall
        auf die Basisgeschwindigkeit). Die Runs bleiben bei ihrer
        zugewiesenen Geschwindigkeit angeheftet -- Swept Areas und
        Animation zeigen die kürzere Klinge L(v).
        """
        tools = _surrogate_tools()
        phys_from_cutter, speed_up_runs = tools.phys_from_cutter, tools.speed_up_runs
        self._reset_run_speeds()
        phys = phys_from_cutter(self.cutter)
        self._run_speeds = speed_up_runs(
            self._runs, np.asarray(self.grid.coords, dtype=float),
            self.material, phys, self.kerf_width)

    def _apply_run_speeds(self, plan: CutPlan) -> None:
        """Rechnet die Schnittzeiten des Plans auf die zugewiesenen
        Regel-4.5-Geschwindigkeiten um.

        Reihenfolge und Verbindungen bleiben gültig: der Sequencer
        minimiert nur die Übergänge (Eilgang + Zündungen), die von der
        Schnittgeschwindigkeit unabhängig sind. Nur ``cut_time`` und
        die Schrittdauern ändern sich.
        """
        plan.cut_time = 0.0
        for step in plan.steps:
            if step.kind != "cut" or step.run is None:
                continue
            run = step.run
            cut_len = run.tcp_length if run.tcp_length > 0 else run.length
            v = self._run_speeds.get(run.run_id, self.cutter.cutting_speed)
            t_cut = cut_len / max(v, 1e-9)
            step.duration = t_cut + (self.cutter.pierce_time()
                                     if step.needs_pierce else 0.0)
            plan.cut_time += t_cut

    def _assign_rule45_chains(self) -> list | None:
        """DP-Split-Kettenausführung für eine Planer-Auswahl (Taste P/S/B).

        Was passiert
        ------------
        Baut aus der gemerkten Segment-Auswahl (``_chain_sel``) die
        exakt verifizierten Ketten der gemeinsamen Planer-Stufe
        (``build_speed_chains``): jede zusammenhängende Kette zerfällt
        per DP in Sub-Runs mit eigener Geschwindigkeit; Übergänge
        zwischen Sub-Runs sind nahtlos (kein Pierce, kein Eilgang, nur
        ``speed_switch_time`` bei Geschwindigkeitswechsel). Genau damit
        rechnet der Brute-Force-Lehrer sein T aus -- Enter reproduziert
        es dadurch exakt.

        Wie umgesetzt
        -------------
        ``self._runs`` wird durch die Sub-Runs ersetzt (Anzeige +
        Coverage laufen wie gewohnt über die Run-Liste), die
        Geschwindigkeiten landen in ``_run_speeds``. Rückgabe ist die
        Kettenliste für ``_build_chain_plan``; None, wenn keine Kette
        gebaut werden konnte (Aufrufer fällt auf den Je-Run-Pfad
        zurück).
        """
        tools = _surrogate_tools()
        phys = tools.phys_from_cutter(self.cutter)
        chains = tools.build_speed_chains(
            self.contour, self._chain_sel, self.material, phys,
            self.kerf_width, np.asarray(self.grid.coords, dtype=float))
        if not chains:
            return None
        self._runs = [r for ch in chains for r in ch.sub_runs]
        self._run_speeds = {r.run_id: float(v)
                            for ch in chains
                            for r, v in zip(ch.sub_runs, ch.speeds)}
        return chains

    def _build_chain_plan(self, chains: list) -> CutPlan:
        """Plant Reihenfolge + Verbindungen für Ketten (CutPlan-Schritte).

        Wie die Planer-Pipeline: der Sequencer (Held-Karp) sieht EINEN
        Makro-Knoten je Kette (Endpunkte, Richtung frei); danach werden
        die Sub-Runs in Kettenreihenfolge vom gewählten Ende aus
        ausgerollt. Nahtlose Sub-Run-Übergänge kosten weder Pierce noch
        Eilgang, nur ``speed_switch_time`` bei Geschwindigkeitswechsel;
        zwischen Ketten wird wie bisher verfahren + neu gezündet. Die
        Schritte tragen die Link-Geometrie -> ``_build_frames`` kann den
        Plan unverändert animieren.
        """
        macros = [ch.macro for ch in chains]
        ordered, is_opt = self.sequencer.order_runs(macros)
        by_id = {ch.macro.run_id: ch for ch in chains}
        t_switch = float(getattr(self.cutter, "speed_switch_time", 0.0))

        plan = CutPlan(is_optimal=is_opt)
        prev_end: np.ndarray | None = None
        prev_v: float | None = None
        for macro in ordered:
            chain = by_id[macro.run_id]
            subs, speeds = chain.rolled_out(macro)
            for run, v in zip(subs, speeds):
                needs_pierce = True
                if prev_end is not None:
                    gap = float(np.linalg.norm(run.tcp_start - prev_end))
                    if gap < CHAIN_TOL:
                        # Nahtlos: Brenner bleibt an; nur t_switch bei
                        # Geschwindigkeitswechsel.
                        needs_pierce = False
                        if prev_v is not None and abs(v - prev_v) > 1e-9:
                            plan.switch_time += t_switch
                            plan.n_switches += 1
                    else:
                        link = self.sequencer.link_between(prev_end,
                                                           run.tcp_start)
                        if link is None:
                            raise LinkInfeasibleError(
                                f"Kein kollisionsfreier Verfahrweg zu "
                                f"R{run.run_id}.")
                        t_link = link.length / self.cutter.rapid_speed
                        plan.steps.append(PlannedStep(
                            kind="link", link=link, duration=t_link))
                        plan.travel_time += t_link
                        plan.travel_length += link.length
                cut_len = run.tcp_length if run.tcp_length > 0 else run.length
                t_cut = cut_len / max(v, 1e-9)
                t_pierce = (self.cutter.pierce_time() if needs_pierce
                            else 0.0)
                plan.steps.append(PlannedStep(
                    kind="cut", run=run, needs_pierce=needs_pierce,
                    duration=t_cut + t_pierce))
                plan.cut_time += t_cut
                plan.cut_length += cut_len
                plan.pierce_time += t_pierce
                if needs_pierce:
                    plan.n_pierces += 1
                prev_end = run.tcp_end
                prev_v = float(v)
        return plan

    def _full_reset(self) -> None:
        """Setzt die Simulation komplett auf Anfang zurück (Taste R).

        Was passiert: stoppt eine laufende Animation, verwirft alle
        gewählten Segmente, Plan, Score, Coverage-Maske und Status und
        stellt die Abspielgeschwindigkeit auf 1x.
        Wie umgesetzt: alle Zustandsfelder zurücksetzen, den Speed-Slider
        (falls vorhanden) auf 1.0 stellen und neu zeichnen.
        """
        self._stop_animation(redraw=False)
        self._runs.clear()
        self._plan = None
        self._score = None
        self._pending = None
        self._anim_mask = None
        self._run_speeds = {}
        self._chain_sel = None
        self._state = _State.IDLE
        self._status_msg = ""
        self._speed = 1.0
        if self._speed_slider is not None:
            self._speed_slider.set_val(1.0)
        self._redraw()

    # ------------------------------------------------------------------
    # Planung + Animation
    # ------------------------------------------------------------------

    def _plan_and_animate(self) -> None:
        """Enter-Aktion: Reihenfolge planen und die Ausführung animieren.

        Was passiert
        ------------
        Aus der aktuellen Auswahl wird ein vollständiger Ablaufplan
        gebaut (beste Reihenfolge + kollisionsfreie Verbindungswege),
        bewertet (Punktezahl) und anschließend Schritt für Schritt als
        Animation abgespielt.

        Wie umgesetzt
        -------------
        1. Vorprüfung: Enthält die Auswahl nicht ausführbare Runs,
           bricht die Methode mit Hinweis ab.
        2. ``sequencer.build_plan`` liefert den geordneten ``CutPlan``;
           eine ``LinkInfeasibleError`` (keine kollisionsfreie Verbindung)
           wird abgefangen und gemeldet.
        3. ``compute_score`` bewertet Zeit + Coverage; Konsolenausgabe der
           Kennzahlen.
        4. ``_build_frames`` diskretisiert den Plan, die Coverage-Maske
           wird geleert, der Zustand auf ANIMATING gesetzt und
           ``_run_animation`` gestartet.
        """
        # Schritt 1: nicht ausführbare Segmente blockieren die Planung.
        infeasible = [r for r in self._runs if not r.is_feasible]
        if infeasible:
            names = ", ".join(f"R{r.run_id}" for r in infeasible)
            self._status_msg = (f"Not feasible: {names} infeasible "
                                f"({infeasible[0].reason}). Remove with U.")
            self._draw_stats()
            return

        # Regel 4.5: Geschwindigkeiten zuweisen (AN) bzw. eine frühere
        # Anhebung zurücknehmen (AUS) -- VOR dem Coverage-Report, damit
        # dieser die tatsächlichen Swept Areas bewertet. Stammt die
        # Auswahl von einem Planer (Taste P/S/B, ``_chain_sel``), wird die
        # DP-Split-Kettenausführung genutzt (Sub-Runs mit eigener
        # Geschwindigkeit, nahtlos) -- dieselbe Semantik, mit der der
        # Brute-Force-Lehrer sein T berechnet. Manuelle Auswahlen sind
        # nicht segment-ausgerichtet -> Regel 4.5 je Run wie bisher.
        chains = None
        if self._use_rule45:
            self._status_msg = "Rule 4.5: assigning speeds ..."
            self._draw_stats()
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            if self._chain_sel:
                chains = self._assign_rule45_chains()
            if chains is None:
                self._assign_rule45_speeds()
        else:
            self._reset_run_speeds()

        report = self.grid_coverage()
        print()
        print(report.summary())

        self._status_msg = "Planning sequence + links ..."
        self._draw_stats()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        # Schritt 2: Reihenfolge + Verbindungen berechnen. Schlägt eine
        # kollisionsfreie Verbindung fehl, ist der Plan nicht umsetzbar.
        try:
            self._plan = (self._build_chain_plan(chains) if chains
                          else self.sequencer.build_plan(self._runs))
        except LinkInfeasibleError as exc:
            self._plan = None
            self._status_msg = f"Not plannable: {exc}"
            print()
            print(f"[INFEASIBLE] {exc}")
            self._redraw()
            return
        # Regel 4.5 je Run: Schnittzeiten des Plans auf die zugewiesenen
        # Geschwindigkeiten umrechnen (Reihenfolge bleibt gültig). Der
        # Ketten-Plan traegt seine Geschwindigkeiten bereits.
        if self._run_speeds and not chains:
            self._apply_run_speeds(self._plan)
        self._score = compute_score(self._plan.total_time, report.fraction)
        print(self._plan.summary())
        order = " -> ".join(f"R{r.run_id}" for r in self._plan.runs_in_order)
        print(f"  Sequence: {order}")
        if self._run_speeds:
            v_txt = ", ".join(
                f"R{r.run_id}: {self._run_speeds.get(r.run_id, self.cutter.cutting_speed):.1f}"
                for r in self._plan.runs_in_order)
            print(f"  Rule 4.5 v [mm/s]: {v_txt}")
        print(f"  Score: {self._score:.0f}")

        # Schritt 4: Plan in Einzelbilder zerlegen, Coverage-Maske leeren
        # und die Wiedergabe starten.
        frames = self._build_frames(self._plan)
        if not frames:
            self._status_msg = "Nothing to simulate."
            self._redraw()
            return

        self._anim_mask = np.zeros(self.grid.total_points, dtype=bool)
        self._state = _State.ANIMATING
        self._run_animation(frames)

    def _build_frames(self, plan: CutPlan) -> list[_Frame]:
        """Diskretisiert den Plan in Animations-Frames.

        Was passiert
        ------------
        Wandelt den abstrakten Plan (eine Folge aus Schnitt- und
        Verfahr-Schritten) in eine flache, zeitlich geordnete Liste von
        Einzelbildern um, die ``_run_animation`` der Reihe nach abspielt.

        Wie umgesetzt
        -------------
        Eine mitlaufende Uhr ``t`` summiert die Dauer jedes Teilstücks.
        Die Hilfsfunktion ``along`` erzeugt Frames entlang einer
        Polylinie: Pro Kante wird die Fahrzeit ``dt = Strecke/Tempo``
        bestimmt und in ``round(dt*fps)`` Frames unterteilt; Position und
        Klingenspitze werden dabei linear interpoliert. Pro Plan-Schritt:
        bei "cut" zuerst optional Zünd-Frames (pierce, am Ort stehend),
        dann der Schnitt mit Schnittgeschwindigkeit; bei "link" ein
        Verfahrweg mit Eilganggeschwindigkeit (ohne Klinge).
        """
        frames: list[_Frame] = []
        t = 0.0

        def along(pts: np.ndarray, tips: np.ndarray | None,
                  speed: float, mode: str, run_id: int = -1) -> None:
            """Frames entlang einer Polylinie (TCP), Klinge interpoliert.

            Geht jede Kante (pts[i] -> pts[i+1]) durch, berechnet die
            Fahrzeit aus Länge/Tempo und verteilt darauf so viele Frames,
            dass die Ziel-Framerate ``fps`` erreicht wird. ``tips`` (die
            Klingenspitzen) werden synchron mit ``f`` linear interpoliert;
            ``None`` bedeutet "keine Klinge" (Verfahren).
            """
            nonlocal t
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                d = float(np.linalg.norm(p2 - p1))
                if d < 1e-9:
                    continue  # Null-Kante (doppelter Punkt) überspringen
                dt = d / speed
                # Anzahl Frames so wählen, dass ~fps Bilder/s entstehen.
                n_f = max(1, int(round(dt * self.fps)))
                for k in range(1, n_f + 1):
                    f = k / n_f  # Fortschritt 0..1 entlang der Kante
                    tip = None
                    if tips is not None:
                        tip = tips[i] + f * (tips[i + 1] - tips[i])
                    frames.append(_Frame(
                        time=t + f * dt,
                        pos=p1 + f * (p2 - p1),
                        tip=tip, mode=mode, run_id=run_id))
                t += dt

        for step in plan.steps:
            if step.kind == "cut" and step.run is not None:
                run = step.run
                # TCP fährt auf dem Offset-Pfad; Fallback auf die Kontur,
                # falls kein eigener TCP-Pfad berechnet wurde.
                tcp = (run.tcp_polyline if run.tcp_polyline is not None
                       else run.polyline)
                tips = run.tip_polyline
                if step.needs_pierce:
                    # Zünden: Brenner steht still (pos = Startpunkt),
                    # Klinge sticht ins Material. Als mehrere Frames über
                    # die Pierce-Zeit, damit die Animation gleichmäßig
                    # weiterläuft.
                    t_p = self.cutter.pierce_time()
                    n_f = max(2, int(round(t_p * self.fps)))
                    tip0 = tips[0] if tips is not None else None
                    for k in range(n_f):
                        frames.append(_Frame(
                            time=t + (k / n_f) * t_p,
                            pos=tcp[0], tip=tip0, mode="pierce",
                            run_id=run.run_id))
                    t += t_p
                # Eigentlicher Schnitt: Regel-4.5-Geschwindigkeit des
                # Runs (falls zugewiesen), sonst Basisgeschwindigkeit.
                v_run = self._run_speeds.get(run.run_id,
                                             self.cutter.cutting_speed)
                along(tcp, tips, v_run, "cut", run_id=run.run_id)
            elif step.kind == "link" and step.link is not None:
                # Verbindungsfahrt zwischen zwei Schnitten: Eilgang, keine
                # Klinge (tips=None) -> Modus "link".
                link = step.link
                along(link.points, None, self.cutter.rapid_speed, "link")
        return frames

    def _run_animation(self, frames: list[_Frame]) -> None:
        """Spielt die Frame-Liste als matplotlib-Animation ab.

        Was passiert
        ------------
        Zeigt den Brenner (Kreis), die Klinge (Linie mit Glow), die
        TCP- und Verfahr-Spuren sowie eine live mitwachsende Coverage
        (gestempelte Punkte) und eine Info-Box mit Zeit/Status/Fortschritt.

        Wie umgesetzt
        -------------
        Zuerst werden alle Animations-Artists einmalig angelegt
        (``torch``, ``blade_line``/``blade_glow``, ``trail_*``,
        ``live_scatter``, ``info``). Die geschachtelte ``update(idx)``
        setzt sie pro Frame neu; ``stamp`` markiert die vom Klingen-
        Viereck überstrichenen Gitterpunkte. ``frame_gen`` liefert die
        Frame-Indizes und überspringt bei hohem Speed Bilder. Eine
        ``FuncAnimation`` treibt das Ganze mit ``blit=False`` (es ändern
        sich zu viele Artists für Blitting).
        """
        self._redraw()
        ax = self._ax_main

        # --- Artists einmalig anlegen (werden pro Frame nur aktualisiert) ---
        torch = mpatches.Circle(
            tuple(frames[0].pos), self.grid.contour_spacing * 0.9,
            facecolor=_C["torch"], edgecolor=_C["torch_edge"],
            linewidth=2.0, alpha=0.9, zorder=25)
        ax.add_patch(torch)
        blade_line, = ax.plot([], [], color=_C["blade"], linewidth=3.0,
                              solid_capstyle="round", alpha=0.85, zorder=24)
        blade_glow, = ax.plot([], [], color=_C["blade_glow"], linewidth=6.5,
                              solid_capstyle="round", alpha=0.20, zorder=23)
        trail_tcp, = ax.plot([], [], color=_C["tcp_trail"], linewidth=1.6,
                             alpha=0.6, zorder=12, solid_capstyle="round")
        trail_link, = ax.plot([], [], "--", color=_C["link"], linewidth=1.4,
                              alpha=0.7, zorder=11)
        info = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=9,
                       va="top", family="monospace", zorder=30,
                       bbox=dict(boxstyle="round", fc="white", alpha=0.88))

        tcp_xs: list[float] = []
        tcp_ys: list[float] = []
        link_xs: list[float] = []
        link_ys: list[float] = []
        live_scatter = ax.scatter([], [], s=30, color=_C["covered"],
                                  edgecolors="#115522", linewidths=0.4,
                                  zorder=14, marker="P", alpha=0.9)
        live_pts: list[np.ndarray] = []

        n_frames = len(frames)
        total_time = frames[-1].time
        sim = self
        coords = self.grid.coords
        # prev hält Modus/Position/Spitze des VORIGEN Frames -- nötig,
        # um den Klingen-Sweep zwischen zwei Frames als Viereck zu stempeln
        # und um Spurenwechsel (NaN-Trenner) zu erkennen.
        prev = {"mode": "", "pos": None, "tip": None}

        import shapely as _shp

        def stamp(p1, t1, p2, t2):
            """Markiert Gitterpunkte im Klingen-Viereck p1-p2-t2-t1.

            Bildet aus zwei aufeinanderfolgenden Klingenlagen (TCP+Spitze
            bei Frame n-1 und n) das überstrichene Viereck, weitet es um
            den halben Kerf auf und markiert alle darin liegenden, noch
            nicht abgedeckten Gitterpunkte. So wächst die Live-Coverage.
            """
            try:
                quad = Polygon([tuple(p1), tuple(p2), tuple(t2), tuple(t1)])
                if not quad.is_valid:
                    quad = quad.buffer(0)  # selbstschneidend -> reparieren
                area = quad.buffer(sim.kerf_width / 2)
            except Exception:
                return
            # Nur noch nicht abgedeckte Punkte testen (spart Rechenzeit).
            rem = ~sim._anim_mask
            if not rem.any():
                return
            idx = np.where(rem)[0]
            # Vektorisierter Point-in-Polygon-Test für alle Restpunkte.
            hit = _shp.contains_xy(area, coords[idx, 0], coords[idx, 1])
            new_idx = idx[hit]
            if len(new_idx):
                sim._anim_mask[new_idx] = True
                live_pts.extend(coords[i] for i in new_idx)
                live_scatter.set_offsets(np.asarray(live_pts))

        def update(idx: int):
            """Zeichnet einen einzelnen Frame (Callback der FuncAnimation).

            Setzt Brennerposition, Klinge, Spuren, Coverage-Stempel und
            die Info-Box für den Frame ``idx`` und gibt am Ende des
            letzten Frames an ``_finish_animation`` ab.
            """
            fr = frames[idx]
            torch.center = tuple(fr.pos)

            # Klinge nur während Schneiden/Zünden zeigen, sonst leeren.
            if fr.tip is not None and fr.mode in ("cut", "pierce"):
                blade_line.set_data([fr.pos[0], fr.tip[0]],
                                    [fr.pos[1], fr.tip[1]])
                blade_glow.set_data([fr.pos[0], fr.tip[0]],
                                    [fr.pos[1], fr.tip[1]])
            else:
                blade_line.set_data([], [])
                blade_glow.set_data([], [])

            # Trails: NaN-Trenner bei Moduswechsel
            if fr.mode in ("cut", "pierce"):
                if prev["mode"] not in ("cut", "pierce") and tcp_xs:
                    tcp_xs.append(float("nan"))
                    tcp_ys.append(float("nan"))
                tcp_xs.append(fr.pos[0])
                tcp_ys.append(fr.pos[1])
                trail_tcp.set_data(tcp_xs, tcp_ys)
            else:
                if prev["mode"] in ("cut", "pierce", "") and link_xs:
                    link_xs.append(float("nan"))
                    link_ys.append(float("nan"))
                link_xs.append(fr.pos[0])
                link_ys.append(fr.pos[1])
                trail_link.set_data(link_xs, link_ys)

            # Live-Coverage: Klingen-Sweep zwischen zwei Frames stempeln.
            # Beim Zünden (pierce) nur einmal die degenerierte Linie
            # stempeln; beim Schneiden (cut) das Viereng vom vorigen zum
            # aktuellen Frame -- aber nur bei stetigem Schnittverlauf.
            if fr.mode == "pierce" and fr.tip is not None:
                if prev["mode"] != "pierce":
                    stamp(fr.pos, fr.tip, fr.pos, fr.tip)
            elif (fr.mode == "cut" and fr.tip is not None
                    and prev["pos"] is not None and prev["tip"] is not None
                    and prev["mode"] in ("cut", "pierce")):
                stamp(prev["pos"], prev["tip"], fr.pos, fr.tip)

            # Vorigen Frame für den nächsten Sweep/Trenner merken.
            prev["mode"] = fr.mode
            prev["pos"] = fr.pos
            prev["tip"] = fr.tip

            n_cov = int(sim._anim_mask.sum())
            n_tot = sim.grid.total_points
            mode_txt = {
                "cut": "Cutting", "pierce": "Piercing",
                "link": "Traversing",
            }[fr.mode]
            info.set_text(
                f"t = {fr.time:5.1f} / {total_time:.1f} s\n"
                f"Status: {mode_txt}\n"
                f"Points: {n_cov}/{n_tot} "
                f"({100.0 * n_cov / max(1, n_tot):.1f} %)")

            # Letzter Frame: Animation sauber beenden (Score, Endstatus).
            if idx >= n_frames - 1:
                sim._finish_animation()
            return (torch, blade_line, blade_glow, trail_tcp, trail_link,
                    info, live_scatter)

        def frame_gen():
            """Liefert die abzuspielenden Frame-Indizes der Reihe nach.

            Bis Speed 2x wird jeder Frame gezeigt (Beschleunigung läuft
            über das Timer-Intervall). Darüber werden Frames in Schritten
            übersprungen (step = speed/2), damit die Wiedergabe trotz
            begrenzter Framerate schneller wird; der letzte Frame wird
            immer ausgegeben, damit das Ende sicher erreicht wird.
            """
            cur = 0
            while cur < n_frames - 1:
                yield cur
                step = 1 if sim._speed <= 2.0 else int(sim._speed / 2.0)
                cur += max(1, step)
            yield n_frames - 1

        # Timer-Intervall: bis 2x echte FPS erhöhen, darüber konstant
        # (dann übernimmt frame_gen das Überspringen).
        effective_fps = self.fps * min(self._speed, 2.0)
        self._anim = FuncAnimation(
            self._fig, update, frames=frame_gen, save_count=n_frames,
            interval=max(1, int(1000 / effective_fps)),
            blit=False, repeat=False)
        self._fig.canvas.draw_idle()

    def _stop_animation(self, redraw: bool = True) -> None:
        """Bricht eine laufende Animation ab (Esc/Reset).

        Was passiert: stoppt die Wiedergabe sofort und schaltet zurück
        in den Bearbeitungszustand IDLE -- im Gegensatz zu
        ``_finish_animation`` OHNE Endbewertung.
        Wie umgesetzt: den Timer der ``FuncAnimation`` stoppen (Fehler
        ignorieren, falls schon weg), Referenz löschen und optional neu
        zeichnen.
        """
        if self._anim is not None:
            try:
                self._anim.event_source.stop()
            except AttributeError:
                pass
            self._anim = None
        self._state = _State.IDLE
        if redraw:
            self._redraw()

    def _finish_animation(self) -> None:
        """Schließt eine vollständig abgespielte Animation regulär ab.

        Was passiert: stoppt den Timer, ermittelt die endgültige
        Coverage und Punktezahl und zeigt eine Abschlussmeldung
        (100 % durchtrennt vs. fehlende Punkte).
        Wie umgesetzt: ``FuncAnimation`` stoppen, ``grid_coverage`` +
        ``compute_score`` auswerten, Statustext setzen und mit
        ``keep_animation_artists=True`` neu zeichnen, damit Brenner/
        Klinge/Spuren des Endbildes sichtbar bleiben.
        """
        if self._anim is not None:
            try:
                self._anim.event_source.stop()
            except AttributeError:
                pass
            self._anim = None
        self._state = _State.IDLE
        report = self.grid_coverage()
        self._score = (compute_score(self._plan.total_time, report.fraction)
                       if self._plan else None)
        score_txt = (f" | Score: {self._score:.0f}"
                     if self._score is not None else "")
        if report.is_complete:
            self._status_msg = (f"Done: cross-section fully cut "
                                f"(100 %)!{score_txt}")
        else:
            self._status_msg = (f"Done, but {report.missing_fraction:.1%} "
                                f"of the points are missing!{score_txt}")
        print(f"\n{self._status_msg}")
        self._redraw(keep_animation_artists=True)

    # ------------------------------------------------------------------
    # Zeichnen
    # ------------------------------------------------------------------

    def _redraw(self, keep_animation_artists: bool = False) -> None:
        """Zeichnet beide Panels neu (Hauptbild + Statistik).

        Bequemer Sammelaufruf nach jeder Zustandsänderung;
        ``keep_animation_artists`` wird an ``_draw_main`` durchgereicht,
        um das Endbild der Animation nicht zu löschen.
        """
        self._draw_main(keep_animation_artists)
        self._draw_stats()

    def _draw_main(self, keep_animation_artists: bool = False) -> None:
        """Zeichnet die große linke Fläche: Geometrie, Auswahl, Plan.

        Was passiert
        ------------
        Stellt den kompletten aktuellen Zustand dar: alle Gitterpunkte
        (grau / grün abgedeckt / rot fehlend), die alternierend
        eingefärbten Segmente, die gewählten Runs (Swept Area, TCP-Pfad,
        Konturbogen mit Richtungspfeil und Label), die Segmentknoten, die
        geplanten Verbindungen, den Pending-Startpunkt sowie Legende und
        Achsen.

        Wie umgesetzt
        -------------
        Bei ``keep_animation_artists`` wird die Achse NICHT geleert
        (sonst verschwände das Endbild der Animation) und nur ein
        ``draw_idle`` ausgelöst. Sonst ``ax.clear()`` und schrittweiser
        Neuaufbau mit ``scatter``/``plot``/``fill``/``annotate``; die
        Achsgrenzen werden aus der Bounding-Box aller Loops plus einem
        Rand (inkl. Mindestabstand) gesetzt.
        """
        ax = self._ax_main
        # Bei Animationsende nicht alles löschen, nur Statistik anpassen
        if keep_animation_artists:
            self._fig.canvas.draw_idle()
            return
        ax.clear()
        ax.set_facecolor(_C["grid_bg"])

        # Querschnitts-Coverage der aktuellen Auswahl
        report = self.grid_coverage() if self._runs else None
        coords = self.grid.coords

        # Alle Gitterpunkte: Basis grau, abgedeckt grün, fehlend rot
        if report is not None:
            cov_idx = np.where(report.mask)[0]
            mis_idx = np.where(~report.mask)[0]
            if len(cov_idx):
                ax.scatter(coords[cov_idx, 0], coords[cov_idx, 1], s=26,
                           color=_C["covered"], edgecolors="#115522",
                           linewidths=0.3, zorder=6, marker="P", alpha=0.85)
            if len(mis_idx):
                ax.scatter(coords[mis_idx, 0], coords[mis_idx, 1], s=22,
                           color=_C["missing"], linewidths=1.0,
                           zorder=5, marker="x", alpha=0.8)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=10,
                       color=_C["inner"], alpha=0.5, zorder=1)

        # Primitiv-Segmente alternierend einfärben
        for seg in self.contour.segments:
            loop = self.contour.loop_by_id(seg.loop_id)
            palette = (_C["seg_colors"] if loop.kind == "outer"
                       else _C["seg_hole"])
            color = palette[seg.seg_id % 2]
            pts = loop.polyline(seg.positions)
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.2,
                    alpha=0.8, zorder=4, solid_capstyle="round")

        # Gewählte Runs: Swept Area + TCP-Pfad + Konturbogen
        for i, run in enumerate(self._runs):
            # Feasible Runs zyklisch aus der Farbpalette, nicht
            # ausführbare Runs einheitlich in der Infeasible-Farbe.
            c = (_C["run_colors"][i % len(_C["run_colors"])]
                 if run.is_feasible else _C["infeasible"])
            # Swept Area (überstrichene Fläche)
            if run.is_feasible and run.swept_polygon is not None:
                geoms = (run.swept_polygon.geoms
                         if hasattr(run.swept_polygon, "geoms")
                         else [run.swept_polygon])
                for g in geoms:
                    try:
                        sx, sy = g.exterior.xy
                        ax.fill(sx, sy, color=c, alpha=0.10, zorder=3)
                    except AttributeError:
                        pass
            # TCP-Offset-Pfad
            if run.tcp_polyline is not None:
                ax.plot(run.tcp_polyline[:, 0], run.tcp_polyline[:, 1],
                        "--" if run.is_feasible else ":",
                        color=c, linewidth=1.6, alpha=0.8, zorder=9)
            # Konturbogen bold
            ax.plot(run.polyline[:, 0], run.polyline[:, 1], color=c,
                    linewidth=4.0, alpha=0.45, zorder=8,
                    solid_capstyle="round")
            # Beschriftung "Rn" in der Bogenmitte ("!" markiert infeasible).
            mid = run.polyline[len(run.polyline) // 2]
            label = f"R{run.run_id}" + ("" if run.is_feasible else " !")
            ax.annotate(label, xy=tuple(mid), fontsize=8.5,
                        fontweight="bold", color=c, zorder=16,
                        xytext=(4, 4), textcoords="offset points")
            # Richtungspfeil am Bogenende zeigt die Schnittrichtung an.
            if len(run.polyline) >= 2:
                p1, p2 = run.polyline[-2], run.polyline[-1]
                ax.annotate("", xy=tuple(p2), xytext=tuple(p1),
                            arrowprops=dict(arrowstyle="-|>", color=c,
                                            lw=2.0), zorder=16)

        # Segment-Knoten (= mögliche Start-/Endpunkte)
        for loop in self.contour.loops:
            nodes = self.contour.nodes.get(loop.loop_id, [])
            if nodes:
                npts = loop.points[nodes]
                ax.scatter(npts[:, 0], npts[:, 1], s=70, color=_C["node"],
                           edgecolors=_C["node_edge"], linewidths=1.3,
                           zorder=10)

        # Geplante Verbindungen (nach Enter)
        if self._plan is not None:
            for step in self._plan.steps:
                if step.kind == "link" and step.link is not None:
                    pts = step.link.points
                    style = dict(linewidth=1.4, alpha=0.65, zorder=9)
                    ax.plot(pts[:, 0], pts[:, 1], "--",
                            color=_C["link"], **style)

        # Pending-Startpunkt
        if self._pending is not None:
            loop = self.contour.loop_by_id(self._pending[0])
            p = loop.points[self._pending[1]]
            ax.plot(p[0], p[1], "*", color=_C["pending"], markersize=20,
                    markeredgecolor="#806000", markeredgewidth=1.2,
                    zorder=18)

        # Legende
        ax.legend(handles=[
            Line2D([0], [0], color=_C["seg_colors"][0], linewidth=2.5,
                   label="Segment (contour)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_C["node"],
                   markeredgecolor=_C["node_edge"], markersize=9,
                   label="Node (start/end)"),
            Line2D([0], [0], linestyle="--", color=_C["run_colors"][0],
                   label="TCP path (standoff)"),
            Line2D([0], [0], color=_C["blade"], linewidth=3,
                   label="Plasma arc L(v)"),
            Line2D([0], [0], marker="P", color="w",
                   markerfacecolor=_C["covered"], markersize=9,
                   label="Point covered"),
            Line2D([0], [0], marker="x", color=_C["missing"], linestyle="",
                   markersize=8, label="Point missing"),
            Line2D([0], [0], linestyle="--", color=_C["link"],
                   label="Rapid traverse (auto)"),
        ], fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.01),
            ncol=4, framealpha=0.92, edgecolor="#CCCCCC",
            labelspacing=0.45, columnspacing=1.4, borderaxespad=0.0)

        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.22, color="#888")

        # Achsgrenzen aus Bounding-Box aller Loops + Rand. Der Rand
        # wächst mit der Objektgröße (span) und enthält den
        # Mindestabstand, damit der außen liegende TCP-Pfad reinpasst.
        all_pts = np.vstack([l.points for l in self.contour.loops])
        span = max(float(np.ptp(all_pts[:, 0])), float(np.ptp(all_pts[:, 1])))
        m = span * 0.09 + 5.0 + self.cutter.minimum_gap
        ax.set_xlim(all_pts[:, 0].min() - m, all_pts[:, 0].max() + m)
        ax.set_ylim(all_pts[:, 1].min() - m, all_pts[:, 1].max() + m)

        ax.text(0.5, -0.048,
                "Click: start/end  |  Right-click/U: undo  |  A: all  |  "
                "P: Greedy+  |  S: surrogate  |  B: brute force  |  "
                "V: rule 4.5  |  "
                "Enter: plan+start  |  R: reset  |  Esc: cancel",
                transform=ax.transAxes, fontsize=7.5, color="#888",
                ha="center", va="top")

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------

    def _draw_stats(self) -> None:
        """Zeichnet das rechte Info-Panel (Kennzahlen + Status).

        Was passiert
        ------------
        Fasst alle Zahlen zur aktuellen Lage zusammen: Querschnitts-
        Coverage, die (letzten) gewählten Segmente, den optimierten Plan
        mit Zeiten und Punktezahl, die Lichtschwert-/Cutter-Parameter und
        unten eine farbige Statuszeile.

        Wie umgesetzt
        -------------
        Die Achse wird geleert und als reines Textpanel genutzt
        (``axis("off")`` + Hintergrund-Box). Zwei lokale Helfer schreiben
        in Achsen-Koordinaten und führen die laufende y-Position ``y``
        nach unten: ``kv`` für Label/Wert-Zeilen, ``header`` für
        Abschnittsüberschriften. Farben signalisieren Vollständigkeit
        (grün), Warnung (orange) oder Fehler (rot).
        """
        ax = self._ax_stats
        ax.clear()
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.01), 0.96, 0.97, boxstyle="round,pad=0.01",
            facecolor=_C["stats_bg"], edgecolor="#B0C4DE",
            linewidth=1.0, transform=ax.transAxes, zorder=0))

        ax.text(0.5, 0.965, "SEGMENT PLAN", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="center", va="top",
                color="#1A2840")
        ax.plot([0.08, 0.92], [0.940, 0.940], color="#B0C4DE",
                linewidth=0.8, transform=ax.transAxes)

        # y = aktuelle Schreibhöhe (1.0 oben), dy = Zeilenabstand.
        y = 0.915
        dy = 0.037

        def kv(label, value, vc="#2D2D2D", bold=False):
            """Schreibt eine Label-links / Wert-rechts-Zeile und rückt y."""
            nonlocal y
            ax.text(0.07, y, label, transform=ax.transAxes,
                    fontsize=7.6, color="#555555", va="top")
            ax.text(0.93, y, value, transform=ax.transAxes,
                    fontsize=7.6 if not bold else 8.3, color=vc,
                    ha="right", va="top",
                    fontweight="bold" if bold else "normal")
            y -= dy

        def header(txt, color="#1A2840"):
            """Schreibt eine zentrierte Abschnittsüberschrift und rückt y."""
            nonlocal y
            y -= dy * 0.2
            ax.text(0.5, y, txt, transform=ax.transAxes, fontsize=8,
                    fontweight="bold", ha="center", va="top", color=color)
            y -= dy * 0.9

        # --- Querschnitts-Coverage (alle Punkte) ---
        report = self.grid_coverage()
        cov_color = "#16A34A" if report.is_complete else (
            "#CC6600" if report.fraction > 0 else "#555555")
        kv("Total points", str(report.total))
        kv("Covered", f"{report.covered} / {report.total}",
           vc=cov_color, bold=True)
        kv("Coverage", f"{report.fraction:.1%}", vc=cov_color, bold=True)
        if not report.is_complete:
            kv("Missing", f"{report.missing_fraction:.1%} "
               f"({report.total - report.covered} pts)",
               vc=_C["missing"], bold=report.fraction > 0)

        # --- Segmente ---
        # Aus Platzgründen nur die letzten n_show Runs auflisten; ein
        # "... N weitere" zeigt an, wie viele davor liegen.
        header("SELECTED SEGMENTS")
        if not self._runs:
            kv("(none)", "")
        n_show = 4
        if len(self._runs) > n_show:
            kv(f"... {len(self._runs) - n_show} more", "")
        offset = max(0, len(self._runs) - n_show)
        for i, run in enumerate(self._runs[-n_show:]):
            c = (_C["run_colors"][(offset + i) % len(_C["run_colors"])]
                 if run.is_feasible else _C["infeasible"])
            feas = "" if run.is_feasible else " !"
            kv(f"R{run.run_id} (Loop {run.loop_id}){feas}",
               f"{(run.tcp_length or run.length):.0f} mm", vc=c)

        # --- Plan + Punktezahl ---
        if self._plan is not None:
            header("PLAN (OPTIMIZED)", "#0B6E2F")
            kv("Sequence",
               "-".join(f"R{r.run_id}" for r in self._plan.runs_in_order),
               bold=True)
            kv("Optimality",
               "exact (time-optimal)" if self._plan.is_optimal else "heuristic",
               vc="#0B6E2F" if self._plan.is_optimal else "#CC6600")
            kv("Pierces", str(self._plan.n_pierces))
            kv("Cut time", f"{self._plan.cut_time:.1f} s")
            kv("Rapid", f"{self._plan.travel_time:.1f} s")
            kv("Pierce", f"{self._plan.pierce_time:.1f} s")
            if self._plan.switch_time > 0:
                kv("Speed changes", f"{self._plan.n_switches} x -> "
                   f"{self._plan.switch_time:.1f} s")
            kv("Total time", f"{self._plan.total_time:.1f} s",
               vc="#0B6E2F", bold=True)
            if self._run_speeds:
                vs = [self._run_speeds.get(r.run_id,
                                           self.cutter.cutting_speed)
                      for r in self._plan.runs_in_order]
                if vs:
                    kv("v per run (4.5)",
                       f"{min(vs):.1f} - {max(vs):.1f} mm/s",
                       vc="#0B6E2F")
            if self._score is not None:
                kv("SCORE", f"{self._score:.0f}",
                   vc="#B8860B", bold=True)

        # --- Plasma torch / cutter ---
        header("PLASMA TORCH")
        kv("Arc length L(v)", f"{self.blade_length:.1f} mm", bold=True)
        kv("Eff. depth", f"{self.kinematics.effective_depth:.1f} mm",
           vc="#16A34A" if self.kinematics.effective_depth > 0
           else _C["missing"])
        vmax = self.cutter.max_cutting_speed
        kv("Cut speed", f"{self.cutter.cutting_speed:.1f}"
           + (f" / max {vmax:.1f} mm/s" if vmax else " mm/s"))
        kv("Rule 4.5 (v per run)", "ON" if self._use_rule45 else "OFF",
           vc="#16A34A" if self._use_rule45 else "#888888",
           bold=self._use_rule45)
        kv("Rapid speed", f"{self.cutter.rapid_speed:.1f} mm/s")
        kv("Standoff (const.)", f"{self.cutter.minimum_gap:.1f} mm",
           vc="#CC6600")
        kv("Kerf", f"{self.kerf_width:.1f} mm")

        # --- Status ---
        # Grundfarbe nach Zustand; bei IDLE wird die Farbe zusätzlich
        # aus Schlüsselwörtern der Statusmeldung verschärft (Erfolg
        # grün, Fehler/fehlende Punkte rot).
        if self._state == _State.IDLE:
            status, sc = (self._status_msg or "Click start point"), "#4B5563"
        elif self._state == _State.PICK_END:
            status, sc = "Click end point ...", "#CC6600"
        else:
            status, sc = f"Simulation running ({self._speed:.2g}x)", "#CC2222"
        if self._status_msg and self._state == _State.IDLE:
            if "100 %" in self._status_msg:
                sc = "#16A34A"
            elif ("missing" in self._status_msg
                  or "feasible" in self._status_msg
                  or "not plannable" in self._status_msg.lower()):
                sc = "#CC2222"

        ax.text(0.5, 0.025, status, transform=ax.transAxes, fontsize=7.0,
                color=sc, ha="center", va="bottom", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=sc,
                          alpha=0.88, linewidth=0.8))
        self._fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Headless-API (für Tests / späteres supervised learning)
# ---------------------------------------------------------------------------

def check_segments(
    grid: PointGrid,
    node_pairs: list[tuple[int, int, int]],
    cutter: Cutter | None = None,
    kerf_width: float = 3.0,
    target_segment_length: float | None = None,
) -> tuple[CutPlan, GridCoverageReport, float]:
    """Minimalanforderung ohne UI: gegebene Segmente prüfen + ordnen.

    Parameters
    ----------
    node_pairs : Liste (loop_id, start_pos, end_pos) -- Knotenpositionen
                 wie von SegmentedContour vergeben.

    Returns
    -------
    (CutPlan, GridCoverageReport, score) -- optimierte Reihenfolge inkl.
    kollisionsfreier Verbindungen, Querschnitts-Abdeckung und
    Punktezahl (Zeit-basiert).

    Wie umgesetzt
    -------------
    Spiegelt den Datenfluss der GUI ohne Fenster: Kontur+Material aus
    dem Gitter, Kinematik-/Planer-Objekte aufbauen, für jedes Knoten-
    paar einen ``CutRun`` erzeugen und mit ``kin.attach`` die Klingen-
    geometrie ergänzen. Nur ausführbare Runs gehen in
    ``build_plan``; ``compute_grid_coverage`` und ``compute_score``
    liefern Abdeckung und Bewertung.
    """
    cutter = cutter or make_default_cutter()
    contour = SegmentedContour.from_grid(
        grid, target_segment_length=target_segment_length)
    material = contour.material_polygon()
    blade_length = cutter.blade_length(cutter.cutting_speed)
    kin = RunKinematics(material, clearance=cutter.minimum_gap,
                        blade_length=blade_length, kerf=kerf_width)
    planner = LinkPlanner(material, clearance=cutter.minimum_gap)
    sequencer = Sequencer(cutter, contour, planner)

    runs: list[CutRun] = []
    for i, (loop_id, a, b) in enumerate(node_pairs):
        covered = covered_positions(contour, runs, loop_id)
        run = contour.make_run(
            run_id=i + 1, loop_id=loop_id, start_pos=a, end_pos=b,
            covered=covered)
        kin.attach(run)
        runs.append(run)

    plan = sequencer.build_plan([r for r in runs if r.is_feasible])
    report = compute_grid_coverage(grid, runs)
    score = compute_score(plan.total_time, report.fraction)
    return plan, report, score


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # CLI-Einstieg: Argumente parsen, daraus einen Cutter bauen, die
    # Lichtschwert-Kennwerte ausgeben und die Simulation starten --
    # entweder mit fester --geometry oder über den Datei-Dialog.
    import argparse

    parser = argparse.ArgumentParser(
        description="Segment-based plasma cutter simulation "
                    "(plasma torch, cross-section coverage)")
    parser.add_argument("--geometry", type=str, default=None,
                        help="Path to the geometry JSON (otherwise a dialog)")
    parser.add_argument("--kerf-width", type=float, default=3.0)
    parser.add_argument("--segment-length", type=float, default=None,
                        help="Target segment length in mm (default: automatic)")
    parser.add_argument("--v-cut", type=float, default=DEFAULT_CUTTING_SPEED,
                        help="Cutting speed in mm/s")
    parser.add_argument("--v-max", type=float,
                        default=DEFAULT_MAX_CUTTING_SPEED,
                        help="Maximum cutting speed in mm/s")
    parser.add_argument("--v-rapid", type=float, default=RAPID_SPEED,
                        help="Rapid traverse speed in mm/s")
    parser.add_argument("--t-switch", type=float, default=SPEED_SWITCH_TIME,
                        help="Time penalty per speed change within a cut [s]")
    parser.add_argument("--blade-length", type=float, default=BLADE_LENGTH,
                        help="Base arc length L(v=0) in mm")
    parser.add_argument("--blade-slope", type=float, default=BLADE_SLOPE,
                        help="Arc-length reduction in mm per mm/s")
    parser.add_argument("--clearance", type=float, default=MINIMUM_GAP,
                        help="Constant standoff TCP-material in mm")
    args = parser.parse_args()

    cutter = make_default_cutter(
        cutting_speed=args.v_cut,
        max_cutting_speed=args.v_max,
        blade_length=args.blade_length,
        blade_slope=args.blade_slope,
        minimum_gap=args.clearance,
        rapid_speed=args.v_rapid,
        speed_switch_time=args.t_switch,
    )
    L = cutter.blade_length(cutter.cutting_speed)
    print(cutter)
    print(f"L(v) = {args.blade_length:.1f} - {args.blade_slope:.2f}*v  ->  "
          f"L({args.v_cut:.1f}) = {L:.1f} mm, "
          f"eff. depth = {L - args.clearance:.1f} mm")

    if args.geometry:
        grid = PointGrid.from_json(args.geometry)
        sim = SegmentCutSimulation(
            grid=grid, cutter=cutter, kerf_width=args.kerf_width,
            target_segment_length=args.segment_length)
        sim.run()
    else:
        SegmentCutSimulation.run_with_dialog(
            cutter=cutter, kerf_width=args.kerf_width,
            target_segment_length=args.segment_length)

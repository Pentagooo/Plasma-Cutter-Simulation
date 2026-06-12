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
  P           : Auto-Planer (waehlt die Segmente automatisch)
  Enter       : Planen + Simulation starten
  R           : alles zuruecksetzen
  Esc         : Auswahl/Animation abbrechen
  +/- / Slider: Animations-Geschwindigkeit
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from shapely.geometry import Polygon

try:
    from ..geometry.point_grid import PointGrid
    from ..cutter.cutter import Cutter
    from ..cutter.assumptions import BladeLengthModel, CuttingAssumptions
    from .segments import (
        SegmentedContour, CutRun, compute_coverage, covered_positions,
        compute_grid_coverage, GridCoverageReport,
    )
    from .planning import (
        LinkPlanner, Sequencer, CutPlan, RunKinematics,
        compute_score, CHAIN_TOL, LinkInfeasibleError,
    )
    from .autoplan import AutoPlanner
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.geometry.point_grid import PointGrid
    from plasma_cutter.cutter.cutter import Cutter
    from plasma_cutter.cutter.assumptions import (
        BladeLengthModel, CuttingAssumptions,
    )
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, CutRun, compute_coverage, covered_positions,
        compute_grid_coverage, GridCoverageReport,
    )
    from plasma_cutter.segment_simulation.planning import (
        LinkPlanner, Sequencer, CutPlan, RunKinematics,
        compute_score, CHAIN_TOL, LinkInfeasibleError,
    )
    from plasma_cutter.segment_simulation.autoplan import AutoPlanner


# ---------------------------------------------------------------------------
# Konstanten / Defaults
# ---------------------------------------------------------------------------

# Konstanter Mindestabstand TCP <-> Materialoberflaeche [mm].
# Wird dem Cutter als minimum_gap mitgegeben (CLI: --clearance).
MINIMUM_GAP = 3.0

# Lineares Klingenmodell  L(v) = BLADE_LENGTH - BLADE_SLOPE * v
BLADE_LENGTH = 27.5   # Grundlaenge der Schneide bei v = 0 [mm]
BLADE_SLOPE = 1.5     # Verkuerzung pro mm/s [mm / (mm/s)]

# Geschwindigkeiten [mm/s]
DEFAULT_CUTTING_SPEED = 5.0
DEFAULT_MAX_CUTTING_SPEED = 10.0


def make_default_cutter(
    cutting_speed: float = DEFAULT_CUTTING_SPEED,
    max_cutting_speed: float = DEFAULT_MAX_CUTTING_SPEED,
    blade_length: float = BLADE_LENGTH,
    blade_slope: float = BLADE_SLOPE,
    minimum_gap: float = MINIMUM_GAP,
    rapid_speed: float = 50.0,
) -> Cutter:
    """Cutter mit linearem L(v)-Klingenmodell fuer die Segment-Simulation."""
    blade = BladeLengthModel(
        mode="linear",
        L_ref=blade_length, v_ref=0.0, slope=-blade_slope,
        L_max=blade_length, L_min=0.0,
    )
    assumptions = CuttingAssumptions(
        blade=blade, use_velocity_dependent_blade=True)
    return Cutter(
        cutting_speed=cutting_speed,
        max_cutting_speed=max_cutting_speed,
        rapid_speed=rapid_speed,
        minimum_gap=minimum_gap,
        assumptions=assumptions,
    )


# ---------------------------------------------------------------------------
# Farben
# ---------------------------------------------------------------------------

_C = dict(
    inner        = "#BBBBBB",
    seg_colors   = ["#1E6FBF", "#6FA8DC"],     # alternierende Segmentfarben
    seg_hole     = ["#C0504D", "#E6A09E"],     # alternierend fuer Lochkontur
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
    IDLE      = auto()   # wartet auf Startpunkt-Klick
    PICK_END  = auto()   # Startpunkt gesetzt, wartet auf Endpunkt
    ANIMATING = auto()


@dataclass
class _Frame:
    """Ein Animations-Frame: Zeitpunkt, TCP, Klingenspitze, Modus."""
    time: float
    pos: np.ndarray              # TCP-Position
    tip: np.ndarray | None       # Klingenspitze (None bei link)
    mode: str                    # "pierce" | "cut" | "link"
    run_id: int = -1


# ---------------------------------------------------------------------------
# SegmentCutSimulation
# ---------------------------------------------------------------------------

class SegmentCutSimulation:
    """Interaktive Simulation: Segmente waehlen, pruefen, ausfuehren.

    Parameters
    ----------
    grid                   : PointGrid der Geometrie
    cutter                 : Cutter; minimum_gap = konstanter Mindestabstand,
                             blade_length(v) = lineare Klingenlaenge.
                             None -> make_default_cutter()
    kerf_width             : Schnittspaltbreite [mm]
    target_segment_length  : Ziel-Segmentlaenge [mm] (None = automatisch)
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
        self.grid = grid
        self.cutter = cutter or make_default_cutter()
        self.kerf_width = kerf_width
        self.fps = fps

        # Klingenlaenge bei der eingestellten Schnittgeschwindigkeit
        self.blade_length = self.cutter.blade_length(self.cutter.cutting_speed)

        self.contour = SegmentedContour.from_grid(
            grid, target_segment_length=target_segment_length)
        self.material = self.contour.material_polygon()
        self.kinematics = RunKinematics(
            self.material,
            clearance=self.cutter.minimum_gap,
            blade_length=self.blade_length,
            kerf=kerf_width,
        )
        self.link_planner = LinkPlanner(
            self.material, clearance=self.cutter.minimum_gap)
        self.sequencer = Sequencer(
            self.cutter, self.contour, self.link_planner)

        if self.kinematics.effective_depth <= 0:
            print(f"WARNUNG: Klinge zu kurz! L(v={self.cutter.cutting_speed}) "
                  f"= {self.blade_length:.1f} mm <= Mindestabstand "
                  f"{self.cutter.minimum_gap:.1f} mm -> kein Schnitt moeglich. "
                  f"Geschwindigkeit verringern oder Klinge verlaengern.")

        self._runs: list[CutRun] = []
        self._plan: CutPlan | None = None
        self._score: float | None = None
        self._state = _State.IDLE
        self._pending: tuple[int, int] | None = None  # (loop_id, pos)
        self._snap_radius = grid.contour_spacing * 4.0

        # Animations-Zustand
        self._anim: FuncAnimation | None = None
        self._speed = 1.0
        self._anim_mask: np.ndarray | None = None
        self._status_msg = ""

        self._fig: plt.Figure | None = None
        self._ax_main: plt.Axes | None = None
        self._ax_stats: plt.Axes | None = None
        self._speed_slider: Slider | None = None

    # ------------------------------------------------------------------

    @classmethod
    def run_with_dialog(cls, **kwargs) -> SegmentCutSimulation | None:
        initial_dir = kwargs.pop("initial_dir", None)
        grid = PointGrid.from_json_dialog(initial_dir=initial_dir)
        if grid is None:
            return None
        sim = cls(grid=grid, **kwargs)
        sim.run()
        return sim

    @property
    def runs(self) -> list[CutRun]:
        return list(self._runs)

    @property
    def plan(self) -> CutPlan | None:
        return self._plan

    def grid_coverage(self) -> GridCoverageReport:
        """Aktuelle Querschnitts-Coverage der gewaehlten Segmente."""
        return compute_grid_coverage(self.grid, self._runs)

    def run(self) -> None:
        self._fig = plt.figure(figsize=(15, 9), facecolor="white")
        self._fig.suptitle(self._title(), fontsize=11, fontweight="bold",
                           y=0.98, color="#1A2840")

        gs = self._fig.add_gridspec(
            1, 2, width_ratios=[4, 1.15],
            left=0.05, right=0.98, top=0.93, bottom=0.12, wspace=0.03)
        self._ax_main = self._fig.add_subplot(gs[0])
        self._ax_stats = self._fig.add_subplot(gs[1])

        ax_speed = self._fig.add_axes([0.15, 0.04, 0.4, 0.03])
        self._speed_slider = Slider(
            ax=ax_speed, label="Sim-Speed ",
            valmin=0.25, valmax=16.0, valinit=1.0,
            valstep=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            color="#0066CC")
        self._speed_slider.on_changed(self._on_speed)

        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._redraw()
        plt.show()

    def _title(self) -> str:
        name = (f"  -  {self.grid._source_path.stem}"
                if hasattr(self.grid, "_source_path") else "")
        return (f"Segment-Simulation{name}  |  "
                f"L(v={self.cutter.cutting_speed:.0f}) = "
                f"{self.blade_length:.1f} mm  |  "
                f"eff. Tiefe: {self.kinematics.effective_depth:.1f} mm  |  "
                f"Min. Abstand: {self.cutter.minimum_gap:.0f} mm (konst.)")

    # ------------------------------------------------------------------
    # Auswahl
    # ------------------------------------------------------------------

    def _add_run(self, loop_id: int, start_pos: int, end_pos: int) -> CutRun:
        """Erzeugt einen CutRun inkl. Lichtschwert-Kinematik."""
        covered = covered_positions(self.contour, self._runs, loop_id)
        run = self.contour.make_run(
            run_id=len(self._runs) + 1,
            loop_id=loop_id, start_pos=start_pos, end_pos=end_pos,
            covered=covered)
        self.kinematics.attach(run)
        self._runs.append(run)
        self._plan = None
        self._score = None
        return run

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_speed(self, val: float) -> None:
        self._speed = val
        if self._anim is not None and self._anim.event_source is not None:
            effective_fps = self.fps * min(self._speed, 2.0)
            self._anim.event_source.interval = max(1, int(1000 / effective_fps))
        self._draw_stats()

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax_main:
            return
        if self._state == _State.ANIMATING:
            return

        if event.button == 3:  # Rechtsklick = Undo
            self._undo_last()
            return
        if event.button != 1:
            return

        xy = np.array([event.xdata, event.ydata])
        snapped = self.contour.snap_node(xy, max_dist=self._snap_radius)
        if snapped is None:
            self._status_msg = "Kein Segmentknoten in der Naehe."
            self._draw_stats()
            return

        if self._state == _State.IDLE:
            self._pending = snapped
            self._state = _State.PICK_END
            self._status_msg = "Endpunkt waehlen ..."
        elif self._state == _State.PICK_END:
            loop_id, start_pos = self._pending
            end_loop, end_pos = snapped
            if end_loop != loop_id:
                self._status_msg = ("Start und Ende muessen auf derselben "
                                    "Kontur liegen!")
                self._draw_stats()
                return
            run = self._add_run(loop_id, start_pos, end_pos)
            self._pending = None
            self._state = _State.IDLE
            if run.is_feasible:
                self._status_msg = f"Segment R{run.run_id} hinzugefuegt."
            else:
                self._status_msg = (f"R{run.run_id} NICHT feasible: "
                                    f"{run.reason}")
        self._redraw()

    def _on_key(self, event) -> None:
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
                self._status_msg = "Auswahl abgebrochen."
                self._redraw()
            return

        if self._state == _State.ANIMATING:
            return

        if event.key == "u":
            self._undo_last()
            return

        if event.key == "a":
            self._select_all_remaining()
            return

        if event.key == "p":
            self._auto_select()
            return

        if event.key == "enter":
            if self._runs:
                self._plan_and_animate()
            else:
                self._status_msg = "Keine Segmente gewaehlt."
                self._draw_stats()
            return

    def _undo_last(self) -> None:
        if self._state == _State.PICK_END:
            self._pending = None
            self._state = _State.IDLE
        elif self._runs:
            removed = self._runs.pop()
            self._plan = None
            self._score = None
            self._status_msg = f"Segment R{removed.run_id} entfernt."
        self._redraw()

    def _select_all_remaining(self) -> None:
        """Waehlt fuer jede Kontur die noch fehlenden Boegen aus."""
        added = 0
        for loop in self.contour.loops:
            covered = covered_positions(self.contour, self._runs, loop.loop_id)
            if len(covered) >= loop.n:
                continue
            nodes = self.contour.nodes[loop.loop_id]
            anchor = nodes[0] if nodes else 0
            if not covered:
                self._add_run(loop.loop_id, anchor, anchor)
                added += 1
            else:
                # Fehlende zusammenhaengende Boegen einzeln hinzufuegen
                for a, b in self._missing_arcs(loop.n, covered):
                    run = self._add_run(loop.loop_id, a, b)
                    covered.update(run.positions)
                    added += 1
        self._status_msg = (f"{added} Segment(e) ergaenzt."
                            if added else "Kontur bereits komplett gewaehlt.")
        self._redraw()

    @staticmethod
    def _missing_arcs(n: int, covered: set[int]) -> list[tuple[int, int]]:
        """Zusammenhaengende unabgedeckte Bereiche als (start, ende)-Paare.

        Start/Ende sind die angrenzenden ABGEDECKTEN Punkte, damit der
        neue Schnitt nahtlos an Bestehendes anschliesst.
        """
        missing = sorted(p for p in range(n) if p not in covered)
        if not missing:
            return []
        groups: list[list[int]] = [[missing[0]]]
        for p in missing[1:]:
            if p == groups[-1][-1] + 1:
                groups[-1].append(p)
            else:
                groups.append([p])
        if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        return [((g[0] - 1) % n, (g[-1] + 1) % n) for g in groups]

    def _auto_select(self) -> None:
        """Auto-Planer (Taste P): waehlt die Segmente automatisch.

        Ersetzt die aktuelle Auswahl durch das Greedy-Set-Cover-Ergebnis
        (siehe autoplan.py). Enter startet danach wie gewohnt die
        Planung + Animation.
        """
        self._status_msg = "Auto-Planer laeuft ..."
        self._draw_stats()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        planner = AutoPlanner(
            self.grid, self.contour, self.cutter,
            self.kinematics, self.sequencer)
        result = planner.plan()

        self._runs = list(result.runs)
        self._plan = None
        self._score = None
        self._pending = None
        self._state = _State.IDLE

        print()
        print(result.summary())
        cov = result.report.fraction if result.report else 0.0
        self._status_msg = (
            f"Auto-Plan: {len(result.runs)} Schnitt(e), "
            f"Coverage {cov:.1%}, {result.elapsed:.2f} s "
            f"-- Enter zum Starten.")
        self._redraw()

    def _full_reset(self) -> None:
        self._stop_animation(redraw=False)
        self._runs.clear()
        self._plan = None
        self._score = None
        self._pending = None
        self._anim_mask = None
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
        infeasible = [r for r in self._runs if not r.is_feasible]
        if infeasible:
            names = ", ".join(f"R{r.run_id}" for r in infeasible)
            self._status_msg = (f"Nicht ausfuehrbar: {names} infeasible "
                                f"({infeasible[0].reason}). Mit U entfernen.")
            self._draw_stats()
            return

        report = self.grid_coverage()
        print()
        print(report.summary())

        self._status_msg = "Plane Reihenfolge + Verbindungen ..."
        self._draw_stats()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        try:
            self._plan = self.sequencer.build_plan(self._runs)
        except LinkInfeasibleError as exc:
            self._plan = None
            self._status_msg = f"Nicht planbar: {exc}"
            print()
            print(f"[INFEASIBLE] {exc}")
            self._redraw()
            return
        self._score = compute_score(self._plan.total_time, report.fraction)
        print(self._plan.summary())
        order = " -> ".join(f"R{r.run_id}" for r in self._plan.runs_in_order)
        print(f"  Reihenfolge: {order}")
        print(f"  Punktezahl: {self._score:.0f}")

        frames = self._build_frames(self._plan)
        if not frames:
            self._status_msg = "Nichts zu simulieren."
            self._redraw()
            return

        self._anim_mask = np.zeros(self.grid.total_points, dtype=bool)
        self._state = _State.ANIMATING
        self._run_animation(frames)

    def _build_frames(self, plan: CutPlan) -> list[_Frame]:
        """Diskretisiert den Plan in Animations-Frames."""
        frames: list[_Frame] = []
        t = 0.0

        def along(pts: np.ndarray, tips: np.ndarray | None,
                  speed: float, mode: str, run_id: int = -1) -> None:
            """Frames entlang einer Polylinie (TCP), Klinge interpoliert."""
            nonlocal t
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                d = float(np.linalg.norm(p2 - p1))
                if d < 1e-9:
                    continue
                dt = d / speed
                n_f = max(1, int(round(dt * self.fps)))
                for k in range(1, n_f + 1):
                    f = k / n_f
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
                tcp = (run.tcp_polyline if run.tcp_polyline is not None
                       else run.polyline)
                tips = run.tip_polyline
                if step.needs_pierce:
                    t_p = self.cutter.pierce_time()
                    n_f = max(2, int(round(t_p * self.fps)))
                    tip0 = tips[0] if tips is not None else None
                    for k in range(n_f):
                        frames.append(_Frame(
                            time=t + (k / n_f) * t_p,
                            pos=tcp[0], tip=tip0, mode="pierce",
                            run_id=run.run_id))
                    t += t_p
                along(tcp, tips, self.cutter.cutting_speed, "cut",
                      run_id=run.run_id)
            elif step.kind == "link" and step.link is not None:
                link = step.link
                along(link.points, None, self.cutter.rapid_speed, "link")
        return frames

    def _run_animation(self, frames: list[_Frame]) -> None:
        self._redraw()
        ax = self._ax_main

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
        prev = {"mode": "", "pos": None, "tip": None}

        import shapely as _shp

        def stamp(p1, t1, p2, t2):
            """Markiert Gitterpunkte im Klingen-Viereck p1-p2-t2-t1."""
            try:
                quad = Polygon([tuple(p1), tuple(p2), tuple(t2), tuple(t1)])
                if not quad.is_valid:
                    quad = quad.buffer(0)
                area = quad.buffer(sim.kerf_width / 2)
            except Exception:
                return
            rem = ~sim._anim_mask
            if not rem.any():
                return
            idx = np.where(rem)[0]
            hit = _shp.contains_xy(area, coords[idx, 0], coords[idx, 1])
            new_idx = idx[hit]
            if len(new_idx):
                sim._anim_mask[new_idx] = True
                live_pts.extend(coords[i] for i in new_idx)
                live_scatter.set_offsets(np.asarray(live_pts))

        def update(idx: int):
            fr = frames[idx]
            torch.center = tuple(fr.pos)

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

            # Live-Coverage: Klingen-Sweep zwischen zwei Frames stempeln
            if fr.mode == "pierce" and fr.tip is not None:
                if prev["mode"] != "pierce":
                    stamp(fr.pos, fr.tip, fr.pos, fr.tip)
            elif (fr.mode == "cut" and fr.tip is not None
                    and prev["pos"] is not None and prev["tip"] is not None
                    and prev["mode"] in ("cut", "pierce")):
                stamp(prev["pos"], prev["tip"], fr.pos, fr.tip)

            prev["mode"] = fr.mode
            prev["pos"] = fr.pos
            prev["tip"] = fr.tip

            n_cov = int(sim._anim_mask.sum())
            n_tot = sim.grid.total_points
            mode_txt = {
                "cut": "Schneiden", "pierce": "Zuenden",
                "link": "Verfahren",
            }[fr.mode]
            info.set_text(
                f"t = {fr.time:5.1f} / {total_time:.1f} s\n"
                f"Status: {mode_txt}\n"
                f"Punkte: {n_cov}/{n_tot} "
                f"({100.0 * n_cov / max(1, n_tot):.1f} %)")

            if idx >= n_frames - 1:
                sim._finish_animation()
            return (torch, blade_line, blade_glow, trail_tcp, trail_link,
                    info, live_scatter)

        def frame_gen():
            cur = 0
            while cur < n_frames - 1:
                yield cur
                step = 1 if sim._speed <= 2.0 else int(sim._speed / 2.0)
                cur += max(1, step)
            yield n_frames - 1

        effective_fps = self.fps * min(self._speed, 2.0)
        self._anim = FuncAnimation(
            self._fig, update, frames=frame_gen, save_count=n_frames,
            interval=max(1, int(1000 / effective_fps)),
            blit=False, repeat=False)
        self._fig.canvas.draw_idle()

    def _stop_animation(self, redraw: bool = True) -> None:
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
        score_txt = (f" | Punktezahl: {self._score:.0f}"
                     if self._score is not None else "")
        if report.is_complete:
            self._status_msg = (f"Fertig: Querschnitt zu 100 % "
                                f"durchtrennt!{score_txt}")
        else:
            self._status_msg = (f"Fertig, aber {report.missing_fraction:.1%} "
                                f"der Punkte fehlen!{score_txt}")
        print(f"\n{self._status_msg}")
        self._redraw(keep_animation_artists=True)

    # ------------------------------------------------------------------
    # Zeichnen
    # ------------------------------------------------------------------

    def _redraw(self, keep_animation_artists: bool = False) -> None:
        self._draw_main(keep_animation_artists)
        self._draw_stats()

    def _draw_main(self, keep_animation_artists: bool = False) -> None:
        ax = self._ax_main
        # Bei Animationsende nicht alles loeschen, nur Statistik anpassen
        if keep_animation_artists:
            self._fig.canvas.draw_idle()
            return
        ax.clear()
        ax.set_facecolor(_C["grid_bg"])

        # Querschnitts-Coverage der aktuellen Auswahl
        report = self.grid_coverage() if self._runs else None
        coords = self.grid.coords

        # Alle Gitterpunkte: Basis grau, abgedeckt gruen, fehlend rot
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

        # Primitiv-Segmente alternierend einfaerben
        for seg in self.contour.segments:
            loop = self.contour.loop_by_id(seg.loop_id)
            palette = (_C["seg_colors"] if loop.kind == "outer"
                       else _C["seg_hole"])
            color = palette[seg.seg_id % 2]
            pts = loop.polyline(seg.positions)
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.2,
                    alpha=0.8, zorder=4, solid_capstyle="round")

        # Gewaehlte Runs: Swept Area + TCP-Pfad + Konturbogen
        for i, run in enumerate(self._runs):
            c = (_C["run_colors"][i % len(_C["run_colors"])]
                 if run.is_feasible else _C["infeasible"])
            # Swept Area (ueberstrichene Flaeche)
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
            mid = run.polyline[len(run.polyline) // 2]
            label = f"R{run.run_id}" + ("" if run.is_feasible else " !")
            ax.annotate(label, xy=tuple(mid), fontsize=8.5,
                        fontweight="bold", color=c, zorder=16,
                        xytext=(4, 4), textcoords="offset points")
            if len(run.polyline) >= 2:
                p1, p2 = run.polyline[-2], run.polyline[-1]
                ax.annotate("", xy=tuple(p2), xytext=tuple(p1),
                            arrowprops=dict(arrowstyle="-|>", color=c,
                                            lw=2.0), zorder=16)

        # Segment-Knoten (= moegliche Start-/Endpunkte)
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
                   label="Segment (Aussenkontur)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=_C["node"],
                   markeredgecolor=_C["node_edge"], markersize=9,
                   label="Knoten (Start/Ende)"),
            Line2D([0], [0], linestyle="--", color=_C["run_colors"][0],
                   label="TCP-Pfad (Mindestabstand)"),
            Line2D([0], [0], color=_C["blade"], linewidth=3,
                   label="Klinge L(v)"),
            Line2D([0], [0], marker="P", color="w",
                   markerfacecolor=_C["covered"], markersize=9,
                   label="Punkt abgedeckt"),
            Line2D([0], [0], marker="x", color=_C["missing"], linestyle="",
                   markersize=8, label="Punkt fehlt"),
            Line2D([0], [0], linestyle="--", color=_C["link"],
                   label="Verfahrweg (auto)"),
        ], fontsize=7.5, loc="upper right", framealpha=0.92,
            edgecolor="#CCCCCC", labelspacing=0.45)

        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.22, color="#888")

        all_pts = np.vstack([l.points for l in self.contour.loops])
        span = max(float(np.ptp(all_pts[:, 0])), float(np.ptp(all_pts[:, 1])))
        m = span * 0.09 + 5.0 + self.cutter.minimum_gap
        ax.set_xlim(all_pts[:, 0].min() - m, all_pts[:, 0].max() + m)
        ax.set_ylim(all_pts[:, 1].min() - m, all_pts[:, 1].max() + m)

        ax.text(0.5, -0.048,
                "Klick: Start/Ende  |  Rechtsklick/U: Undo  |  A: alles  |  "
                "P: Auto-Plan  |  Enter: Planen+Start  |  R: Reset  |  "
                "Esc: Abbruch",
                transform=ax.transAxes, fontsize=7.5, color="#888",
                ha="center", va="top")

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------

    def _draw_stats(self) -> None:
        ax = self._ax_stats
        ax.clear()
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.01), 0.96, 0.97, boxstyle="round,pad=0.01",
            facecolor=_C["stats_bg"], edgecolor="#B0C4DE",
            linewidth=1.0, transform=ax.transAxes, zorder=0))

        ax.text(0.5, 0.965, "SEGMENT-PLAN", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="center", va="top",
                color="#1A2840")
        ax.plot([0.08, 0.92], [0.940, 0.940], color="#B0C4DE",
                linewidth=0.8, transform=ax.transAxes)

        y = 0.915
        dy = 0.037

        def kv(label, value, vc="#2D2D2D", bold=False):
            nonlocal y
            ax.text(0.07, y, label, transform=ax.transAxes,
                    fontsize=7.6, color="#555555", va="top")
            ax.text(0.93, y, value, transform=ax.transAxes,
                    fontsize=7.6 if not bold else 8.3, color=vc,
                    ha="right", va="top",
                    fontweight="bold" if bold else "normal")
            y -= dy

        def header(txt, color="#1A2840"):
            nonlocal y
            y -= dy * 0.2
            ax.text(0.5, y, txt, transform=ax.transAxes, fontsize=8,
                    fontweight="bold", ha="center", va="top", color=color)
            y -= dy * 0.9

        # --- Querschnitts-Coverage (alle Punkte) ---
        report = self.grid_coverage()
        cov_color = "#16A34A" if report.is_complete else (
            "#CC6600" if report.fraction > 0 else "#555555")
        kv("Punkte gesamt", str(report.total))
        kv("Abgedeckt", f"{report.covered} / {report.total}",
           vc=cov_color, bold=True)
        kv("Coverage", f"{report.fraction:.1%}", vc=cov_color, bold=True)
        if not report.is_complete:
            kv("Fehlt", f"{report.missing_fraction:.1%} "
               f"({report.total - report.covered} Pkt)",
               vc=_C["missing"], bold=report.fraction > 0)

        # --- Segmente ---
        header("GEWAEHLTE SEGMENTE")
        if not self._runs:
            kv("(keine)", "")
        n_show = 4
        if len(self._runs) > n_show:
            kv(f"... {len(self._runs) - n_show} weitere", "")
        offset = max(0, len(self._runs) - n_show)
        for i, run in enumerate(self._runs[-n_show:]):
            c = (_C["run_colors"][(offset + i) % len(_C["run_colors"])]
                 if run.is_feasible else _C["infeasible"])
            feas = "" if run.is_feasible else " !"
            kv(f"R{run.run_id} (Loop {run.loop_id}){feas}",
               f"{(run.tcp_length or run.length):.0f} mm", vc=c)

        # --- Plan + Punktezahl ---
        if self._plan is not None:
            header("PLAN (OPTIMIERT)", "#0B6E2F")
            kv("Reihenfolge",
               "-".join(f"R{r.run_id}" for r in self._plan.runs_in_order),
               bold=True)
            kv("Optimalitaet",
               "exakt zeitminimal" if self._plan.is_optimal else "heuristisch",
               vc="#0B6E2F" if self._plan.is_optimal else "#CC6600")
            kv("Zuendungen", str(self._plan.n_pierces))
            kv("Schnittzeit", f"{self._plan.cut_time:.1f} s")
            kv("Eilgang", f"{self._plan.travel_time:.1f} s")
            kv("Pierce", f"{self._plan.pierce_time:.1f} s")
            kv("Gesamtzeit", f"{self._plan.total_time:.1f} s",
               vc="#0B6E2F", bold=True)
            if self._score is not None:
                kv("PUNKTEZAHL", f"{self._score:.0f}",
                   vc="#B8860B", bold=True)

        # --- Lichtschwert / Cutter ---
        header("LICHTSCHWERT")
        kv("Klinge L(v)", f"{self.blade_length:.1f} mm", bold=True)
        kv("eff. Tiefe", f"{self.kinematics.effective_depth:.1f} mm",
           vc="#16A34A" if self.kinematics.effective_depth > 0
           else _C["missing"])
        vmax = self.cutter.max_cutting_speed
        kv("v Schneiden", f"{self.cutter.cutting_speed:.1f}"
           + (f" / max {vmax:.1f} mm/s" if vmax else " mm/s"))
        kv("v Eilgang", f"{self.cutter.rapid_speed:.1f} mm/s")
        kv("Min. Abstand (konst.)", f"{self.cutter.minimum_gap:.1f} mm",
           vc="#CC6600")
        kv("Kerf", f"{self.kerf_width:.1f} mm")

        # --- Status ---
        if self._state == _State.IDLE:
            status, sc = (self._status_msg or "Startpunkt klicken"), "#4B5563"
        elif self._state == _State.PICK_END:
            status, sc = "Endpunkt klicken ...", "#CC6600"
        else:
            status, sc = f"Simulation laeuft ({self._speed:.2g}x)", "#CC2222"
        if self._status_msg and self._state == _State.IDLE:
            if "100 %" in self._status_msg:
                sc = "#16A34A"
            elif ("fehlen" in self._status_msg
                  or "feasible" in self._status_msg
                  or "Nicht ausfuehrbar" in self._status_msg):
                sc = "#CC2222"

        ax.text(0.5, 0.025, status, transform=ax.transAxes, fontsize=7.0,
                color=sc, ha="center", va="bottom", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=sc,
                          alpha=0.88, linewidth=0.8))
        self._fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Headless-API (fuer Tests / spaeteres supervised learning)
# ---------------------------------------------------------------------------

def check_segments(
    grid: PointGrid,
    node_pairs: list[tuple[int, int, int]],
    cutter: Cutter | None = None,
    kerf_width: float = 3.0,
    target_segment_length: float | None = None,
) -> tuple[CutPlan, GridCoverageReport, float]:
    """Minimalanforderung ohne UI: gegebene Segmente pruefen + ordnen.

    Parameters
    ----------
    node_pairs : Liste (loop_id, start_pos, end_pos) -- Knotenpositionen
                 wie von SegmentedContour vergeben.

    Returns
    -------
    (CutPlan, GridCoverageReport, score) -- optimierte Reihenfolge inkl.
    kollisionsfreier Verbindungen, Querschnitts-Abdeckung und
    Punktezahl (Zeit-basiert).
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Segment-basierte Plasmaschneider-Simulation "
                    "(Lichtschwert, Querschnitts-Coverage)")
    parser.add_argument("--geometry", type=str, default=None,
                        help="Pfad zur Geometrie-JSON (sonst Dialog)")
    parser.add_argument("--kerf-width", type=float, default=3.0)
    parser.add_argument("--segment-length", type=float, default=None,
                        help="Ziel-Segmentlaenge in mm (Standard: automatisch)")
    parser.add_argument("--v-cut", type=float, default=DEFAULT_CUTTING_SPEED,
                        help="Schnittgeschwindigkeit in mm/s")
    parser.add_argument("--v-max", type=float,
                        default=DEFAULT_MAX_CUTTING_SPEED,
                        help="Maximale Schnittgeschwindigkeit in mm/s")
    parser.add_argument("--v-rapid", type=float, default=50.0,
                        help="Eilgang-Geschwindigkeit in mm/s")
    parser.add_argument("--blade-length", type=float, default=BLADE_LENGTH,
                        help="Klingen-Grundlaenge L(v=0) in mm")
    parser.add_argument("--blade-slope", type=float, default=BLADE_SLOPE,
                        help="Klingenverkuerzung in mm pro mm/s")
    parser.add_argument("--clearance", type=float, default=MINIMUM_GAP,
                        help="Konstanter Mindestabstand TCP-Material in mm")
    args = parser.parse_args()

    cutter = make_default_cutter(
        cutting_speed=args.v_cut,
        max_cutting_speed=args.v_max,
        blade_length=args.blade_length,
        blade_slope=args.blade_slope,
        minimum_gap=args.clearance,
        rapid_speed=args.v_rapid,
    )
    L = cutter.blade_length(cutter.cutting_speed)
    print(cutter)
    print(f"L(v) = {args.blade_length:.1f} - {args.blade_slope:.2f}*v  ->  "
          f"L({args.v_cut:.1f}) = {L:.1f} mm, "
          f"eff. Tiefe = {L - args.clearance:.1f} mm")

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

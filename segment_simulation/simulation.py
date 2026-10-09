"""Simulation of cutting primitives.

Model
-----
  - The TCP keeps a constant gap ``MINIMUM_GAP`` to the material.
  - The plasma jet extends from the TCP toward the part; its length drops
    linearly with speed: L(v) = blade_length - blade_slope * v.

Workflow
--------
1. The contour is split into segments; the nodes are the possible
   start/end points.
2. Two clicks pick start and end node; the arc in between is cut
   (same node twice = whole loop).
3. The panel shows the coverage; missing points are red.
4. Enter: plan order and links, animate the sequence.

Controls
--------
  Left click        pick start/end point
  Right click / U   remove last segment
  A                 select all remaining contours
  P / S / B         Automatic Planner / Surrogate / Brute Force (also buttons)
  V                 speed rule on/off (also button)
  Enter             plan and start simulation
  R / Esc           reset / cancel
  + / - / Slider    animation speed
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

# Force TkAgg when run directly, since some IDEs set a non-interactive
# backend. matplotlib.use() must come before importing pyplot.
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
# Constants / defaults
# ---------------------------------------------------------------------------
# Any change here invalidates labels and model (phys_hash).

MINIMUM_GAP = 3.6

BLADE_LENGTH = 29.9
BLADE_SLOPE = 0.327

DEFAULT_CUTTING_SPEED = 19.4
DEFAULT_MAX_CUTTING_SPEED = 34.7
RAPID_SPEED = 100.0

T_SWITCH = 0.0

PIERCE_T0 = 1.0
PIERCE_K = 0.0
SHEET_THICKNESS = 15.0


def make_default_cutter(
    cutting_speed: float = DEFAULT_CUTTING_SPEED,
    max_cutting_speed: float = DEFAULT_MAX_CUTTING_SPEED,
    blade_length: float = BLADE_LENGTH,
    blade_slope: float = BLADE_SLOPE,
    minimum_gap: float = MINIMUM_GAP,
    rapid_speed: float = RAPID_SPEED,
    t_switch: float = T_SWITCH,
    pierce_t0: float = PIERCE_T0,
    pierce_k: float = PIERCE_K,
    sheet_thickness: float = SHEET_THICKNESS,
) -> Cutter:
    """Cutter with linear L(v) blade model and the values above."""
    blade = BladeLengthModel(L0=blade_length, slope=blade_slope)
    assumptions = CuttingAssumptions(
        blade=blade,
        pierce=PierceTimeModel(t0=pierce_t0, k=pierce_k),
        sheet_thickness=sheet_thickness)
    return Cutter(
        cutting_speed=cutting_speed,
        max_cutting_speed=max_cutting_speed,
        rapid_speed=rapid_speed,
        minimum_gap=minimum_gap,
        t_switch=t_switch,
        assumptions=assumptions,
    )


def _surrogate_tools():
    """Lazy import of the planner helpers (fast start, no import cycle)."""
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
# Colors
# ---------------------------------------------------------------------------

_C = dict(
    inner        = "#BBBBBB",
    seg_colors   = ["#1E6FBF", "#6FA8DC"],     # alternating segment colors
    seg_hole     = ["#C0504D", "#E6A09E"],     # alternating, for hole contour
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
    """UI states: waiting for start, waiting for end, animation running."""
    IDLE      = auto()   # waiting for start point click
    PICK_END  = auto()   # start point set, waiting for end point
    ANIMATING = auto()


@dataclass
class _Frame:
    """One animation frame (time, TCP, blade tip, mode, run)."""
    time: float
    pos: np.ndarray              # TCP position
    tip: np.ndarray | None       # blade tip (None for link)
    mode: str                    # "pierce" | "cut" | "link"
    run_id: int = -1


# ---------------------------------------------------------------------------
# SegmentCutSimulation
# ---------------------------------------------------------------------------

class SegmentCutSimulation:
    """Interactive simulation: pick, check and execute segments.

    Parameters
    ----------
    grid                   : PointGrid of the geometry
    cutter                 : Cutter (None -> make_default_cutter())
    kerf_width             : kerf width [mm]
    target_segment_length  : target segment length [mm] (None = automatic)
    fps                    : animation frame rate
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
            print(f"WARNING: plasma arc too short! "
                  f"L(v={self.cutter.cutting_speed}) "
                  f"= {self.blade_length:.1f} mm <= standoff "
                  f"{self.cutter.minimum_gap:.1f} mm -> no cut possible. "
                  f"Reduce the speed or increase the arc length.")

        # Selection and plan; snap radius = four contour point spacings.
        self._runs: list[CutRun] = []
        self._plan: CutPlan | None = None
        self._score: float | None = None
        self._state = _State.IDLE
        self._pending: tuple[int, int] | None = None  # (loop_id, pos)
        self._snap_radius = grid.contour_spacing * 4.0

        # Animation; _anim_mask = grid points already swept.
        self._anim: FuncAnimation | None = None
        self._speed = 1.0
        self._anim_mask: np.ndarray | None = None
        self._status_msg = ""

        # Speed rule: _run_speeds = run_id -> v [mm/s];
        # _chain_sel = seg_ids from a planner (P/S/B), None = manual selection.
        self._use_rule45 = False
        self._run_speeds: dict[int, float] = {}
        self._chain_sel: list[int] | None = None
        # surrogate model, loaded on first S
        self._model = None

        # matplotlib handles, set only in run()
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
        """Pick geometry via file dialog and start the simulation."""
        initial_dir = kwargs.pop("initial_dir", None)
        grid = PointGrid.from_json_dialog(initial_dir=initial_dir)
        if grid is None:
            return None
        sim = cls(grid=grid, **kwargs)
        sim.run()
        return sim

    @property
    def runs(self) -> list[CutRun]:
        """Copy of the selected runs."""
        return list(self._runs)

    @property
    def plan(self) -> CutPlan | None:
        """Most recently built plan (None until planned)."""
        return self._plan

    def grid_coverage(self) -> GridCoverageReport:
        """Cross-section coverage of the selected runs."""
        return compute_grid_coverage(self.grid, self._runs)

    def run(self) -> None:
        """Build the window and start the event loop."""
        # free 's' (save) and 'p' (pan) for surrogate and Automatic Planner
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

        ax_speed = self._fig.add_axes([0.10, 0.04, 0.19, 0.03])
        self._speed_slider = Slider(
            ax=ax_speed, label="Sim speed ",
            valmin=0.25, valmax=16.0, valinit=1.0,
            valstep=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            color="#0066CC")
        self._speed_slider.on_changed(self._on_speed)

        ax_reset = self._fig.add_axes([0.337, 0.035, 0.075, 0.045])
        self._reset_button = Button(
            ax_reset, "Reset (R)", color="#F0D0D0", hovercolor="#E0A0A0")
        self._reset_button.label.set_fontsize(9)
        self._reset_button.on_clicked(lambda _evt: self._full_reset())

        ax_gp = self._fig.add_axes([0.419, 0.035, 0.165, 0.045])
        self._gp_button = Button(
            ax_gp, "Automatic planner (P)", color="#D0DEF0", hovercolor="#A8C4E4")
        self._gp_button.label.set_fontsize(9)
        self._gp_button.on_clicked(lambda _evt: self._greedy_select())

        ax_sur = self._fig.add_axes([0.591, 0.035, 0.105, 0.045])
        self._sur_button = Button(
            ax_sur, "Surrogate (S)", color="#D0DEF0", hovercolor="#A8C4E4")
        self._sur_button.label.set_fontsize(9)
        self._sur_button.on_clicked(lambda _evt: self._surrogate_select())

        ax_bf = self._fig.add_axes([0.703, 0.035, 0.115, 0.045])
        self._bf_button = Button(
            ax_bf, "Brute force (B)", color="#D0DEF0", hovercolor="#A8C4E4")
        self._bf_button.label.set_fontsize(9)
        self._bf_button.on_clicked(lambda _evt: self._brute_force_select())

        ax_rule = self._fig.add_axes([0.825, 0.035, 0.155, 0.045])
        self._rule_button = Button(ax_rule, "", color="#E0E0E0",
                                   hovercolor="#C8C8C8")
        self._rule_button.label.set_fontsize(9)
        self._rule_button.on_clicked(lambda _evt: self._toggle_rule45())
        self._style_rule_button()

        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._redraw()

        # warn if a non-interactive backend is active after all
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
        """Title line with file name, L(v), effective depth and minimum gap."""
        name = (f"  -  {self.grid._source_path.stem}"
                if hasattr(self.grid, "_source_path") else "")
        return (f"Segment Simulation{name}  |  "
                f"L(v={self.cutter.cutting_speed:.0f}) = "
                f"{self.blade_length:.1f} mm  |  "
                f"Eff. depth: {self.kinematics.effective_depth:.1f} mm  |  "
                f"Standoff: {self.cutter.minimum_gap:.0f} mm (const.)")

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _add_run(self, loop_id: int, start_pos: int, end_pos: int) -> CutRun:
        """Create a run between two nodes and discard the plan."""
        covered = covered_positions(self.contour, self._runs, loop_id)
        run = self.contour.make_run(
            run_id=len(self._runs) + 1,
            loop_id=loop_id, start_pos=start_pos, end_pos=end_pos,
            covered=covered)
        self.kinematics.attach(run)
        self._runs.append(run)
        self._plan = None
        self._score = None
        self._chain_sel = None   # manual change: planner selection dropped
        return run

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_speed(self, val: float) -> None:
        """Speed slider: up to 2x via the frame rate, beyond that
        ``frame_gen`` skips frames.
        """
        self._speed = val
        if self._anim is not None and self._anim.event_source is not None:
            effective_fps = self.fps * min(self._speed, 2.0)
            self._anim.event_source.interval = max(1, int(1000 / effective_fps))
        self._draw_stats()

    def _on_click(self, event) -> None:
        """Left click sets start/end node, right click undoes."""
        if event.inaxes is not self._ax_main:
            return
        if self._state == _State.ANIMATING:
            return

        if event.button == 3:  # right click = undo
            self._undo_last()
            return
        if event.button != 1:
            return

        xy = np.array([event.xdata, event.ydata])
        snapped = self.contour.snap_node(xy, max_dist=self._snap_radius)
        if snapped is None:
            self._status_msg = "No segment node nearby."
            self._draw_stats()
            return

        if self._state == _State.IDLE:
            self._pending = snapped
            self._state = _State.PICK_END
            self._status_msg = "Select end point ..."
        elif self._state == _State.PICK_END:
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
        """Keyboard handler (controls: see module header).

        During animation only +/-, R and Esc are active.
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

        # from here on, not during animation
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
        """Discard the pending start point or the last segment."""
        if self._state == _State.PICK_END:
            self._pending = None
            self._state = _State.IDLE
        elif self._runs:
            removed = self._runs.pop()
            self._plan = None
            self._score = None
            self._chain_sel = None   # selection changed manually
            self._status_msg = f"Segment R{removed.run_id} removed."
        self._redraw()

    def _select_all_remaining(self) -> None:
        """Key A: add the still missing arcs on each contour."""
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
                for a, b in self._missing_arcs(loop.n, covered):
                    run = self._add_run(loop.loop_id, a, b)
                    covered.update(run.positions)
                    added += 1
        self._status_msg = (f"{added} segment(s) added."
                            if added else "Contour already fully selected.")
        self._redraw()

    @staticmethod
    def _missing_arcs(n: int, covered: set[int]) -> list[tuple[int, int]]:
        """Uncovered arcs of a ring as (start, end) pairs.

        Start/end are the adjacent covered points, so the new cut joins
        seamlessly.
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
        # close the seam 0 <-> n-1
        if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == n - 1:
            groups[0] = groups[-1] + groups[0]
            groups.pop()
        return [((g[0] - 1) % n, (g[-1] + 1) % n) for g in groups]

    def _greedy_select(self) -> None:
        """Key P: Automatic Planner selection (``autoplan.AutoPlanner``).

        The segment selection goes to ``_chain_sel``; with the speed rule,
        Enter uses the same DP split stage as surrogate and Brute Force.
        """
        self._status_msg = "Automatic planner running ..."
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
        self._run_speeds = {}
        self._chain_sel = list(result.selected_segments) or None
        self._state = _State.IDLE

        print()
        print(result.summary())
        cov = result.report.fraction if result.report else 0.0
        self._status_msg = (
            f"Automatic planner: {len(result.runs)} cut(s), "
            f"coverage {cov:.1%}, {result.elapsed:.2f} s "
            f"-- press Enter to start.")
        self._redraw()

    def _brute_force_select(self) -> None:
        """Key B: exact time-minimal segment selection
        (``surrogate.teacher.exhaustive_plan``).

        Enumerates all subsets, same teacher as for the labels. With too
        many segments (``TeacherSkipped``) the selection stays unchanged.
        """
        if self._state == _State.ANIMATING:
            return  # button is clickable during animation as well
        tools = _surrogate_tools()
        phys_from_cutter = tools.phys_from_cutter
        merge_covering_runs = tools.merge_covering_runs
        exhaustive_plan = tools.exhaustive_plan
        TeacherSkipped = tools.TeacherSkipped
        label = "Brute force"

        n_seg = len(self.contour.segments)
        rule_txt = "with" if self._use_rule45 else "without"
        self._status_msg = (f"{label} running ({n_seg} segments, "
                            f"{rule_txt} speed rule) ...")
        self._draw_stats()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        try:
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
        # check_partial=True: same merge decision as the teacher
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
        self._chain_sel = list(result.selected)
        self._state = _State.IDLE

        print()
        print(f"{label} ({rule_txt} speed rule): "
              f"{len(result.selected)}/{result.n_segments} segments "
              f"-> {len(runs)} cut(s), T = {result.total_time:.1f} s, "
              f"coverage {result.coverage:.1%}, {detail} in "
              f"{result.plan_time:.2f} s")
        if not runs:
            self._status_msg = f"{label}: no feasible segments."
        else:
            self._status_msg = (
                f"{label} ({rule_txt} speed rule): {len(runs)} "
                f"cut(s), T={result.total_time:.1f} s, coverage "
                f"{result.coverage:.1%}, {result.plan_time:.1f} s "
                f"-- press Enter to start.")
        self._redraw()

    def _surrogate_model(self):
        """Lazily load the surrogate model once and cache it on the instance."""
        if self._model is None:
            self._model = _surrogate_tools().load_model()
        return self._model

    def _surrogate_select(self) -> None:
        """Key S: surrogate planner selection (``surrogate.planner.surrogate_plan``).

        The model only sets the order; the coverage guarantee never depends
        on the model (fallback to the greedy selection).
        """
        if self._state == _State.ANIMATING:
            return
        tools = _surrogate_tools()
        label = "Surrogate"
        n_seg = len(self.contour.segments)
        rule_txt = "with" if self._use_rule45 else "without"
        self._status_msg = (f"{label} running ({n_seg} segments, "
                            f"{rule_txt} speed rule) ...")
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
        print(f"{label} ({rule_txt} speed rule): " + result.summary())
        if not runs:
            self._status_msg = f"{label}: no feasible segments."
        else:
            fb = " [fallback to Greedy]" if result.used_fallback else ""
            self._status_msg = (
                f"{label} ({rule_txt} speed rule): {len(runs)} cut(s), "
                f"T={result.T:.1f} s, coverage {result.coverage:.1%}, "
                f"{result.t_plan * 1e3:.0f} ms{fb} -- press Enter to start.")
        self._redraw()

    # ------------------------------------------------------------------
    # Speed rule
    # ------------------------------------------------------------------

    def _toggle_rule45(self) -> None:
        """Key V: speed rule on/off.

        ON: during planning each run gets the fastest coverage-preserving
        speed. OFF: all runs at base speed. An existing plan is discarded.
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
            "Speed rule ON: v per run assigned during planning."
            if self._use_rule45 else
            "Speed rule OFF: all runs at base speed.")
        self._redraw()

    def _style_rule_button(self) -> None:
        """Label and color of the rule button (green = ON)."""
        if self._rule_button is None:
            return
        on = self._use_rule45
        self._rule_button.label.set_text(
            f"Speed rule: {'ON' if on else 'OFF'} (V)")
        self._rule_button.color = "#C8E6C9" if on else "#E0E0E0"
        self._rule_button.hovercolor = "#A5D6A7" if on else "#C8C8C8"
        self._rule_button.ax.set_facecolor(self._rule_button.color)

    def _reset_run_speeds(self) -> None:
        """Re-attach all runs at base speed."""
        if not self._run_speeds:
            return
        for run in self._runs:
            self.kinematics.attach(run)
        self._run_speeds = {}

    def _assign_rule45_speeds(self) -> None:
        """Speed rule per run (``speed_up_runs``); the runs stay attached
        at their speed.
        """
        tools = _surrogate_tools()
        phys_from_cutter, speed_up_runs = tools.phys_from_cutter, tools.speed_up_runs
        self._reset_run_speeds()
        phys = phys_from_cutter(self.cutter)
        self._run_speeds = speed_up_runs(
            self._runs, np.asarray(self.grid.coords, dtype=float),
            self.material, phys, self.kerf_width)

    def _apply_run_speeds(self, plan: CutPlan) -> None:
        """Convert the plan's cut times to the rule speeds; order and links
        stay valid.
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
        """DP split chains for a planner selection (P/S/B).

        Replaces ``_runs`` with the sub-runs, so Enter reproduces the
        planner's T exactly. None if no chain could be built.
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
        """Plan for chains: Held-Karp over one macro node per chain, then
        roll out the sub-runs.

        Within a chain a transition costs only ``t_switch``; between chains,
        rapid traverse and pierce.
        """
        macros = [ch.macro for ch in chains]
        ordered, is_opt = self.sequencer.order_runs(macros)
        by_id = {ch.macro.run_id: ch for ch in chains}
        t_switch = float(getattr(self.cutter, "t_switch", 0.0))

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
                        # seamless: torch stays on
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
        """Key R: reset everything to the initial state."""
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
    # Planning + animation
    # ------------------------------------------------------------------

    def _plan_and_animate(self) -> None:
        """Enter: assign speeds, build the plan, score it and animate it."""
        infeasible = [r for r in self._runs if not r.is_feasible]
        if infeasible:
            names = ", ".join(f"R{r.run_id}" for r in infeasible)
            self._status_msg = (f"Not feasible: {names} infeasible "
                                f"({infeasible[0].reason}). Remove with U.")
            self._draw_stats()
            return

        # Before the coverage report: planner selection via DP split chains,
        # manual selection via the per-run rule.
        chains = None
        if self._use_rule45:
            self._status_msg = "Speed rule: assigning speeds ..."
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
        # the chain plan already carries its speeds
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
            print(f"  Speed rule v [mm/s]: {v_txt}")
        print(f"  Score: {self._score:.0f}")

        frames = self._build_frames(self._plan)
        if not frames:
            self._status_msg = "Nothing to simulate."
            self._redraw()
            return

        self._anim_mask = np.zeros(self.grid.total_points, dtype=bool)
        self._state = _State.ANIMATING
        self._run_animation(frames)

    def _build_frames(self, plan: CutPlan) -> list[_Frame]:
        """Split the plan into animation frames (pierce stationary, cut at
        the run's speed, link at rapid traverse).
        """
        frames: list[_Frame] = []
        t = 0.0

        def along(pts: np.ndarray, tips: np.ndarray | None,
                  speed: float, mode: str, run_id: int = -1) -> None:
            """Frames along a polyline; blade tips linearly interpolated
            (None = no blade).
            """
            nonlocal t
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                d = float(np.linalg.norm(p2 - p1))
                if d < 1e-9:
                    continue  # duplicate point
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
                    # pierce: torch stationary, frames over the pierce time
                    t_p = self.cutter.pierce_time()
                    n_f = max(2, int(round(t_p * self.fps)))
                    tip0 = tips[0] if tips is not None else None
                    for k in range(n_f):
                        frames.append(_Frame(
                            time=t + (k / n_f) * t_p,
                            pos=tcp[0], tip=tip0, mode="pierce",
                            run_id=run.run_id))
                    t += t_p
                v_run = self._run_speeds.get(run.run_id,
                                             self.cutter.cutting_speed)
                along(tcp, tips, v_run, "cut", run_id=run.run_id)
            elif step.kind == "link" and step.link is not None:
                link = step.link
                along(link.points, None, self.cutter.rapid_speed, "link")
        return frames

    def _run_animation(self, frames: list[_Frame]) -> None:
        """Play the frames as a FuncAnimation (torch, plasma jet, trails,
        live coverage, info box).
        """
        self._redraw()
        ax = self._ax_main

        # create artists once, only update them per frame
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
        # previous frame, for the sweep stamp and the trail separators
        prev = {"mode": "", "pos": None, "tip": None}

        import shapely as _shp

        def stamp(p1, t1, p2, t2):
            """Mark the grid points in the swept quad p1-p2-t2-t1
            (plus half the kerf).
            """
            try:
                quad = Polygon([tuple(p1), tuple(p2), tuple(t2), tuple(t1)])
                if not quad.is_valid:
                    quad = quad.buffer(0)  # self-intersecting
                area = quad.buffer(sim.kerf_width / 2)
            except Exception:
                return
            # test only points not yet covered
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
            """Draw frame ``idx`` (FuncAnimation callback)."""
            fr = frames[idx]
            torch.center = tuple(fr.pos)

            # blade only while cutting/piercing
            if fr.tip is not None and fr.mode in ("cut", "pierce"):
                blade_line.set_data([fr.pos[0], fr.tip[0]],
                                    [fr.pos[1], fr.tip[1]])
                blade_glow.set_data([fr.pos[0], fr.tip[0]],
                                    [fr.pos[1], fr.tip[1]])
            else:
                blade_line.set_data([], [])
                blade_glow.set_data([], [])

            # trails: NaN separator on mode change
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

            # live coverage: once when piercing; when cutting, the quad
            # from the previous to the current frame
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
                "cut": "Cutting", "pierce": "Piercing",
                "link": "Traversing",
            }[fr.mode]
            info.set_text(
                f"t = {fr.time:5.1f} / {total_time:.1f} s\n"
                f"Status: {mode_txt}\n"
                f"Points: {n_cov}/{n_tot} "
                f"({100.0 * n_cov / max(1, n_tot):.1f} %)")

            if idx >= n_frames - 1:
                sim._finish_animation()
            return (torch, blade_line, blade_glow, trail_tcp, trail_link,
                    info, live_scatter)

        def frame_gen():
            """Frame indices; above 2x frames are skipped, the last one is
            always included.
            """
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
        """Abort the animation without final scoring (Esc/Reset)."""
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
        """Finish the animation normally: coverage, score, final message."""
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
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self, keep_animation_artists: bool = False) -> None:
        """Redraw the main view and the statistics."""
        self._draw_main(keep_animation_artists)
        self._draw_stats()

    def _draw_main(self, keep_animation_artists: bool = False) -> None:
        """Draw grid points, segments, runs, nodes, links, legend.

        With ``keep_animation_artists`` the final image of the animation stays.
        """
        ax = self._ax_main
        if keep_animation_artists:
            self._fig.canvas.draw_idle()
            return
        ax.clear()
        ax.set_facecolor(_C["grid_bg"])

        # cross-section coverage of the current selection
        report = self.grid_coverage() if self._runs else None
        coords = self.grid.coords

        # all grid points: base gray, covered green, missing red
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

        # color primitive segments alternately
        for seg in self.contour.segments:
            loop = self.contour.loop_by_id(seg.loop_id)
            palette = (_C["seg_colors"] if loop.kind == "outer"
                       else _C["seg_hole"])
            color = palette[seg.seg_id % 2]
            pts = loop.polyline(seg.positions)
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2.2,
                    alpha=0.8, zorder=4, solid_capstyle="round")

        # selected runs: swept area + TCP path + contour arc
        for i, run in enumerate(self._runs):
            c = (_C["run_colors"][i % len(_C["run_colors"])]
                 if run.is_feasible else _C["infeasible"])
            # swept area
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
            # TCP offset path
            if run.tcp_polyline is not None:
                ax.plot(run.tcp_polyline[:, 0], run.tcp_polyline[:, 1],
                        "--" if run.is_feasible else ":",
                        color=c, linewidth=1.6, alpha=0.8, zorder=9)
            # bold contour arc
            ax.plot(run.polyline[:, 0], run.polyline[:, 1], color=c,
                    linewidth=4.0, alpha=0.45, zorder=8,
                    solid_capstyle="round")
            # label "Rn" ("!" = not feasible)
            mid = run.polyline[len(run.polyline) // 2]
            label = f"R{run.run_id}" + ("" if run.is_feasible else " !")
            ax.annotate(label, xy=tuple(mid), fontsize=8.5,
                        fontweight="bold", color=c, zorder=16,
                        xytext=(4, 4), textcoords="offset points")
            # arrow = cut direction
            if len(run.polyline) >= 2:
                p1, p2 = run.polyline[-2], run.polyline[-1]
                ax.annotate("", xy=tuple(p2), xytext=tuple(p1),
                            arrowprops=dict(arrowstyle="-|>", color=c,
                                            lw=2.0), zorder=16)

        # segment nodes (= possible start/end points)
        for loop in self.contour.loops:
            nodes = self.contour.nodes.get(loop.loop_id, [])
            if nodes:
                npts = loop.points[nodes]
                ax.scatter(npts[:, 0], npts[:, 1], s=70, color=_C["node"],
                           edgecolors=_C["node_edge"], linewidths=1.3,
                           zorder=10)

        # planned links (after Enter)
        if self._plan is not None:
            for step in self._plan.steps:
                if step.kind == "link" and step.link is not None:
                    pts = step.link.points
                    style = dict(linewidth=1.4, alpha=0.65, zorder=9)
                    ax.plot(pts[:, 0], pts[:, 1], "--",
                            color=_C["link"], **style)

        # pending start point
        if self._pending is not None:
            loop = self.contour.loop_by_id(self._pending[0])
            p = loop.points[self._pending[1]]
            ax.plot(p[0], p[1], "*", color=_C["pending"], markersize=20,
                    markeredgecolor="#806000", markeredgewidth=1.2,
                    zorder=18)

        # legend
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

        # margin incl. minimum gap, so the TCP path fits
        all_pts = np.vstack([l.points for l in self.contour.loops])
        span = max(float(np.ptp(all_pts[:, 0])), float(np.ptp(all_pts[:, 1])))
        m = span * 0.09 + 5.0 + self.cutter.minimum_gap
        ax.set_xlim(all_pts[:, 0].min() - m, all_pts[:, 0].max() + m)
        ax.set_ylim(all_pts[:, 1].min() - m, all_pts[:, 1].max() + m)

        ax.text(0.5, -0.048,
                "Click: start/end  |  Right-click/U: undo  |  A: all  |  "
                "P: automatic planner  |  S: surrogate  |  B: brute force  |  "
                "V: speed rule  |  "
                "Enter: plan+start  |  R: reset  |  Esc: cancel",
                transform=ax.transAxes, fontsize=7.5, color="#888",
                ha="center", va="top")

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------

    def _draw_stats(self) -> None:
        """Draw the right info panel (coverage, segments, plan, torch)."""
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

        # y = current text position (1.0 = top), dy = line spacing
        y = 0.915
        dy = 0.037

        def kv(label, value, vc="#2D2D2D", bold=False):
            """Write a label-left / value-right line and advance y."""
            nonlocal y
            ax.text(0.07, y, label, transform=ax.transAxes,
                    fontsize=7.6, color="#555555", va="top")
            ax.text(0.93, y, value, transform=ax.transAxes,
                    fontsize=7.6 if not bold else 8.3, color=vc,
                    ha="right", va="top",
                    fontweight="bold" if bold else "normal")
            y -= dy

        def header(txt, color="#1A2840"):
            """Write a centered section heading and advance y."""
            nonlocal y
            y -= dy * 0.2
            ax.text(0.5, y, txt, transform=ax.transAxes, fontsize=8,
                    fontweight="bold", ha="center", va="top", color=color)
            y -= dy * 0.9

        # --- cross-section coverage (all points) ---
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

        # --- segments ---
        # only the last n_show runs
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

        # --- plan + score ---
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
                    kv("v per run (speed rule)",
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
        kv("Speed rule (v per run)", "ON" if self._use_rule45 else "OFF",
           vc="#16A34A" if self._use_rule45 else "#888888",
           bold=self._use_rule45)
        kv("Rapid speed", f"{self.cutter.rapid_speed:.1f} mm/s")
        kv("Standoff (const.)", f"{self.cutter.minimum_gap:.1f} mm",
           vc="#CC6600")
        kv("Kerf", f"{self.kerf_width:.1f} mm")

        # --- Status ---
        # color by state; in IDLE by keywords of the message
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
# Headless API
# ---------------------------------------------------------------------------

def check_segments(
    grid: PointGrid,
    node_pairs: list[tuple[int, int, int]],
    cutter: Cutter | None = None,
    kerf_width: float = 3.0,
    target_segment_length: float | None = None,
) -> tuple[CutPlan, GridCoverageReport, float]:
    """Check and order given segments without UI.

    Parameters
    ----------
    node_pairs : list of (loop_id, start_pos, end_pos) node positions

    Returns
    -------
    (CutPlan, GridCoverageReport, score)
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
    parser.add_argument("--t-switch", type=float, default=T_SWITCH,
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
        t_switch=args.t_switch,
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

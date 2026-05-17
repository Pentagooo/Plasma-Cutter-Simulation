from __future__ import annotations

"""Interaktive Simulation fuer kontinuierliche Schneidpfade (Lichtschwert).

Der Brenner wird als Lichtschwert visualisiert:
  - Griff (goldener Kreis) = TCP, faehrt entlang der Kontur
  - Klinge (roter Strich) = Plasmastrahl, ragt senkrecht nach innen

Waehrend der Griff sich bewegt, ueberstreicht die Klinge Material.
Alle Grid-Punkte in der Swept Area werden gruen markiert.

Bedienung
---------
  Linksklick : Griff-Wegpunkt setzen (manueller Pfad)
  Enter      : Manuellen Pfad ausfuehren
  R          : Alles zuruecksetzen
  Esc        : Wegpunkte verwerfen / Animation abbrechen
"""

from dataclasses import dataclass
from pathlib import Path
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

try:
    from ..geometry.point_grid import PointGrid, GridPoint, PointStatus
    from ..cutter.cutter import Cutter
    from ..cutter.continuous_path import (
        ContinuousPlanner, ContinuousPathResult, ContinuousCut,
        calculate_reward, _polyline_length,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.geometry.point_grid import PointGrid, GridPoint, PointStatus
    from plasma_cutter.cutter.cutter import Cutter
    from plasma_cutter.cutter.continuous_path import (
        ContinuousPlanner, ContinuousPathResult, ContinuousCut,
        calculate_reward, _polyline_length,
    )


# ---------------------------------------------------------------------------
# Farben
# ---------------------------------------------------------------------------

_C = dict(
    outer       = "#1E6FBF",
    inner       = "#AAAAAA",
    cut_point   = "#22A84E",
    grip        = "#FFD700",
    grip_edge   = "#B8860B",
    blade       = "#FF2222",
    blade_glow  = "#FF6644",
    swept_fill  = "#FF8888",
    trail_grip  = "#FF8C00",
    trail_air   = "#88CC88",
    waypoint    = "#E02020",
    preview     = "#FF8C00",
    stats_bg    = "#EEF4FF",
    grid_bg     = "#F9FAFB",
    ring_colors = [
        "#FF4444", "#FF7733", "#FFAA22", "#DDCC11",
        "#88BB22", "#33AA55", "#2299AA", "#3366CC",
        "#6644BB", "#AA33AA",
    ],
)


class _State(Enum):
    IDLE      = auto()
    DRAWING   = auto()
    ANIMATING = auto()


@dataclass
class _AnimWP:
    time: float
    grip_pos: np.ndarray
    blade_tip: np.ndarray
    is_cutting: bool
    cut_idx: int = 0
    progress: float = 0.0


# ---------------------------------------------------------------------------
# ContinuousCutSimulation
# ---------------------------------------------------------------------------

class ContinuousCutSimulation:

    def __init__(
        self,
        grid: PointGrid,
        cutter: Cutter | None = None,
        kerf_width: float = 3.0,
        cutter_radius: float | None = None,
        fps: int = 30,
    ) -> None:
        self.grid = grid
        self.cutter = cutter or Cutter()
        self.kerf_width = kerf_width
        self.minimum_gap = self.cutter.minimum_gap
        self.fps = fps
        self.cutter_radius = cutter_radius or grid.contour_spacing * 1.0

        self._planner = ContinuousPlanner(
            grid=grid, cutter=self.cutter,
            kerf_width=kerf_width,
        )

        self._state = _State.IDLE
        self._results: list[ContinuousPathResult] = []
        self._cum_coverages: list[float] = []  # kumulative Coverage pro Ergebnis
        self._draw_waypoints: list[np.ndarray] = []
        self._draw_artists: list = []
        self._current_result: ContinuousPathResult | None = None
        self._anim: FuncAnimation | None = None
        self._speed: float = 1.0  # Animations-Geschwindigkeitsfaktor
        self._fig: plt.Figure | None = None
        self._ax_main: plt.Axes | None = None
        self._ax_stats: plt.Axes | None = None
        self._ax_speed: plt.Axes | None = None
        self._speed_slider: Slider | None = None

    # ------------------------------------------------------------------

    @classmethod
    def run_with_dialog(cls, **kwargs) -> ContinuousCutSimulation | None:
        initial_dir = kwargs.pop("initial_dir", None)
        grid = PointGrid.from_json_dialog(initial_dir=initial_dir)
        if grid is None:
            return None
        sim = cls(grid=grid, **kwargs)
        sim.run()
        return sim

    @property
    def results(self) -> list[ContinuousPathResult]:
        return list(self._results)

    def run(self) -> None:
        self.grid.reset()
        self._results.clear()
        self._state = _State.IDLE
        self._draw_waypoints.clear()

        self._fig = plt.figure(figsize=(15, 9), facecolor="white")
        self._fig.suptitle(self._title(), fontsize=11,
                           fontweight="bold", y=0.98, color="#1A2840")

        gs = self._fig.add_gridspec(
            1, 2, width_ratios=[4, 1],
            left=0.05, right=0.98, top=0.93, bottom=0.12, wspace=0.03,
        )
        self._ax_main = self._fig.add_subplot(gs[0])
        self._ax_stats = self._fig.add_subplot(gs[1])

        self._ax_speed = self._fig.add_axes([0.15, 0.04, 0.4, 0.03])
        self._speed_slider = Slider(
            ax=self._ax_speed,
            label='Sim-Speed ',
            valmin=0.25,
            valmax=16.0,
            valinit=1.0,
            valstep=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            color="#0066CC"
        )

        def _update_slider(val):
            self._speed = val
            self._apply_speed()
            self._draw_stats()

        self._speed_slider.on_changed(_update_slider)

        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._redraw()
        plt.show()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax_main:
            return
        if event.button not in (1, 3):
            return
        if self._state == _State.ANIMATING:
            return
        pos = np.array([event.xdata, event.ydata])
        is_cutting = (event.button == 1)

        if self._state == _State.IDLE:
            self._state = _State.DRAWING
            self._draw_waypoints = [{"wps": [pos], "is_cutting": is_cutting}]
        elif self._state == _State.DRAWING:
            if not self._draw_waypoints:
                self._draw_waypoints = [{"wps": [pos], "is_cutting": is_cutting}]
            else:
                last_seg = self._draw_waypoints[-1]
                if last_seg["is_cutting"] == is_cutting:
                    last_seg["wps"].append(pos)
                else:
                    if last_seg["wps"]:
                        self._draw_waypoints.append({"wps": [last_seg["wps"][-1], pos], "is_cutting": is_cutting})
                    else:
                        self._draw_waypoints.append({"wps": [pos], "is_cutting": is_cutting})

        self._update_preview()
        self._draw_stats()

    def _on_key(self, event) -> None:
        # Geschwindigkeit aendern (funktioniert immer, auch waehrend Animation)
        if event.key in ("+", "="):
            new_speed = min(self._speed * 2.0, 16.0)
            if self._speed_slider is not None:
                self._speed_slider.set_val(new_speed)
            return
        if event.key in ("-", "_"):
            new_speed = max(self._speed / 2.0, 0.25)
            if self._speed_slider is not None:
                self._speed_slider.set_val(new_speed)
            return

        if event.key == "r":
            self._full_reset()
            return

        if event.key == "escape":
            if self._state == _State.ANIMATING:
                if self._anim is not None:
                    self._anim.event_source.stop()
                    self._anim = None
                if self._current_result:
                    for cut in self._current_result.continuous_cuts:
                        self.grid.apply_swept_area(cut.swept_polygon)
                self._state = _State.IDLE
                self._current_result = None
                self._redraw()
            elif self._state == _State.DRAWING:
                self._draw_waypoints.clear()
                self._clear_draw_artists()
                self._state = _State.IDLE
                self._draw_stats()
                self._fig.canvas.draw_idle()
            return

        if self._state == _State.ANIMATING:
            return

        if event.key == "enter" and self._state == _State.DRAWING:
            if any(len(s["wps"]) >= 2 for s in self._draw_waypoints):
                self._execute_custom()
            return

    def _apply_speed(self) -> None:
        """Aktualisiert das Animations-Intervall basierend auf _speed."""
        if self._anim is not None and self._anim.event_source is not None:
            # Bei _speed > 2 werden zusatzlich Frames uebersprungen (siehe _run_animation)
            effective_fps = self.fps * min(self._speed, 2.0)
            self._anim.event_source.interval = max(1, int(1000 / effective_fps))

    def _full_reset(self) -> None:
        if self._anim is not None:
            self._anim.event_source.stop()
            self._anim = None
        self._speed = 1.0
        if self._speed_slider is not None:
            self._speed_slider.set_val(1.0)
        self.grid.reset()
        self._results.clear()
        self._cum_coverages.clear()
        self._draw_waypoints.clear()
        self._draw_artists.clear()
        self._state = _State.IDLE
        self._current_result = None
        self._redraw()

    # ------------------------------------------------------------------
    # Vorschau
    # ------------------------------------------------------------------

    def _clear_draw_artists(self) -> None:
        for a in self._draw_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._draw_artists.clear()

    def _update_preview(self) -> None:
        self._clear_draw_artists()
        ax = self._ax_main
        wps_lists = self._draw_waypoints
        if not wps_lists:
            self._fig.canvas.draw_idle()
            return

        poly = self._planner._build_polygon_from_grid()
        all_feasible = True
        total_len = 0.0
        total_time = 0.0

        for segment in wps_lists:
            wps = segment["wps"]
            is_cutting = segment["is_cutting"]
            if not wps:
                continue
            
            # Waypoint-Marker
            for wp in wps:
                marker_color = _C["waypoint"] if is_cutting else "#666666"
                a, = ax.plot(wp[0], wp[1], "o", color=marker_color,
                             markersize=8, markeredgecolor="#800000",
                             markeredgewidth=1.2, zorder=15, alpha=0.9)
                self._draw_artists.append(a)

            if len(wps) >= 2:
                # TCP-Clearance pruefen
                _, is_feasible = self._planner._validate_grip_path(wps, poly)
                if not is_feasible:
                    all_feasible = False

                if is_cutting:
                    # Griff-Pfad (rot wenn infeasible)
                    path_color = _C["preview"] if is_feasible else "#CC0000"
                    xs = [w[0] for w in wps]
                    ys = [w[1] for w in wps]
                    ln, = ax.plot(xs, ys, color=path_color, linewidth=2.5,
                                  alpha=0.7, zorder=14, solid_capstyle="round")
                    self._draw_artists.append(ln)

                    # Klingen-Vorschau: Swept Area berechnen
                    normals = self._planner._compute_inward_normals(wps, poly)
                    blade_wps, swept = self._planner._compute_blade_sweep(wps, normals)

                    # Klingen als Striche zeichnen (jeden 3. Waypoint)
                    step = max(1, len(wps) // 10)
                    for i in range(0, len(wps), step):
                        gp = wps[i]
                        bp = blade_wps[i]
                        bl, = ax.plot([gp[0], bp[0]], [gp[1], bp[1]],
                                      color=_C["blade"], linewidth=1.5,
                                      alpha=0.4, zorder=13)
                        self._draw_artists.append(bl)

                    # Swept Area Vorschau
                    if swept is not None and not swept.is_empty:
                        try:
                            if hasattr(swept, "geoms"):
                                polys = list(swept.geoms)
                            else:
                                polys = [swept]
                            for p in polys:
                                sx, sy = p.exterior.xy
                                fill = ax.fill(sx, sy, color=_C["swept_fill"],
                                               alpha=0.10, zorder=4)[0]
                                self._draw_artists.append(fill)
                                edge, = ax.plot(sx, sy, color=_C["swept_fill"],
                                                linewidth=0.8, alpha=0.35, zorder=5)
                                self._draw_artists.append(edge)
                        except Exception:
                            pass

                    path_len = _polyline_length(wps)
                    cut_time = self.cutter.time_for_length(path_len, mode="cut")
                    total_len += path_len
                    total_time += cut_time
                else:
                    # Eilgang (rapid) Vorschau
                    rapid_color = "#666666" if is_feasible else "#CC0000"
                    xs = [w[0] for w in wps]
                    ys = [w[1] for w in wps]
                    a, = ax.plot(xs, ys, "--", color=rapid_color,
                                 linewidth=1.5, alpha=0.6, zorder=13)
                    self._draw_artists.append(a)
                    rapid_len = _polyline_length(wps)
                    total_time += self.cutter.time_for_length(rapid_len, mode="rapid")

        # Eilgang-Linien (rapid) zwischen Sub-Pfaden (z.B. nach einem Ende des Segments)
        for i in range(len(wps_lists) - 1):
            wps1 = wps_lists[i]["wps"]
            wps2 = wps_lists[i+1]["wps"]
            if wps1 and wps2:
                p1 = wps1[-1]
                p2 = wps2[0]
                air_dist = float(np.linalg.norm(p2 - p1))
                if air_dist > 1e-6:
                    _, rapid_feasible = self._planner._validate_grip_path([p1, p2], poly)
                    if not rapid_feasible:
                        all_feasible = False
                        rapid_color = "#CC0000"
                    else:
                        rapid_color = "#666666"
                    a, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color=rapid_color,
                                 linewidth=1.5, alpha=0.6, zorder=13)
                    self._draw_artists.append(a)
                    total_time += self.cutter.time_for_length(air_dist, mode="rapid")

        last_wps = wps_lists[-1]["wps"] if wps_lists[-1]["wps"] else (wps_lists[-2]["wps"] if len(wps_lists) > 1 else [])
        if len(last_wps) >= 2:
            if all_feasible:
                label = f"{total_len:.0f} mm | ~{total_time:.1f} s | Enter: ausfuehren"
                label_color = _C["preview"]
            else:
                label = (f"TCP zu nah! (min {self._planner.minimum_gap:.0f} mm)")
                label_color = "#CC0000"

            txt = ax.text(
                last_wps[-1][0], last_wps[-1][1] - self.grid.contour_spacing * 1.5,
                label,
                fontsize=7.5, color=label_color, ha="center", va="top",
                zorder=16,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=label_color, alpha=0.88, linewidth=0.8))
            self._draw_artists.append(txt)

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Ausfuehrung
    # ------------------------------------------------------------------

    def _execute_custom(self) -> None:
        segments = [(s["wps"], s["is_cutting"]) for s in self._draw_waypoints]
        self._draw_waypoints.clear()
        self._clear_draw_artists()

        saved_is_cut = self.grid._is_cut.copy()
        saved_cp = list(self.grid._cut_points)

        result = self._planner.plan_custom(segments, apply=True)

        # Pruefen ob der Schnitt die TCP-Clearance verletzt
        infeasible = [c for c in result.continuous_cuts if not c.is_feasible]
        if infeasible:
            # Schnitt abbrechen: Grid zuruecksetzen, Fehlermeldung
            self.grid._is_cut = saved_is_cut
            self.grid._cut_points = saved_cp
            self._state = _State.IDLE
            self._show_error(
                f"ABBRUCH: TCP-Pfad verletzt den Mindestabstand "
                f"({self._planner.minimum_gap:.0f} mm) zum Material!\n"
                f"Der Schneider darf nicht durch die Geometrie fahren."
            )
            return

        self._results.append(result)
        self._cum_coverages.append(
            self.grid.n_cut / max(1, self.grid.total_points))

        # Grid zuruecksetzen fuer schrittweise Animation
        self.grid._is_cut = saved_is_cut
        self.grid._cut_points = saved_cp
        self._start_animation(result)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _start_animation(self, result: ContinuousPathResult) -> None:
        self._current_result = result
        self._state = _State.ANIMATING

        print(result.summary())
        cum_cov = self.grid.n_cut / max(1, self.grid.total_points)
        print(f"  Coverage (kumulativ): {cum_cov:.1%}")
        print(f"  Reward: {calculate_reward(result, coverage_override=cum_cov):.2f}")

        if not result.continuous_cuts:
            print("  Kein Pfad!")
            self._state = _State.IDLE
            self._redraw()
            return

        anim_wps = self._build_anim_wps(result)
        if not anim_wps:
            self._finish_animation(result)
            return

        self._run_animation(result, anim_wps)

    def _build_anim_wps(self, result: ContinuousPathResult) -> list[_AnimWP]:
        out: list[_AnimWP] = []
        cum_time = 0.0

        for ci, cut in enumerate(result.continuous_cuts):
            g_wps = cut.grip_waypoints
            b_wps = cut.blade_waypoints
            if len(g_wps) < 2 or len(b_wps) < 2:
                continue

            # Verfahren (Luft) zum Anfang
            if ci > 0 and out:
                prev = out[-1].grip_pos
                start = np.asarray(g_wps[0])
                air_d = float(np.linalg.norm(start - prev))
                if air_d > 1e-3:
                    t_air = self.cutter.time_for_length(air_d, mode="rapid")
                    n = max(2, int(t_air * self.fps))
                    for s in range(n + 1):
                        f = s / n
                        gp = prev + f * (start - prev)
                        # Klinge auch interpolieren
                        bp_start = np.asarray(b_wps[0])
                        bp_prev = out[-1].blade_tip
                        bp = bp_prev + f * (bp_start - bp_prev)
                        out.append(_AnimWP(
                            time=cum_time + f * t_air,
                            grip_pos=gp, blade_tip=bp,
                            is_cutting=False, cut_idx=ci))
                    cum_time += t_air

            # Schneidphase
            seg_lens = [
                float(np.linalg.norm(
                    np.asarray(g_wps[i + 1]) - np.asarray(g_wps[i])))
                for i in range(len(g_wps) - 1)
            ]
            total_len = sum(seg_lens)
            if total_len < 1e-6:
                continue

            cut_time = self.cutter.time_for_length(total_len, mode="cut" if cut.is_cutting else "rapid")
            n_frames = max(4, int(cut_time * self.fps))

            cum_seg = [0.0]
            for sl in seg_lens:
                cum_seg.append(cum_seg[-1] + sl)

            for frame in range(n_frames + 1):
                frac = frame / n_frames
                dist = frac * total_len

                seg_i = 0
                for i in range(len(cum_seg) - 1):
                    if cum_seg[i] <= dist <= cum_seg[i + 1]:
                        seg_i = i
                        break

                seg_len = cum_seg[seg_i + 1] - cum_seg[seg_i]
                if seg_len < 1e-12:
                    lf = 0.0
                else:
                    lf = (dist - cum_seg[seg_i]) / seg_len

                gp = (np.asarray(g_wps[seg_i], dtype=float)
                      + lf * (np.asarray(g_wps[seg_i + 1], dtype=float)
                              - np.asarray(g_wps[seg_i], dtype=float)))
                bp = (np.asarray(b_wps[seg_i], dtype=float)
                      + lf * (np.asarray(b_wps[seg_i + 1], dtype=float)
                              - np.asarray(b_wps[seg_i], dtype=float)))

                out.append(_AnimWP(
                    time=cum_time + frac * cut_time,
                    grip_pos=gp, blade_tip=bp,
                    is_cutting=cut.is_cutting, cut_idx=ci, progress=frac))

            cum_time += cut_time

        return out

    def _run_animation(self, result: ContinuousPathResult,
                       anim_wps: list[_AnimWP]) -> None:
        self._redraw()
        ax = self._ax_main

        # Geplante Griff-Pfade (transparent)
        for ci, cut in enumerate(result.continuous_cuts):
            if len(cut.grip_waypoints) >= 2:
                w = np.array(cut.grip_waypoints)
                if cut.is_cutting:
                    c = _C["ring_colors"][ci % len(_C["ring_colors"])]
                else:
                    c = "#666666"
                ax.plot(w[:, 0], w[:, 1], color=c, linewidth=0.8,
                        alpha=0.2, zorder=5, linestyle="--")

        # Animierte Artists
        wp0 = anim_wps[0]

        grip_circle = mpatches.Circle(
            tuple(wp0.grip_pos), self.cutter_radius,
            facecolor=_C["grip"], edgecolor=_C["grip_edge"],
            linewidth=2.0, alpha=0.85, zorder=25)
        ax.add_patch(grip_circle)

        blade_line, = ax.plot(
            [wp0.grip_pos[0], wp0.blade_tip[0]],
            [wp0.grip_pos[1], wp0.blade_tip[1]],
            color=_C["blade"], linewidth=3.5, solid_capstyle="round",
            alpha=0.85, zorder=24)

        blade_glow, = ax.plot(
            [wp0.grip_pos[0], wp0.blade_tip[0]],
            [wp0.grip_pos[1], wp0.blade_tip[1]],
            color=_C["blade_glow"], linewidth=7, solid_capstyle="round",
            alpha=0.2, zorder=23)

        trail_grip, = ax.plot(
            [], [], color=_C["trail_grip"], linewidth=1.8,
            alpha=0.5, zorder=11, solid_capstyle="round")

        time_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.88),
            zorder=30)

        grip_xs: list[float] = []
        grip_ys: list[float] = []
        n_frames = len(anim_wps)
        sim = self
        completed: set[int] = set()

        def init():
            trail_grip.set_data([], [])
            time_text.set_text("")
            return grip_circle, blade_line, blade_glow, trail_grip, time_text

        def update(frame):
            wp = anim_wps[frame]
            gx, gy = wp.grip_pos
            bx, by = wp.blade_tip

            grip_circle.center = (gx, gy)
            blade_line.set_data([gx, bx], [gy, by])
            blade_glow.set_data([gx, bx], [gy, by])

            # Griff-Trail
            prev_cut = (frame > 0 and anim_wps[frame - 1].is_cutting)
            if not wp.is_cutting and prev_cut and grip_xs:
                grip_xs.append(float("nan"))
                grip_ys.append(float("nan"))
            grip_xs.append(gx)
            grip_ys.append(gy)
            trail_grip.set_data(grip_xs, grip_ys)

            if wp.is_cutting:
                # Inkrementell Punkte schneiden
                if frame % 3 == 0 or wp.progress >= 0.99:
                    cut = result.continuous_cuts[wp.cut_idx]
                    
                    # Partial swept area genau berechnen
                    n_wp = len(cut.grip_waypoints)
                    total_len = _polyline_length(cut.grip_waypoints)
                    cur_len = wp.progress * total_len
                    
                    partial_g = [cut.grip_waypoints[0]]
                    partial_b = [cut.blade_waypoints[0]]
                    accum = 0.0
                    for i in range(n_wp - 1):
                        p1 = np.asarray(cut.grip_waypoints[i])
                        p2 = np.asarray(cut.grip_waypoints[i+1])
                        seg_len = float(np.linalg.norm(p2 - p1))
                        if accum + seg_len < cur_len - 1e-6:
                            partial_g.append(cut.grip_waypoints[i+1])
                            partial_b.append(cut.blade_waypoints[i+1])
                            accum += seg_len
                        else:
                            break
                    partial_g.append(wp.grip_pos)
                    partial_b.append(wp.blade_tip)

                    # Band-Polygon aus partial grip + blade
                    from shapely.geometry import Polygon as ShapelyPolygon
                    gc = [(float(w[0]), float(w[1])) for w in partial_g]
                    bc = [(float(w[0]), float(w[1])) for w in partial_b]
                    band_c = gc + list(reversed(bc))
                    band_c.append(band_c[0])
                    try:
                        band = ShapelyPolygon(band_c)
                        if not band.is_valid:
                            band = band.buffer(0)
                        partial_swept = band.buffer(sim.kerf_width / 2)
                        new_cps = sim.grid.apply_swept_area(partial_swept)
                        if new_cps:
                            ax.scatter(
                                [p.x for p in new_cps],
                                [p.y for p in new_cps],
                                s=35, color=_C["cut_point"],
                                edgecolors="#115522", linewidths=0.4,
                                zorder=6, marker="P", alpha=0.8)
                    except Exception:
                        pass

                # Ring fertig -> Swept Area permanent zeichnen
                if wp.progress >= 0.99 and wp.cut_idx not in completed:
                    completed.add(wp.cut_idx)
                    cut = result.continuous_cuts[wp.cut_idx]
                    sim.grid.apply_swept_area(cut.swept_polygon)
                    if cut.swept_polygon is not None:
                        try:
                            sx, sy = cut.swept_polygon.exterior.xy
                            c = _C["ring_colors"][wp.cut_idx % len(_C["ring_colors"])]
                            ax.fill(sx, sy, color=c, alpha=0.10, zorder=3)
                            ax.plot(sx, sy, color=c, linewidth=0.6,
                                    alpha=0.3, zorder=4)
                        except Exception:
                            pass

            # Info
            status = "Schneiden" if wp.is_cutting else "Verfahren"
            pct = sim.grid.cut_percentage
            time_text.set_text(
                f"t = {wp.time:.1f} s\n"
                f"Status: {status}\n"
                f"Ring: {wp.cut_idx + 1}/{len(result.continuous_cuts)}\n"
                f"Geschnitten: {pct:.1f}%")

            if getattr(sim, "_last_drawn_pct", None) != pct:
                sim._last_drawn_pct = pct
                sim._draw_stats()

            if frame == n_frames - 1:
                sim._finish_animation(result)

            return grip_circle, blade_line, blade_glow, trail_grip, time_text

        def frame_generator():
            current = 0
            while current < n_frames - 1:
                yield current
                step = 1 if sim._speed <= 2.0 else int(sim._speed / 2.0)
                current += max(1, step)
            yield n_frames - 1

        effective_fps = self.fps * min(self._speed, 2.0)
        self._anim = FuncAnimation(
            self._fig, update, init_func=init,
            frames=frame_generator,
            save_count=n_frames,
            interval=max(1, int(1000 / effective_fps)),
            blit=False, repeat=False)
        self._fig.canvas.draw_idle()

    def _finish_animation(self, result: ContinuousPathResult) -> None:
        for cut in result.continuous_cuts:
            self.grid.apply_swept_area(cut.swept_polygon)
        self._state = _State.IDLE
        self._current_result = None
        self._redraw()

    # ------------------------------------------------------------------
    # Fehlermeldung
    # ------------------------------------------------------------------

    def _show_error(self, msg: str) -> None:
        """Zeigt eine Fehlermeldung in der Konsole und im Hauptfenster."""
        print(f"\n  *** FEHLER: {msg} ***\n")

        ax = self._ax_main
        if ax is not None:
            ax.text(
                0.5, 0.5, msg,
                transform=ax.transAxes, fontsize=11,
                color="#CC0000", ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.6", fc="#FFEEEE",
                          ec="#CC0000", linewidth=2.0, alpha=0.95),
                zorder=50, wrap=True,
            )
            self._fig.canvas.draw_idle()

        self._draw_stats()

    # ------------------------------------------------------------------
    # Zeichnen
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        self._draw_main()
        self._draw_stats()

    def _draw_main(self) -> None:
        ax = self._ax_main
        ax.clear()
        ax.set_facecolor(_C["grid_bg"])

        all_pts = self.grid.points
        if not all_pts:
            ax.text(0.5, 0.5, "Keine Punkte",
                    transform=ax.transAxes, ha="center", va="center")
            self._fig.canvas.draw_idle()
            return

        inner = self.grid.inner_points
        if inner:
            ax.scatter([p.x for p in inner], [p.y for p in inner],
                       s=14, color=_C["inner"], edgecolors="none",
                       alpha=0.50, zorder=2)

        outer = self.grid.outer_points
        if outer:
            ax.scatter([p.x for p in outer], [p.y for p in outer],
                       s=48, color=_C["outer"], edgecolors="#0D3A73",
                       linewidths=0.7, zorder=3, marker="s")

        cut_pts = self.grid.cut_points
        if cut_pts:
            ax.scatter([p.x for p in cut_pts], [p.y for p in cut_pts],
                       s=35, color=_C["cut_point"], edgecolors="#115522",
                       linewidths=0.4, zorder=6, marker="P", alpha=0.8)

        # Fruehere Ergebnisse
        results_to_draw = self._results
        if self._state == _State.ANIMATING and self._results:
            results_to_draw = self._results[:-1]

        for res in results_to_draw:
            for ci, cut in enumerate(res.continuous_cuts):
                c = _C["ring_colors"][ci % len(_C["ring_colors"])]
                if cut.swept_polygon is not None:
                    try:
                        sx, sy = cut.swept_polygon.exterior.xy
                        ax.fill(sx, sy, color=c, alpha=0.10, zorder=3)
                    except Exception:
                        pass
                if len(cut.grip_waypoints) >= 2:
                    w = np.array(cut.grip_waypoints)
                    ax.plot(w[:, 0], w[:, 1], color=c, linewidth=1.2,
                            alpha=0.3, zorder=5)

        # Legende
        ax.legend(
            handles=[
                Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=_C["outer"], markeredgecolor="#0D3A73",
                       markersize=9, label="Aussenpunkt"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=_C["inner"], markersize=7,
                       label="Innenpunkt"),
                Line2D([0], [0], marker="P", color="w",
                       markerfacecolor=_C["cut_point"], markeredgecolor="#115522",
                       markersize=9, label="Geschnitten"),
                mpatches.Patch(facecolor=_C["grip"],
                               edgecolor=_C["grip_edge"], label="Griff (TCP)"),
                Line2D([0], [0], color=_C["blade"], linewidth=3,
                       label="Klinge (Plasma)"),
                mpatches.Patch(facecolor=_C["swept_fill"], alpha=0.25,
                               label="Swept Area"),
            ],
            fontsize=7.5, loc="upper right",
            framealpha=0.92, edgecolor="#CCCCCC", labelspacing=0.45)

        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.22, color="#888")

        xs = [p.x for p in all_pts]
        ys = [p.y for p in all_pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        m = span * 0.09 + 5.0
        ax.set_xlim(min(xs) - m, max(xs) + m)
        ax.set_ylim(min(ys) - m, max(ys) + m)

        ax.text(
            0.5, -0.048,
            "Klick: Wegpunkt  |  Enter: ausfuehren  |  "
            "R: Reset  |  Esc: Abbruch  |  +/-: Speed",
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

        ax.text(0.5, 0.965, "STATISTIK", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="center", va="top",
                color="#1A2840")
        ax.plot([0.08, 0.92], [0.940, 0.940], color="#B0C4DE",
                linewidth=0.8, transform=ax.transAxes)

        total = self.grid.total_points
        n_cut = self.grid.n_cut
        pct = self.grid.cut_percentage
        y, dy = 0.915, 0.050

        def kv(label, value, vc="#2D2D2D", bold=False):
            nonlocal y
            ax.text(0.07, y, label, transform=ax.transAxes,
                    fontsize=7.8, color="#555555", va="top")
            ax.text(0.93, y, value, transform=ax.transAxes,
                    fontsize=7.8 if not bold else 8.5, color=vc,
                    ha="right", va="top",
                    fontweight="bold" if bold else "normal")
            y -= dy

        def sep():
            nonlocal y
            y -= dy * 0.2
            ax.plot([0.08, 0.92], [y + dy * 0.85, y + dy * 0.85],
                    color="#B0C4DE", linewidth=0.5, transform=ax.transAxes)

        kv("Punkte gesamt", str(total))
        kv("Geschnitten", f"{n_cut} / {total}",
           vc="#16A34A" if n_cut > 0 else "#555555", bold=True)
        kv("Anteil", f"{pct:.1f} %",
           vc="#16A34A" if pct > 0 else "#555555", bold=pct > 0)

        est_vol = n_cut * (self.grid.contour_spacing ** 2)
        kv("Volumen (ca.)", f"{est_vol:.0f} mm²", 
           vc="#16A34A" if est_vol > 0 else "#555555")

        if self._state == _State.DRAWING:
            n_pierces = sum(1 for s in self._draw_waypoints if s["is_cutting"] and len(s["wps"]) >= 2)
            kv("Zuendungen", str(n_pierces))
        elif self._current_result:
            kv("Zuendungen", str(self._current_result.n_pierces))
        elif self._results:
            kv("Zuendungen gesamt", str(sum(r.n_pierces for r in self._results)))
            
        sep()

        ax.text(0.5, y, "LICHTSCHWERT", transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="center", va="top",
                color="#1A2840")
        y -= dy * 0.85
        kv("Klingen-Laenge", f"{self._planner.blade_length:.0f} mm", bold=True)
        kv("Kerf-Breite", f"{self.kerf_width:.1f} mm")
        kv("Min. Abstand", f"{self.minimum_gap:.1f} mm",
           vc="#CC6600", bold=True)
        kv("Eff. Tiefe", f"{self._planner._effective_depth:.1f} mm")
        kv("v Schneiden", f"{self.cutter.cutting_speed:.1f} mm/s")
        kv("v Bewegen", f"{self.cutter.moving_speed:.1f} mm/s")
        kv("v Eilgang", f"{self.cutter.rapid_speed:.1f} mm/s")
        kv("Sim-Speed", f"{self._speed:.2g}x",
           vc="#0066CC" if self._speed != 1.0 else "#555555",
           bold=self._speed != 1.0)
        sep()

        if self._state == _State.DRAWING:
            ax.text(0.5, y, "PFAD-ZEICHNUNG", transform=ax.transAxes,
                    fontsize=8, fontweight="bold", ha="center", va="top",
                    color="#E02020")
            y -= dy * 0.85
            n_wps = sum(len(s["wps"]) for s in self._draw_waypoints)
            kv("Wegpunkte", str(n_wps), bold=True)
            if n_wps >= 2:
                total_len = sum(_polyline_length(s["wps"]) for s in self._draw_waypoints)
                kv("Pfadlaenge", f"{total_len:.1f} mm")
            sep()

        ax.text(0.5, y, "ERGEBNISSE", transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="center", va="top",
                color="#1A2840")
        y -= dy * 0.85

        n_res = len(self._results)
        for i, res in enumerate(reversed(self._results[-5:])):
            if y < 0.08:
                break
            idx = n_res - i
            # Kumulative Coverage fuer dieses Ergebnis
            cov_idx = idx - 1  # 0-basiert
            cum_cov = (self._cum_coverages[cov_idx]
                       if cov_idx < len(self._cum_coverages) else 0.0)

            ax.text(0.07, y, f"#{idx} ({res.strategy.upper()})",
                    transform=ax.transAxes, fontsize=7.5,
                    color=_C["trail_grip"], va="top", fontweight="bold")
            y -= dy * 0.65
            ax.text(0.10, y,
                    f"{res.n_pierces}x | {res.total_time:.1f}s | "
                    f"{res.total_points_cut}p | {cum_cov:.0%}",
                    transform=ax.transAxes, fontsize=6.5, color="#666", va="top")
            y -= dy * 0.55
            reward = calculate_reward(res, coverage_override=cum_cov)
            ax.text(0.10, y, f"Reward: {reward:.1f}",
                    transform=ax.transAxes, fontsize=6.8,
                    color="#16A34A" if reward > 0 else "#D93535",
                    va="top", fontweight="bold")
            y -= dy * 0.75

        if self._state == _State.IDLE:
            status, sc = "Klick oder P druecken", "#4B5563"
        elif self._state == _State.DRAWING:
            status = f"{sum(len(s['wps']) for s in self._draw_waypoints)} Punkt(e) | Enter"
            sc = _C["waypoint"]
        elif self._state == _State.ANIMATING:
            spd = f" ({self._speed:.2g}x)" if self._speed != 1.0 else ""
            status, sc = f"Animation{spd} ...", _C["blade"]
        else:
            status, sc = "", "#4B5563"

        ax.text(0.5, 0.025, status, transform=ax.transAxes,
                fontsize=7.5, color=sc, ha="center", va="bottom",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=sc, alpha=0.88, linewidth=0.8))
        self._fig.canvas.draw_idle()

    def _title(self) -> str:
        name = (f"  -  {self.grid._source_path.stem}"
                if hasattr(self.grid, "_source_path") else "")
        return (
            f"Lichtschwert-Simulation{name}  |  "
            f"Klinge: {self._planner.blade_length:.0f} mm  |  "
            f"Kerf: {self.kerf_width:.1f} mm  |  "
            f"Min. Abstand: {self.minimum_gap:.0f} mm")

    def print_summary(self) -> None:
        print("=" * 72)
        print("LICHTSCHWERT-SIMULATION - ZUSAMMENFASSUNG")
        print("=" * 72)
        cum_cov = self.grid.n_cut / max(1, self.grid.total_points)
        total_reward = calculate_reward(
            self._results[-1], coverage_override=cum_cov,
        ) if self._results else 0.0
        print(f"  Punkte: {self.grid.n_cut}/{self.grid.total_points} "
              f"({self.grid.cut_percentage:.1f}%)")
        print(f"  Gesamt-Reward: {total_reward:.2f}")
        for i, r in enumerate(self._results):
            cov = self._cum_coverages[i] if i < len(self._cum_coverages) else 0.0
            rw = calculate_reward(r, coverage_override=cov)
            print(f"    #{i+1}: {r.summary()} | Reward: {rw:.1f}")
        print("=" * 72)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--kerf-width", type=float, default=3.0)
    parser.add_argument("--v-cut", type=float, default=None)
    parser.add_argument("--v-move", type=float, default=None)
    parser.add_argument("--v-rapid", type=float, default=None)
    args = parser.parse_args()

    kw: dict = {}
    if args.max_depth is not None:
        kw["max_depth"] = args.max_depth
    if args.v_cut is not None:
        kw["cutting_speed"] = args.v_cut
    if args.v_move is not None:
        kw["moving_speed"] = args.v_move
    if args.v_rapid is not None:
        kw["rapid_speed"] = args.v_rapid

    ContinuousCutSimulation.run_with_dialog(
        cutter=Cutter(**kw) if kw else Cutter(),
        kerf_width=args.kerf_width)

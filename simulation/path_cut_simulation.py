from __future__ import annotations

"""Interaktive Pfad-Schnitt-Simulation auf einem Punktgitter.

Bedienung
---------
  1. Klick : Startpunkt A setzen (snappt zum naechsten Aussenpunkt)
  2. Klick : Endpunkt B setzen -> Pfad wird berechnet und animiert
  R        : Simulation zuruecksetzen
  Esc      : Auswahl verwerfen

Der Schneider (Kreis) folgt der Aussenkontur von A nach B und setzt
in regelmaessigen Abstaenden Schnitte senkrecht in das Material.

Architektur
-----------
Die Berechnung (``CuttingPath``) ist von der Visualisierung getrennt,
damit ein ML-Modell spaeter ``CuttingPath.compute(apply=False)`` aufrufen
kann, um Pfade zu bewerten, ohne die Simulation zu benoetigen.
"""

from dataclasses import dataclass
from pathlib import Path
from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

try:
    from ..geometry.point_grid import PointGrid, GridPoint, PointStatus
    from ..cutter.cutter import Cutter
    from ..cutter.cutting_path import (
        CuttingPath, PathResult, PathCut,
        polyline_length, cumulative_dists, interpolate_at,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.geometry.point_grid import PointGrid, GridPoint, PointStatus
    from plasma_cutter.cutter.cutter import Cutter
    from plasma_cutter.cutter.cutting_path import (
        CuttingPath, PathResult, PathCut,
        polyline_length, cumulative_dists, interpolate_at,
    )


# ---------------------------------------------------------------------------
# Farbpalette
# ---------------------------------------------------------------------------

_C = dict(
    outer       = "#1E6FBF",
    inner       = "#AAAAAA",
    cut_ok      = "#22A84E",
    cut_fail    = "#D93535",
    path_line   = "#FF8C00",
    cutter_fill = "#FFD700",
    cutter_edge = "#B8860B",
    cut_line    = "#E02020",
    preview     = "#E07B10",
    start       = "#E0A010",
    end_pt      = "#10A0E0",
    stats_bg    = "#EEF4FF",
    grid_bg     = "#F9FAFB",
)


# ---------------------------------------------------------------------------
# Simulationszustaende
# ---------------------------------------------------------------------------

class _State(Enum):
    IDLE      = auto()   # warten auf Startpunkt-Klick
    START_SET = auto()   # Startpunkt gewaehlt, warten auf Endpunkt
    ANIMATING = auto()   # Animation laeuft


# ---------------------------------------------------------------------------
# Animations-Waypoint
# ---------------------------------------------------------------------------

@dataclass
class _Waypoint:
    time: float                # kumulierte Zeit [s]
    cutter_pos: np.ndarray     # (x, y) Schneider-Position
    is_cutting: bool           # True waehrend Schneidphase
    cut_idx: int = -1          # Index in PathResult.cuts
    cut_progress: float = 0.0  # 0..1 fuer Schnittausdehnung


# ---------------------------------------------------------------------------
# PathCutSimulation
# ---------------------------------------------------------------------------

class PathCutSimulation:
    """Interaktive Pfad-Simulation auf einem PointGrid.

    Parameters
    ----------
    grid            : PointGrid mit der Geometrie
    cutter          : Cutter-Objekt (None -> Standardwerte)
    cut_spacing     : Abstand zwischen Schnitten entlang der Kontur [mm]
    cutter_radius   : visueller Radius des Schneider-Kreises [mm]
    cut_tolerance   : Treff-Toleranz Innen [mm]
    outer_tolerance : Treff-Toleranz Aussen [mm]
    fps             : Animations-Framerate
    """

    def __init__(
        self,
        grid: PointGrid,
        cutter: Cutter | None = None,
        cut_spacing: float | None = None,
        cutter_radius: float | None = None,
        cut_tolerance: float | None = None,
        outer_tolerance: float | None = None,
        fps: int = 30,
    ) -> None:
        self.grid = grid
        self.cutter = cutter or Cutter()
        self.fps = fps
        self.cutter_radius = cutter_radius or grid.contour_spacing * 1.2

        self._path_engine = CuttingPath(
            grid=grid,
            cutter=self.cutter,
            cut_spacing=cut_spacing,
            cut_tolerance=cut_tolerance,
            outer_tolerance=outer_tolerance,
        )

        # Zustand
        self._state = _State.IDLE
        self._start_gpt: GridPoint | None = None
        self._start_pos: np.ndarray | None = None
        self._path_results: list[PathResult] = []

        # Animations-Zustand
        self._current_result: PathResult | None = None
        self._applied_cuts: set[int] = set()
        self._cut_base_idx: int = 0
        self._anim: FuncAnimation | None = None

        # Matplotlib
        self._fig: plt.Figure | None = None
        self._ax_main: plt.Axes | None = None
        self._ax_stats: plt.Axes | None = None
        self._preview_artists: list = []
        self._hover_artist = None
        self._start_artist = None

        # Snap-Toleranz
        self._snap_tol = grid.contour_spacing * 1.5

    # ------------------------------------------------------------------
    # Klassenmethoden
    # ------------------------------------------------------------------

    @classmethod
    def run_with_dialog(
        cls,
        cutter: Cutter | None = None,
        cut_spacing: float | None = None,
        cutter_radius: float | None = None,
        cut_tolerance: float | None = None,
        outer_tolerance: float | None = None,
        fps: int = 30,
        initial_dir: str | Path | None = None,
    ) -> "PathCutSimulation | None":
        """Datei-Dialog oeffnen, Geometrie laden und Simulation starten."""
        grid = PointGrid.from_json_dialog(initial_dir=initial_dir)
        if grid is None:
            return None
        sim = cls(
            grid=grid, cutter=cutter, cut_spacing=cut_spacing,
            cutter_radius=cutter_radius, cut_tolerance=cut_tolerance,
            outer_tolerance=outer_tolerance, fps=fps,
        )
        sim.run()
        return sim

    # ------------------------------------------------------------------
    # Oeffentliche Schnittstelle
    # ------------------------------------------------------------------

    @property
    def path_results(self) -> list[PathResult]:
        """Alle bisherigen Pfad-Ergebnisse."""
        return list(self._path_results)

    def run(self) -> None:
        """Startet die interaktive Simulation (blockierend)."""
        self.grid.reset()
        self._path_results.clear()
        self._state = _State.IDLE
        self._start_gpt = None
        self._start_pos = None

        self._fig = plt.figure(figsize=(14, 8), facecolor="white")
        self._fig.suptitle(self._figure_title(), fontsize=11,
                           fontweight="bold", y=0.98, color="#1A2840")

        gs = self._fig.add_gridspec(
            1, 2, width_ratios=[4, 1],
            left=0.05, right=0.98, top=0.93, bottom=0.06, wspace=0.03,
        )
        self._ax_main = self._fig.add_subplot(gs[0])
        self._ax_stats = self._fig.add_subplot(gs[1])

        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw_main()
        self._draw_stats()
        plt.show()

    # ------------------------------------------------------------------
    # Event-Handler
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax_main or event.button != 1:
            return
        if self._state == _State.ANIMATING:
            return

        pos = np.array([event.xdata, event.ydata])

        if self._state == _State.IDLE:
            # -- Erster Klick: Startpunkt --
            snapped, gpt = self._snap_to_outer(pos)
            if gpt is None:
                return
            self._start_pos = snapped
            self._start_gpt = gpt
            self._state = _State.START_SET

            self._remove_hover()
            self._start_artist = self._ax_main.plot(
                snapped[0], snapped[1], "o",
                color=_C["start"], markersize=14,
                markeredgecolor="#7A4A00", markeredgewidth=1.5,
                zorder=11, alpha=0.92,
            )[0]
            self._draw_stats()
            self._fig.canvas.draw_idle()

        elif self._state == _State.START_SET:
            # -- Zweiter Klick: Endpunkt -> Pfad berechnen & animieren --
            snapped, gpt = self._snap_to_outer(pos)
            if gpt is None or gpt.index == self._start_gpt.index:
                return

            self._clear_preview()
            self._compute_and_animate(self._start_gpt, gpt)

    def _on_motion(self, event) -> None:
        if self._state == _State.ANIMATING:
            return
        if event.inaxes is not self._ax_main:
            self._remove_hover()
            return

        cursor = np.array([event.xdata, event.ydata])

        if self._state == _State.START_SET:
            self._update_path_preview(cursor)
        else:
            self._update_hover(cursor)

    def _on_key(self, event) -> None:
        if event.key == "r":
            # Reset
            if self._anim is not None:
                self._anim.event_source.stop()
                self._anim = None
            self.grid.reset()
            self._path_results.clear()
            self._state = _State.IDLE
            self._start_gpt = None
            self._start_pos = None
            self._start_artist = None
            self._preview_artists.clear()
            self._hover_artist = None
            self._current_result = None
            self._applied_cuts.clear()
            self._draw_main()
            self._draw_stats()

        elif event.key == "escape":
            if self._state == _State.START_SET:
                self._clear_preview()
                self._remove_hover()
                self._state = _State.IDLE
                self._start_gpt = None
                self._start_pos = None
                if self._start_artist is not None:
                    try:
                        self._start_artist.remove()
                    except ValueError:
                        pass
                    self._start_artist = None
                self._draw_stats()
                self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Snapping
    # ------------------------------------------------------------------

    def _snap_to_outer(
        self, pos: np.ndarray,
    ) -> tuple[np.ndarray, GridPoint | None]:
        """Snappt zum naechsten verbleibenden Aussenpunkt."""
        outer = self.grid.outer_points
        if not outer:
            return pos, None
        coords = np.array([[p.x, p.y] for p in outer])
        dists = np.linalg.norm(coords - pos, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= self._snap_tol:
            pt = outer[idx]
            return np.array([pt.x, pt.y]), pt
        return pos, None

    # ------------------------------------------------------------------
    # Vorschau (Hover in START_SET-Zustand)
    # ------------------------------------------------------------------

    def _clear_preview(self) -> None:
        for a in self._preview_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._preview_artists.clear()

    def _remove_hover(self) -> None:
        if self._hover_artist is not None:
            try:
                self._hover_artist.remove()
            except (ValueError, AttributeError):
                pass
            self._hover_artist = None
        if self._fig is not None:
            self._fig.canvas.draw_idle()

    def _update_hover(self, cursor: np.ndarray) -> None:
        """Hebt den naechsten Snap-Kandidaten hervor."""
        self._remove_hover()
        outer = self.grid.outer_points
        if not outer:
            return
        coords = np.array([[p.x, p.y] for p in outer])
        dists = np.linalg.norm(coords - cursor, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= self._snap_tol:
            pt = outer[idx]
            self._hover_artist = self._ax_main.plot(
                pt.x, pt.y, "o", color=_C["outer"], markersize=19,
                alpha=0.25, markeredgecolor=_C["outer"],
                markeredgewidth=2.0, zorder=8,
            )[0]
            self._fig.canvas.draw_idle()

    def _update_path_preview(self, cursor: np.ndarray) -> None:
        """Zeigt leichte Vorschau des Konturpfads von Start zum Hover-Punkt."""
        self._clear_preview()
        ax = self._ax_main

        snapped, gpt = self._snap_to_outer(cursor)
        if gpt is None or gpt.index == self._start_gpt.index:
            self._fig.canvas.draw_idle()
            return

        # Kontur-Teilpfad berechnen (ohne Schnitte auszufuehren)
        path_pts = self._path_engine.contour_subpath(self._start_gpt, gpt)
        if len(path_pts) < 2:
            self._fig.canvas.draw_idle()
            return

        coords = np.array([p.coords for p in path_pts])

        # Pfad zeichnen
        ln, = ax.plot(
            coords[:, 0], coords[:, 1],
            color=_C["path_line"], linewidth=2.5, alpha=0.6,
            zorder=7, solid_capstyle="round",
        )
        self._preview_artists.append(ln)

        # Endpunkt-Marker
        ep, = ax.plot(
            snapped[0], snapped[1], "o",
            color=_C["end_pt"], markersize=14,
            markeredgecolor="#0A6090", markeredgewidth=1.5,
            zorder=11, alpha=0.85,
        )
        self._preview_artists.append(ep)

        # Schnitt-Positionen als kleine Striche anzeigen
        path_len = polyline_length([p.coords for p in path_pts])
        if path_len > self._path_engine.cut_spacing * 0.5:
            n_cuts = max(1, round(path_len / self._path_engine.cut_spacing))
            spacing = path_len / (n_cuts + 1)
            cum = cumulative_dists([p.coords for p in path_pts])
            for k in range(1, n_cuts + 1):
                d = k * spacing
                cpos = interpolate_at([p.coords for p in path_pts], cum, d)
                tick, = ax.plot(
                    cpos[0], cpos[1], "+",
                    color=_C["cut_line"], markersize=8, markeredgewidth=1.5,
                    zorder=10, alpha=0.6,
                )
                self._preview_artists.append(tick)

        # Info-Text
        n_cuts_est = max(1, round(path_len / self._path_engine.cut_spacing)) \
            if path_len > self._path_engine.cut_spacing * 0.5 else 0
        travel_t = self.cutter.time_for_length(path_len, mode="move")

        txt = ax.text(
            snapped[0], snapped[1] - self.grid.contour_spacing * 2.0,
            f"Pfad: {path_len:.0f} mm | ~{n_cuts_est} Schnitte | ~{travel_t:.1f} s",
            fontsize=7.5, color=_C["path_line"], ha="center", va="top",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=_C["path_line"], alpha=0.88, linewidth=0.8),
        )
        self._preview_artists.append(txt)

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Pfad berechnen & Animation starten
    # ------------------------------------------------------------------

    def _compute_and_animate(
        self, start: GridPoint, end: GridPoint,
    ) -> None:
        # Grid-Zustand sichern
        saved_ci = self.grid._cut_indices.copy()
        saved_cp = list(self.grid._cut_points)

        # Berechnung (modifiziert Grid temporaer)
        result = self._path_engine.compute(start, end, apply=True)
        self._path_results.append(result)

        # Grid zuruecksetzen (Schnitte werden waehrend Animation re-applied)
        self.grid._cut_indices = saved_ci
        self.grid._cut_points = saved_cp

        self._current_result = result
        self._applied_cuts = set()
        self._cut_base_idx = self.grid.n_cut
        self._state = _State.ANIMATING

        print(result.summary())

        # Waypoints bauen
        waypoints = self._build_waypoints(result)
        if not waypoints:
            self._finish_animation()
            return

        # Basis neu zeichnen (ohne die neuen Schnitte)
        self._draw_main()
        self._draw_stats()

        ax = self._ax_main

        # Konturpfad (statisch)
        if result.contour_points:
            c = np.array(result.contour_points)
            ax.plot(
                c[:, 0], c[:, 1],
                color=_C["path_line"], linewidth=2.0, alpha=0.4,
                zorder=6, solid_capstyle="round",
            )

        # Start- und Endmarker
        ax.plot(
            result.contour_points[0][0], result.contour_points[0][1],
            "o", color=_C["start"], markersize=12,
            markeredgecolor="#7A4A00", markeredgewidth=1.3,
            zorder=12, alpha=0.9,
        )
        ax.plot(
            result.contour_points[-1][0], result.contour_points[-1][1],
            "o", color=_C["end_pt"], markersize=12,
            markeredgecolor="#0A6090", markeredgewidth=1.3,
            zorder=12, alpha=0.9,
        )

        # Animierte Artists
        cutter_circle = mpatches.Circle(
            tuple(waypoints[0].cutter_pos), self.cutter_radius,
            facecolor=_C["cutter_fill"], edgecolor=_C["cutter_edge"],
            linewidth=2.0, alpha=0.85, zorder=15,
        )
        ax.add_patch(cutter_circle)

        cut_line_artist, = ax.plot(
            [], [], color=_C["cut_line"], linewidth=3.0,
            solid_capstyle="round", zorder=14,
        )

        time_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes,
            fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            zorder=20,
        )

        n_frames = len(waypoints)
        sim_ref = self  # Referenz fuer Closure

        def init():
            cut_line_artist.set_data([], [])
            time_text.set_text("")
            return cutter_circle, cut_line_artist, time_text

        def update(frame):
            wp = waypoints[frame]
            cx, cy = wp.cutter_pos

            # Schneider bewegen
            cutter_circle.center = (cx, cy)

            # Schnittlinie
            if wp.is_cutting and wp.cut_idx >= 0:
                pc = result.cuts[wp.cut_idx]
                end_pt = pc.position + wp.cut_progress * pc.cut_length * pc.direction
                cut_line_artist.set_data(
                    [pc.position[0], end_pt[0]],
                    [pc.position[1], end_pt[1]],
                )

                # Schnitt abgeschlossen -> permanent zeichnen & Grid aktualisieren
                if wp.cut_progress >= 0.99 and wp.cut_idx not in sim_ref._applied_cuts:
                    sim_ref._applied_cuts.add(wp.cut_idx)

                    # Schnitt auf Grid anwenden
                    sim_ref.grid.apply_cut(
                        pc.points_hit, sim_ref._cut_base_idx + wp.cut_idx,
                    )

                    # Permanente Schnittlinie
                    color = _C["cut_ok"] if pc.is_feasible else _C["cut_fail"]
                    full_end = pc.position + pc.cut_length * pc.direction
                    ax.plot(
                        [pc.position[0], full_end[0]],
                        [pc.position[1], full_end[1]],
                        color=color, linewidth=2.0, alpha=0.6,
                        zorder=10, solid_capstyle="round",
                    )

                    # Getroffene Punkte markieren
                    if pc.points_hit:
                        hx = [p.x for p in pc.points_hit]
                        hy = [p.y for p in pc.points_hit]
                        marker_c = _C["cut_ok"] if pc.is_feasible else _C["cut_fail"]
                        ax.scatter(
                            hx, hy, s=40, color=marker_c,
                            edgecolors="#111", linewidths=0.4,
                            zorder=11, marker="P",
                        )
            else:
                cut_line_artist.set_data([], [])

            # Info-Text
            n_done = len(sim_ref._applied_cuts)
            status = "Schneiden" if wp.is_cutting else "Verfahren"
            time_text.set_text(
                f"t = {wp.time:.1f} s\n"
                f"Status: {status}\n"
                f"Schnitte: {n_done} / {result.n_total}"
            )

            # Letzter Frame: Animation beenden
            if frame == n_frames - 1:
                sim_ref._finish_animation()

            return cutter_circle, cut_line_artist, time_text

        self._anim = FuncAnimation(
            self._fig, update, init_func=init,
            frames=n_frames, interval=1000 // self.fps,
            blit=False, repeat=False,
        )
        self._fig.canvas.draw_idle()

    def _finish_animation(self) -> None:
        """Animation abschliessen, restliche Schnitte anwenden."""
        if self._current_result is not None:
            for i, pc in enumerate(self._current_result.cuts):
                if i not in self._applied_cuts:
                    self.grid.apply_cut(
                        pc.points_hit, self._cut_base_idx + i,
                    )
                    self._applied_cuts.add(i)

        self._state = _State.IDLE
        self._start_gpt = None
        self._start_pos = None
        self._start_artist = None
        self._current_result = None
        self._draw_stats()

    # ------------------------------------------------------------------
    # Waypoint-Konstruktion
    # ------------------------------------------------------------------

    def _build_waypoints(self, result: PathResult) -> list[_Waypoint]:
        """Baut Animations-Waypoints: Verfahren + Schneiden abwechselnd."""
        if not result.contour_points:
            return []

        coords = result.contour_points
        cum = cumulative_dists([np.array(c) for c in coords])
        path_length = cum[-1]

        waypoints: list[_Waypoint] = []
        cum_time = 0.0
        prev_dist = 0.0

        for cut_idx, pc in enumerate(result.cuts):
            # Verfahren von prev_dist nach pc.path_distance
            travel_dist = pc.path_distance - prev_dist
            if travel_dist > 1e-6:
                travel_time = self.cutter.time_for_length(travel_dist, mode="move")
                n_steps = max(2, int(travel_time * self.fps))
                for s in range(n_steps + 1):
                    frac = s / n_steps
                    d = prev_dist + frac * travel_dist
                    pos = interpolate_at(coords, cum, d)
                    waypoints.append(_Waypoint(
                        time=cum_time + frac * travel_time,
                        cutter_pos=pos,
                        is_cutting=False,
                    ))
                cum_time += travel_time

            # Schneidphase
            if pc.cut_length > 1e-6:
                n_steps = max(2, int(pc.cutting_time * self.fps))
                for s in range(n_steps + 1):
                    frac = s / n_steps
                    waypoints.append(_Waypoint(
                        time=cum_time + frac * pc.cutting_time,
                        cutter_pos=pc.position.copy(),
                        is_cutting=True,
                        cut_idx=cut_idx,
                        cut_progress=frac,
                    ))
                cum_time += pc.cutting_time

            prev_dist = pc.path_distance

        # Restliches Verfahren zum Ende
        final_dist = path_length - prev_dist
        if final_dist > 1e-6:
            travel_time = self.cutter.time_for_length(final_dist, mode="move")
            n_steps = max(2, int(travel_time * self.fps))
            for s in range(n_steps + 1):
                frac = s / n_steps
                d = prev_dist + frac * final_dist
                pos = interpolate_at(coords, cum, d)
                waypoints.append(_Waypoint(
                    time=cum_time + frac * travel_time,
                    cutter_pos=pos,
                    is_cutting=False,
                ))
            cum_time += travel_time

        return waypoints

    # ------------------------------------------------------------------
    # Hauptplot
    # ------------------------------------------------------------------

    def _draw_main(self) -> None:
        ax = self._ax_main
        ax.clear()
        ax.set_facecolor(_C["grid_bg"])

        all_pts = self.grid.points
        if not all_pts:
            ax.text(0.5, 0.5, "Keine Punkte geladen",
                    transform=ax.transAxes, ha="center", va="center")
            self._fig.canvas.draw_idle()
            return

        # Innenpunkte
        inner = self.grid.inner_points
        if inner:
            ax.scatter(
                [p.x for p in inner], [p.y for p in inner],
                s=12, color=_C["inner"], edgecolors="none",
                alpha=0.50, zorder=2,
            )

        # Aussenpunkte
        outer = self.grid.outer_points
        if outer:
            ax.scatter(
                [p.x for p in outer], [p.y for p in outer],
                s=48, color=_C["outer"], edgecolors="#0D3A73",
                linewidths=0.7, zorder=3, marker="s",
            )

        # Bereits geschnittene Punkte
        cut_pts = self.grid.cut_points
        if cut_pts:
            ax.scatter(
                [p.x for p in cut_pts], [p.y for p in cut_pts],
                s=55, color=_C["cut_ok"], edgecolors="#111",
                linewidths=0.5, zorder=4, marker="P",
            )

        # Fruehere Pfade (statisch)
        # Waehrend der Animation: aktuellen Pfad nicht zeichnen (wird animiert)
        results_to_draw = self._path_results
        if self._state == _State.ANIMATING and self._path_results:
            results_to_draw = self._path_results[:-1]

        for pr in results_to_draw:
            if pr.contour_points:
                c = np.array(pr.contour_points)
                ax.plot(
                    c[:, 0], c[:, 1], color=_C["path_line"],
                    linewidth=1.5, alpha=0.3, zorder=5,
                )
            for pc in pr.cuts:
                if pc.cut_length > 0:
                    end_pt = pc.position + pc.cut_length * pc.direction
                    color = _C["cut_ok"] if pc.is_feasible else _C["cut_fail"]
                    ax.plot(
                        [pc.position[0], end_pt[0]],
                        [pc.position[1], end_pt[1]],
                        color=color, linewidth=1.5, alpha=0.4,
                        zorder=6, solid_capstyle="round",
                    )

        # Startpunkt-Marker (falls in START_SET)
        if self._start_pos is not None and self._state == _State.START_SET:
            self._start_artist = ax.plot(
                self._start_pos[0], self._start_pos[1], "o",
                color=_C["start"], markersize=14,
                markeredgecolor="#7A4A00", markeredgewidth=1.5,
                zorder=11, alpha=0.93,
            )[0]

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
                       markerfacecolor=_C["cut_ok"], markeredgecolor="#111",
                       markersize=9, label="Geschnitten"),
                Line2D([0], [0], color=_C["path_line"], linewidth=2,
                       label="Schnittpfad"),
                Line2D([0], [0], color=_C["cut_line"], linewidth=2,
                       label="Schnitt"),
                mpatches.Patch(facecolor=_C["cutter_fill"],
                               edgecolor=_C["cutter_edge"],
                               label="Schneide"),
            ],
            fontsize=7.5, loc="upper right",
            framealpha=0.92, edgecolor="#CCCCCC", labelspacing=0.45,
        )

        # Achsen
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

        hint = ("1. Klick: Startpunkt (A)  |  2. Klick: Endpunkt (B)  |  "
                "R: Reset  |  Esc: Abbrechen")
        ax.text(
            0.5, -0.048, hint, transform=ax.transAxes,
            fontsize=7.5, color="#888", ha="center", va="top",
        )

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Statistik-Panel
    # ------------------------------------------------------------------

    def _draw_stats(self) -> None:
        ax = self._ax_stats
        ax.clear()
        ax.axis("off")

        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, 0.01), 0.96, 0.97,
            boxstyle="round,pad=0.01",
            facecolor=_C["stats_bg"], edgecolor="#B0C4DE",
            linewidth=1.0, transform=ax.transAxes, zorder=0,
        ))

        ax.text(0.5, 0.965, "STATISTIK", transform=ax.transAxes,
                fontsize=9.5, fontweight="bold", ha="center", va="top",
                color="#1A2840")
        ax.plot([0.08, 0.92], [0.940, 0.940], color="#B0C4DE",
                linewidth=0.8, transform=ax.transAxes)

        total = self.grid.total_points
        n_cut = self.grid.n_cut
        pct = self.grid.cut_percentage

        y, dy = 0.915, 0.055

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
        kv("Geschnitten",
           f"{n_cut} / {total}",
           vc="#16A34A" if n_cut > 0 else "#555555", bold=True)
        kv("Anteil", f"{pct:.1f} %",
           vc="#16A34A" if pct > 0 else "#555555", bold=pct > 0)
        sep()

        # Schneider-Parameter
        ax.text(0.5, y, "SCHNEIDER", transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="center", va="top",
                color="#1A2840")
        y -= dy * 0.85
        kv("Schnitttiefe", f"{self.cutter.max_depth:.0f} mm", bold=True)
        kv("max. Winkel", f"{np.degrees(self.cutter.max_tilt):.0f}\u00b0")
        kv("v Schneiden", f"{self.cutter.cutting_speed:.1f} mm/s")
        kv("v Bewegen", f"{self.cutter.moving_speed:.1f} mm/s")
        kv("v Eilgang", f"{self.cutter.rapid_speed:.1f} mm/s")
        kv("Schnittabstand", f"{self._path_engine.cut_spacing:.1f} mm")
        sep()

        # Pfad-Ergebnisse
        ax.text(0.5, y, "PFADE", transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="center", va="top",
                color="#1A2840")
        y -= dy * 0.85

        for i, pr in enumerate(reversed(self._path_results[-6:])):
            if y < 0.07:
                break
            idx = len(self._path_results) - i
            ax.text(0.07, y, f"Pfad #{idx}",
                    transform=ax.transAxes, fontsize=7.5,
                    color=_C["path_line"], va="top", fontweight="bold")
            ax.text(0.93, y,
                    f"{pr.n_successful}/{pr.n_total} OK",
                    transform=ax.transAxes, fontsize=6.8,
                    color="#444444", ha="right", va="top")
            y -= dy * 0.75
            ax.text(0.10, y,
                    f"{pr.total_path_length:.0f}mm  {pr.total_time:.1f}s  "
                    f"{pr.total_points_cut}p",
                    transform=ax.transAxes, fontsize=6.5,
                    color="#888888", va="top")
            y -= dy * 0.75

        # Statuszeile
        if self._state == _State.IDLE:
            status, s_col = "Startpunkt (A) klicken", "#4B5563"
        elif self._state == _State.START_SET:
            status, s_col = "Endpunkt (B) klicken ...", _C["start"]
        elif self._state == _State.ANIMATING:
            status, s_col = "Animation ...", _C["path_line"]
        else:
            status, s_col = "", "#4B5563"

        ax.text(
            0.5, 0.025, status, transform=ax.transAxes,
            fontsize=7.5, color=s_col, ha="center", va="bottom",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=s_col, alpha=0.88, linewidth=0.8),
        )

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _figure_title(self) -> str:
        name = (f"  \u2013  {self.grid._source_path.stem}"
                if hasattr(self.grid, "_source_path") else "")
        return (
            f"Pfad-Schnitt-Simulation{name}  \u2502  "
            f"Tiefe: {self.cutter.max_depth:.0f} mm  \u2502  "
            f"max. Winkel: {np.degrees(self.cutter.max_tilt):.0f}\u00b0  \u2502  "
            f"Abstand: {self._path_engine.cut_spacing:.1f} mm"
        )

    def print_summary(self) -> None:
        """Zusammenfassung auf der Konsole ausgeben."""
        print("=" * 72)
        print("PFAD-SCHNITT-SIMULATION \u2013 ZUSAMMENFASSUNG")
        print("=" * 72)
        print(f"  Punkte gesamt  : {self.grid.total_points}")
        print(f"  Geschnitten    : {self.grid.n_cut} "
              f"({self.grid.cut_percentage:.1f}%)")
        print(f"  Pfade          : {len(self._path_results)}")
        for i, pr in enumerate(self._path_results):
            print(f"    Pfad #{i + 1}: {pr.summary()}")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Direktaufruf
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interaktive Pfad-Schnitt-Simulation."
    )
    parser.add_argument("--max-depth", type=float, default=None,
                        help="Maximale Schnitttiefe [mm]")
    parser.add_argument("--max-tilt", type=float, default=None,
                        help="Maximaler Kippwinkel [\u00b0]")
    parser.add_argument("--v-cut", type=float, default=None,
                        help="Schneidgeschwindigkeit [mm/s]")
    parser.add_argument("--v-move", type=float, default=None,
                        help="Bewegungsgeschwindigkeit (ohne Schneiden) [mm/s]")
    parser.add_argument("--v-rapid", type=float, default=None,
                        help="Eilgang-Geschwindigkeit [mm/s]")
    parser.add_argument("--cut-spacing", type=float, default=None,
                        help="Abstand zwischen Schnitten [mm]")
    parser.add_argument("--cutter-radius", type=float, default=None,
                        help="Visueller Schneider-Radius [mm]")
    args = parser.parse_args()

    cutter_kwargs: dict = {}
    if args.max_depth is not None:
        cutter_kwargs["max_depth"] = args.max_depth
    if args.max_tilt is not None:
        cutter_kwargs["max_tilt"] = np.radians(args.max_tilt)
    if args.v_cut is not None:
        cutter_kwargs["cutting_speed"] = args.v_cut
    if args.v_move is not None:
        cutter_kwargs["moving_speed"] = args.v_move
    if args.v_rapid is not None:
        cutter_kwargs["rapid_speed"] = args.v_rapid

    cutter = Cutter(**cutter_kwargs) if cutter_kwargs else Cutter()

    PathCutSimulation.run_with_dialog(
        cutter=cutter,
        cut_spacing=args.cut_spacing,
        cutter_radius=args.cutter_radius,
    )
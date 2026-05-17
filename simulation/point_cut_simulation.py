from __future__ import annotations

"""Interaktive Schnitt-Simulation auf einem Punktgitter.

Bedienung
---------
  1. Klick   : Startpunkt setzen (snappt zum nächsten Außenpunkt)
  Maus       : Live-Vorschau — Schnittstrahl, Reichweite, erlaubter Winkelbereich,
               Kandidaten-Punkte mit Anzahl und Länge
  2. Klick   : Richtung bestätigen → Schnitt ausführen
  R          : Simulation zurücksetzen
  Esc        : Startpunkt verwerfen

Erfolgskriterien
----------------
  ✓  Der Endpunkt des Schnitts ist kein Innenpunkt
     (= Schnitt erreicht die gegenüberliegende Seite)

Physikalische Constraints (aus Cutter-Parametern)
--------------------------------------------------
  • max_depth  → maximale Schnittlänge [mm]  (Schnitt endet spätestens hier)
  • max_tilt   → Winkel zwischen Schnittrichtung und Kontur-Normalen am
                 Startpunkt darf max_tilt nicht überschreiten
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    from ..geometry.point_grid import PointGrid, GridPoint, CutPoint, PointStatus
    from ..cutter.cutter import Cutter
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.geometry.point_grid import PointGrid, GridPoint, CutPoint, PointStatus
    from plasma_cutter.cutter.cutter import Cutter


# ---------------------------------------------------------------------------
# Farbpalette
# ---------------------------------------------------------------------------

_C = dict(
    outer      = "#1E6FBF",
    inner      = "#AAAAAA",
    cut_ok     = "#22A84E",
    cut_fail   = "#D93535",
    preview    = "#E07B10",
    start      = "#E0A010",
    reach_fill = "#FDE8C8",
    cone_ok    = "#A8E6B0",   # erlaubter Winkelbereich (grün)
    cone_err   = "#F5AAAA",   # Winkelbereich verletzt (rot)
    normal_line= "#338844",   # Kontur-Normale
    stats_bg   = "#EEF4FF",
    grid_bg    = "#F9FAFB",
)


# ---------------------------------------------------------------------------
# CutResult
# ---------------------------------------------------------------------------

@dataclass
class CutResult:
    """Ergebnis eines einzelnen Schnittversuchs."""
    cut_index:      int
    points_cut:     list[CutPoint]
    is_successful:  bool
    failure_reason: str
    start_point:    Optional[CutPoint]
    end_point:      Optional[CutPoint]
    start_pos:      np.ndarray
    direction:      np.ndarray
    cut_length:     float       # Abstand Start→Ende [mm]
    angle_deg:      float       # Winkel zur Kontur-Normalen [°]
    truncated:      bool        # True wenn max_depth begrenzt hat

    @property
    def n_points(self) -> int:
        return len(self.points_cut)

    @property
    def n_outer_cut(self) -> int:
        return sum(1 for p in self.points_cut if p.was_outer)

    @property
    def n_inner_cut(self) -> int:
        return sum(1 for p in self.points_cut if p.was_inner)

    def summary(self) -> str:
        status = "OK" if self.is_successful else f"FAIL – {self.failure_reason}"
        return (
            f"Schnitt #{self.cut_index + 1}: {status}  |  "
            f"{self.n_points} Pkt ({self.n_outer_cut} außen, {self.n_inner_cut} innen)  |  "
            f"Länge: {self.cut_length:.1f} mm  |  Winkel: {self.angle_deg:.1f}°"
        )


# ---------------------------------------------------------------------------
# PointCutSimulation
# ---------------------------------------------------------------------------

class PointCutSimulation:
    """Interaktive manuelle Schnitt-Simulation auf einem PointGrid.

    Parameters
    ----------
    grid           : das zu schneidende PointGrid
    cutter         : Cutter-Objekt mit max_depth [mm] und max_tilt [rad].
                     Wird über ``Cutter()`` erzeugt.  None → keine Constraints.
    cut_tolerance  : Treff-Toleranz für Innenpunkte (senkr. Abstand) [mm].
                     Standard: 55 % des Punktabstands.
    outer_tolerance: Treff-Toleranz für Außenpunkte [mm] – großzügiger.
                     Standard: 60 % des Konturabstands.
    """

    def __init__(
        self,
        grid:            PointGrid,
        cutter:          Cutter | None = None,
        cut_tolerance:   float | None = None,
        outer_tolerance: float | None = None,
    ) -> None:
        self.grid    = grid
        self.cutter  = cutter or Cutter()

        self.max_depth: float        = self.cutter.max_depth
        self.max_tilt:  float        = self.cutter.max_tilt   # [rad]

        # Toleranzen
        self.cut_tolerance   = cut_tolerance   or grid.point_spacing   * 0.55
        self.outer_tolerance = outer_tolerance or grid.contour_spacing * 0.60

        # Snap-Radius: innerhalb davon wird zum nächsten Außenpunkt gesprungen
        self.snap_tolerance = grid.contour_spacing * 1.5

        self._cuts: list[CutResult] = []

        # Interaktionszustand
        self._start_pos:  np.ndarray | None = None
        self._start_gpt:  GridPoint  | None = None

        # Matplotlib
        self._fig:      plt.Figure | None = None
        self._ax_main:  plt.Axes   | None = None
        self._ax_stats: plt.Axes   | None = None

        self._preview_artists: list = []
        self._hover_artist    = None
        self._start_artist    = None

    # ------------------------------------------------------------------
    # Klassenmethoden
    # ------------------------------------------------------------------

    @classmethod
    def run_with_dialog(
        cls,
        cutter:          Cutter | None = None,
        cut_tolerance:   float | None = None,
        outer_tolerance: float | None = None,
        initial_dir:     str | Path | None = None,
    ) -> "PointCutSimulation | None":
        """Datei-Dialog öffnen, Geometrie laden und Simulation starten."""
        grid = PointGrid.from_json_dialog(initial_dir=initial_dir)
        if grid is None:
            return None
        sim = cls(grid=grid, cutter=cutter,
                  cut_tolerance=cut_tolerance, outer_tolerance=outer_tolerance)
        sim.run()
        return sim

    # ------------------------------------------------------------------
    # Öffentliche Schnittstelle
    # ------------------------------------------------------------------

    @property
    def cuts(self) -> list[CutResult]:
        return list(self._cuts)

    @property
    def n_successful(self) -> int:
        return sum(1 for c in self._cuts if c.is_successful)

    @property
    def n_unsuccessful(self) -> int:
        return sum(1 for c in self._cuts if not c.is_successful)

    def run(self) -> None:
        """Startet die interaktive Simulation (blockierend)."""
        self.grid.reset()
        self._cuts.clear()
        self._start_pos = None
        self._start_gpt = None

        self._fig = plt.figure(figsize=(14, 8), facecolor="white")
        self._fig.suptitle(self._figure_title(), fontsize=11,
                           fontweight="bold", y=0.98, color="#1A2840")

        gs = self._fig.add_gridspec(
            1, 2, width_ratios=[4, 1],
            left=0.05, right=0.98, top=0.93, bottom=0.06, wspace=0.03,
        )
        self._ax_main  = self._fig.add_subplot(gs[0])
        self._ax_stats = self._fig.add_subplot(gs[1])

        self._fig.canvas.mpl_connect("button_press_event",  self._on_click)
        self._fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._fig.canvas.mpl_connect("key_press_event",     self._on_key)

        self._draw_main()
        self._draw_stats()
        plt.show()

    # ------------------------------------------------------------------
    # Event-Handler
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes is not self._ax_main or event.button != 1:
            return

        pos = np.array([event.xdata, event.ydata])

        if self._start_pos is None:
            # ── Erster Klick: Startpunkt ──
            snapped_pos, snapped_pt = self._snap_to_outer(pos)
            self._start_pos = snapped_pos
            self._start_gpt = snapped_pt

            self._remove_hover()
            self._start_artist = self._ax_main.plot(
                snapped_pos[0], snapped_pos[1], "o",
                color=_C["start"], markersize=13, markeredgecolor="#7A4A00",
                markeredgewidth=1.5, zorder=11, alpha=0.92,
            )[0]
            self._draw_stats()
            self._fig.canvas.draw_idle()

        else:
            # ── Zweiter Klick: Richtung → Schnitt ──
            d = pos - self._start_pos
            if np.linalg.norm(d) < 1e-9:
                return

            self._clear_preview()
            result = self._evaluate_cut(self._start_pos, d, self._start_gpt)
            self._cuts.append(result)
            print(result.summary())

            self._start_pos    = None
            self._start_gpt    = None
            self._start_artist = None

            self._draw_main()
            self._draw_stats()

    def _on_motion(self, event) -> None:
        if event.inaxes is not self._ax_main:
            self._remove_hover()
            return

        cursor = np.array([event.xdata, event.ydata])

        if self._start_pos is not None:
            d = cursor - self._start_pos
            if np.linalg.norm(d) > 1e-9:
                self._update_preview(cursor)
        else:
            self._update_hover(cursor)

    def _on_key(self, event) -> None:
        if event.key == "r":
            self.grid.reset()
            self._cuts.clear()
            self._start_pos    = None
            self._start_gpt    = None
            self._start_artist = None
            self._preview_artists.clear()
            self._hover_artist = None
            self._draw_main()
            self._draw_stats()
        elif event.key == "escape":
            self._clear_preview()
            self._remove_hover()
            self._start_pos = None
            self._start_gpt = None
            if self._start_artist is not None:
                try:
                    self._start_artist.remove()
                except ValueError:
                    pass
                self._start_artist = None
            self._draw_stats()
            self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Snapping (nur zu Außenpunkten)
    # ------------------------------------------------------------------

    def _snap_to_outer(
        self, pos: np.ndarray
    ) -> tuple[np.ndarray, GridPoint | None]:
        """Snappt zur nächsten verbleibenden Außenpunkt (bevorzugt) oder
        Innenpunkt (wenn kein Außenpunkt in Reichweite)."""
        remaining = self.grid.remaining_points
        if not remaining:
            return pos, None

        # Erst Außenpunkte prüfen
        outer = self.grid.outer_points
        if outer:
            coords = np.array([[p.x, p.y] for p in outer])
            dists  = np.linalg.norm(coords - pos, axis=1)
            idx    = int(np.argmin(dists))
            if dists[idx] <= self.snap_tolerance:
                pt = outer[idx]
                return np.array([pt.x, pt.y]), pt

        # Fallback: nächster beliebiger Punkt
        coords = np.array([[p.x, p.y] for p in remaining])
        dists  = np.linalg.norm(coords - pos, axis=1)
        idx    = int(np.argmin(dists))
        if dists[idx] <= self.snap_tolerance * 2:
            pt = remaining[idx]
            return np.array([pt.x, pt.y]), pt

        return pos, None

    # ------------------------------------------------------------------
    # Kandidaten-Suche (vektorisiert, duale Toleranz)
    # ------------------------------------------------------------------

    def _find_candidates(
        self,
        start:       np.ndarray,
        d_unit:      np.ndarray,
        apply_depth: bool = True,
    ) -> list[GridPoint]:
        """Findet verbleibende Punkte auf dem Schnittstrahl ab *start*.

        Außenpunkte werden mit der großzügigeren ``outer_tolerance`` geprüft,
        Innenpunkte mit ``cut_tolerance``.
        Mit ``apply_depth=True`` wird der Strahl auf ``max_depth`` begrenzt.
        """
        remaining = self.grid.remaining_points
        if not remaining:
            return []

        coords  = np.array([[p.x, p.y] for p in remaining])
        is_outer = np.array([p.is_outer for p in remaining])

        v    = coords - start
        proj = v @ d_unit                                            # (n,)
        perp = np.linalg.norm(v - np.outer(proj, d_unit), axis=1)   # (n,)

        # Nur vorwärts (proj ≥ -kleines Epsilon)
        fwd = proj >= -min(self.cut_tolerance, self.outer_tolerance) * 0.15

        # Tiefenbegrenzung auf max_depth
        depth_ok = (proj <= self.max_depth) if apply_depth else np.ones(len(remaining), bool)

        # Duale Toleranz: Außenpunkte großzügiger
        tol_ok = np.where(is_outer, perp <= self.outer_tolerance, perp <= self.cut_tolerance)

        mask    = fwd & depth_ok & tol_ok
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []

        order = indices[np.argsort(proj[indices])]
        return [remaining[i] for i in order]

    # ------------------------------------------------------------------
    # Winkelberechnung
    # ------------------------------------------------------------------

    def _angle_to_normal(
        self, d_unit: np.ndarray, start_gpt: GridPoint | None
    ) -> float:
        """Winkel [rad] zwischen Schnittrichtung und Kontur-Normalen am Startpunkt.

        Gibt 0.0 zurück wenn kein Außenpunkt als Start gesetzt ist.
        """
        if start_gpt is None or not start_gpt.is_outer:
            return 0.0
        normal = self.grid.contour_normal_at(start_gpt)
        # Winkel zur Normalen, unabhängig von der Richtung (abs)
        cos_a = float(np.clip(abs(np.dot(d_unit, normal)), 0.0, 1.0))
        return float(np.arccos(cos_a))

    def _inward_normal(self, start_gpt: GridPoint) -> np.ndarray:
        """Normalen-Einheitsvektor, der ins Innere der Geometrie zeigt."""
        normal = self.grid.contour_normal_at(start_gpt)
        # Schwerpunkt aller Punkte als Innen-Referenz
        all_coords = np.array([[p.x, p.y] for p in self.grid.points])
        centroid   = all_coords.mean(axis=0)
        toward     = centroid - start_gpt.coords
        if np.dot(normal, toward) < 0:
            normal = -normal
        return normal

    # ------------------------------------------------------------------
    # Schnitt auswerten
    # ------------------------------------------------------------------

    def _evaluate_cut(
        self,
        start:     np.ndarray,
        direction: np.ndarray,
        start_gpt: GridPoint | None,
    ) -> CutResult:
        """Wertet den Schnitt aus und wendet ihn aufs Grid an."""
        cut_idx = len(self._cuts)
        d_unit  = direction / np.linalg.norm(direction)

        # Winkel zur Kontur-Normalen
        angle_rad = self._angle_to_normal(d_unit, start_gpt)
        angle_deg = float(np.degrees(angle_rad))

        # Kandidaten (komplett, um Truncation zu erkennen)
        all_cands = self._find_candidates(start, d_unit, apply_depth=False)
        # Mit Tiefenbegrenzung (max_depth = physikalische Länge)
        cands     = self._find_candidates(start, d_unit, apply_depth=True)
        truncated = len(all_cands) > len(cands)

        # ── Erfolgskriterien ──
        # Einziges Kriterium: Schnitt darf nicht an einem Innenpunkt enden.
        # Zusätzlich: Winkel darf max_tilt nicht überschreiten.
        failure_parts: list[str] = []

        if len(cands) == 0:
            failure_parts.append("keine Punkte auf der Linie")
        else:
            last = cands[-1]
            if last.is_inner:
                if truncated:
                    failure_parts.append(
                        f"Tiefenlimit ({self.max_depth:.0f} mm) — endet an Innenpunkt"
                    )
                else:
                    failure_parts.append("endet an Innenpunkt")

        if angle_rad > self.max_tilt and start_gpt is not None and start_gpt.is_outer:
            failure_parts.append(
                f"Winkel zur Normalen {angle_deg:.1f}° > "
                f"{np.degrees(self.max_tilt):.1f}° (max_tilt)"
            )

        is_successful  = len(failure_parts) == 0
        failure_reason = "; ".join(failure_parts)

        # Schnitt auf Grid anwenden
        new_cut_pts = self.grid.apply_cut(cands, cut_idx)

        cut_length = (
            float(np.linalg.norm(new_cut_pts[-1].coords - new_cut_pts[0].coords))
            if len(new_cut_pts) >= 2 else 0.0
        )

        return CutResult(
            cut_index     = cut_idx,
            points_cut    = new_cut_pts,
            is_successful = is_successful,
            failure_reason= failure_reason,
            start_point   = new_cut_pts[0]  if new_cut_pts else None,
            end_point     = new_cut_pts[-1] if new_cut_pts else None,
            start_pos     = start.copy(),
            direction     = d_unit.copy(),
            cut_length    = cut_length,
            angle_deg     = angle_deg,
            truncated     = truncated,
        )

    # ------------------------------------------------------------------
    # Vorschau
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

    def _update_preview(self, cursor: np.ndarray) -> None:
        """Zeichnet Schnittstrahl, Reichweiten-Kreis, Winkelkegel, Kandidaten."""
        self._clear_preview()

        start  = self._start_pos
        d      = cursor - start
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-9:
            return
        d_unit = d / d_norm

        ax = self._ax_main

        # ── 1. Erlaubter Winkelbereich (Kegel) ──
        if self._start_gpt is not None and self._start_gpt.is_outer:
            inward = self._inward_normal(self._start_gpt)
            angle_rad = self._angle_to_normal(d_unit, self._start_gpt)
            in_cone   = angle_rad <= self.max_tilt

            # Zwei Kegellinien bei ±max_tilt um die Inward-Normale
            for sign in (-1, +1):
                a = sign * self.max_tilt
                c, s = np.cos(a), np.sin(a)
                rotated = np.array([c * inward[0] - s * inward[1],
                                    s * inward[0] + c * inward[1]])
                end_cone = start + self.max_depth * rotated
                ln, = ax.plot(
                    [start[0], end_cone[0]], [start[1], end_cone[1]],
                    color="#33AA55", linewidth=1.2, linestyle=":",
                    alpha=0.55, zorder=6,
                )
                self._preview_artists.append(ln)

            # Mittelachse (Inward-Normale)
            end_norm = start + self.max_depth * inward
            ln, = ax.plot(
                [start[0], end_norm[0]], [start[1], end_norm[1]],
                color=_C["normal_line"], linewidth=1.0, linestyle="-.",
                alpha=0.45, zorder=6,
            )
            self._preview_artists.append(ln)

            # Winkelanzeige-Text
            cone_color = _C["cut_ok"] if in_cone else _C["cut_fail"]
            txt = ax.text(
                start[0], start[1] - self.grid.contour_spacing * 1.8,
                f"Winkel: {np.degrees(angle_rad):.1f}°  "
                f"(max {np.degrees(self.max_tilt):.0f}°)",
                fontsize=7.5, color=cone_color, ha="center", va="top",
                zorder=11,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                         ec=cone_color, alpha=0.88, linewidth=0.8),
            )
            self._preview_artists.append(txt)

        # ── 2. Reichweiten-Kreis ──
        circle = mpatches.Circle(
            start, self.max_depth,
            color=_C["preview"], fill=True,
            facecolor=_C["reach_fill"], linewidth=1.3,
            linestyle=":", alpha=0.30, zorder=5,
        )
        ax.add_patch(circle)
        self._preview_artists.append(circle)

        # ── 3. Schnittstrahl-Linie ──
        end_ray = start + self.max_depth * d_unit
        line, = ax.plot(
            [start[0], end_ray[0]], [start[1], end_ray[1]],
            color=_C["preview"], linewidth=2.2, linestyle="--",
            alpha=0.75, zorder=7,
        )
        self._preview_artists.append(line)

        # Pfeilspitze
        tip_base = start + self.max_depth * 0.70 * d_unit
        arr = ax.annotate(
            "", xy=tuple(end_ray), xytext=tuple(tip_base),
            arrowprops=dict(arrowstyle="->", color=_C["preview"],
                           lw=2.0, mutation_scale=14),
            zorder=8,
        )
        self._preview_artists.append(arr)

        # ── 4. Kandidaten hervorheben ──
        cands = self._find_candidates(start, d_unit, apply_depth=True)

        if cands:
            cx = [p.x for p in cands]
            cy = [p.y for p in cands]
            sc = ax.scatter(
                cx, cy, s=95,
                facecolors="none", edgecolors=_C["preview"],
                linewidths=2.2, zorder=9, alpha=0.90,
            )
            self._preview_artists.append(sc)

            # Länge des Schnitts (Abstand erster→letzter Kandidat)
            if len(cands) >= 2:
                preview_len = float(np.linalg.norm(cands[-1].coords - cands[0].coords))
            else:
                preview_len = 0.0
            n_outer_c = sum(1 for p in cands if p.is_outer)
            lbl = (f"{len(cands)} Pkt  ({n_outer_c} außen)  "
                   f"{preview_len:.0f} mm")
        else:
            lbl = "0 Punkte"

        # Info-Label neben Startpunkt
        offset = np.array([self.grid.contour_spacing * 1.0,
                           self.grid.contour_spacing * 1.0])
        txt = ax.text(
            *(start + offset), lbl,
            fontsize=8, color=_C["preview"], zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                     ec=_C["preview"], alpha=0.87, linewidth=0.8),
        )
        self._preview_artists.append(txt)

        self._fig.canvas.draw_idle()

    def _update_hover(self, cursor: np.ndarray) -> None:
        """Hebt den nächsten Snap-Kandidaten (Außenpunkt) hervor."""
        self._remove_hover()
        outer = self.grid.outer_points
        if not outer:
            return

        coords = np.array([[p.x, p.y] for p in outer])
        dists  = np.linalg.norm(coords - cursor, axis=1)
        idx    = int(np.argmin(dists))

        if dists[idx] <= self.snap_tolerance:
            pt = outer[idx]
            self._hover_artist = self._ax_main.plot(
                pt.x, pt.y, "o",
                color=_C["outer"], markersize=19,
                alpha=0.25, markeredgecolor=_C["outer"],
                markeredgewidth=2.0, zorder=8,
            )[0]
            self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Hauptplot zeichnen
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

        # ── Innenpunkte ──
        inner = self.grid.inner_points
        if inner:
            ax.scatter([p.x for p in inner], [p.y for p in inner],
                      s=12, color=_C["inner"], edgecolors="none",
                      alpha=0.50, zorder=2)

        # ── Außenpunkte ──
        outer = self.grid.outer_points
        if outer:
            ax.scatter([p.x for p in outer], [p.y for p in outer],
                      s=48, color=_C["outer"], edgecolors="#0D3A73",
                      linewidths=0.7, zorder=3, marker="s")

        # ── Geschnittene Punkte ──
        cut_map: dict[int, list[CutPoint]] = {}
        for cp in self.grid.cut_points:
            cut_map.setdefault(cp.cut_index, []).append(cp)

        for cidx, pts in cut_map.items():
            result = self._cuts[cidx] if cidx < len(self._cuts) else None
            color  = _C["cut_ok"] if (result and result.is_successful) else _C["cut_fail"]
            ax.scatter([p.x for p in pts], [p.y for p in pts],
                      s=55, color=color, edgecolors="#111",
                      linewidths=0.5, zorder=4, marker="P")

        # ── Schnittlinien ──
        for result in self._cuts:
            self._draw_cut_on(ax, result)

        # ── Startpunkt-Marker wiederherstellen ──
        if self._start_pos is not None:
            self._start_artist = ax.plot(
                self._start_pos[0], self._start_pos[1], "o",
                color=_C["start"], markersize=13,
                markeredgecolor="#7A4A00", markeredgewidth=1.5,
                zorder=11, alpha=0.93,
            )[0]

        # ── Legende ──
        ax.legend(
            handles=[
                Line2D([0],[0], marker="s", color="w",
                       markerfacecolor=_C["outer"], markeredgecolor="#0D3A73",
                       markersize=9, label="Außenpunkt"),
                Line2D([0],[0], marker="o", color="w",
                       markerfacecolor=_C["inner"], markersize=7,
                       label="Innenpunkt"),
                Line2D([0],[0], marker="P", color="w",
                       markerfacecolor=_C["cut_ok"], markeredgecolor="#111",
                       markersize=9, label="Geschnitten ✓"),
                Line2D([0],[0], marker="P", color="w",
                       markerfacecolor=_C["cut_fail"], markeredgecolor="#111",
                       markersize=9, label="Geschnitten ✗"),
            ],
            fontsize=7.5, loc="upper right",
            framealpha=0.92, edgecolor="#CCCCCC", labelspacing=0.45,
        )

        # ── Achsen ──
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]", fontsize=9)
        ax.set_ylabel("y [mm]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.22, color="#888")

        xs = [p.x for p in all_pts]; ys = [p.y for p in all_pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        m    = span * 0.09 + 5.0
        ax.set_xlim(min(xs) - m, max(xs) + m)
        ax.set_ylim(min(ys) - m, max(ys) + m)

        ax.text(0.5, -0.048,
               "1. Klick: Startpunkt  │  2. Klick: Richtung  │  R: Reset  │  Esc: Abbrechen",
               transform=ax.transAxes, fontsize=7.5, color="#888", ha="center", va="top")

        self._fig.canvas.draw_idle()

    def _draw_cut_on(self, ax: plt.Axes, result: CutResult) -> None:
        if not result.points_cut:
            return

        color   = _C["cut_ok"]   if result.is_successful else _C["cut_fail"]
        style   = "-"            if result.is_successful else "--"
        bg_face = "#E8FFF0"      if result.is_successful else "#FFE8E8"

        s = result.points_cut[0].coords
        e = result.points_cut[-1].coords

        ax.plot([s[0], e[0]], [s[1], e[1]],
               color=color, linewidth=2.0, linestyle=style,
               alpha=0.62, zorder=3, solid_capstyle="round")

        if np.linalg.norm(e - s) > 1e-3:
            ax.annotate("", xy=tuple(e), xytext=tuple(s + (e-s)*0.55),
                       arrowprops=dict(arrowstyle="->", color=color,
                                      lw=1.8, mutation_scale=13),
                       zorder=5)

        mid = (s + e) / 2
        ax.text(mid[0], mid[1], f"#{result.cut_index + 1}",
               fontsize=7.5, color=color, ha="center", va="center",
               fontweight="bold", zorder=6,
               bbox=dict(boxstyle="round,pad=0.25", fc=bg_face,
                        ec=color, alpha=0.88, linewidth=0.9))

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
               fontsize=9.5, fontweight="bold", ha="center", va="top", color="#1A2840")
        ax.plot([0.08, 0.92], [0.940, 0.940], color="#B0C4DE",
               linewidth=0.8, transform=ax.transAxes)

        total     = self.grid.total_points
        n_cut     = self.grid.n_cut
        pct       = self.grid.cut_percentage
        n_outer_t = sum(1 for p in self.grid.points if p.is_outer)
        n_inner_t = sum(1 for p in self.grid.points if p.is_inner)
        total_len = sum(c.cut_length for c in self._cuts)

        y, dy = 0.915, 0.060

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
        kv("  Außenpunkte",  str(n_outer_t))
        kv("  Innenpunkte",  str(n_inner_t))
        sep()
        kv("Geschnitten",
           f"{n_cut} / {total}",
           vc="#16A34A" if n_cut > 0 else "#555555", bold=True)
        kv("Anteil",
           f"{pct:.1f} %",
           vc="#16A34A" if pct > 0 else "#555555", bold=pct > 0)
        sep()
        kv("Schnittweg ges.", f"{total_len:.1f} mm")
        sep()
        kv("Schnitte ges.", str(len(self._cuts)))
        kv("  ✓ Erfolg",   str(self.n_successful),
           vc="#22A84E" if self.n_successful > 0 else "#555555")
        kv("  ✗ Fehler",   str(self.n_unsuccessful),
           vc="#D93535" if self.n_unsuccessful > 0 else "#555555")
        sep()

        # ── Cutter-Parameter ──
        ax.text(0.5, y, "CUTTER", transform=ax.transAxes,
               fontsize=8, fontweight="bold", ha="center", va="top", color="#1A2840")
        y -= dy * 0.85
        kv("Schnitttiefe", f"{self.max_depth:.0f} mm", bold=True)
        kv("max. Winkel",  f"{np.degrees(self.max_tilt):.0f}°")
        kv("Tol. Innen",   f"{self.cut_tolerance:.2f} mm")
        kv("Tol. Außen",   f"{self.outer_tolerance:.2f} mm")
        sep()

        # ── Schnittliste ──
        ax.text(0.5, y, "SCHNITTE", transform=ax.transAxes,
               fontsize=8, fontweight="bold", ha="center", va="top", color="#1A2840")
        y -= dy * 0.85

        for r in reversed(self._cuts[-9:]):
            if y < 0.07:
                break
            color = "#22A84E" if r.is_successful else "#D93535"
            sym   = "✓" if r.is_successful else "✗"
            ax.text(0.07, y, f"{sym} #{r.cut_index+1}",
                   transform=ax.transAxes, fontsize=7.5,
                   color=color, va="top", fontweight="bold")
            ax.text(0.93, y,
                   f"{r.n_points}p  {r.cut_length:.0f}mm  {r.angle_deg:.0f}°",
                   transform=ax.transAxes, fontsize=6.8,
                   color="#444444", ha="right", va="top")
            y -= dy * 0.80
            if not r.is_successful and r.failure_reason:
                short = r.failure_reason[:34] + ("…" if len(r.failure_reason) > 34 else "")
                ax.text(0.10, y, f"↳ {short}",
                       transform=ax.transAxes, fontsize=5.8,
                       color="#888888", va="top", style="italic")
                y -= dy * 0.72

        # ── Statuszeile ──
        if self._start_pos is not None:
            status, s_col = "→ Richtung klicken …", _C["start"]
        else:
            status, s_col = "→ Startpunkt klicken", "#4B5563"

        ax.text(0.5, 0.025, status, transform=ax.transAxes,
               fontsize=7.5, color=s_col, ha="center", va="bottom",
               style="italic",
               bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        ec=s_col, alpha=0.88, linewidth=0.8))

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _figure_title(self) -> str:
        name = (f"  –  {self.grid._source_path.stem}"
                if hasattr(self.grid, "_source_path") else "")
        return (
            f"Schnitt-Simulation{name}  │  "
            f"Tiefe: {self.max_depth:.0f} mm  │  "
            f"max. Winkel: {np.degrees(self.max_tilt):.0f}°"
        )

    def print_summary(self) -> None:
        print("=" * 72)
        print("SCHNITT-SIMULATION – ZUSAMMENFASSUNG")
        print("=" * 72)
        print(f"  Punkte gesamt  : {self.grid.total_points}")
        print(f"  Geschnitten    : {self.grid.n_cut} ({self.grid.cut_percentage:.1f}%)")
        print(f"  Schnittweg ges.: {sum(c.cut_length for c in self._cuts):.1f} mm")
        print(f"  Schnitte       : {len(self._cuts)}")
        print(f"    erfolgreich  : {self.n_successful}")
        print(f"    fehlgesch.   : {self.n_unsuccessful}")
        print("-" * 72)
        for r in self._cuts:
            print(f"  {r.summary()}")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Direktaufruf
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interaktive Schnitt-Simulation (Cutter-Parameter aus Cutter-Klasse)."
    )
    parser.add_argument("--max-depth",  type=float, default=None,
                        help="Maximale Schnitttiefe [mm]  (Standard: Cutter-Default 40 mm)")
    parser.add_argument("--max-tilt",   type=float, default=None,
                        help="Maximaler Kippwinkel [°]     (Standard: Cutter-Default 20°)")
    parser.add_argument("--v-cut",      type=float, default=None,
                        help="Schneidgeschwindigkeit [mm/s]")
    parser.add_argument("--tolerance",  type=float, default=None,
                        help="Treff-Toleranz Innen [mm]")
    parser.add_argument("--outer-tol",  type=float, default=None,
                        help="Treff-Toleranz Außen [mm]")
    args = parser.parse_args()

    cutter_kwargs: dict = {}
    if args.max_depth is not None:
        cutter_kwargs["max_depth"] = args.max_depth
    if args.max_tilt is not None:
        cutter_kwargs["max_tilt"] = np.radians(args.max_tilt)
    if args.v_cut is not None:
        cutter_kwargs["cutting_speed"] = args.v_cut

    cutter = Cutter(**cutter_kwargs) if cutter_kwargs else Cutter()

    PointCutSimulation.run_with_dialog(
        cutter=cutter,
        cut_tolerance=args.tolerance,
        outer_tolerance=args.outer_tol,
    )

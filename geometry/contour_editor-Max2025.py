"""Interaktiver 2D-Kontur-Editor für Querschnittsgeometrien. 

Dieses Skript öffnet ein Matplotlib-Fenster, in dem Konturen per Maus
gezeichnet und bearbeitet werden können. Das Ergebnis wird als JSON-Datei
gespeichert und kann anschließend vom geometry_processor.py weiterverarbeitet
werden.

Bedienung
---------
* Linksklick auf einen Punkt      → Punkt auswählen & verschieben
* Linksklick auf eine Kante       → Neuen Punkt an dieser Stelle einfügen
* Rechtsklick auf einen Punkt     → Punkt löschen (mind. 3 Punkte bleiben)
* Taste  'e'                       → Punkte als JSON in die Konsole ausgeben
* Taste  'r'                       → Ansicht zurücksetzen

Vorlagen
--------
IBeam, T-Träger, L-Profil, Flachstahl, Mauseloch-Aussparung, Platte+Ausschnitt
"""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib
# Interaktives Backend muss VOR dem pyplot-Import gesetzt werden.
# TkAgg nutzt Tkinter als Fenster-Backend — am stabilsten unter Windows.
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")   # Fallback auf Qt falls Tkinter fehlt
    except Exception:
        pass  # Letzter Fallback: Standard-Backend von Matplotlib

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Vordefinierte Vorlagen  (x, y) in [mm]
# ---------------------------------------------------------------------------
# Jede Vorlage gibt ein Tupel (outer, inner) zurück:
#   outer = Liste von (x,y)-Punkten der Außenkontur
#   inner = Liste von (x,y)-Punkten der Innenkontur (Loch) oder None

def _t_beam(h=200, b=100, tf=10, tw=12):
    """T-Träger: Flansch oben, Steg nach unten."""
    xf, xt = b / 2, tw / 2
    yt, yb = h / 2, -h / 2
    yfi = yt - tf   # Unterkante des Flansches (= Steg-Oberkante)
    return [
        (-xt, yb), (xt, yb),
        (xt, yfi), (xf, yfi),
        (xf, yt), (-xf, yt),
        (-xf, yfi), (-xt, yfi),
    ]


def _l_profile(h=150, b=100, t1=10, t2=10):
    """L-Profil: Ecke im Ursprung, Schenkel in +x und +y."""
    return [(0, 0), (b, 0), (b, t2), (t1, t2), (t1, h), (0, h)]


def _hp_profile(h=180, b=90, tf=14, tw=9):
    """Holland-Profil (HP / Bulb-Flat): Steg mit verdicktem Fuß (Bulb).

    Typisch im Schiffbau — der untere Stegabschluss ist verbreitert
    um die Knicklast zu erhöhen.
    """
    xf, xt = b / 2, tw / 2
    yt, yb = h / 2, -h / 2
    yfi = yt - tf
    bw = tw * 2.2   # Bulb-Breite (Verbreiterung am Fuß)
    bh = tw * 1.8   # Bulb-Höhe
    return [
        (-bw/2, yb), (bw/2, yb),
        (bw/2, yb + bh), (xt, yb + bh),
        (xt, yfi), (xf, yfi),
        (xf, yt), (-xf, yt),
        (-xf, yfi), (-xt, yfi),
        (-xt, yb + bh), (-bw/2, yb + bh),
    ]


def _u_profile(h=200, b=80, tf=10, tw=12):
    """U-Profil: nach rechts offen."""
    xl, xr = -b/2, b/2
    yt, yb = h/2, -h/2
    return [
        (xl, yb), (xr, yb), (xr, yb + tf), (xl + tw, yb + tf),
        (xl + tw, yt - tf), (xr, yt - tf), (xr, yt), (xl, yt),
    ]


def _plate_cutout(W=300, H=200, cw=80, ch=60):
    """Rechteckige Platte mit rechteckigem Ausschnitt in der Mitte.

    Gibt ein Tupel (outer, inner) zurück — inner ist das Loch.
    """
    outer = [(-W/2, -H/2), (W/2, -H/2), (W/2, H/2), (-W/2, H/2)]
    inner = [(-cw/2, -ch/2), (cw/2, -ch/2), (cw/2, ch/2), (-cw/2, ch/2)]
    return outer, inner


# Alle Vorlagen in einem Dict: Name → (outer, inner)
# Vorlagen ohne Loch haben inner=None
TEMPLATES: dict[str, tuple] = {
    "T-Träger":           (_t_beam(),    None),
    "L-Profil":           (_l_profile(), None),
    "Holland-Profil":     (_hp_profile(), None),
    "U-Profil":           (_u_profile(), None),
    "Platte+Ausparung":   _plate_cutout(),
}


# ---------------------------------------------------------------------------
# Hilfsfunktionen für die Mausinteraktion
# ---------------------------------------------------------------------------

def _snap(pts: np.ndarray, x: float, y: float, tol: float) -> int | None:
    """Findet den nächsten Punkt innerhalb der Toleranz tol.

    Berechnet euklidische Abstände von (x,y) zu allen Punkten in pts
    und gibt den Index des nächsten zurück — oder None wenn keiner
    nah genug ist.

    Parameters
    ----------
    pts : Array der Form (N, 2) mit allen Konturpunkten
    x, y: Mausposition in Daten-Koordinaten (mm)
    tol : maximaler Abstand für einen Treffer (mm)
    """
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    idx = int(np.argmin(d))    # Index des nächsten Punktes
    return idx if d[idx] < tol else None


def _edge_snap(pts: np.ndarray, x: float, y: float, tol: float
               ) -> tuple[int, float, float] | None:
    """Prüft ob (x,y) nahe einer Kante liegt und gibt den Einfügepunkt zurück.

    Projiziert den Mausklick auf jede Kante des Polygons. Wenn die
    Projektion näher als tol ist, wird der projizierte Punkt als
    neuer Eckpunkt vorgeschlagen.

    Returns
    -------
    (einfüge_index, px, py) wenn Treffer, sonst None.
    einfüge_index: Position im Array wo der neue Punkt eingefügt wird.
    px, py: Koordinaten des projizierten Punkts auf der Kante.
    """
    n = len(pts)
    best = (None, np.inf, 0.0, 0.0)
    for i in range(n):
        a, b_ = pts[i], pts[(i + 1) % n]   # Kante von Punkt i zu Punkt i+1
        ab = b_ - a
        # Parameter t der Projektion (0=Punkt a, 1=Punkt b)
        t = np.dot(np.array([x, y]) - a, ab) / (np.dot(ab, ab) + 1e-12)
        t = float(np.clip(t, 0, 1))         # auf Segment begrenzen
        proj = a + t * ab                    # projizierter Punkt auf der Kante
        dist = np.hypot(proj[0] - x, proj[1] - y)
        if dist < best[1]:
            best = (i + 1, dist, proj[0], proj[1])  # i+1: nach Punkt i einfügen
    idx, dist, px, py = best
    return (idx, px, py) if dist < tol else None


def _poly_stats(pts: np.ndarray, hole: np.ndarray | None) -> dict:
    """Berechnet geometrische Kenngrößen für die Info-Anzeige.

    Nutzt Shapely um Fläche, Umfang, Schwerpunkt und Bounding-Box
    zu berechnen. Bei ungültigen Polygonen (selbstüberschneidend)
    wird buffer(0) als Reparatur versucht.

    Returns
    -------
    Dict mit: area, perimeter, cx, cy, width, height — oder leeres Dict
    bei Fehler.
    """
    try:
        if hole is not None:
            poly = Polygon(pts, [hole])   # Polygon mit Loch
        else:
            poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)         # Shapely-Trick: repariert viele Fehler
        c = poly.centroid
        b = poly.bounds
        return {
            "area":      poly.area,
            "perimeter": poly.length,
            "cx":        c.x,
            "cy":        c.y,
            "width":     b[2] - b[0],    # x_max - x_min
            "height":    b[3] - b[1],    # y_max - y_min
        }
    except Exception:
        return {}   # Bei jedem Fehler leeres Dict zurückgeben


# ---------------------------------------------------------------------------
# Haupt-Editor-Klasse
# ---------------------------------------------------------------------------

class ContourEditor:
    """Interaktiver Matplotlib-Editor für 2D-Querschnittskonturen.

    Aufbau des Fensters:
    - Links (60 %): Hauptzeichenfläche mit der Kontur
    - Rechts oben: Radio-Buttons zur Vorlagenauswahl
    - Rechts mitte: Speichern / Laden Buttons
    - Rechts unten: Info-Box mit geometrischen Kenngrößen

    Der interne Zustand wird in zwei NumPy-Arrays gehalten:
    - self._pts  : Außenkontur (N×2)
    - self._hole : Innenkontur (M×2) oder None

    Parameters
    ----------
    template:
        Name einer vordefinierten Vorlage. Muss ein Schlüssel aus
        TEMPLATES sein (Standard: "T-Träger").
    figsize:
        Fenstergröße in Zoll (Breite, Höhe).
    """

    # Klassen-Konstanten für die Interaktionstoleranzen
    _PICK_TOL   = 12   # Pixel: wie nah muss man an einen Punkt klicken
    _EDGE_TOL   = 10   # Pixel: wie nah muss man an eine Kante klicken
    _POINT_SIZE = 8    # Matplotlib-Markergröße der Konturpunkte

    def __init__(self,
                 template: str = "T-Träger",
                 figsize: tuple[float, float] = (13, 8)) -> None:

        # Matplotlib-Figur mit dunklem Hintergrund anlegen
        self._fig = plt.figure(figsize=figsize)
        self._fig.patch.set_facecolor("#1e1e2e")

        # --- Layout: Axes-Positionen manuell definieren [left, bottom, width, height] ---
        # Hauptzeichenfläche (links)
        self._ax = self._fig.add_axes([0.03, 0.12, 0.60, 0.85])
        self._ax.set_facecolor("#2b2b3b")

        # Vorlagen-Auswahl (Radio-Buttons, rechts oben)
        ax_radio  = self._fig.add_axes([0.66, 0.40, 0.30, 0.48])
        # Speichern-Button
        ax_export = self._fig.add_axes([0.66, 0.28, 0.14, 0.07])
        # Laden-Button
        ax_import = self._fig.add_axes([0.82, 0.28, 0.14, 0.07])
        # Reset-Button
        ax_reset  = self._fig.add_axes([0.66, 0.20, 0.30, 0.07])

        # Alle Hilfs-Axes dunkel einfärben
        for a in (ax_radio, ax_export, ax_import, ax_reset):
            a.set_facecolor("#2b2b3b")

        # --- Radio-Buttons für Vorlagenauswahl ---
        self._radio = RadioButtons(
            ax_radio,
            list(TEMPLATES.keys()),   # Vorlagennamen als Labels
            activecolor="#89b4fa",    # Farbe des aktiven Buttons
        )
        ax_radio.set_facecolor("#2b2b3b")
        for lbl in self._radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(9)
        self._radio.on_clicked(self._on_template)   # Callback bei Auswahl

        # --- Aktions-Buttons ---
        self._btn_export = Button(ax_export, "Speichern",
                                  color="#1a6b3c", hovercolor="#28a860")
        self._btn_import = Button(ax_import, "Laden",
                                  color="#1a3c6b", hovercolor="#2860a8")
        self._btn_reset  = Button(ax_reset,  "↺ Reset",
                                  color="#313244", hovercolor="#45475a")
        for btn in (self._btn_export, self._btn_import, self._btn_reset):
            btn.label.set_color("white")
        # Callbacks registrieren
        self._btn_export.on_clicked(self._on_export)
        self._btn_import.on_clicked(self._on_import)
        self._btn_reset.on_clicked(self._on_reset)

        # --- Info-Box (rechts unten): zeigt Fläche, Umfang, etc. ---
        self._ax_info = self._fig.add_axes([0.66, 0.02, 0.30, 0.17])
        self._ax_info.set_facecolor("#313244")
        self._ax_info.set_xticks([])
        self._ax_info.set_yticks([])
        self._info_txt = self._ax_info.text(
            0.05, 0.95, "", transform=self._ax_info.transAxes,
            va="top", ha="left", fontsize=8.5, color="white",
            fontfamily="monospace",
        )

        # --- Interner Zustand ---
        self._pts: np.ndarray         = np.empty((0, 2))   # Außenkontur
        self._hole: np.ndarray | None = None               # Innenkontur (Loch)
        self._drag_idx: int | None    = None               # Index des gezogenen Punkts
        self._drag_hole: bool         = False              # True wenn Innenpunkt gezogen wird

        # --- Zeichenobjekte (werden beim Neuzeichnen ersetzt) ---
        self._patch:       mpatches.Polygon | None = None  # gefüllte Außenfläche
        self._hole_patch:  mpatches.Polygon | None = None  # gefülltes Loch
        self._pt_line:     plt.Line2D | None       = None  # Außenkontur-Linie+Punkte
        self._hole_line:   plt.Line2D | None       = None  # Innenkontur-Linie+Punkte
        self._centroid_pt: plt.Line2D | None       = None  # Schwerpunkt-Marker
        self._pt_labels:   list                    = []    # Punkt-Nummern
        self._sel_marker:  plt.Line2D | None       = None  # Markierung des ausgewählten Punkts

        # --- Maus- und Tastatur-Events verbinden ---
        self._fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self._fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self._fig.canvas.mpl_connect("key_press_event",      self._on_key)

        # Toolbar-Navigationsmodus deaktivieren: Zoom/Pan würde mit
        # unseren Klick-Events kollidieren
        try:
            toolbar = self._fig.canvas.toolbar
            if toolbar is not None:
                toolbar.mode = ""
        except Exception:
            pass

        # Startvorlage laden und zeichnen
        self._load_template(template)

    # ------------------------------------------------------------------
    # Vorlagen laden
    # ------------------------------------------------------------------

    def _load_template(self, name: str) -> None:
        """Lädt eine Vorlage aus TEMPLATES und zeichnet sie neu."""
        outer, inner = TEMPLATES[name]
        self._pts  = np.array(outer, dtype=float)
        self._hole = np.array(inner, dtype=float) if inner is not None else None
        self._auto_zoom()   # Achsen automatisch anpassen
        self._redraw()      # Kontur neu zeichnen

    def _auto_zoom(self) -> None:
        """Passt die Achsengrenzen automatisch an alle Konturpunkte an."""
        # Alle Punkte (Außen + Innen) zusammenfassen für Bounding Box
        all_pts = self._pts if self._hole is None else np.vstack([self._pts, self._hole])
        xmin, ymin = all_pts.min(axis=0)
        xmax, ymax = all_pts.max(axis=0)
        # 20 % Rand damit die Kontur nicht am Achsenrand klebt
        m = max(xmax - xmin, ymax - ymin) * 0.2
        self._ax.set_xlim(xmin - m, xmax + m)
        self._ax.set_ylim(ymin - m, ymax + m)

    # ------------------------------------------------------------------
    # Zeichnen
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        """Löscht alle alten Zeichenobjekte und zeichnet die Kontur neu.

        Wird nach jeder Änderung der Kontur aufgerufen (Punkt verschieben,
        einfügen, löschen, Vorlage wechseln).
        """
        ax = self._ax

        # Alle alten Grafik-Objekte aus der Axes entfernen
        for artist in (self._patch, self._hole_patch,
                       self._pt_line, self._hole_line,
                       self._centroid_pt, self._sel_marker):
            if artist is not None:
                artist.remove()
        for lbl in self._pt_labels:
            lbl.remove()
        self._pt_labels = []

        pts  = self._pts
        hole = self._hole
        # Kontur schließen: letzter Punkt = erster Punkt für durchgehende Linie
        closed_pts = np.vstack([pts, pts[0]])

        # --- Gefüllte Außenfläche ---
        self._patch = mpatches.Polygon(
            pts, closed=True,
            facecolor="#89b4fa", edgecolor="#cdd6f4",
            linewidth=1.5, alpha=0.35, zorder=1,
        )
        ax.add_patch(self._patch)

        # --- Gefülltes Loch (weiß übermalt die Fläche) ---
        if hole is not None:
            self._hole_patch = mpatches.Polygon(
                hole, closed=True,
                facecolor="#1e1e2e", edgecolor="#f38ba8",
                linewidth=1.5, alpha=1.0, zorder=2,
            )
            ax.add_patch(self._hole_patch)
        else:
            self._hole_patch = None

        # --- Außenkontur-Linie mit Punktmarkierungen ---
        self._pt_line, = ax.plot(
            closed_pts[:, 0], closed_pts[:, 1],
            "o-", color="#89b4fa", markersize=self._POINT_SIZE,
            markerfacecolor="#cdd6f4", markeredgecolor="#89b4fa",
            linewidth=1.5, zorder=3,
        )

        # --- Innenkontur-Linie (falls Loch vorhanden) ---
        if hole is not None:
            closed_hole = np.vstack([hole, hole[0]])
            self._hole_line, = ax.plot(
                closed_hole[:, 0], closed_hole[:, 1],
                "o-", color="#f38ba8", markersize=self._POINT_SIZE - 1,
                markerfacecolor="#fab387", markeredgecolor="#f38ba8",
                linewidth=1.2, zorder=4,
            )
        else:
            self._hole_line = None

        # --- Punkt-Nummern (1-basiert) neben jedem Außenpunkt ---
        for i, (x, y) in enumerate(pts):
            lbl = ax.annotate(
                str(i + 1), (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7, color="#a6e3a1", zorder=5,
            )
            self._pt_labels.append(lbl)

        # --- Schwerpunkt als Kreuz einzeichnen ---
        stats = _poly_stats(pts, hole)
        if stats:
            self._centroid_pt, = ax.plot(
                stats["cx"], stats["cy"], "+",
                color="#f9e2af", markersize=10, markeredgewidth=2, zorder=6,
            )
        else:
            self._centroid_pt = None

        # Auswahlmarker zurücksetzen (wird bei Klick gesetzt)
        self._sel_marker = None

        # --- Achsen-Styling ---
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]", color="#cdd6f4")
        ax.set_ylabel("y [mm]", color="#cdd6f4")
        ax.tick_params(colors="#cdd6f4")
        ax.spines[:].set_color("#45475a")
        ax.grid(True, linestyle="--", alpha=0.25, color="#cdd6f4")
        ax.set_title(
            "2D Kontur-Editor   |   LMB: Punkt ziehen / Kante einfügen   "
            "|   RMB: Punkt löschen",
            color="#cdd6f4", fontsize=9,
        )

        # Info-Box aktualisieren
        self._update_info(stats, pts, hole)
        self._fig.canvas.draw_idle()   # Fenster neu rendern (nicht-blockierend)

    def _update_info(self,
                     stats: dict,
                     pts:   np.ndarray,
                     hole:  np.ndarray | None) -> None:
        """Aktualisiert den Info-Text mit geometrischen Kenngrößen."""
        n_outer = len(pts)
        n_hole  = len(hole) if hole is not None else 0
        lines = [
            f"Punkte (außen): {n_outer}",
            f"Punkte (innen): {n_hole}" if n_hole else "",
            "",
        ]
        if stats:
            lines += [
                f"Fläche:      {stats['area']:.1f} mm²",
                f"Umfang:      {stats['perimeter']:.1f} mm",
                f"Breite:      {stats['width']:.1f} mm",
                f"Höhe:        {stats['height']:.1f} mm",
                f"Schwerpunkt: ({stats['cx']:.1f}, {stats['cy']:.1f})",
            ]
        self._info_txt.set_text("\n".join(l for l in lines))
        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Maus-Events
    # ------------------------------------------------------------------

    def _display_to_data(self, event) -> tuple[float, float, float]:
        """Wandelt Pixel-Koordinaten in Daten-Koordinaten (mm) um.

        Berechnet auch die Toleranz in mm, die den Pixel-Toleranzen
        (_PICK_TOL, _EDGE_TOL) entspricht — da die Achsenskalierung
        vom Zoom-Level abhängt.

        Returns
        -------
        (x_data, y_data, tol_data) oder (None, None, None) wenn der
        Mauszeiger außerhalb der Axes ist.
        """
        ax = self._ax
        if event.inaxes is not ax:
            return None, None, None
        try:
            # Fenster-Pixel-Größe der Axes bestimmen
            bbox = ax.get_window_extent()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            # Pixel pro mm in x- und y-Richtung
            px_per_unit_x = bbox.width  / (xlim[1] - xlim[0])
            px_per_unit_y = bbox.height / (ylim[1] - ylim[0])
            # Toleranz in mm = Pixel-Toleranz / (Pixel pro mm)
            tol = self._PICK_TOL / min(px_per_unit_x, px_per_unit_y)
        except Exception:
            tol = 10.0  # Fallback: 10 mm Toleranz
        return event.xdata, event.ydata, tol

    def _on_press(self, event) -> None:
        """Verarbeitet Mausklicks: Punkt auswählen, Kante aufteilen, Punkt löschen."""
        x, y, tol = self._display_to_data(event)
        if x is None:
            return   # Klick außerhalb der Axes ignorieren

        # --- Rechtsklick: Punkt löschen ---
        if event.button == 3:
            # Außenkontur: Mindestens 3 Punkte müssen bleiben
            idx = _snap(self._pts, x, y, tol * 2)
            if idx is not None and len(self._pts) > 3:
                self._pts = np.delete(self._pts, idx, axis=0)
                self._redraw()
            elif self._hole is not None:
                # Innenkontur: ebenfalls mind. 3 Punkte
                idx = _snap(self._hole, x, y, tol * 2)
                if idx is not None and len(self._hole) > 3:
                    self._hole = np.delete(self._hole, idx, axis=0)
                    self._redraw()
            return

        # --- Linksklick: Punkt ziehen oder Kante aufteilen ---
        if event.button == 1:
            # 1. Versuch: Außenpunkt auswählen
            idx = _snap(self._pts, x, y, tol * 1.5)
            if idx is not None:
                self._drag_idx  = idx
                self._drag_hole = False
                self._highlight(idx, hole=False)
                return

            # 2. Versuch: Innenpunkt auswählen
            if self._hole is not None:
                idx = _snap(self._hole, x, y, tol * 1.5)
                if idx is not None:
                    self._drag_idx  = idx
                    self._drag_hole = True
                    self._highlight(idx, hole=True)
                    return

            # 3. Versuch: Außenkante aufteilen (neuer Punkt)
            res = _edge_snap(self._pts, x, y, tol * 1.5)
            if res is not None:
                i, px, py = res
                # Neuen Punkt an Position i einfügen
                self._pts = np.insert(self._pts, i, [px, py], axis=0)
                self._drag_idx  = i
                self._drag_hole = False
                self._redraw()
                self._highlight(i, hole=False)
                return

            # 4. Versuch: Innenkante aufteilen
            if self._hole is not None:
                res = _edge_snap(self._hole, x, y, tol * 1.5)
                if res is not None:
                    i, px, py = res
                    self._hole = np.insert(self._hole, i, [px, py], axis=0)
                    self._drag_idx  = i
                    self._drag_hole = True
                    self._redraw()
                    self._highlight(i, hole=True)

    def _highlight(self, idx: int, hole: bool) -> None:
        """Zeichnet einen gelben Kreis um den ausgewählten Punkt."""
        pts = self._hole if hole else self._pts
        if self._sel_marker is not None:
            self._sel_marker.remove()   # alten Marker entfernen
        self._sel_marker, = self._ax.plot(
            pts[idx, 0], pts[idx, 1], "o",
            color="#f9e2af", markersize=self._POINT_SIZE + 4,
            markerfacecolor="none", markeredgewidth=2, zorder=10,
        )
        self._fig.canvas.draw_idle()

    def _on_motion(self, event) -> None:
        """Verschiebt den ausgewählten Punkt mit der Maus (Drag)."""
        if self._drag_idx is None:
            return   # Kein Punkt ausgewählt → nichts tun
        x, y, _ = self._display_to_data(event)
        if x is None:
            return
        # Punkt-Koordinaten aktualisieren
        if self._drag_hole:
            self._hole[self._drag_idx] = [x, y]
        else:
            self._pts[self._drag_idx] = [x, y]
        # Sofort neu zeichnen für flüssiges Verschieben
        self._redraw()
        self._highlight(self._drag_idx, self._drag_hole)

    def _on_release(self, event) -> None:
        """Beendet das Ziehen eines Punkts bei Maustaste loslassen."""
        self._drag_idx = None

    # ------------------------------------------------------------------
    # Tastatur-Events & Button-Callbacks
    # ------------------------------------------------------------------

    def _on_key(self, event) -> None:
        """Tastaturkürzel: 'e' = exportieren, 'r' = Ansicht zurücksetzen."""
        if event.key == "e":
            self._on_export(None)
        elif event.key == "r":
            self._auto_zoom()
            self._fig.canvas.draw_idle()

    def _on_export(self, _) -> None:
        """Speichert die aktuelle Kontur als JSON-Datei (öffnet Datei-Dialog).

        Format:
            {"outer": [[x,y], ...], "hole": [[x,y], ...]}
        "hole" ist nur vorhanden wenn eine Innenkontur existiert.
        """
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="Kontur speichern",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Alle Dateien", "*.*")],
            initialfile="kontur.json",
        )
        root.destroy()
        if not path:
            return   # Abgebrochen

        data = {"outer": self._pts.tolist()}
        if self._hole is not None:
            data["hole"] = self._hole.tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Gespeichert: {path}")

    def _on_import(self, _) -> None:
        """Lädt eine Kontur aus einer JSON-Datei (öffnet Datei-Dialog)."""
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Kontur laden",
            filetypes=[("JSON", "*.json"), ("Alle Dateien", "*.*")],
        )
        root.destroy()
        if not path:
            return   # Abgebrochen

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "outer" not in data:
            print(f"Ungültige Datei: 'outer'-Schlüssel fehlt in {path}")
            return

        self._pts  = np.array(data["outer"], dtype=float)
        self._hole = np.array(data["hole"],  dtype=float) if "hole" in data else None
        self._auto_zoom()
        self._redraw()
        print(f"Geladen: {path}  ({len(self._pts)} Punkte)")

    def _on_template(self, label: str) -> None:
        """Callback wenn ein Radio-Button geklickt wird."""
        self._load_template(label)

    def _on_reset(self, _) -> None:
        """Setzt die aktive Vorlage auf den Ursprungszustand zurück."""
        label = self._radio.value_selected   # welcher Radio-Button ist aktiv?
        self._load_template(label)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def get_points(self) -> list[tuple[float, float]]:
        """Gibt die aktuelle Außenkontur als Liste von (x, y)-Tupeln zurück."""
        return self._pts.tolist()

    def get_hole_points(self) -> list[tuple[float, float]] | None:
        """Gibt die Innenkontur zurück, oder None wenn kein Loch vorhanden."""
        return self._hole.tolist() if self._hole is not None else None

    def show(self) -> None:
        """Öffnet das Editor-Fenster (blockiert bis das Fenster geschlossen wird)."""
        plt.show()


# ---------------------------------------------------------------------------
# Direktaufruf
# ---------------------------------------------------------------------------

def run(template: str = "T-Träger") -> ContourEditor:
    """Öffnet den Kontur-Editor mit der angegebenen Vorlage.

    Kann direkt aus anderen Skripten importiert und aufgerufen werden:
        from plasma_cutter.geometry import run_contour_editor
        editor = run_contour_editor()
        points = editor.get_points()
    """
    editor = ContourEditor(template=template)
    editor.show()
    return editor


if __name__ == "__main__":
    # Direkt ausführbar: python contour_editor.py
    run()
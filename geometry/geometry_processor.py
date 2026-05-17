"""
Geometry Processor
==================
Verarbeitet eine rohe Kontur-JSON-Datei (aus dem Kontur-Editor) in zwei Schritten:

  1. Kontur verdichten  -- Außenpunkte mit max. 5 mm Abstand
  2. Innenpunkte       -- gleichmaessiges 2.5 mm Raster innerhalb der Geometrie

Eingabe-Ordner : Geometrie_Konturen_ungeprüft
Ausgabe-Ordner : Geometrie_Konturen geprüft

Ausgabe-JSON
------------
    {
        "points": [
            {"x": …, "y": …, "type": "outer"},
            {"x": …, "y": …, "type": "hole"},   (falls Loch vorhanden)
            {"x": …, "y": …, "type": "inner"},
            ...
        ],
        "n_outer":          …,
        "n_hole":           …,   (falls Loch vorhanden)
        "n_inner":          …,
        "contour_spacing":  5.0,
        "grid_spacing":     2.5,
        "unit":             "mm"
    }

Visualisierung (3 Bilder nebeneinander)
----------------------------------------
  1. Eingang    -- Originalkontur mit wenigen Punkten (direkt aus Editor)
  2. Kontur     -- Verdichtete Außenkontur (5 mm Abstand)
  3. Geometrie  -- Verdichtete Kontur + Innenpunkte-Raster
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, Polygon

# ---------------------------------------------------------------------------
# Pfade & Konstanten
# ---------------------------------------------------------------------------

BASE_DIR          = Path(__file__).parent                        # Ordner dieser Datei (geometry/)
DIR_INPUT         = BASE_DIR / "Geometrie_Konturen_ungeprüft"    # Roh-JSONs aus dem ContourEditor
DIR_OUTPUT        = BASE_DIR / "Geometrie_Konturen_geprüft"      # Verarbeitete JSONs (Ausgabe)

CONTOUR_SPACING   = 5.0    # mm – maximaler Abstand zwischen zwei Außenpunkten nach Verdichtung
GRID_SPACING      = 2.5    # mm – Rasterabstand der Innenpunkte

# Farben für die Visualisierung
COLOR_OUTER       = "steelblue"   # Außenkontur
COLOR_HOLE        = "tomato"      # Innenkontur (Loch)
COLOR_INNER       = "seagreen"    # Innenpunkte (Raster)


# ---------------------------------------------------------------------------
# Schritt 1 | Laden
# ---------------------------------------------------------------------------

def load_raw_contour(path: Path) -> tuple[list[list[float]], list[list[float]] | None]:
    """Lädt eine Roh-Kontur aus einer JSON-Datei.

    Unterstützt zwei Formate:
    - Flache Liste: [[x,y], [x,y], ...]  (älteres Format)
    - Dict mit Schlüsseln: "outer", "hole" (ContourEditor-Format)

    Mehrere Schlüsselnamen werden versucht damit alte und neue
    Dateien gleichermassen geladen werden können.

    Returns
    -------
    outer : Außenpunkte als [[x, y], ...]
    hole  : Innenkontur-Punkte oder None wenn kein Loch vorhanden
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Format 1: einfache Liste von Punkten → nur Außenkontur, kein Loch
    if isinstance(data, list):
        return [[float(p[0]), float(p[1])] for p in data], None

    # Format 2: Dict mit benannten Schlüsseln
    if isinstance(data, dict):
        # Verschiedene Schlüsselnamen für die Außenkontur versuchen
        outer_raw = (
            data.get("outer")
            or data.get("points")
            or data.get("exterior")
            or data.get("contour")
        )
        if outer_raw is None:
            raise ValueError(
                f"Kein bekannter Schluessel in JSON. "
                f"Vorhandene Schluessel: {list(data.keys())}"
            )
        outer = [[float(p[0]), float(p[1])] for p in outer_raw]
        # Innenkontur (Loch) ist optional
        hole_raw = data.get("hole") or data.get("interior")
        hole = [[float(p[0]), float(p[1])] for p in hole_raw] if hole_raw else None
        return outer, hole

    raise ValueError("Unbekanntes JSON-Format.")


# ---------------------------------------------------------------------------
# Schritt 2 | Kontur verdichten
# ---------------------------------------------------------------------------

def densify_contour(
    points: list[list[float]],
    max_dist: float,
) -> tuple[list[list[float]], int]:
    """Verdichtet eine Kontur: fügt Zwischenpunkte ein wo Segmente zu lang sind.

    Jedes Segment das länger als max_dist ist wird gleichmäßig unterteilt.
    Dadurch wird sichergestellt dass kein zwei benachbarte Punkte weiter
    als max_dist mm voneinander entfernt sind.

    Beispiel: Segment 18 mm lang, max_dist=5 mm → wird in 4 Teile à 4.5 mm geteilt.

    Returns
    -------
    densified : neue Punktliste mit Zwischenpunkten
    added     : Anzahl der neu hinzugefügten Punkte (zur Info-Ausgabe)
    """
    result: list[list[float]] = []
    added = 0
    n = len(points)

    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]   # nächster Punkt (zyklisch: nach letztem kommt erster)
        result.append(p0)           # aktuellen Punkt immer übernehmen

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = math.hypot(dx, dy)   # Segmentlänge

        if dist > max_dist:
            # Anzahl gleichmäßiger Teilintervalle berechnen
            n_intervals = math.ceil(dist / max_dist)
            # Zwischenpunkte bei t = 1/n, 2/n, ..., (n-1)/n einfügen
            for k in range(1, n_intervals):
                t = k / n_intervals
                result.append([p0[0] + t * dx, p0[1] + t * dy])
                added += 1

    return result, added


# ---------------------------------------------------------------------------
# Schritt 3 | Innenpunkte erzeugen
# ---------------------------------------------------------------------------

def generate_inner_points(
    outer: list[list[float]],
    hole:  list[list[float]] | None,
    spacing: float,
) -> list[list[float]]:
    """Erzeugt ein gleichmäßiges Punktraster innerhalb der Geometrie.

    Das Raster überspannt die gesamte Bounding Box der Geometrie.
    Nur Punkte die tatsächlich innerhalb des Polygons liegen (und
    nicht im Loch) werden behalten.

    Der Rasterursprung wird um spacing/2 versetzt damit keine Punkte
    genau auf der Konturlinie selbst liegen (was numerisch problematisch
    wäre für Schneider-Berechnungen).

    Parameters
    ----------
    outer   : verdichtete Außenkontur
    hole    : verdichtete Innenkontur (Loch) oder None
    spacing : Abstand zwischen den Rasterpunkten in mm

    Returns
    -------
    Liste von [x, y]-Punkten innerhalb der Geometrie.
    """
    # Shapely-Polygon erstellen (mit Loch falls vorhanden)
    # buffer(0) repariert mögliche geometrische Fehler (z.B. leicht selbstüberschneidend)
    polygon = Polygon(outer, [hole] if hole else []).buffer(0)

    xmin, ymin, xmax, ymax = polygon.bounds
    inner: list[list[float]] = []

    # Raster zeilenweise aufbauen (x äußere Schleife, y innere Schleife)
    x = xmin + spacing / 2   # Versatz um halben Rasterabstand vom Rand
    while x < xmax:
        y = ymin + spacing / 2
        while y < ymax:
            # Punkt nur aufnehmen wenn er tatsächlich im Polygon liegt
            if polygon.contains(Point(x, y)):
                inner.append([round(x, 6), round(y, 6)])   # auf 6 Stellen runden
            y += spacing
        x += spacing

    return inner


# ---------------------------------------------------------------------------
# Kern-Funktion
# ---------------------------------------------------------------------------

def prepare_geometry_points(
    input_path: Path,
    contour_spacing: float = CONTOUR_SPACING,
    grid_spacing: float    = GRID_SPACING,
    show_plot: bool        = True,
) -> Path:
    """Vollstaendige Verarbeitung: Laden -> Verdichten -> Innenpunkte -> Speichern.

    Parameters
    ----------
    input_path       : Pfad zur Roh-JSON-Datei
    contour_spacing  : maximaler Abstand Außenpunkte [mm]
    grid_spacing     : Rasterabstand Innenpunkte [mm]
    show_plot        : Visualisierung anzeigen

    Returns
    -------
    output_path : Pfad der gespeicherten Ergebnis-Datei
    """
    # --- Laden ---
    print(f"Lade:  {input_path.name}")
    orig_outer, orig_hole = load_raw_contour(input_path)
    print(f"  Eingang Außenkontur : {len(orig_outer)} Punkte")
    if orig_hole:
        print(f"  Eingang Innenkontur  : {len(orig_hole)} Punkte")

    # --- Kontur verdichten ---
    dense_outer, added_outer = densify_contour(orig_outer, contour_spacing)
    dense_hole,  added_hole  = (
        densify_contour(orig_hole, contour_spacing)
        if orig_hole else (None, 0)
    )

    print(f"  Außenkontur verdichtet: {len(dense_outer)} Punkte (+{added_outer})")
    if dense_hole:
        print(f"  Innenkontur verdichtet : {len(dense_hole)} Punkte (+{added_hole})")

    # --- Innenpunkte ---
    print(f"  Erzeuge Innenpunkte (Raster {grid_spacing} mm) ...")
    inner = generate_inner_points(dense_outer, dense_hole, grid_spacing)
    print(f"  Innenpunkte : {len(inner)}")

    # --- Speichern ---
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = DIR_OUTPUT / input_path.name

    points_list = []
    for x, y in dense_outer:
        points_list.append({"x": round(x, 6), "y": round(y, 6), "type": "outer"})
    if dense_hole:
        for x, y in dense_hole:
            points_list.append({"x": round(x, 6), "y": round(y, 6), "type": "hole"})
    for x, y in inner:
        points_list.append({"x": round(x, 6), "y": round(y, 6), "type": "inner"})

    payload: dict = {
        "points":           points_list,
        "n_outer":          len(dense_outer),
        "n_inner":          len(inner),
        "contour_spacing":  contour_spacing,
        "grid_spacing":     grid_spacing,
        "unit":             "mm",
    }
    if dense_hole:
        payload["n_hole"] = len(dense_hole)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Gespeichert: {output_path}")

    # --- Visualisierung ---
    if show_plot:
        _visualize(
            orig_outer, orig_hole,
            dense_outer, dense_hole,
            inner,
            title=input_path.stem,
        )

    return output_path


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------

def _draw_contour(
    ax: plt.Axes,
    outer: list[list[float]],
    hole: list[list[float]] | None,
    inner: list[list[float]] | None,
    show_inner_label: bool = True,
) -> None:
    """Zeichnet Kontur (+ optionales Loch + optionale Innenpunkte) auf ax."""

    # Gefuellte Flaeche
    outer_patch = mpatches.Polygon(
        outer, closed=True,
        facecolor=COLOR_OUTER, edgecolor=COLOR_OUTER,
        linewidth=1.4, alpha=0.12,
    )
    ax.add_patch(outer_patch)

    if hole:
        hole_patch = mpatches.Polygon(
            hole, closed=True,
            facecolor="white", edgecolor=COLOR_HOLE,
            linewidth=1.4, alpha=1.0,
        )
        ax.add_patch(hole_patch)

    # Konturlinien
    ox = [p[0] for p in outer] + [outer[0][0]]
    oy = [p[1] for p in outer] + [outer[0][1]]
    ax.plot(ox, oy, "-", color=COLOR_OUTER, linewidth=1.3,
            label=f"Außenkontur ({len(outer)} Pkt)")

    if hole:
        hx = [p[0] for p in hole] + [hole[0][0]]
        hy = [p[1] for p in hole] + [hole[0][1]]
        ax.plot(hx, hy, "-", color=COLOR_HOLE, linewidth=1.3,
                label=f"Innenkontur ({len(hole)} Pkt)")

    # Außenpunkte
    ax.scatter(
        [p[0] for p in outer], [p[1] for p in outer],
        s=22, color=COLOR_OUTER, edgecolors="black", linewidths=0.4,
        zorder=4,
    )
    if hole:
        ax.scatter(
            [p[0] for p in hole], [p[1] for p in hole],
            s=22, color=COLOR_HOLE, edgecolors="black", linewidths=0.4,
            zorder=4,
        )

    # Innenpunkte (Raster)
    if inner:
        label = f"Innenpunkte ({len(inner)})" if show_inner_label else None
        ax.scatter(
            [p[0] for p in inner], [p[1] for p in inner],
            s=10, color=COLOR_INNER, edgecolors="none",
            zorder=3, alpha=0.85, label=label,
        )

    # Achsgrenzen
    all_pts = outer + (hole or []) + (inner or [])
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    m = span * 0.1 + 5
    ax.set_xlim(min(xs) - m, max(xs) + m)
    ax.set_ylim(min(ys) - m, max(ys) + m)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="upper right")


def _visualize(
    orig_outer:  list[list[float]],
    orig_hole:   list[list[float]] | None,
    dense_outer: list[list[float]],
    dense_hole:  list[list[float]] | None,
    inner:       list[list[float]],
    title: str   = "",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- Bild 1: Eingang ---
    axes[0].set_title(
        f"1 | Eingang\n{len(orig_outer)} Außenpunkte"
        + (f"  +  {len(orig_hole)} Innenkontupunkte" if orig_hole else ""),
        fontsize=10,
    )
    _draw_contour(axes[0], orig_outer, orig_hole, inner=None)

    # --- Bild 2: Verdichtete Kontur ---
    axes[1].set_title(
        f"2 | Kontur verdichtet (max {CONTOUR_SPACING} mm)\n"
        f"{len(dense_outer)} Außenpunkte"
        + (f"  +  {len(dense_hole)} Innenkontupunkte" if dense_hole else ""),
        fontsize=10,
    )
    _draw_contour(axes[1], dense_outer, dense_hole, inner=None)

    # --- Bild 3: Vollstaendige Geometrie ---
    axes[2].set_title(
        f"3 | Geometrie komplett\n"
        f"Außen {len(dense_outer)} Pkt  |  Innen {len(inner)} Pkt (Raster {GRID_SPACING} mm)",
        fontsize=10,
    )
    _draw_contour(axes[2], dense_outer, dense_hole, inner=inner)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Geometry Processor: verdichtet Kontur (5 mm) und erzeugt "
            "Innenpunkte-Raster (2.5 mm)."
        )
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Pfad zur Roh-JSON-Datei (optional; Datei-Dialog wenn weggelassen)",
    )
    parser.add_argument(
        "--contour-spacing", type=float, default=CONTOUR_SPACING,
        help=f"Max. Abstand Außenpunkte in mm (Standard: {CONTOUR_SPACING})",
    )
    parser.add_argument(
        "--grid-spacing", type=float, default=GRID_SPACING,
        help=f"Rasterabstand Innenpunkte in mm (Standard: {GRID_SPACING})",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Visualisierung unterdruecken",
    )
    args = parser.parse_args()

    # Datei-Dialog wenn kein Pfad angegeben
    if args.input is None:
        import tkinter as tk
        from tkinter import filedialog
        DIR_INPUT.mkdir(parents=True, exist_ok=True)
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Roh-Kontur auswaehlen",
            filetypes=[("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")],
            initialdir=str(DIR_INPUT),
        )
        root.destroy()
        if not chosen:
            print("Keine Datei ausgewaehlt. Abbruch.", file=sys.stderr)
            sys.exit(0)
        args.input = chosen

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Fehler: Datei nicht gefunden: {input_path}", file=sys.stderr)
        sys.exit(1)

    prepare_geometry_points(
        input_path,
        contour_spacing=args.contour_spacing,
        grid_spacing=args.grid_spacing,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()

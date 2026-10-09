"""
Geometry Processor
==================
Processes a raw contour JSON file:

  1. Densify contour  -- outer points at most 5 mm apart
  2. Inner points     -- uniform 2.5 mm grid inside the geometry

Input folder  : Geometrie_Konturen_ungeprüft
Output folder : Geometrie_Konturen_geprüft

Output JSON
-----------
    {
        "points": [
            {"x": …, "y": …, "type": "outer"},
            {"x": …, "y": …, "type": "hole"},
            {"x": …, "y": …, "type": "inner"},
            ...
        ],
        "n_outer":          …,
        "n_hole":           …,
        "n_inner":          …,
        "contour_spacing":  5.0,
        "grid_spacing":     2.5,
        "unit":             "mm"
    }

What happens: corner points → densified cross-section.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Window backend: TkAgg first, then Qt5Agg, else default.
# matplotlib.use() must come before the pyplot import.
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
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR          = Path(__file__).parent                        # this file's folder (geometry/)
DIR_INPUT         = BASE_DIR / "Geometrie_Konturen_ungeprüft"    # raw JSONs from the contour editor
DIR_OUTPUT        = BASE_DIR / "Geometrie_Konturen_geprüft"      # processed JSONs (output)

CONTOUR_SPACING   = 5.0    # mm – max. spacing of outer points
GRID_SPACING      = 2.5    # mm – grid spacing of inner points

# Colors for the visualization
COLOR_OUTER       = "steelblue"   # outer contour
COLOR_HOLE        = "tomato"      # hole contour
COLOR_INNER       = "seagreen"    # inner points (grid)


# ---------------------------------------------------------------------------
# Step 1 | Load
# ---------------------------------------------------------------------------

def load_raw_contour(path: Path) -> tuple[list[list[float]], list[list[float]] | None]:
    """Loads a raw contour from a JSON file.

    Returns
    -------
    outer : outer points as [[x, y], ...]
    hole  : hole contour points, or None if there is no hole
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Format 1: plain list of points → outer contour only, no hole
    if isinstance(data, list):
        return [[float(p[0]), float(p[1])] for p in data], None

    # Format 2: dict with named keys
    if isinstance(data, dict):
        # Try different key names for the outer contour
        # How it works: the "or" chain takes the first key that
        #   yields a value (anything else is None/empty = "falsy").
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
        # Hole contour is optional
        hole_raw = data.get("hole") or data.get("interior")
        hole = [[float(p[0]), float(p[1])] for p in hole_raw] if hole_raw else None
        return outer, hole

    raise ValueError("Unbekanntes JSON-Format.")


# ---------------------------------------------------------------------------
# Step 2 | Densify contour
# ---------------------------------------------------------------------------

def densify_contour(
    points: list[list[float]],
    max_dist: float,
) -> tuple[list[list[float]], int]:
    """Densifies a contour: inserts intermediate points where segments are too long.

    Every segment longer than max_dist is subdivided evenly.

    How it works:
        Loop over all points; consider the segment to the next point
        (cyclic, so that the last and the first point form a closed
        ring). If the segment is too long, evenly spaced intermediate
        points (t = 1/n ... (n-1)/n) are inserted by linear
        interpolation p0 + t*(p1-p0). math.ceil guarantees that the
        pieces are <= max_dist.

    Returns
    -------
    densified : new point list with intermediate points
    added     : number of newly added points (for info output)
    """
    result: list[list[float]] = []
    added = 0
    n = len(points)

    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]   # next point (cyclic: the first follows the last)
        result.append(p0)           # always keep the current point

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = math.hypot(dx, dy)   # edge length

        if dist > max_dist:
            # number of equal subintervals
            n_intervals = math.ceil(dist / max_dist)    #round up
            # insert intermediate points at t = 1/n, 2/n, ..., (n-1)/n
            for k in range(1, n_intervals):
                t = k / n_intervals
                result.append([p0[0] + t * dx, p0[1] + t * dy])
                added += 1

    return result, added


# ---------------------------------------------------------------------------
# Step 3 | Generate inner points
# ---------------------------------------------------------------------------

def generate_inner_points(
    outer: list[list[float]],
    hole:  list[list[float]] | None,
    spacing: float,
) -> list[list[float]]:
    """Generates a uniform point grid inside the geometry.

    How it works:
        A shapely polygon is built from the contour points (hole as
        second ring). polygon.bounds gives the enclosing rectangle
        (bounding box). A nested while loop runs over it in x and y
        with step size "spacing". For each grid point,
        polygon.contains(Point(x, y)) checks whether it really lies in
        the workpiece (holes automatically count as "outside"). Only
        these hits end up in the result list.

    Parameters
    ----------
    outer   : densified outer contour
    hole    : densified hole contour or None
    spacing : spacing between the grid points in mm

    Returns
    -------
    List of [x, y] points inside the geometry.
    """
    # Build shapely polygon (with hole if present)
    # buffer(0) repairs possible geometric defects (e.g. slightly self-intersecting)
    polygon = Polygon(outer, [hole] if hole else []).buffer(0)

    xmin, ymin, xmax, ymax = polygon.bounds
    inner: list[list[float]] = []

    # Build the grid row by row (x outer loop, y inner loop)
    x = xmin + spacing / 2   # offset by half the grid spacing from the edge
    while x < xmax:
        y = ymin + spacing / 2
        while y < ymax:
            # add the point only if it actually lies in the polygon
            if polygon.contains(Point(x, y)):
                inner.append([round(x, 6), round(y, 6)])   # round to 6 decimals
            y += spacing
        x += spacing

    return inner


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def prepare_geometry_points(
    input_path: Path,
    contour_spacing: float = CONTOUR_SPACING,
    grid_spacing: float    = GRID_SPACING,
    show_plot: bool        = True,
) -> Path:
    """Full processing: load -> densify -> inner points -> save.

    Parameters
    ----------
    input_path       : path to the raw JSON file
    contour_spacing  : maximum spacing of outer points [mm]
    grid_spacing     : grid spacing of inner points [mm]
    show_plot        : show visualization

    Returns
    -------
    output_path : path of the saved result file
    """
    # --- Load ---
    print(f"Lade:  {input_path.name}")
    orig_outer, orig_hole = load_raw_contour(input_path)
    print(f"  Eingang Außenkontur : {len(orig_outer)} Punkte")
    if orig_hole:
        print(f"  Eingang Innenkontur  : {len(orig_hole)} Punkte")

    # --- Densify contour ---
    dense_outer, added_outer = densify_contour(orig_outer, contour_spacing)
    dense_hole,  added_hole  = (
        densify_contour(orig_hole, contour_spacing)
        if orig_hole else (None, 0)
    )

    print(f"  Außenkontur verdichtet: {len(dense_outer)} Punkte (+{added_outer})")
    if dense_hole:
        print(f"  Innenkontur verdichtet : {len(dense_hole)} Punkte (+{added_hole})")

    # --- Inner points ---
    print(f"  Erzeuge Innenpunkte (Raster {grid_spacing} mm) ...")
    inner = generate_inner_points(dense_outer, dense_hole, grid_spacing)
    print(f"  Innenpunkte : {len(inner)}")

    # --- Save ---
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = DIR_OUTPUT / input_path.name

    # Write all point types into ONE flat list, each with a "type"
    # marker. Order: outer contour first, then hole, then grid.
    # A reader can later tell a point's role from the "type" field
    # without keeping the lists separate.
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

    # --- Visualization ---
    if show_plot:
        _visualize(
            orig_outer, orig_hole,
            dense_outer, dense_hole,
            inner,
            title=input_path.stem,
        )

    return output_path


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_contour(
    ax: plt.Axes,
    outer: list[list[float]],
    hole: list[list[float]] | None,
    inner: list[list[float]] | None,
    show_inner_label: bool = True,
) -> None:
    """Draws contour (+ optional hole + optional inner points) on ax.
    """

    # Filled area
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

    # Contour lines
    ox = [p[0] for p in outer] + [outer[0][0]]
    oy = [p[1] for p in outer] + [outer[0][1]]
    ax.plot(ox, oy, "-", color=COLOR_OUTER, linewidth=1.3,
            label=f"Außenkontur ({len(outer)} Pkt)")

    if hole:
        hx = [p[0] for p in hole] + [hole[0][0]]
        hy = [p[1] for p in hole] + [hole[0][1]]
        ax.plot(hx, hy, "-", color=COLOR_HOLE, linewidth=1.3,
                label=f"Innenkontur ({len(hole)} Pkt)")

    # Outer points
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

    # Inner points (grid)
    if inner:
        label = f"Innenpunkte ({len(inner)})" if show_inner_label else None
        ax.scatter(
            [p[0] for p in inner], [p[1] for p in inner],
            s=10, color=COLOR_INNER, edgecolors="none",
            zorder=3, alpha=0.85, label=label,
        )

    # Derive axis limits automatically from all visible points.
    # How it works: determine the largest extent (span) and compute a
    # margin "m" from it (10 % plus a 5 mm base) so the geometry does not
    # stick to the image border. set_aspect("equal") keeps x/y to scale.
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

    # --- Panel 1: input ---
    axes[0].set_title(
        f"1 | Eingang\n{len(orig_outer)} Außenpunkte"
        + (f"  +  {len(orig_hole)} Innenkontupunkte" if orig_hole else ""),
        fontsize=10,
    )
    _draw_contour(axes[0], orig_outer, orig_hole, inner=None)

    # --- Panel 2: densified contour ---
    axes[1].set_title(
        f"2 | Kontur verdichtet (max {CONTOUR_SPACING} mm)\n"
        f"{len(dense_outer)} Außenpunkte"
        + (f"  +  {len(dense_hole)} Innenkontupunkte" if dense_hole else ""),
        fontsize=10,
    )
    _draw_contour(axes[1], dense_outer, dense_hole, inner=None)

    # --- Panel 3: complete geometry ---
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
    """Entry point for command-line invocation.
    """
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

    # File dialog if no path is given
    # How it works: tkinter is imported only HERE (not at the top), so
    #   the script also runs without a GUI as long as a path is passed.
    #   root.withdraw() hides the empty main window; "-topmost" brings
    #   the dialog to the front.
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

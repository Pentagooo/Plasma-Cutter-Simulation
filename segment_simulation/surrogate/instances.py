"""Trainings- und Benchmark-Instanzen (BA Kap. 5).

* ``generate_instances``   -- parametrischer Katalog von Strukturelementen
  (Flachstahl, T-Profil, Doppel-T, Hollandprofil-Naeherung) mit seedbar
  variierten Massen PLUS die geprueften Testgeometrien (Familie "real").
* ``make_catalog_instance`` -- Instanz ``idx`` deterministisch und stabil
  ueber (seed, idx, SHAPE_VERSION); Basis des resumebaren Label-Caches.
* ``grid_from_polygon``     -- PointGrid aus einem Shapely-Polygon (spiegelt
  ``geometry_processor`` wider: Aussenkontur -> Loch -> Innenraster).
* ``default_cutter``        -- der Standard-Cutter des Simulators (lazy).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

try:
    from ...geometry.point_grid import PointGrid
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.geometry.point_grid import PointGrid


TEST_GEOMETRY_DIR = (
    Path(__file__).resolve().parents[2] / "geometry"
    / "Geometrie_Konturen_geprüft"
)

# ---------------------------------------------------------------------------
# Katalog-Massbereiche [mm]
# ---------------------------------------------------------------------------
# TODO(Projektwerte): Bereiche final an die real vorkommenden Profilmasse
# anpassen. Bewusst BREIT gewaehlt (mehr Varianz -> bessere Lernkurve).

# SHAPE_VERSION versioniert den GEOMETRIE-Katalog (Massbereiche, Formen).
# ACHTUNG: ``make_catalog_instance`` bildet idx -> SHAPE_ORDER[idx % len]
# und seedet mit SHAPE_VERSION. Eine neue Familie oder ein neuer
# Massbereich aendert damit JEDE Instanz -> alle Labels neu (der Stempel
# ``params.params_hash`` enthaelt SHAPE_VERSION). Familien und Bereiche
# muessen vor dem Start eines Label-Laufs final sein.
# Bewusst getrennt von der Label-Version des Lehrers (``teacher.LABEL_VERSION``):
# eine Aenderung am Zeitmodell darf die Instanzen nicht umwuerfeln, sonst
# sind Benchmarks ueber Versionen hinweg nicht mehr vergleichbar. Nur
# erhoehen, wenn sich Formen/Massbereiche aendern (dann sind auch die
# Label-Caches hinfaellig).
SHAPE_VERSION = 6
CONTOUR_SPACING = 5.0     # wie im geometry_processor (Aussenpunkt-Abstand)
GRID_SPACING = 2.5        # Rasterabstand der Innenpunkte

# Katalog nach Normprofilen (Max, 18.09.2026). Massbereiche = uebliche
# Lieferbereiche der jeweiligen Norm, oben so gekappt, dass Punktzahl und
# Lehrerzeit vertretbar bleiben. Ecken sind scharf (keine Ausrundungen).
#
# Flachstahl (Breitflachstahl DIN 59200): Breite b x Dicke t
FLAT_WIDTH = (100.0, 400.0)
FLAT_THICK = (6.0, 40.0)
# Winkelprofil (DIN EN 10056-1, gleich- und ungleichschenklig):
# Schenkel a, Verhaeltnis b/a (1.0 = gleichschenklig), Dicke t/a
L_A = (30.0, 200.0)
L_RATIO = (0.5, 1.0)          # b = a * ratio; 50 % der Instanzen gleichschenklig
L_T_REL = (0.06, 0.12)        # t = a * rel, mindestens 3 mm
# T-Profil (EN 10055, hochstegig, b = h): Hoehe h, Dicke t = h * rel
T_H = (30.0, 140.0)
T_T_REL = (0.09, 0.12)
# U-Profil (UPN/U/CH nach DIN EN 10365): Hoehe h; b, tw, tf aus h abgeleitet
# (UPN 100: b 50 / tw 6 / tf 8.5; UPN 300: b 100 / tw 10 / tf 16)
U_H = (40.0, 300.0)
U_B_REL = (0.23, 0.33)        # b  = h * rel (+ Streuung)
U_TW_REL = (0.028, 0.045)     # tw = h * rel, mindestens 4 mm
U_TF_REL = (0.045, 0.065)     # tf = h * rel, mindestens 5 mm
# Wulstflachstahl / Hollandprofil (DIN EN 10067): Steg h x t, Wulst einseitig
HP_H = (80.0, 300.0)
HP_T = (5.0, 16.0)
HP_W = (1.6, 2.6)             # Auskragung des Wulsts als Vielfaches von t
HP_HB = (0.12, 0.22)          # Wulsthoehe als Anteil der Steghoehe h
# H-Profil (HEA/HEB nach DIN EN 10365): Hoehe h, b/h, tw/h, tf/h
# (HEB 100: b 100 / tw 6 / tf 10; HEB 300: b 300 / tw 11 / tf 19)
H_H = (100.0, 300.0)
H_B_REL = (0.85, 1.0)
H_TW_REL = (0.03, 0.045)
H_TF_REL = (0.055, 0.075)
# Anbauten = Schweissnaehte und kleine Anschweissteile an einem Basisprofil:
#   * Kehlnaht: kleines Dreieck in (fast) jeder Innenecke (Steg/Flansch-Uebergang)
#   * gelegentlich ein kleiner Halbkreis (Schweissraupe, Bolzen) auf einer Kante
#   * gelegentlich ein kleines Rechteck (Lasche, Steife) auf einer Kante
ATT_WELD_P = 0.75              # Wahrscheinlichkeit je Innenecke fuer eine Kehlnaht
ATT_WELD_LEG = (4.0, 10.0)     # Schenkel der Kehlnaht [mm]
ATT_EXTRA_N = (0, 2)           # Anzahl zusaetzlicher Halbkreise/Rechtecke (inkl.)
ATT_R = (2.5, 5.0)             # Radius Halbkreis (Schweissraupe) [mm]
ATT_W = (8.0, 25.0)            # Rechteck: Breite entlang der Kante [mm]
ATT_H = (4.0, 10.0)            # Rechteck: Auskragung [mm]

# Familien-IDs (fuer GroupKFold nach Formfamilie). "real" bleibt 4; IDs
# frueherer Familien bleiben reserviert, neue nur ANHAENGEN.
FAMILIES = {"flat": 0, "tprofile": 1, "hbeam": 2, "holland": 3, "real": 4,
            "angle": 5, "channel": 6, "attached": 7}

# ---------------------------------------------------------------------------
# Grid-Bau aus Shapely-Polygon (spiegelt geometry_processor wider)
# ---------------------------------------------------------------------------

def _densify_ring(points: np.ndarray, max_dist: float) -> np.ndarray:
    """Verdichtet einen geschlossenen Ring (offene, geordnete Punktfolge),
    so dass keine Kante laenger als ``max_dist`` ist (zyklisch)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    out: list[np.ndarray] = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        out.append(p0)
        d = float(np.hypot(*(p1 - p0)))
        if d > max_dist:
            k = int(np.ceil(d / max_dist))
            for j in range(1, k):
                out.append(p0 + (j / k) * (p1 - p0))
    return np.asarray(out, dtype=float)


def _inner_grid(poly: Polygon, spacing: float) -> np.ndarray:
    """Gleichmaessiges Punktraster innerhalb ``poly`` (vektorisiert)."""
    xmin, ymin, xmax, ymax = poly.bounds
    xs = np.arange(xmin + spacing / 2, xmax, spacing)
    ys = np.arange(ymin + spacing / 2, ymax, spacing)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2))
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.ravel()
    gy = gy.ravel()
    inside = shapely.contains_xy(poly, gx, gy)
    return np.column_stack([gx[inside], gy[inside]])


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    if isinstance(geom, Polygon):
        return geom
    return None


def grid_from_polygon(
    poly: Polygon | MultiPolygon,
    contour_spacing: float = CONTOUR_SPACING,
    grid_spacing: float = GRID_SPACING,
) -> PointGrid | None:
    """Baut ein ``PointGrid`` aus einem Shapely-Polygon.

    Reihenfolge der Punkte: Aussenkontur -> (max. eine) Lochkontur ->
    Innenraster. Genau die Reihenfolge, auf die sich ``PointGrid``/
    ``SegmentedContour`` verlassen.
    """
    poly = _largest_polygon(poly.buffer(0) if poly is not None else None)
    if poly is None or poly.area < 1e-6:
        return None

    ext = np.asarray(poly.exterior.coords, dtype=float)[:-1]
    outer = _densify_ring(ext, contour_spacing)
    if len(outer) < 3:
        return None

    hole = None
    if poly.interiors:
        big = max(poly.interiors, key=lambda r: Polygon(r).area)
        hcoords = np.asarray(big.coords, dtype=float)[:-1]
        if len(hcoords) >= 3:
            hole = _densify_ring(hcoords, contour_spacing)

    inner = _inner_grid(poly, grid_spacing)

    parts = [outer]
    status = [np.zeros(len(outer), dtype=np.int8)]
    if hole is not None:
        parts.append(hole)
        status.append(np.full(len(hole), PointGrid._STATUS_HOLE, dtype=np.int8))
    if len(inner):
        parts.append(inner)
        status.append(np.full(len(inner), PointGrid._STATUS_INNER, dtype=np.int8))

    coords = np.vstack(parts)
    status_arr = np.concatenate(status)
    return PointGrid(coords=coords, status=status_arr,
                     point_spacing=grid_spacing, contour_spacing=contour_spacing)


# ---------------------------------------------------------------------------
# Katalog-Formen (Querschnitte)
# ---------------------------------------------------------------------------

def _u(rng, lo_hi) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def shape_flat(rng) -> Polygon:
    """Flachstahl DIN 59200: Rechteck b x t."""
    w, t = _u(rng, FLAT_WIDTH), _u(rng, FLAT_THICK)
    return box(-w / 2, -t / 2, w / 2, t / 2)


def shape_angle(rng) -> Polygon:
    """Winkelprofil DIN EN 10056-1: Schenkel a (horizontal), b (vertikal),
    Dicke t; jede zweite Instanz gleichschenklig."""
    a = _u(rng, L_A)
    ratio = 1.0 if rng.random() < 0.5 else _u(rng, L_RATIO)
    b = a * ratio
    t = max(3.0, a * _u(rng, L_T_REL))
    return unary_union([box(0.0, 0.0, a, t), box(0.0, 0.0, t, b)])


def shape_tprofile(rng) -> Polygon:
    """T-Profil EN 10055: Flanschbreite = Hoehe = h, Steg- und
    Flanschdicke t."""
    h = _u(rng, T_H)
    t = max(3.0, h * _u(rng, T_T_REL))
    web = box(-t / 2, 0.0, t / 2, h - t)
    flange = box(-h / 2, h - t, h / 2, h)
    return unary_union([web, flange])


def shape_channel(rng) -> Polygon:
    """U-Profil UPN/U/CH (DIN EN 10365): Steg (Hoehe h, Dicke tw) mit zwei
    Flanschen (Breite b, Dicke tf) nach +x."""
    h = _u(rng, U_H)
    b = h * _u(rng, U_B_REL) + 20.0
    tw = max(4.0, h * _u(rng, U_TW_REL))
    tf = max(5.0, h * _u(rng, U_TF_REL))
    web = box(0.0, 0.0, tw, h)
    bot = box(0.0, 0.0, b, tf)
    top = box(0.0, h - tf, b, h)
    return unary_union([web, bot, top])


def shape_holland(rng) -> Polygon:
    """Wulstflachstahl DIN EN 10067 (HP-Profil).

    Steg = Rechteck (Hoehe h, Dicke t); die Seite x=-t/2 bleibt ueber die
    volle Hoehe gerade. Wulst = Keil an der OBERkante, der nur nach +x
    auskragt: entlang der Steg-Aussenkante von h-hb bis h, oben bis
    x = t/2 + w, die Spitze durch einen einbeschriebenen Kreisbogen
    gerundet. Am Wulst ist das Profil damit deutlich dicker als der Steg.
    """
    h, t = _u(rng, HP_H), _u(rng, HP_T)
    w = t * _u(rng, HP_W)                  # einseitige Auskragung
    hb = h * _u(rng, HP_HB)                # Hoehe des Wulstansatzes
    web = box(-t / 2, 0.0, t / 2, h)
    tip = np.array([t / 2 + w, h])
    u_top = np.array([-1.0, 0.0])                    # entlang der Oberkante
    hyp_len = float(np.hypot(w, hb))
    u_hyp = np.array([-w, -hb]) / hyp_len            # entlang der Hypotenuse
    theta = float(np.arccos(np.clip(np.dot(u_top, u_hyp), -1.0, 1.0)))
    tau = 0.30 * min(w, hyp_len)                     # Tangentenlaenge
    rr = tau * np.tan(theta / 2.0)                   # Rundungsradius
    bis = u_top + u_hyp
    bis /= np.linalg.norm(bis)
    ctr = tip + bis * (rr / np.sin(theta / 2.0))     # Bogenmittelpunkt
    t1 = tip + u_top * tau                           # Tangente Oberkante
    t2 = tip + u_hyp * tau                           # Tangente Hypotenuse
    a1 = float(np.arctan2(*(t1 - ctr)[::-1]))
    a2 = float(np.arctan2(*(t2 - ctr)[::-1]))
    while a1 <= a2:
        a1 += 2.0 * np.pi
    arc = [ctr + rr * np.array([np.cos(a), np.sin(a)])
           for a in np.linspace(a1, a2, 14)]
    bulb = Polygon([(t / 2, h - hb), (t / 2, h), tuple(t1),
                    *map(tuple, arc), tuple(t2)])
    return unary_union([web, bulb])


def shape_hbeam(rng) -> Polygon:
    """H-Profil HEA/HEB (DIN EN 10365): Hoehe h, Flanschbreite b, Steg tw,
    Flansch tf; Flansche oben und unten."""
    h = _u(rng, H_H)
    b = h * _u(rng, H_B_REL)
    tw = max(5.0, h * _u(rng, H_TW_REL))
    tf = max(8.0, h * _u(rng, H_TF_REL))
    bot = box(-b / 2, 0.0, b / 2, tf)
    web = box(-tw / 2, tf, tw / 2, h - tf)
    top = box(-b / 2, h - tf, b / 2, h)
    return unary_union([bot, web, top])


_BASE_BUILDERS = None   # wird unten gesetzt (nach den Basisformen)


def _ring_ccw(poly: Polygon) -> np.ndarray:
    """Aussenring gegen den Uhrzeigersinn, ohne doppelten Schlusspunkt."""
    ring = np.asarray(orient(poly, sign=1.0).exterior.coords, dtype=float)[:-1]
    return ring


def _attach_on_edge(rng, ring: np.ndarray, kind: str) -> Polygon | None:
    """Kleines Rechteck oder Halbkreis auf einer Aussenkante."""
    n = len(ring)
    edges = [(i, float(np.linalg.norm(ring[(i + 1) % n] - ring[i]))) for i in range(n)]
    if kind == "round":
        r = _u(rng, ATT_R)
        need = 2.0 * r + 6.0
    else:
        w = _u(rng, ATT_W)
        need = w + 6.0
    long_edges = [i for i, L in edges if L >= need]
    if not long_edges:
        return None
    i = int(rng.choice(long_edges))
    p0, p1 = ring[i], ring[(i + 1) % n]
    d = p1 - p0
    L = float(np.linalg.norm(d))
    d = d / L
    nrm = np.array([d[1], -d[0]])          # bei CCW-Ring zeigt (dy,-dx) nach aussen
    if kind == "round":
        s0 = _u(rng, (r + 3.0, L - r - 3.0))
        centre = p0 + d * s0
        return Point(tuple(centre)).buffer(r, quad_segs=6)
    s0 = _u(rng, (3.0, L - w - 3.0))
    a = p0 + d * s0
    b = a + d * w
    h = _u(rng, ATT_H)
    inset = nrm * 1.0                      # 1 mm ins Material, damit die Union sicher zusammenhaengt
    return Polygon([tuple(a - inset), tuple(b - inset), tuple(b + nrm * h), tuple(a + nrm * h)])


def _weld_fillets(rng, ring: np.ndarray) -> list[Polygon]:
    """Kehlnaht-Dreiecke in den Innenecken (einspringende Ecken), je Ecke
    mit Wahrscheinlichkeit ``ATT_WELD_P``."""
    n = len(ring)
    out = []
    for i in range(n):
        c = ring[i]
        e_prev = ring[(i - 1) % n] - c          # zurueck entlang der Vorgaengerkante
        e_next = ring[(i + 1) % n] - c
        cross = float((-e_prev[0]) * e_next[1] - (-e_prev[1]) * e_next[0])
        if cross >= -1e-9:                      # keine einspringende Ecke
            continue
        if rng.random() > ATT_WELD_P:
            continue
        leg = _u(rng, ATT_WELD_LEG)
        l1 = min(leg, 0.45 * float(np.linalg.norm(e_prev)))
        l2 = min(leg, 0.45 * float(np.linalg.norm(e_next)))
        u1 = e_prev / float(np.linalg.norm(e_prev))
        u2 = e_next / float(np.linalg.norm(e_next))
        bis = -(u1 + u2)
        bis = bis / max(1e-9, float(np.linalg.norm(bis)))
        out.append(Polygon([tuple(c + bis * 1.0), tuple(c + u1 * l1), tuple(c + u2 * l2)]))
    return out


def shape_attached(rng) -> Polygon:
    """Basisprofil (eine der sechs Normfamilien) mit Schweissnaehten:
    Kehlnaht-Dreiecke in den Innenecken, dazu 0..2 kleine Halbkreise oder
    Rechtecke auf Aussenkanten."""
    base_name = str(rng.choice(list(_BASE_BUILDERS)))
    poly = _largest_polygon(_BASE_BUILDERS[base_name](rng).buffer(0))
    parts = [poly] + _weld_fillets(rng, _ring_ccw(poly))
    n_extra = int(rng.integers(ATT_EXTRA_N[0], ATT_EXTRA_N[1] + 1))
    for _ in range(n_extra):
        ring = _ring_ccw(_largest_polygon(unary_union(parts).buffer(0)))
        kind = "round" if rng.random() < 0.5 else "rect"
        piece = _attach_on_edge(rng, ring, kind)
        if piece is not None and not piece.is_empty:
            parts.append(piece)
    merged = _largest_polygon(unary_union(parts).buffer(0))
    return Polygon(merged.exterior)         # keine eingeschlossenen Hohlraeume


_BASE_BUILDERS = {
    "flat": shape_flat,
    "angle": shape_angle,
    "tprofile": shape_tprofile,
    "channel": shape_channel,
    "holland": shape_holland,
    "hbeam": shape_hbeam,
}
SHAPE_BUILDERS = {**_BASE_BUILDERS, "attached": shape_attached}
# Reihum-Reihenfolge der Katalogfamilien: NUR anhaengen (siehe SHAPE_VERSION).
SHAPE_ORDER = ["flat", "angle", "tprofile", "channel", "holland", "hbeam",
               "attached"]


def tag_instance(grid: PointGrid, family: str, name: str) -> PointGrid:
    grid._family = FAMILIES[family]      # additive Attribute (nur intern)
    grid._family_name = family
    grid._instance_name = name
    return grid


def load_test_geometries() -> list[PointGrid]:
    """Die 6 vorhandenen, geprueften Testgeometrien."""
    out: list[PointGrid] = []
    for p in sorted(TEST_GEOMETRY_DIR.glob("*.json")):
        try:
            g = PointGrid.from_json(p)
        except Exception:
            continue
        out.append(tag_instance(g, "real", p.stem))
    return out


def make_catalog_instance(idx: int, seed: int = 0) -> PointGrid | None:
    """Eine Katalog-Instanz, DETERMINISTISCH und STABIL ueber ``idx``.

    Jede Instanz haengt nur von (seed, idx, SHAPE_VERSION) ab -- nicht von n
    und NICHT von der Label-Version: ein neues Zeitmodell darf dieselben
    Geometrien neu labeln, ohne sie zu veraendern.
    So kann man inkrementell mehr Instanzen erzeugen, ohne die vorhandenen zu
    veraendern (Voraussetzung fuer den resumebaren Label-Cache).
    """
    fam = SHAPE_ORDER[idx % len(SHAPE_ORDER)]
    rng = np.random.default_rng(
        (int(seed) & 0xFFFFFFFF) * 1_000_003 + idx * 131 + SHAPE_VERSION)
    poly = SHAPE_BUILDERS[fam](rng)
    grid = grid_from_polygon(poly)
    if grid is None:
        return None
    grid._catalog_idx = int(idx)          # fuer dataset.spec_of_grid
    return tag_instance(grid, fam, f"{fam}_{idx:05d}")


def generate_instances(n: int, seed: int = 0,
                       include_real: bool = True) -> list[PointGrid]:
    """Erzeugt ``n`` (stabile) Katalog-Instanzen + optional die 6 Testgeometrien.

    Die Katalogformen werden reihum aus ``SHAPE_ORDER`` gezogen und in ihren
    Massen seedbar variiert; Instanz ``i`` ist ueber ``make_catalog_instance``
    stabil (unabhaengig von n). Zusaetzlich werden die realen Testgeometrien
    angehaengt (Familie "real").
    """
    out: list[PointGrid] = []
    for i in range(n):
        grid = make_catalog_instance(i, seed)
        if grid is not None:
            out.append(grid)
    if include_real:
        out.extend(load_test_geometries())
    return out


def default_cutter():
    """Standard-Cutter des Simulators (lazy importiert, um den Import-Zyklus
    surrogate <-> simulation zu vermeiden)."""
    try:
        from ..simulation import make_default_cutter
    except ImportError:
        from plasma_cutter.segment_simulation.simulation import (
            make_default_cutter,
        )
    return make_default_cutter()

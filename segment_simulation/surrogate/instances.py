"""Training and benchmark instances.

generate_instances    : catalog (standard sections, welded parts, assemblies)
                        with seeded dimension variation + the verified test
                        geometries (family "real")
make_catalog_instance : instance idx, stable over (seed, idx, SHAPE_VERSION);
                        basis of the resumable label cache
grid_from_polygon     : PointGrid from Shapely polygon (like geometry_processor:
                        outer contour -> hole -> inner grid)
default_cutter        : default cutter of the simulator (lazy)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
import shapely.affinity
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

try:
    from ...geometry.point_grid import PointGrid
except ImportError:  # direct run without package context
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.geometry.point_grid import PointGrid


TEST_GEOMETRY_DIR = (
    Path(__file__).resolve().parents[2] / "geometry"
    / "Geometrie_Konturen_geprüft"
)

# ---------------------------------------------------------------------------
# Catalog dimension ranges [mm]
# ---------------------------------------------------------------------------

# Version of the geometry catalog (shapes, dimension ranges)
#   - instances are seeded with SHAPE_VERSION: incrementing it changes every
#     instance, all labels must be redone (part of params_hash)
#   - separate from LABEL_VERSION: a new time model does not reshuffle the
#     instances, benchmarks stay comparable
#   - fix families and ranges before a labeling run
SHAPE_VERSION = 7         # 7: family "assembly", double weight
CONTOUR_SPACING = 5.0     # as in geometry_processor (outer point spacing)
GRID_SPACING = 2.5        # grid spacing of the inner points

# Catalog based on standard sections
#   - dimension ranges = usual commercially available ranges, capped at the top
#     (point count, teacher time)
#   - sharp corners (no radii)
#
# Flat bar (wide flat bar DIN 59200): width b x thickness t
FLAT_WIDTH = (100.0, 400.0)
FLAT_THICK = (6.0, 40.0)
# Angle section (DIN EN 10056-1, equal-leg and unequal-leg):
# leg a, ratio b/a (1.0 = equal-leg), thickness t/a
L_A = (30.0, 200.0)
L_RATIO = (0.5, 1.0)          # b = a * ratio; 50 % of instances equal-leg
L_T_REL = (0.06, 0.12)        # t = a * rel, at least 3 mm
# T-section (EN 10055, equal flange tee, b = h): height h, thickness t = h * rel
T_H = (30.0, 140.0)
T_T_REL = (0.09, 0.12)
# Channel (UPN/U/CH per DIN EN 10365): height h; b, tw, tf derived from h
# (UPN 100: b 50 / tw 6 / tf 8.5; UPN 300: b 100 / tw 10 / tf 16)
U_H = (40.0, 300.0)
U_B_REL = (0.23, 0.33)        # b  = h * rel (+ scatter)
U_TW_REL = (0.028, 0.045)     # tw = h * rel, at least 4 mm
U_TF_REL = (0.045, 0.065)     # tf = h * rel, at least 5 mm
# Bulb flat (HP, DIN EN 10067): web h x t, one-sided bulb
HP_H = (80.0, 300.0)
HP_T = (5.0, 16.0)
HP_W = (1.6, 2.6)             # bulb projection as a multiple of t
HP_HB = (0.12, 0.22)          # bulb height as a fraction of web height h
# H-beam (HEA/HEB per DIN EN 10365): height h, b/h, tw/h, tf/h
# (HEB 100: b 100 / tw 6 / tf 10; HEB 300: b 300 / tw 11 / tf 19)
H_H = (100.0, 300.0)
H_B_REL = (0.85, 1.0)
H_TW_REL = (0.03, 0.045)
H_TF_REL = (0.055, 0.075)
# Attachments = welds and small welded-on parts on a base section:
#   * fillet weld: small triangle in (almost) every inner corner (web/flange joint)
#   * occasionally a small semicircle (weld bead, stud) on an edge
#   * occasionally a small rectangle (lug, stiffener) on an edge
ATT_WELD_P = 0.75              # probability of a fillet weld per inner corner
ATT_WELD_LEG = (4.0, 10.0)     # fillet weld leg [mm]
ATT_EXTRA_N = (0, 2)           # number of extra semicircles/rectangles (inclusive)
ATT_R = (2.5, 5.0)             # semicircle radius (weld bead) [mm]
ATT_W = (8.0, 25.0)            # rectangle: width along the edge [mm]
ATT_H = (4.0, 10.0)            # rectangle: projection [mm]
# Assembly ("assembly"): welded assembly on a base section
#   - at least a drilled hole/slot or a second welded-on section
#   - always fillet welds + 1..3 attachments (also lugs, ribs, gusset plates)
#   - at most one hole (PointGrid supports only one hole contour)
ASM_P_HOLE = 0.5               # probability of drilled hole/slot
ASM_P_SECOND = 0.5             # probability of second section
ASM_SECOND_W = (0.3, 0.8)      # second section width / edge length
ASM_WELD_P = 0.75
ASM_WELD_LEG = (4.0, 14.0)
ASM_EXTRA_N = (1, 3)
ASM_LUG_W = (15.0, 60.0)       # lug/rib: width along the edge [mm]
ASM_LUG_H = (10.0, 40.0)       # lug/rib: projection [mm]
ASM_GUSSET_LEG = (15.0, 50.0)  # gusset plate in an inner corner [mm]
ASM_HOLE_R = (3.0, 12.0)       # drilled hole radius [mm]
ASM_SLOT_P = 0.35              # share of slots instead of drilled holes
ASM_SLOT_L = (10.0, 40.0)      # slot: center distance [mm]
ASM_LIGAMENT = 5.0             # minimum ligament hole <-> outer contour [mm]

# Family IDs (GroupKFold by shape family); only append new families
FAMILIES = {"flat": 0, "tprofile": 1, "hbeam": 2, "holland": 3, "real": 4,
            "angle": 5, "channel": 6, "attached": 7, "assembly": 8}

# ---------------------------------------------------------------------------
# Grid construction from Shapely polygon (mirrors geometry_processor)
# ---------------------------------------------------------------------------

def _densify_ring(points: np.ndarray, max_dist: float) -> np.ndarray:
    """Densifies a closed ring so that no edge is longer than
    ``max_dist`` (cyclic).
    """
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
    """Uniform point grid inside ``poly`` (vectorized)."""
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
    """``PointGrid`` from a Shapely polygon.

    Point order as expected by ``PointGrid``/``SegmentedContour``:
    outer contour -> (at most one) hole contour -> inner grid.
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
# Catalog shapes (cross-sections)
# ---------------------------------------------------------------------------

def _u(rng, lo_hi) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def shape_flat(rng) -> Polygon:
    """Flat bar DIN 59200: rectangle b x t."""
    w, t = _u(rng, FLAT_WIDTH), _u(rng, FLAT_THICK)
    return box(-w / 2, -t / 2, w / 2, t / 2)


def shape_angle(rng) -> Polygon:
    """Angle section DIN EN 10056-1: legs a (horizontal), b (vertical),
    thickness t; half of the instances equal-leg.
    """
    a = _u(rng, L_A)
    ratio = 1.0 if rng.random() < 0.5 else _u(rng, L_RATIO)
    b = a * ratio
    t = max(3.0, a * _u(rng, L_T_REL))
    return unary_union([box(0.0, 0.0, a, t), box(0.0, 0.0, t, b)])


def shape_tprofile(rng) -> Polygon:
    """T-section EN 10055: flange width = height = h, thickness t."""
    h = _u(rng, T_H)
    t = max(3.0, h * _u(rng, T_T_REL))
    web = box(-t / 2, 0.0, t / 2, h - t)
    flange = box(-h / 2, h - t, h / 2, h)
    return unary_union([web, flange])


def shape_channel(rng) -> Polygon:
    """Channel UPN/U/CH (DIN EN 10365): web (height h, thickness tw) with two
    flanges (width b, thickness tf) toward +x."""
    h = _u(rng, U_H)
    b = h * _u(rng, U_B_REL) + 20.0
    tw = max(4.0, h * _u(rng, U_TW_REL))
    tf = max(5.0, h * _u(rng, U_TF_REL))
    web = box(0.0, 0.0, tw, h)
    bot = box(0.0, 0.0, b, tf)
    top = box(0.0, h - tf, b, h)
    return unary_union([web, bot, top])


def shape_holland(rng) -> Polygon:
    """Bulb flat DIN EN 10067 (HP section).

    - web: rectangle h x t, side x = -t/2 straight
    - bulb: wedge at the top edge, toward +x only (up to x = t/2 + w),
      tip rounded with a circular arc
    """
    h, t = _u(rng, HP_H), _u(rng, HP_T)
    w = t * _u(rng, HP_W)                  # one-sided projection
    hb = h * _u(rng, HP_HB)                # height of the bulb root
    web = box(-t / 2, 0.0, t / 2, h)
    tip = np.array([t / 2 + w, h])
    u_top = np.array([-1.0, 0.0])                    # along the top edge
    hyp_len = float(np.hypot(w, hb))
    u_hyp = np.array([-w, -hb]) / hyp_len            # along the hypotenuse
    theta = float(np.arccos(np.clip(np.dot(u_top, u_hyp), -1.0, 1.0)))
    tau = 0.30 * min(w, hyp_len)                     # tangent length
    rr = tau * np.tan(theta / 2.0)                   # rounding radius
    bis = u_top + u_hyp
    bis /= np.linalg.norm(bis)
    ctr = tip + bis * (rr / np.sin(theta / 2.0))     # arc center
    t1 = tip + u_top * tau                           # tangent point, top edge
    t2 = tip + u_hyp * tau                           # tangent point, hypotenuse
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
    """H-beam HEA/HEB (DIN EN 10365): height h, flange width b, web tw,
    flange tf; flanges at top and bottom."""
    h = _u(rng, H_H)
    b = h * _u(rng, H_B_REL)
    tw = max(5.0, h * _u(rng, H_TW_REL))
    tf = max(8.0, h * _u(rng, H_TF_REL))
    bot = box(-b / 2, 0.0, b / 2, tf)
    web = box(-tw / 2, tf, tw / 2, h - tf)
    top = box(-b / 2, h - tf, b / 2, h)
    return unary_union([bot, web, top])


_BASE_BUILDERS = None   # set below (after the base shapes)


def _ring_ccw(poly: Polygon) -> np.ndarray:
    """Exterior ring counterclockwise, without duplicate closing point."""
    ring = np.asarray(orient(poly, sign=1.0).exterior.coords, dtype=float)[:-1]
    return ring


def _attach_on_edge(rng, ring: np.ndarray, kind: str,
                    w_rng=ATT_W, h_rng=ATT_H) -> Polygon | None:
    """Small rectangle or semicircle on an outer edge (``w_rng``,
    ``h_rng``: rectangle dimensions, larger for "assembly").
    """
    n = len(ring)
    edges = [(i, float(np.linalg.norm(ring[(i + 1) % n] - ring[i]))) for i in range(n)]
    if kind == "round":
        r = _u(rng, ATT_R)
        need = 2.0 * r + 6.0
    else:
        w = _u(rng, w_rng)
        need = w + 6.0
    long_edges = [i for i, L in edges if L >= need]
    if not long_edges:
        return None
    i = int(rng.choice(long_edges))
    p0, p1 = ring[i], ring[(i + 1) % n]
    d = p1 - p0
    L = float(np.linalg.norm(d))
    d = d / L
    nrm = np.array([d[1], -d[0]])          # for a CCW ring, (dy,-dx) points outward
    if kind == "round":
        s0 = _u(rng, (r + 3.0, L - r - 3.0))
        centre = p0 + d * s0
        return Point(tuple(centre)).buffer(r, quad_segs=6)
    s0 = _u(rng, (3.0, L - w - 3.0))
    a = p0 + d * s0
    b = a + d * w
    h = _u(rng, h_rng)
    inset = nrm * 1.0                      # 1 mm into material (union stays connected)
    return Polygon([tuple(a - inset), tuple(b - inset), tuple(b + nrm * h), tuple(a + nrm * h)])


def _weld_fillets(rng, ring: np.ndarray, p_weld: float = ATT_WELD_P,
                  leg_rng=ATT_WELD_LEG) -> list[Polygon]:
    """Fillet weld triangles in the inner corners, each corner with probability
    ``p_weld``.
    """
    n = len(ring)
    out = []
    for i in range(n):
        c = ring[i]
        e_prev = ring[(i - 1) % n] - c          # back along the previous edge
        e_next = ring[(i + 1) % n] - c
        cross = float((-e_prev[0]) * e_next[1] - (-e_prev[1]) * e_next[0])
        if cross >= -1e-9:                      # not a re-entrant corner
            continue
        if rng.random() > p_weld:
            continue
        leg = _u(rng, leg_rng)
        l1 = min(leg, 0.45 * float(np.linalg.norm(e_prev)))
        l2 = min(leg, 0.45 * float(np.linalg.norm(e_next)))
        u1 = e_prev / float(np.linalg.norm(e_prev))
        u2 = e_next / float(np.linalg.norm(e_next))
        bis = -(u1 + u2)
        bis = bis / max(1e-9, float(np.linalg.norm(bis)))
        out.append(Polygon([tuple(c + bis * 1.0), tuple(c + u1 * l1), tuple(c + u2 * l2)]))
    return out


def shape_attached(rng) -> Polygon:
    """Base section + fillet welds in the inner corners + 0..2 semicircles or
    rectangles on outer edges.
    """
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
    return Polygon(merged.exterior)         # no enclosed voids


def _inner_corners(ring: np.ndarray) -> list[int]:
    """Indices of the re-entrant corners of a CCW ring."""
    n = len(ring)
    out = []
    for i in range(n):
        a = ring[i] - ring[(i - 1) % n]
        b = ring[(i + 1) % n] - ring[i]
        if float(a[0] * b[1] - a[1] * b[0]) < -1e-9:
            out.append(i)
    return out


def _gusset(rng, ring: np.ndarray) -> Polygon | None:
    """Gusset plate: larger triangle in a random inner corner."""
    corners = _inner_corners(ring)
    if not corners:
        return None
    n = len(ring)
    i = int(rng.choice(corners))
    c = ring[i]
    e1, e2 = ring[(i - 1) % n] - c, ring[(i + 1) % n] - c
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    leg = _u(rng, ASM_GUSSET_LEG)
    a, b = min(leg, 0.7 * l1), min(leg * _u(rng, (0.6, 1.0)), 0.7 * l2)
    if a < 3.0 or b < 3.0:
        return None
    u1, u2 = e1 / l1, e2 / l2
    bis = -(u1 + u2)
    bis = bis / max(1e-9, float(np.linalg.norm(bis)))
    return Polygon([tuple(c + bis * 1.0), tuple(c + u1 * a), tuple(c + u2 * b)])


def _second_profile(rng, ring: np.ndarray) -> Polygon | None:
    """Second section (flat, angle, T) welded onto an outer edge
    (width ``ASM_SECOND_W`` of the edge, 1 mm into the material).
    """
    n = len(ring)
    edges = [(i, float(np.linalg.norm(ring[(i + 1) % n] - ring[i])))
             for i in range(n)]
    long_edges = [i for i, L in edges if L >= 20.0]
    if not long_edges:
        return None
    i = int(rng.choice(long_edges))
    p0, p1 = ring[i], ring[(i + 1) % n]
    L = float(np.linalg.norm(p1 - p0))
    d = (p1 - p0) / L
    nrm = np.array([d[1], -d[0]])            # CCW ring: outward
    kind = str(rng.choice(["flat", "angle", "tprofile"]))
    sec = _largest_polygon(_BASE_BUILDERS[kind](rng).buffer(0))
    k = int(rng.integers(0, 4))              # orientation: 0/90/180/270 deg
    sec = shapely.affinity.rotate(sec, 90.0 * k, origin=(0.0, 0.0))
    if rng.random() < 0.5:
        sec = shapely.affinity.scale(sec, xfact=-1.0, origin=(0.0, 0.0))
    x0, y0, x1, y1 = sec.bounds
    s = min(1.0, L * _u(rng, ASM_SECOND_W) / max(1e-9, x1 - x0))
    sec = shapely.affinity.scale(sec, xfact=s, yfact=s, origin=(x0, y0))
    x0, y0, x1, y1 = sec.bounds
    off = _u(rng, (0.0, max(0.0, L - (x1 - x0))))
    base = p0 + d * off - nrm * 1.0
    # local (x, y) -> base + x * d + y * nrm
    loc = np.asarray(sec.exterior.coords, dtype=float) - [x0, y0]
    world = base + loc[:, :1] * d + loc[:, 1:2] * nrm
    return Polygon(world).buffer(0)


def _cut_hole(rng, poly: Polygon) -> Polygon:
    """Drilled hole or slot with minimum ligament ``ASM_LIGAMENT``; if
    nothing fits, the section stays without a hole.
    """
    slot = rng.random() < ASM_SLOT_P
    r = _u(rng, ASM_HOLE_R)
    half = 0.5 * _u(rng, ASM_SLOT_L) if slot else 0.0
    ang = (float(rng.choice([0.0, 90.0])) if rng.random() < 0.7
           else _u(rng, (0.0, 180.0)))
    for shrink in (1.0, 0.7, 0.5):
        rr, hh = max(3.0, r * shrink), half * shrink
        region = poly.buffer(-(rr + hh + ASM_LIGAMENT))
        if region.is_empty or region.area < 1e-6:
            continue
        xmin, ymin, xmax, ymax = region.bounds
        for _ in range(200):
            x, y = _u(rng, (xmin, xmax)), _u(rng, (ymin, ymax))
            if not region.contains(Point(x, y)):
                continue
            if hh > 0:
                t = np.radians(ang)
                dd = np.array([np.cos(t), np.sin(t)]) * hh
                c = np.array([x, y])
                hole = shapely.LineString([tuple(c - dd), tuple(c + dd)]).buffer(
                    rr, quad_segs=8)
            else:
                hole = Point(x, y).buffer(rr, quad_segs=8)
            return poly.difference(hole)
    return poly


def shape_assembly(rng) -> Polygon:
    """Welded assembly: base section + hole and/or second section +
    fillet welds + 1..3 attachments (semicircle, rectangle, lug/rib,
    gusset plate), see ASM_*.
    """
    base_name = str(rng.choice(list(_BASE_BUILDERS)))
    poly = _largest_polygon(_BASE_BUILDERS[base_name](rng).buffer(0))
    with_hole = rng.random() < ASM_P_HOLE
    with_second = rng.random() < ASM_P_SECOND
    if not (with_hole or with_second):
        if rng.random() < 0.5:
            with_hole = True
        else:
            with_second = True
    parts = [poly]
    if with_second:
        sec = _second_profile(rng, _ring_ccw(poly))
        if sec is not None and not sec.is_empty:
            parts.append(sec)
    cur = _largest_polygon(unary_union(parts).buffer(0))
    parts = [cur] + _weld_fillets(rng, _ring_ccw(cur), ASM_WELD_P, ASM_WELD_LEG)
    n_extra = int(rng.integers(ASM_EXTRA_N[0], ASM_EXTRA_N[1] + 1))
    for _ in range(n_extra):
        ring = _ring_ccw(_largest_polygon(unary_union(parts).buffer(0)))
        kind = str(rng.choice(["round", "rect", "lug", "gusset"]))
        if kind == "gusset":
            piece = _gusset(rng, ring)
        elif kind == "lug":
            piece = _attach_on_edge(rng, ring, "rect", ASM_LUG_W, ASM_LUG_H)
        else:
            piece = _attach_on_edge(rng, ring, kind)
        if piece is not None and not piece.is_empty:
            parts.append(piece)
    merged = _largest_polygon(unary_union(parts).buffer(0))
    out = Polygon(merged.exterior)           # drop enclosed voids
    if with_hole:
        out = _cut_hole(rng, out)
    return out


_BASE_BUILDERS = {
    "flat": shape_flat,
    "angle": shape_angle,
    "tprofile": shape_tprofile,
    "channel": shape_channel,
    "holland": shape_holland,
    "hbeam": shape_hbeam,
}
SHAPE_BUILDERS = {**_BASE_BUILDERS, "attached": shape_attached,
                  "assembly": shape_assembly}
# Catalog families (each once, for figures/evaluation): append ONLY.
SHAPE_ORDER = ["flat", "angle", "tprofile", "channel", "holland", "hbeam",
               "attached", "assembly"]
# Round-robin slots of ``make_catalog_instance``: "assembly" in 3 of 10 slots,
# since many assembly instances exceed k_max -> in the dataset about twice
# as often as any other family
CATALOG_SLOTS = SHAPE_ORDER + ["assembly", "assembly"]


def tag_instance(grid: PointGrid, family: str, name: str) -> PointGrid:
    grid._family = FAMILIES[family]      # additive attributes (internal only)
    grid._family_name = family
    grid._instance_name = name
    return grid


def load_test_geometries() -> list[PointGrid]:
    """The verified test geometries (family "real")."""
    out: list[PointGrid] = []
    for p in sorted(TEST_GEOMETRY_DIR.glob("*.json")):
        try:
            g = PointGrid.from_json(p)
        except Exception:
            continue
        out.append(tag_instance(g, "real", p.stem))
    return out


def make_catalog_instance(idx: int, seed: int = 0) -> PointGrid | None:
    """Catalog instance ``idx``, deterministic and stable.

    - depends only on (seed, idx, SHAPE_VERSION), not on n or
      LABEL_VERSION
    - generating more instances does not change existing ones (resumable
      label cache)
    """
    fam = CATALOG_SLOTS[idx % len(CATALOG_SLOTS)]
    rng = np.random.default_rng(
        (int(seed) & 0xFFFFFFFF) * 1_000_003 + idx * 131 + SHAPE_VERSION)
    poly = SHAPE_BUILDERS[fam](rng)
    grid = grid_from_polygon(poly)
    if grid is None:
        return None
    grid._catalog_idx = int(idx)          # for dataset.spec_of_grid
    return tag_instance(grid, fam, f"{fam}_{idx:05d}")


def generate_instances(n: int, seed: int = 0,
                       include_real: bool = True) -> list[PointGrid]:
    """``n`` stable catalog instances (round-robin over ``CATALOG_SLOTS``) +
    optionally the test geometries (family "real").
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
    """Default cutter of the simulator (lazy, avoids the import cycle
    surrogate <-> simulation).
    """
    try:
        from ..simulation import make_default_cutter
    except ImportError:
        from plasma_cutter.segment_simulation.simulation import (
            make_default_cutter,
        )
    return make_default_cutter()

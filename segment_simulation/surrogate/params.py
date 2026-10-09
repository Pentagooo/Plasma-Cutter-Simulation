"""Parameter stamp for labels, dataset and model.

``LabelParams`` collects everything that affects the teacher's evaluation
(physics, kerf, sampling, point density, segmentation, catalog) from the
modules where it is defined. Two hashes of it are stored in every label
(``params_hash`` also in its file name), in ``dataset_meta.json`` and in
the model:

  phys_hash   : physics + kerf + sampling + corner angle; must match for
                labels, model and code
  params_hash : additionally segmentation, point density, catalog; only a
                warning when loading a model

The hash detects parameter changes; changes to the pipeline itself
(time model, evaluation) -> increment ``LABEL_VERSION``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

try:
    from ..planning import TCP_SAMPLE_STEP
    from ..segments import SegmentedContour
    from .features import V_MIN_DEFAULT, phys_from_cutter
    from .instances import (
        CONTOUR_SPACING, GRID_SPACING, SHAPE_VERSION, default_cutter,
    )
except ImportError:  # direct run without package context
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plasma_cutter.segment_simulation.planning import TCP_SAMPLE_STEP
    from plasma_cutter.segment_simulation.segments import SegmentedContour
    from plasma_cutter.segment_simulation.surrogate.features import (
        V_MIN_DEFAULT, phys_from_cutter,
    )
    from plasma_cutter.segment_simulation.surrogate.instances import (
        CONTOUR_SPACING, GRID_SPACING, SHAPE_VERSION, default_cutter,
    )

# Label pipeline version: increment if the same parameters yield different
# labels or features (v2: parameter stamp, v3: nested TCP sampling)
LABEL_VERSION = 3

# Kerf width [mm]
KERF = 3.0

# Segmentation: target length = perimeter / SEG_DIVISOR,
# at least SEG_MIN_SPACINGS * point spacing
SEG_DIVISOR_DEFAULT = 12.0
SEG_MIN_SPACINGS_DEFAULT = 4.0
CORNER_ANGLE_DEG = 30.0

_PHYS_FIELDS = (
    "v_cut", "v_max", "v_min", "blade0", "slope", "gap", "t_switch",
    "rapid_speed", "pierce_time", "kerf", "tcp_sample_step",
    "corner_angle_deg",
)


@dataclass(frozen=True)
class LabelParams:
    """All label-relevant values (see module docstring)."""
    label_version: int
    # cutter physics
    v_cut: float
    v_max: float
    v_min: float
    blade0: float
    slope: float
    gap: float
    t_switch: float
    rapid_speed: float
    pierce_t0: float
    pierce_k: float
    sheet_thickness: float
    pierce_time: float
    kerf: float
    tcp_sample_step: float
    # point density / catalog
    contour_spacing: float
    grid_spacing: float
    shape_version: int
    # segmentation
    corner_angle_deg: float
    seg_divisor: float
    seg_min_spacings: float


def label_params(seg_divisor: float = SEG_DIVISOR_DEFAULT,
                 seg_min_spacings: float = SEG_MIN_SPACINGS_DEFAULT,
                 cutter=None) -> LabelParams:
    """Current values from the code (default cutter, planning, catalog
    and segmentation constants).
    """
    if cutter is None:
        cutter = default_cutter()
    phys = phys_from_cutter(cutter)
    a = cutter.assumptions
    return LabelParams(
        label_version=LABEL_VERSION,
        v_cut=float(phys.v_cut), v_max=float(phys.v_max), v_min=float(V_MIN_DEFAULT),
        blade0=float(phys.blade0), slope=float(phys.slope), gap=float(phys.gap),
        t_switch=float(phys.t_switch), rapid_speed=float(cutter.rapid_speed),
        pierce_t0=float(a.pierce.t0), pierce_k=float(a.pierce.k),
        sheet_thickness=float(a.sheet_thickness),
        pierce_time=float(cutter.pierce_time()),
        kerf=float(KERF), tcp_sample_step=float(TCP_SAMPLE_STEP),
        contour_spacing=float(CONTOUR_SPACING), grid_spacing=float(GRID_SPACING),
        shape_version=int(SHAPE_VERSION),
        corner_angle_deg=float(CORNER_ANGLE_DEG),
        seg_divisor=float(seg_divisor), seg_min_spacings=float(seg_min_spacings),
    )


def _canon(v):
    """Round floats to 9 decimal places: floating-point noise does not
    change the hash.
    """
    if isinstance(v, float):
        v = round(v, 9)
        return 0.0 if v == 0 else v
    return v


def _digest(d: dict) -> str:
    canon = {k: _canon(v) for k, v in d.items()}
    return hashlib.sha1(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]


def phys_hash(p: LabelParams) -> str:
    """8 hex characters over physics + kerf + sampling + corner angle."""
    d = asdict(p)
    return _digest({k: d[k] for k in _PHYS_FIELDS})


def params_hash(p: LabelParams) -> str:
    """8 hex characters over ALL fields (incl. segmentation, catalog)."""
    return _digest(asdict(p))


def stamp(p: LabelParams) -> dict:
    """Stamp for dataset_meta.json / model_meta.json / joblib."""
    return {"label_version": int(p.label_version), "phys_hash": phys_hash(p),
            "params_hash": params_hash(p), "params": asdict(p)}


def from_stamp(found: dict) -> LabelParams | None:
    """LabelParams from a stored stamp (None if incomplete)."""
    d = found.get("params") if isinstance(found, dict) else None
    if not isinstance(d, dict):
        return None
    try:
        return LabelParams(**{f.name: d[f.name] for f in fields(LabelParams)})
    except (KeyError, TypeError):
        return None


def check_stamp(found: dict, expected: LabelParams, what: str,
                strict_seg: bool = True) -> None:
    """Compares a stored stamp with ``expected``.

    - missing stamp, different label_version or phys_hash -> SystemExit
      with a list of the differing fields
    - only params_hash differs: error with ``strict_seg``, otherwise warning
      (e.g. finely segmented model in the simulator)
    """
    found = found or {}
    lv = found.get("label_version", -1)
    ph = found.get("phys_hash")
    pa = found.get("params_hash")
    if int(lv) != int(expected.label_version) or ph is None or pa is None:
        raise SystemExit(
            f"FEHLER: {what} hat label_version={lv}, phys_hash={ph} -- "
            f"erwartet LABEL_VERSION={expected.label_version} mit Stempel. "
            f"Neu labeln und trainieren.")
    if ph == phys_hash(expected) and pa == params_hash(expected):
        return
    old = from_stamp(found)
    diff = []
    if old is not None:
        for f in fields(LabelParams):
            a, b = getattr(old, f.name), getattr(expected, f.name)
            if a != b:
                diff.append(f"{f.name}: {a} (gespeichert) != {b} (Code)")
    diff_txt = ("\n  " + "\n  ".join(diff)) if diff else " (Details unbekannt)"
    if ph != phys_hash(expected):
        raise SystemExit(
            f"FEHLER: {what} passt nicht zur Physik im Code "
            f"(phys_hash {ph} != {phys_hash(expected)}):{diff_txt}\n"
            f"Entweder die Konstanten in simulation.py zuruecksetzen oder "
            f"neu labeln und trainieren.")
    msg = (f"{what}: andere Segmentierung/Punktdichte/Katalog als der Code "
           f"(params_hash {pa} != {params_hash(expected)}):{diff_txt}")
    if strict_seg:
        raise SystemExit("FEHLER: " + msg)
    print("WARNUNG: " + msg)


def make_contour(grid, p: LabelParams) -> SegmentedContour:
    """Segmentation exactly as the labels of ``p`` assume it."""
    return SegmentedContour.from_grid(
        grid, corner_angle_deg=p.corner_angle_deg,
        segment_divisor=p.seg_divisor, seg_min_spacings=p.seg_min_spacings)


if __name__ == "__main__":
    p = label_params()
    print(json.dumps(stamp(p), indent=2))

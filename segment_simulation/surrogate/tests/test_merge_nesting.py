"""Verschachtelte TCP-Abtastung: ein verschmolzener Run deckt mindestens
die Vereinigung seiner Einzelsegmente ab.

Sonst kann ein einzelner Abtastpunkt am Knoten den coverage-erhaltenden
Merge kippen (Fallback auf Einzelruns = eine Zuendung je Segment). Geprueft
fuer alle zusammenhaengenden Gruppen der Laenge 2 und 3 sowie den vollen
Loop, auf dem kleinen Flachstahl und auf kontur.json.
"""
from __future__ import annotations

import pytest

from plasma_cutter.segment_simulation.planning import RunKinematics
from plasma_cutter.segment_simulation.segments import (
    SegmentedContour, compute_grid_coverage,
)
from plasma_cutter.segment_simulation.simulation import make_default_cutter
from ._helpers import real_geometries, small_flat_bar


def _cases():
    yield "small_flat_bar", small_flat_bar()
    for g in real_geometries():
        if getattr(g, "_source_path", None) is not None and g._source_path.stem == "kontur":
            yield "kontur", g


@pytest.mark.parametrize("name,grid", list(_cases()))
def test_merged_run_covers_union_of_singletons(name, grid):
    contour = SegmentedContour.from_grid(grid)
    material = contour.material_polygon()
    cutter = make_default_cutter()
    kin = RunKinematics(material, clearance=cutter.minimum_gap,
                        blade_length=cutter.blade_length(), kerf=3.0)
    n_checked = 0
    for loop_id in contour.nodes:
        segs = [s for s in contour.segments if s.loop_id == loop_id]
        for i in range(len(segs)):
            for size in (2, 3, len(segs)):
                grp = [segs[(i + j) % len(segs)] for j in range(min(size, len(segs)))]
                full = len(grp) == len(segs)
                merged = kin.attach(contour.make_run(
                    1, loop_id, grp[0].start_pos,
                    grp[0].start_pos if full else grp[-1].end_pos))
                singles = [kin.attach(contour.make_run(2 + j, loop_id,
                                                       s.start_pos, s.end_pos))
                           for j, s in enumerate(grp)]
                if not merged.is_feasible or any(not r.is_feasible for r in singles):
                    continue
                m = compute_grid_coverage(grid, [merged]).mask
                u = compute_grid_coverage(grid, singles).mask
                lost = int((u & ~m).sum())
                assert lost == 0, (f"{name}: Merge {[s.seg_id for s in grp]} "
                                   f"verliert {lost} Punkte gegenueber den Einzelsegmenten")
                n_checked += 1
                if full:
                    break
    assert n_checked > 0

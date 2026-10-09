"""Tests of the DP split stage (speed rule + t_switch).

Covered:
  * Edge cases: t_switch = inf -> 1 block (old merge behavior);
    t_switch = 0 -> fully split time (sum len_i / v_i).
  * Optimality: DP result == brute-force enumeration of all
    2^(m-1) partitions of small chains (m <= 10).
  * Coverage preservation: the exactly verified sub-runs of a chain
    (``build_speed_chains``) cover the singleton masks of the selection.
  * Time breakdown: total = cut + travel + pierce + switch;
    switch_time = t_switch * n_switches.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from plasma_cutter.segment_simulation.segments import (
    SegmentedContour, compute_grid_coverage,
)
from plasma_cutter.segment_simulation.planning import LinkPlanner
from plasma_cutter.segment_simulation.surrogate.instances import default_cutter
from plasma_cutter.segment_simulation.surrogate.features import phys_from_cutter
from plasma_cutter.segment_simulation.surrogate.runutils import (
    ExactSequencer, build_plan_with_speeds, build_singletons,
    build_speed_chains, clamp_rule_speed, split_group_for_speed,
)
from ._helpers import small_flat_bar

PHYS = phys_from_cutter(default_cutter())


def _rand_chain(rng, m):
    lengths = rng.uniform(10.0, 60.0, m).tolist()
    depths = rng.uniform(5.0, 18.0, m).tolist()
    return lengths, depths


def _partition_cost(lengths, depths, cuts, t_switch):
    """Cost of an explicit partition (cuts = split points 1..m-1)."""
    bounds = [0, *sorted(cuts), len(lengths)]
    total = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = clamp_rule_speed(PHYS, max(depths[a:b]))
        total += sum(lengths[a:b]) / v
    return total + t_switch * (len(bounds) - 2)


def test_tswitch_inf_is_single_block():
    """t_switch = inf => old merge behavior (exactly 1 block)."""
    rng = np.random.default_rng(1)
    for m in (1, 3, 7):
        lengths, depths = _rand_chain(rng, m)
        blocks, speeds, total = split_group_for_speed(
            lengths, depths, PHYS, math.inf)
        assert blocks == [(0, m)]
        assert len(speeds) == 1
        v = clamp_rule_speed(PHYS, max(depths))
        assert total == pytest.approx(sum(lengths) / v)


def test_tswitch_zero_is_fully_split():
    """t_switch = 0 => time of the complete decomposition (each segment
    runs at its own speed); equally fast neighbors may be combined
    by the tie-break (same time)."""
    rng = np.random.default_rng(2)
    for m in (2, 5, 9):
        lengths, depths = _rand_chain(rng, m)
        _, _, total = split_group_for_speed(lengths, depths, PHYS, 0.0)
        t_full = sum(l / clamp_rule_speed(PHYS, d)
                     for l, d in zip(lengths, depths))
        assert total == pytest.approx(t_full, abs=1e-9)


def test_dp_matches_bruteforce_enumeration():
    """Optimality of the partition against complete enumeration
    (m <= 10, several t_switch values)."""
    rng = np.random.default_rng(3)
    for m in range(2, 11):
        lengths, depths = _rand_chain(rng, m)
        for t_switch in (0.0, 0.5, 2.0, 10.0):
            _, _, total = split_group_for_speed(
                lengths, depths, PHYS, t_switch)
            best = math.inf
            for k in range(m):
                for cuts in itertools.combinations(range(1, m), k):
                    best = min(best, _partition_cost(
                        lengths, depths, cuts, t_switch))
            assert total == pytest.approx(best, abs=1e-9), (
                f"m={m}, t_switch={t_switch}: DP {total} != BF {best}")


def test_dp_monotone_in_t_switch():
    """A higher t_switch must never lower the optimal time."""
    rng = np.random.default_rng(4)
    lengths, depths = _rand_chain(rng, 8)
    totals = [split_group_for_speed(lengths, depths, PHYS, ts)[2]
              for ts in (0.0, 0.5, 1.0, 5.0, 50.0, math.inf)]
    assert all(a <= b + 1e-9 for a, b in zip(totals[:-1], totals[1:]))


def _setup_geometry():
    grid = small_flat_bar()
    cutter = default_cutter()
    phys = phys_from_cutter(cutter)
    contour = SegmentedContour.from_grid(grid)
    material = contour.material_polygon()
    coords = np.asarray(grid.coords, dtype=float)
    return grid, cutter, phys, contour, material, coords


def test_chain_coverage_preserved_after_split():
    """The exactly verified sub-runs cover the singleton masks of the
    selection (promised coverage is preserved after the split)."""
    grid, cutter, phys, contour, material, coords = _setup_geometry()
    selected = [s.seg_id for s in contour.segments]
    seg_run, seg_mask = build_singletons(
        contour, selected, material, phys, phys.v_cut, 3.0, coords)
    promised = np.zeros(len(coords), dtype=bool)
    for s in selected:
        promised |= seg_mask[s]

    chains = build_speed_chains(contour, selected, material, phys, 3.0,
                                coords, seg_run=seg_run, seg_mask=seg_mask)
    assert chains, "keine Ketten gebaut"
    runs = [r for ch in chains for r in ch.sub_runs]
    mask = compute_grid_coverage(grid, runs).mask
    assert bool(np.all(promised <= mask)), (
        "Split verliert zugesagte Singleton-Coverage")


def test_plan_accounts_switch_time():
    """SpeedPlan reports t_switch separately; the time breakdown is consistent
    and the macro-node invariant holds (sub-runs > chains, pierces = chains)."""
    grid, cutter, phys, contour, material, coords = _setup_geometry()
    selected = [s.seg_id for s in contour.segments]
    t_switch = 3.0
    chains = build_speed_chains(contour, selected, material, phys, 3.0,
                                coords, t_switch=t_switch)
    seq = ExactSequencer(cutter, contour,
                         LinkPlanner(material, clearance=phys.gap))
    plan = build_plan_with_speeds(seq, chains, {}, cutter,
                                  drop_unlinkable=True, t_switch=t_switch)
    assert plan.ordered_runs
    assert plan.total_time == pytest.approx(
        plan.cut_time + plan.travel_time + plan.pierce_time
        + plan.switch_time)
    assert plan.switch_time == pytest.approx(t_switch * plan.n_switches)
    # No pierce between sub-runs of the same chain: pierces <= chains
    assert plan.n_pierces <= len(chains)
    # Speed changes only at sub-run boundaries
    assert plan.n_switches <= max(0, len(plan.ordered_runs) - len(chains))


def test_speed_rule_off_collapses_to_base():
    """v_max = v_cut (rule off) => 1 block per chain, all v = v_cut,
    no speed changes."""
    from dataclasses import replace
    grid, cutter, phys, contour, material, coords = _setup_geometry()
    phys_off = replace(phys, v_max=phys.v_cut)
    selected = [s.seg_id for s in contour.segments]
    chains = build_speed_chains(contour, selected, material, phys_off, 3.0,
                                coords)
    for ch in chains:
        assert all(abs(v - phys.v_cut) < 1e-9 for v in ch.speeds)
    lengths = [40.0, 50.0]
    depths = [8.0, 17.0]
    blocks, speeds, _ = split_group_for_speed(lengths, depths, phys_off, 0.0)
    assert blocks == [(0, 2)] and speeds == [phys.v_cut]

"""Tests der DP-Split-Stufe (Kap. 4.5 + t_switch).

Abgedeckt:
  * Grenzfaelle: t_switch = inf -> 1 Block (altes Merge-Verhalten);
    t_switch = 0 -> voll gesplittete Zeit (Summe len_i / v_i).
  * Optimalitaet: DP-Ergebnis == Brute-Force-Enumeration aller
    2^(m-1) Partitionen kleiner Ketten (m <= 10).
  * Coverage-Erhalt: die exakt verifizierten Sub-Runs einer Kette
    (``build_speed_chains``) decken die Singleton-Masken der Auswahl ab.
  * Zeitbilanz: total = cut + travel + pierce + switch;
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
    """Kosten einer expliziten Partition (cuts = Schnittstellen 1..m-1)."""
    bounds = [0, *sorted(cuts), len(lengths)]
    total = 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = clamp_rule_speed(PHYS, max(depths[a:b]))
        total += sum(lengths[a:b]) / v
    return total + t_switch * (len(bounds) - 2)


def test_tswitch_inf_is_single_block():
    """t_switch = inf => altes Merge-Verhalten (genau 1 Block)."""
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
    """t_switch = 0 => Zeit der vollstaendigen Zerlegung (jedes Segment
    faehrt seine eigene Geschwindigkeit); gleich schnelle Nachbarn duerfen
    per Tie-Break zusammengefasst sein (gleiche Zeit)."""
    rng = np.random.default_rng(2)
    for m in (2, 5, 9):
        lengths, depths = _rand_chain(rng, m)
        _, _, total = split_group_for_speed(lengths, depths, PHYS, 0.0)
        t_full = sum(l / clamp_rule_speed(PHYS, d)
                     for l, d in zip(lengths, depths))
        assert total == pytest.approx(t_full, abs=1e-9)


def test_dp_matches_bruteforce_enumeration():
    """Optimalitaet der Partition gegen vollstaendige Enumeration
    (m <= 10, mehrere t_switch-Werte)."""
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
    """Hoeheres t_switch darf die optimale Zeit nie senken."""
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
    """Die exakt verifizierten Sub-Runs decken die Singleton-Masken der
    Auswahl ab (zugesagte Coverage bleibt nach dem Split erhalten)."""
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
    """SpeedPlan weist t_switch separat aus; Bilanz ist konsistent und
    Makro-Knoten-Invariante haelt (Sub-Runs > Ketten, Pierces = Ketten)."""
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
    # Kein Pierce zwischen Sub-Runs derselben Kette: Zuendungen <= Ketten
    assert plan.n_pierces <= len(chains)
    # Geschwindigkeitswechsel nur an Sub-Run-Grenzen
    assert plan.n_switches <= max(0, len(plan.ordered_runs) - len(chains))


def test_speed_rule_off_collapses_to_base():
    """v_max = v_cut (Regel aus) => 1 Block je Kette, alle v = v_cut,
    keine Wechsel."""
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

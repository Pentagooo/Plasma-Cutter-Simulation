from __future__ import annotations

"""Automatische Segmentwahl (Auto-Planer, Stufe 1).

Das Wunschziel: Geometrie rein, fertiger Schnittplan raus -- ohne dass
der Benutzer Segmente anklicken muss. Strukturell ist das ein
Covering-Tour-Problem (Set Cover + TSP, NP-hart); fuer die typische
Problemgroesse (K = 13..25 Primitiv-Segmente) genuegt ein Greedy-
Verfahren mit Pruning, um in deutlich unter 1 Sekunde einen
fast-optimalen Plan zu finden.

Ablauf (bewusst einfach gehalten)
---------------------------------
1. **Kandidaten + Abdeckbarkeits-Matrix**: Jedes automatisch erzeugte
   Primitiv-Segment wird EINMAL als CutRun durchgerechnet
   (``RunKinematics.attach`` -> Swept Area). Daraus entsteht die
   bool-Matrix ``A[p, s]`` = "Gitterpunkt p liegt in der Swept Area
   von Segment s". Danach ist jede Coverage-Frage eine reine
   numpy-Mengenoperation -- kein Shapely mehr im Optimierungs-Loop.
2. **Greedy Set-Cover**: Wiederholt das Segment mit dem besten
   Verhaeltnis "neue Punkte / Zusatzzeit" waehlen, bis keine neuen
   Punkte mehr erreichbar sind. Die Zusatzzeit beruecksichtigt das
   Verketten: schliesst ein Segment nahtlos an ein bereits gewaehltes
   Nachbarsegment an, entfaellt die Pierce-Pauschale.
   (Approximationsguete des Greedy-Set-Cover: ln(n) -- BA-Kap. 4.4.)
3. **Pruning**: Redundante Segmente (alle ihre Punkte sind auch von
   anderen gewaehlten Segmenten abgedeckt) werden wieder entfernt,
   teuerste zuerst. So muss z.B. ein duenner Steg nur von EINER
   Seite geschnitten werden.
4. **Verschmelzen + Sequencer**: Zusammenhaengende gewaehlte Segmente
   verschmelzen zu einem CutRun (eine Zuendung); der vorhandene
   Sequencer (Held-Karp, exakt bis 11 Runs) ordnet die Runs
   zeitminimal, LinkPlanner verbindet kollisionsfrei.

Punkte, die physikalisch unerreichbar sind (tiefer im Material als die
effektive Klingentiefe), werden ehrlich als ``n_unreachable`` gemeldet
-- genau wie bei der manuellen Auswahl.

Einstieg
--------
Headless:  ``result = auto_plan(grid)``  ->  AutoPlanResult
UI:        Taste **P** in der SegmentCutSimulation.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import shapely

try:
    from ..geometry.point_grid import PointGrid
    from ..cutter.cutter import Cutter
    from .segments import (
        SegmentedContour, CutRun, compute_grid_coverage, GridCoverageReport,
    )
    from .planning import (
        LinkPlanner, Sequencer, CutPlan, RunKinematics, compute_score,
    )
except ImportError:  # Direktstart ohne Paket-Kontext
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.geometry.point_grid import PointGrid
    from plasma_cutter.cutter.cutter import Cutter
    from plasma_cutter.segment_simulation.segments import (
        SegmentedContour, CutRun, compute_grid_coverage, GridCoverageReport,
    )
    from plasma_cutter.segment_simulation.planning import (
        LinkPlanner, Sequencer, CutPlan, RunKinematics, compute_score,
    )


# ---------------------------------------------------------------------------
# Ergebnis
# ---------------------------------------------------------------------------

@dataclass
class AutoPlanResult:
    """Ergebnis des Auto-Planers.

    Attributes
    ----------
    runs          : die automatisch gewaehlten (verschmolzenen) CutRuns
    plan          : zeitminimal geordneter Ausfuehrungsplan
    report        : Querschnitts-Coverage der gewaehlten Runs
    score         : Punktezahl (compute_score)
    elapsed       : Gesamtlaufzeit des Auto-Planers [s]
    n_candidates  : Anzahl Primitiv-Segmente (Kandidaten)
    n_selected    : Anzahl gewaehlter Primitiv-Segmente (nach Pruning)
    n_unreachable : Gitterpunkte, die KEIN Segment erreichen kann
                    (physikalisch unerreichbar, z.B. zu tief)
    """
    runs: list[CutRun] = field(default_factory=list)
    plan: CutPlan = field(default_factory=CutPlan)
    report: GridCoverageReport | None = None
    score: float = 0.0
    elapsed: float = 0.0
    n_candidates: int = 0
    n_selected: int = 0
    n_unreachable: int = 0

    def summary(self) -> str:
        lines = [
            f"Auto-Plan: {self.n_selected}/{self.n_candidates} Segmente "
            f"gewaehlt -> {len(self.runs)} Schnitt(e), "
            f"geplant in {self.elapsed:.2f} s",
        ]
        if self.n_unreachable:
            lines.append(
                f"  {self.n_unreachable} Punkt(e) physikalisch "
                f"unerreichbar (Klinge zu kurz)")
        if self.report is not None:
            lines.append("  " + self.report.summary().replace("\n", "\n  "))
        lines.append("  " + self.plan.summary())
        lines.append(f"  Punktezahl: {self.score:.0f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AutoPlanner
# ---------------------------------------------------------------------------

class AutoPlanner:
    """Automatische Segmentwahl per Greedy Set-Cover + Pruning.

    Parameters
    ----------
    grid       : PointGrid der Geometrie (liefert die Zielpunkte)
    contour    : SegmentedContour mit den Primitiv-Segmenten
    cutter     : Cutter (Zeiten fuer die Kostenfunktion)
    kinematics : RunKinematics (Swept Areas der Kandidaten)
    sequencer  : Sequencer fuer den finalen Plan
    """

    def __init__(
        self,
        grid: PointGrid,
        contour: SegmentedContour,
        cutter: Cutter,
        kinematics: RunKinematics,
        sequencer: Sequencer,
    ) -> None:
        self.grid = grid
        self.contour = contour
        self.cutter = cutter
        self.kinematics = kinematics
        self.sequencer = sequencer

        # Segment-Reihenfolge je Loop (fuer Nachbarschaft + Verschmelzen):
        # contour.segments ist je Loop in Knoten-Reihenfolge angelegt.
        self._loop_segs: dict[int, list[int]] = {}
        for seg in contour.segments:
            self._loop_segs.setdefault(seg.loop_id, []).append(seg.seg_id)

        self._build_candidates()

    # ------------------------------------------------------------------
    # Schritt 1: Kandidaten + Abdeckbarkeits-Matrix A[p, s]
    # ------------------------------------------------------------------

    def _build_candidates(self) -> None:
        """Rechnet jedes Primitiv-Segment einmal als CutRun durch und
        stempelt seine Swept Area in die Abdeckbarkeits-Matrix."""
        coords = self.grid.coords
        n_pts = len(coords)
        n_seg = len(self.contour.segments)

        self._A = np.zeros((n_pts, n_seg), dtype=bool)
        self._cut_time = np.full(n_seg, np.inf)
        self._feasible = np.zeros(n_seg, dtype=bool)
        self._cand_runs: list[CutRun | None] = [None] * n_seg

        for seg in self.contour.segments:
            loop = self.contour.loop_by_id(seg.loop_id)
            run = CutRun(
                run_id=seg.seg_id + 1,
                loop_id=seg.loop_id,
                start_pos=seg.start_pos,
                end_pos=seg.end_pos,
                direction=+1,
                positions=list(seg.positions),
                polyline=loop.polyline(seg.positions),
                length=seg.length,
            )
            self.kinematics.attach(run)
            if not run.is_feasible or run.swept_polygon is None:
                continue
            s = seg.seg_id
            self._feasible[s] = True
            self._cand_runs[s] = run
            self._A[:, s] = shapely.contains_xy(
                run.swept_polygon, coords[:, 0], coords[:, 1])
            cut_len = run.tcp_length if run.tcp_length > 0 else run.length
            self._cut_time[s] = self.cutter.time_for_length(
                cut_len, mode="cut")

    # ------------------------------------------------------------------
    # Schritt 2: Greedy Set-Cover ("neue Punkte / Zusatzzeit")
    # ------------------------------------------------------------------

    def _neighbors(self, s: int) -> tuple[int, int]:
        """Die beiden zyklischen Nachbar-Segmente von s auf seinem Loop."""
        order = self._loop_segs[self.contour.segments[s].loop_id]
        i = order.index(s)
        k = len(order)
        return order[(i - 1) % k], order[(i + 1) % k]

    def _extra_time(self, s: int, selected: set[int]) -> float:
        """Zusatzzeit, wenn Segment s zur Auswahl hinzukommt:
        Schnittzeit + Pierce -- Pierce entfaellt, wenn s nahtlos an ein
        bereits gewaehltes Nachbarsegment anschliesst (Verkettung)."""
        t = self._cut_time[s]
        prev_s, next_s = self._neighbors(s)
        if prev_s not in selected and next_s not in selected:
            t += self.cutter.pierce_time()
        return float(t)

    def _greedy(self) -> list[int]:
        """Waehlt Segmente, bis alle erreichbaren Punkte abgedeckt sind."""
        covered = np.zeros(self._A.shape[0], dtype=bool)
        selected: list[int] = []
        sel_set: set[int] = set()

        while True:
            best_s, best_ratio, best_cost = -1, 0.0, np.inf
            for s in range(self._A.shape[1]):
                if s in sel_set or not self._feasible[s]:
                    continue
                new = int(np.count_nonzero(self._A[:, s] & ~covered))
                if new == 0:
                    continue
                cost = self._extra_time(s, sel_set)
                ratio = new / cost
                # Bestes Verhaeltnis; bei Gleichstand das billigere Segment
                if (ratio > best_ratio + 1e-12
                        or (abs(ratio - best_ratio) <= 1e-12
                            and cost < best_cost)):
                    best_s, best_ratio, best_cost = s, ratio, cost
            if best_s < 0:
                break  # kein Segment bringt mehr neue Punkte
            selected.append(best_s)
            sel_set.add(best_s)
            covered |= self._A[:, best_s]

        return selected

    # ------------------------------------------------------------------
    # Schritt 3: Pruning redundanter Segmente
    # ------------------------------------------------------------------

    def _prune(self, selected: list[int]) -> list[int]:
        """Entfernt Segmente, deren Punkte alle mehrfach abgedeckt sind
        (teuerste zuerst) -- die Coverage bleibt dabei exakt gleich."""
        if not selected:
            return selected
        keep = set(selected)
        # Wie oft ist jeder Punkt von der aktuellen Auswahl abgedeckt?
        cover_count = self._A[:, selected].sum(axis=1).astype(int)
        for s in sorted(selected, key=lambda i: self._cut_time[i],
                        reverse=True):
            col = self._A[:, s]
            if np.all(cover_count[col] >= 2):
                keep.discard(s)
                cover_count[col] -= 1
        return [s for s in selected if s in keep]

    # ------------------------------------------------------------------
    # Schritt 4: Zusammenhaengende Segmente zu CutRuns verschmelzen
    # ------------------------------------------------------------------

    def _merge_to_runs(self, selected: list[int]) -> list[CutRun]:
        """Verschmilzt zusammenhaengende gewaehlte Primitiv-Segmente je
        Loop zu einem CutRun (eine Zuendung pro Gruppe).

        Sicherheitsnetz: Deckt ein verschmolzener Run NICHT alles ab,
        was seine Einzelsegmente abdecken (passiert bei degeneriertem /
        mehrteiligem Material, wenn der laengere Run auf einen anderen
        Offset-Ring projiziert), zerfaellt die Gruppe zurueck in die
        Einzelsegment-Runs -- die Coverage der Auswahl bleibt garantiert.
        """
        runs: list[CutRun] = []
        sel_set = set(selected)

        for loop_id, order in self._loop_segs.items():
            loop = self.contour.loop_by_id(loop_id)
            chosen = [s for s in order if s in sel_set]
            if not chosen:
                continue

            groups: list[list[int]]
            if len(chosen) == len(order):
                groups = [list(order)]  # kompletter Loop: einmal herum
            else:
                # Zyklisch zusammenhaengende Gruppen: Gruppenstart =
                # gewaehltes Segment, dessen Vorgaenger NICHT gewaehlt ist.
                groups = []
                k = len(order)
                for i, s in enumerate(order):
                    if s not in sel_set or order[(i - 1) % k] in sel_set:
                        continue
                    group = [s]
                    j = i
                    while order[(j + 1) % k] in sel_set:
                        j += 1
                        group.append(order[j % k])
                    groups.append(group)

            for group in groups:
                runs.extend(self._runs_for_group(loop, loop_id, group,
                                                 next_id=len(runs) + 1))

        return runs

    def _runs_for_group(self, loop, loop_id: int, group: list[int],
                        next_id: int) -> list[CutRun]:
        """Ein verschmolzener Run fuer die Gruppe -- oder die
        Einzelsegment-Runs, falls der Merge Coverage verlieren wuerde."""
        first = self.contour.segments[group[0]]
        last = self.contour.segments[group[-1]]
        full_loop = len(group) == len(self._loop_segs[loop_id])
        merged = self._make_run(
            next_id, loop, loop_id, first.start_pos,
            first.start_pos if full_loop else last.end_pos)

        if merged.is_feasible and merged.swept_polygon is not None:
            group_mask = self._A[:, group].any(axis=1)
            pts = self.grid.coords[group_mask]
            ok = bool(np.all(shapely.contains_xy(
                merged.swept_polygon, pts[:, 0], pts[:, 1])))
            if ok:
                return [merged]

        # Fallback: Einzelsegmente behalten (kein Coverage-Verlust)
        out: list[CutRun] = []
        for s in group:
            cand = self._cand_runs[s]
            if cand is None:
                continue
            cand.run_id = next_id + len(out)
            out.append(cand)
        return out

    def _make_run(self, run_id: int, loop, loop_id: int,
                  start_pos: int, end_pos: int) -> CutRun:
        """CutRun in +1-Richtung von start_pos nach end_pos, inkl.
        Lichtschwert-Kinematik."""
        positions = loop.arc_positions(start_pos, end_pos, +1)
        run = CutRun(
            run_id=run_id, loop_id=loop_id,
            start_pos=start_pos, end_pos=end_pos, direction=+1,
            positions=positions,
            polyline=loop.polyline(positions),
            length=(loop.length if start_pos == end_pos
                    else loop.arc_length(start_pos, end_pos, +1)),
        )
        return self.kinematics.attach(run)

    # ------------------------------------------------------------------
    # Gesamtablauf
    # ------------------------------------------------------------------

    def plan(self) -> AutoPlanResult:
        """Fuehrt Greedy + Pruning + Verschmelzen + Sequencing aus."""
        t0 = time.perf_counter()

        selected = self._greedy()
        selected = self._prune(selected)
        runs = self._merge_to_runs(selected)

        plan = self.sequencer.build_plan(
            [r for r in runs if r.is_feasible])
        report = compute_grid_coverage(self.grid, runs)
        score = compute_score(plan.total_time, report.fraction)

        reachable = self._A.any(axis=1)
        return AutoPlanResult(
            runs=runs,
            plan=plan,
            report=report,
            score=score,
            elapsed=time.perf_counter() - t0,
            n_candidates=len(self.contour.segments),
            n_selected=len(selected),
            n_unreachable=int(np.count_nonzero(~reachable)),
        )


# ---------------------------------------------------------------------------
# Headless-Einstieg (analog zu simulation.check_segments)
# ---------------------------------------------------------------------------

def auto_plan(
    grid: PointGrid,
    cutter: Cutter | None = None,
    kerf_width: float = 3.0,
    target_segment_length: float | None = None,
) -> AutoPlanResult:
    """Wunschziel ohne UI: Geometrie rein, fertiger Schnittplan raus.

    Parameters
    ----------
    grid : PointGrid der Geometrie
    cutter, kerf_width, target_segment_length : wie bei check_segments

    Returns
    -------
    AutoPlanResult mit Runs, zeitminimal geordnetem CutPlan,
    Coverage-Report und Punktezahl.
    """
    if cutter is None:
        try:
            from .simulation import make_default_cutter
        except ImportError:
            from plasma_cutter.segment_simulation.simulation import (
                make_default_cutter,
            )
        cutter = make_default_cutter()

    contour = SegmentedContour.from_grid(
        grid, target_segment_length=target_segment_length)
    material = contour.material_polygon()
    blade_length = cutter.blade_length(cutter.cutting_speed)
    kinematics = RunKinematics(
        material, clearance=cutter.minimum_gap,
        blade_length=blade_length, kerf=kerf_width)
    link_planner = LinkPlanner(material, clearance=cutter.minimum_gap)
    sequencer = Sequencer(cutter, contour, link_planner)

    planner = AutoPlanner(grid, contour, cutter, kinematics, sequencer)
    return planner.plan()
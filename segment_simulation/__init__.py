"""Segment-basierte Plasmaschneider-Simulation (Konturschnitt).

  - Der Brenner steht orthogonal zum Blech und schneidet ENTLANG der
    Kontur (klassischer Plasma-Konturschnitt, 2D-Draufsicht).
  - Die Außen- und Lochkonturen werden automatisch in **Segmente**
    unterteilt (Sammlung von Konturpunkten)
  - Coverage-Prüfung: Schneiden die gewählten Segmente das Objekt
    zu 100 % durch? Falls nicht: wieviel fehlt
  - Beim Start ordnet der Sequencer die Segmente optimal an und der
    LinkPlanner verbindet sie automatisch kollisionsfrei
    (Mindestabstand ``cutter.minimum_gap`` zum Material).

Module
------
segments.py      Konturen, Segmentierung, CutRuns, Coverage
planning.py      Kinematik, LinkPlanner (kollisionsfreie Verbindungen),
                 Sequencer (Held-Karp), Score
autoplan.py      Greedy-Auswahl (Greedy Set-Cover + Pruning + Merge)
simulation.py    Interaktive Matplotlib-Simulation + Animation; vier
                 Auswahlverfahren: manuell (Klick), Greedy+ (P),
                 Surrogat (S), Brute Force (B)
surrogate/       Brute-Force-Lehrer, Greedy+, Surrogat-Planer, Modell,
                 Label-Pipeline, Benchmark, Lernkurve
thesis_figures/  Abbildungsskripte der Bachelorarbeit
"""

from .segments import (
    ContourLoop,
    Segment,
    SegmentedContour,
    CutRun,
    CoverageReport,
    compute_coverage,
    GridCoverageReport,
    compute_grid_coverage,
)
from .planning import (
    LinkPath,
    LinkPlanner,
    LinkInfeasibleError,
    Sequencer,
    CutPlan,
    PlannedStep,
    RunKinematics,
    compute_score,
)
from .autoplan import AutoPlanner, AutoPlanResult, auto_plan

__all__ = [
    "ContourLoop",
    "Segment",
    "SegmentedContour",
    "CutRun",
    "CoverageReport",
    "compute_coverage",
    "GridCoverageReport",
    "compute_grid_coverage",
    "LinkPath",
    "LinkPlanner",
    "LinkInfeasibleError",
    "Sequencer",
    "CutPlan",
    "PlannedStep",
    "RunKinematics",
    "compute_score",
    "AutoPlanner",
    "AutoPlanResult",
    "auto_plan",
]

"""Segment-basierte Plasmaschneider-Simulation (Konturschnitt).

Neuentwicklung der Simulation mit folgendem Modell:

  - Der Brenner steht orthogonal zum Blech und schneidet ENTLANG der
    Kontur (klassischer Plasma-Konturschnitt, 2D-Draufsicht).
  - Die Aussen- und Lochkonturen werden automatisch in **Segmente**
    unterteilt (Sammlung von Konturpunkten, sinnvolle Groesse).
  - Der Benutzer waehlt Schnitt-Segmente per Start-/Endpunkt aus;
    die Klicks werden auf die moeglichen Segmentknoten gesnappt.
  - Coverage-Pruefung: Schneiden die gewaehlten Segmente das Objekt
    zu 100 % durch? Falls nicht: wieviel fehlt (visuell + Zahl).
  - Beim Start ordnet der Sequencer die Segmente optimal an und der
    LinkPlanner verbindet sie automatisch kollisionsfrei
    (Mindestabstand ``cutter.minimum_gap`` zum Material).

Module
------
segments.py    Konturen, Segmentierung, CutRuns, Coverage
planning.py    LinkPlanner (kollisionsfreie Verbindungen) + Sequencer
simulation.py  Interaktive Matplotlib-Simulation + Animation
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
    Sequencer,
    CutPlan,
    PlannedStep,
    RunKinematics,
    compute_score,
)

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
    "Sequencer",
    "CutPlan",
    "PlannedStep",
    "RunKinematics",
    "compute_score",
]

"""Segment-based simulation of plasma contour cutting."""

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

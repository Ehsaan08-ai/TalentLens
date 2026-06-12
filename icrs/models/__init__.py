"""Typed data models for ICRS (RequirementVector, candidate profiles, results).

This package holds the pure data models (pydantic v2) the pipeline operates on:
job-description models (``icrs.models.job``), candidate profile models
(``icrs.models.candidate``), ranking output models (``icrs.models.ranking``),
and shared enums (``icrs.models.enums``).
"""

from icrs.models.candidate import (
    BehavioralSignal,
    Education,
    EnrichedProfile,
    NormalizedProfile,
    RawCandidate,
    Role,
)
from icrs.models.enums import DepthBreadth, SignalTier, TrajectoryArc
from icrs.models.job import (
    WEIGHT_SUM_TOLERANCE,
    JobDescription,
    JobType,
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)
from icrs.models.ranking import (
    MAX_SUMMARY_CHARS,
    Explanation,
    RankingResult,
    SignalBreakdown,
)

__all__ = [
    # job
    "WEIGHT_SUM_TOLERANCE",
    "JobDescription",
    "JobType",
    "Requirement",
    "RequirementCategory",
    "RequirementTier",
    "RequirementVector",
    "SeniorityBand",
    # candidate
    "BehavioralSignal",
    "Education",
    "EnrichedProfile",
    "NormalizedProfile",
    "RawCandidate",
    "Role",
    # enums
    "DepthBreadth",
    "SignalTier",
    "TrajectoryArc",
    # ranking
    "MAX_SUMMARY_CHARS",
    "Explanation",
    "RankingResult",
    "SignalBreakdown",
]

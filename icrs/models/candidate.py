"""Candidate profile data models for ICRS (Task 2.2).

Three representations of a candidate flow through the pipeline:

1. :class:`RawCandidate` — the heterogeneous, as-submitted profile (structured
   fields, free text, external handles).
2. :class:`NormalizedProfile` — the canonical schema produced by
   ``CandidateEnricher.normalize`` (roles, education, certifications, explicit
   skills, total tenure in whole months).
3. :class:`EnrichedProfile` — the normalized profile augmented with inferred
   Tier 2 semantic signals, Tier 3 behavioral signals, per-tier signal
   availability, and an embedding.

This module defines only the data models and their validation. Enrichment logic
(deriving the signals, computing availability, producing the embedding) is
implemented in Task 4.x — here we only enforce the structural and range
invariants the rest of the pipeline relies on.

Key validation guarantees (Requirements 3.1, 3.2, 3.5):
    - ``signal_availability`` is a per-tier map whose every value is in [0,1].
    - Absent structured scalar fields are represented as ``None`` ("not-present")
      rather than being silently defaulted to a sentinel value such as 0 or "".
"""

from __future__ import annotations

from datetime import date
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from icrs.models.enums import DepthBreadth, SignalTier, TrajectoryArc
from icrs.providers.base import Vector


class Role(BaseModel):
    """A single position held by a candidate.

    ``title`` and ``company`` are the identifying fields. ``start``, ``end`` and
    ``prestige_tier`` are optional: an absent value is recorded as ``None``
    ("not-present") rather than being defaulted, so downstream structural-signal
    derivation can distinguish "unknown" from a real value (Requirement 3.2).
    An ``end`` of ``None`` conventionally denotes a currently-held role.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    company: str
    start: date | None = None
    end: date | None = None
    # Tiered company prestige reference (not a hard gate). Absent => not-present.
    prestige_tier: int | None = Field(default=None, ge=1)

    @field_validator("title", "company")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Role title and company must be non-empty")
        return value


class Education(BaseModel):
    """A single education record.

    Every field is optional: a candidate may list a degree without a field of
    study, or dates without an institution. Absent fields are ``None``
    ("not-present"), never defaulted (Requirement 3.2).
    """

    model_config = ConfigDict(extra="forbid")

    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start: date | None = None
    end: date | None = None


class RawCandidate(BaseModel):
    """A heterogeneous candidate profile exactly as submitted.

    No normalization is applied here; the structured fields are an open map so
    that arbitrary source schemas can be ingested before canonicalization.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    # Open key/value map of source-provided structured data (titles, dates,
    # education, skills, ...). Kept untyped at ingestion; normalized later.
    structured_fields: dict[str, Any] = Field(default_factory=dict)
    # Free-text prose (summaries, role descriptions) for Tier 2 inference.
    free_text: str = ""
    # External platform handles keyed by source ("github", "linkedin", ...).
    external_handles: dict[str, str] = Field(default_factory=dict)


class NormalizedProfile(BaseModel):
    """The canonical candidate schema produced by normalization (Requirement 3.1).

    ``total_tenure_months`` is expressed in whole months and must be
    non-negative. List fields default to empty collections (an empty list is a
    legitimately-known "none present", distinct from an absent scalar field).
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    roles: list[Role] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    explicit_skills: list[str] = Field(default_factory=list)
    total_tenure_months: int = Field(default=0, ge=0)


class BehavioralSignal(BaseModel):
    """A Tier 3 behavioral / activity signal derived from an external platform.

    ``recency_days`` (the age of the activity) drives freshness weighting in the
    scoring engine and must be non-negative. ``corroborates_skill`` lists the
    explicit skills this activity provides evidence for (consistency check).
    """

    model_config = ConfigDict(extra="forbid")

    source: str  # e.g. "github", "linkedin", "publications"
    metric: str  # e.g. "commit_frequency", "endorsements"
    value: float
    recency_days: int = Field(ge=0)
    corroborates_skill: list[str] = Field(default_factory=list)

    @field_validator("source", "metric")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("BehavioralSignal source and metric must be non-empty")
        return value


class EnrichedProfile(BaseModel):
    """A normalized profile augmented with inferred and behavioral signals.

    The inferred Tier 2 fields (``inferred_responsibilities``, ``implicit_skills``,
    ``trajectory_arc``, ``depth_breadth``) and Tier 3 ``behavioral_signals`` are
    populated by the enricher (Task 4.x); here they carry only their types and
    constraints.

    ``signal_availability`` records, per :class:`SignalTier`, the fraction of that
    tier's expected fields that are populated — a value in [0,1] so that missing
    data is treated as "unknown" (0 coverage) rather than as a zero score
    (Requirement 3.5). Each value is validated to the inclusive range [0,1].

    ``embedding`` is optional and defaults to ``None`` ("not-present") because it
    is produced by a later pipeline stage (Task 5); it is never defaulted to a
    zero vector.
    """

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    base: NormalizedProfile
    # Tier 2 — semantic, inferred from free text.
    inferred_responsibilities: list[str] = Field(default_factory=list)
    implicit_skills: list[str] = Field(default_factory=list)
    trajectory_arc: TrajectoryArc | None = None
    depth_breadth: DepthBreadth | None = None
    # Tier 3 — behavioral / activity.
    behavioral_signals: list[BehavioralSignal] = Field(default_factory=list)
    # Per-tier coverage in [0,1]; missing data recorded as unknown, not zero.
    signal_availability: dict[SignalTier, float] = Field(default_factory=dict)
    # Produced by the Embedding Generator (Task 5); absent until then.
    embedding: Vector | None = None

    @field_validator("signal_availability")
    @classmethod
    def _availability_per_tier_in_unit_interval(
        cls, value: dict[SignalTier, float]
    ) -> dict[SignalTier, float]:
        """Enforce that every recorded per-tier availability lies in [0,1]."""

        for tier, coverage in value.items():
            if not (0.0 <= coverage <= 1.0):
                raise ValueError(
                    f"signal_availability for tier {tier} must be in [0,1], "
                    f"got {coverage}"
                )
        return value


__all__ = [
    "Role",
    "Education",
    "RawCandidate",
    "NormalizedProfile",
    "BehavioralSignal",
    "EnrichedProfile",
]

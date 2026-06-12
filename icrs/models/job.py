"""Job-description data models for ICRS (Phase 1 foundation).

This module defines the structured representation of a job description that the
JD Decomposer (Task 3) will populate from raw text:

    - :class:`Requirement`         - a single classified, weighted requirement
    - :class:`RequirementVector`   - the decomposed, weighted requirement schema
    - :class:`JobDescription`      - the submitted JD plus its parsed vector

These are pure data models with validation only — no LLM/decomposition logic
lives here (that is Task 3.1).

Validation rules enforced here (mirroring the design's "Validation Rules"):
    - ``raw_text`` is non-empty / non-whitespace (Requirement 1.7)
    - ``role_intent`` is stored distinct from the job title (Requirement 1.1)
    - every requirement is classified as exactly one category and tier
      (Requirement 1.2)
    - a :class:`RequirementVector` contains at least one MUST_HAVE
      (Requirements 1.5, 1.8)
    - ``implicit_expectations`` and ``culture_signals`` default to empty
      collections (Requirement 1.3)
    - within each *weighted* category (MUST_HAVE, NICE_TO_HAVE) the requirement
      weights are normalized to sum to 1.0; DISQUALIFYING requirements are
      absolute gates and are excluded from any weighted contribution
      (Requirements 1.4)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from icrs.providers.base import Vector

# Tolerance used when checking/normalizing per-category weight sums to 1.0.
WEIGHT_SUM_TOLERANCE = 1e-3


class RequirementCategory(str, Enum):
    """How a single requirement participates in scoring.

    - ``MUST_HAVE``    - mandatory; contributes to weighted scoring.
    - ``NICE_TO_HAVE`` - optional; contributes to weighted scoring.
    - ``DISQUALIFYING`` - an absolute gate. A positive match removes the
      candidate; it is *excluded* from weighted scoring contribution
      (Requirement 1.4).
    """

    MUST_HAVE = "MUST_HAVE"
    NICE_TO_HAVE = "NICE_TO_HAVE"
    DISQUALIFYING = "DISQUALIFYING"


class RequirementTier(str, Enum):
    """The signal tier a requirement is evaluated against.

    Mirrors the three candidate signal tiers so a requirement can be matched
    against the appropriate evidence:

    - ``STRUCTURAL`` - Tier 1, structured profile fields.
    - ``SEMANTIC``   - Tier 2, inferred from free text.
    - ``BEHAVIORAL`` - Tier 3, external platform activity.
    """

    STRUCTURAL = "STRUCTURAL"
    SEMANTIC = "SEMANTIC"
    BEHAVIORAL = "BEHAVIORAL"


class SeniorityBand(str, Enum):
    """The predefined enumerated set of seniority bands (Requirement 1.3)."""

    JUNIOR = "JUNIOR"
    MID = "MID"
    SENIOR = "SENIOR"
    STAFF = "STAFF"
    LEAD = "LEAD"
    EXECUTIVE = "EXECUTIVE"


class JobType(str, Enum):
    """Role family that drives weight-profile selection (design Scoring section).

    Unknown / future types fall back to the default weight profile in the
    scoring layer (Task 9); the model itself only needs the enumerated set.
    """

    TECHNICAL = "TECHNICAL"
    LEADERSHIP = "LEADERSHIP"
    GENERALIST = "GENERALIST"
    SALES = "SALES"


class Requirement(BaseModel):
    """A single extracted, classified, and weighted job requirement.

    ``weight`` is the requirement's relative importance *within its category*.
    For the weighted categories (MUST_HAVE, NICE_TO_HAVE) the weights are
    normalized at the :class:`RequirementVector` level so each such category
    sums to 1.0. DISQUALIFYING requirements carry a weight for completeness but
    it never contributes to scoring (they act as absolute gates).

    ``embedding`` is optional and populated later by the Embedding Generator
    (Task 5) for dense matching; it is ``None`` until then.
    """

    model_config = ConfigDict(use_enum_values=False)

    text: str = Field(..., description="The requirement text as extracted from the JD.")
    category: RequirementCategory = Field(
        ..., description="Exactly one of MUST_HAVE / NICE_TO_HAVE / DISQUALIFYING."
    )
    tier: RequirementTier = Field(
        ..., description="Signal tier this requirement is evaluated against."
    )
    weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relative importance within the category, in [0,1].",
    )
    embedding: Vector | None = Field(
        default=None,
        description="Optional dense vector for matching; populated by the "
        "Embedding Generator (Task 5).",
    )

    @field_validator("text")
    @classmethod
    def _text_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Requirement text must be a non-empty, non-whitespace string")
        return value

    @property
    def is_weighted(self) -> bool:
        """Whether this requirement contributes to weighted scoring.

        DISQUALIFYING requirements are absolute gates and never contribute
        (Requirement 1.4).
        """

        return self.category is not RequirementCategory.DISQUALIFYING


class RequirementVector(BaseModel):
    """The structured, weighted decomposition of a job description.

    Produced by the JD Decomposer (Task 3). This model enforces only the
    structural/validation invariants; population from raw text is out of scope
    here.
    """

    model_config = ConfigDict(use_enum_values=False)

    job_id: UUID = Field(default_factory=uuid4, description="Owning JobDescription id.")
    role_intent: str = Field(
        ...,
        description="The job to be done, derived from the JD body and stored "
        "distinct from the title (Requirement 1.1).",
    )
    seniority_band: SeniorityBand = Field(
        ..., description="Seniority band inferred from scope language (Requirement 1.3)."
    )
    requirements: list[Requirement] = Field(
        default_factory=list, description="All classified requirements."
    )
    implicit_expectations: list[str] = Field(
        default_factory=list,
        description="e.g. 'adaptability', 'ownership'. Defaults to empty (Requirement 1.3).",
    )
    culture_signals: list[str] = Field(
        default_factory=list,
        description="Culture/domain fit signals. Defaults to empty (Requirement 1.3).",
    )

    @field_validator("role_intent")
    @classmethod
    def _role_intent_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("role_intent must be a non-empty, non-whitespace string")
        return value

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> "RequirementVector":
        """Enforce the must-have requirement and normalize per-category weights.

        - At least one MUST_HAVE requirement must be present (Requirements 1.5,
          1.8). A vector with zero MUST_HAVEs is invalid.
        - Within each *weighted* category (MUST_HAVE, NICE_TO_HAVE) the weights
          are normalized to sum to 1.0 so each category contributes a unit of
          relative importance. DISQUALIFYING requirements are excluded from this
          normalization entirely (Requirement 1.4).
        """

        must_haves = [
            r for r in self.requirements if r.category is RequirementCategory.MUST_HAVE
        ]
        if not must_haves:
            raise ValueError(
                "RequirementVector must contain at least one MUST_HAVE requirement"
            )

        for category in (RequirementCategory.MUST_HAVE, RequirementCategory.NICE_TO_HAVE):
            members = [r for r in self.requirements if r.category is category]
            if not members:
                continue
            total = sum(r.weight for r in members)
            if total <= 0.0:
                # No usable signal: distribute importance equally so the
                # category still sums to 1.0.
                equal = 1.0 / len(members)
                for r in members:
                    r.weight = equal
            elif abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
                # Proportionally normalize so the category sums to exactly 1.0.
                for r in members:
                    r.weight = r.weight / total

        return self

    @property
    def must_haves(self) -> list[Requirement]:
        """All MUST_HAVE requirements."""

        return [r for r in self.requirements if r.category is RequirementCategory.MUST_HAVE]

    @property
    def nice_to_haves(self) -> list[Requirement]:
        """All NICE_TO_HAVE requirements."""

        return [
            r for r in self.requirements if r.category is RequirementCategory.NICE_TO_HAVE
        ]

    @property
    def disqualifiers(self) -> list[Requirement]:
        """All DISQUALIFYING requirements (absolute gates, not weighted)."""

        return [
            r for r in self.requirements if r.category is RequirementCategory.DISQUALIFYING
        ]

    def weighted_requirements(self) -> list[Requirement]:
        """Requirements that contribute to weighted scoring.

        Excludes DISQUALIFYING requirements per Requirement 1.4.
        """

        return [r for r in self.requirements if r.is_weighted]


class JobDescription(BaseModel):
    """A submitted job description and (once decomposed) its requirement vector.

    ``parsed`` is ``None`` until the JD Decomposer succeeds; ``job_type`` drives
    weight-profile selection in the scoring layer.
    """

    model_config = ConfigDict(use_enum_values=False)

    id: UUID = Field(default_factory=uuid4)
    raw_text: str = Field(..., description="The original JD text as submitted.")
    title: str = Field(default="", description="The job title (distinct from role intent).")
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the JD was submitted.",
    )
    parsed: RequirementVector | None = Field(
        default=None,
        description="The decomposed requirement vector; None until decomposition succeeds.",
    )
    job_type: JobType = Field(
        default=JobType.GENERALIST,
        description="Role family driving weight-profile selection.",
    )

    @field_validator("raw_text")
    @classmethod
    def _raw_text_non_empty(cls, value: str) -> str:
        # Requirement 1.7: empty/whitespace-only JD is invalid.
        if not value or not value.strip():
            raise ValueError("raw_text must contain at least one non-whitespace character")
        return value


__all__ = [
    "WEIGHT_SUM_TOLERANCE",
    "RequirementCategory",
    "RequirementTier",
    "SeniorityBand",
    "JobType",
    "Requirement",
    "RequirementVector",
    "JobDescription",
]

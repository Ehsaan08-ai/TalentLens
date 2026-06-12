"""Ranking output data models for ICRS.

This module defines the models the pipeline emits as its final product: the
per-signal :class:`SignalBreakdown`, the recruiter-facing :class:`Explanation`,
and the :class:`RankingResult` that bundles a candidate's final score, rank,
breakdown, explanation, and confidence for a given job.

The models are intentionally self-contained — they depend only on the standard
library and pydantic v2 — so they can be developed in parallel with the job and
candidate profile models without import-time coupling. Concrete cross-model
wiring happens later in the pipeline layers.

Validation here enforces the design's "Model: RankingResult" rules and the
output contract from Requirements 5.1 and 2.4:

    - every sub-score in :class:`SignalBreakdown` is in ``[0.0, 1.0]``
    - ``final_score`` and ``confidence`` are in ``[0.0, 1.0]``
    - ``rank`` is a positive integer (ranks are unique/contiguous from 1, which
      the orchestrator guarantees across the result set)
    - the recruiter-facing ``summary`` is a non-empty string of at most 1000
      characters
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Inclusive bounds shared by every normalized score / sub-score in this module.
SCORE_MIN = 0.0
SCORE_MAX = 1.0

# Upper bound on the length of a recruiter-facing summary (Requirement 5.1).
MAX_SUMMARY_CHARS = 1000


class SignalBreakdown(BaseModel):
    """Per-tier sub-score breakdown contributing to a candidate's final score.

    Each field is a normalized sub-score in ``[0.0, 1.0]``. ``disqualifying_penalty``
    captures soft red flags (it is subtracted during composite fusion); the hard
    disqualification gate is handled separately by the scoring engine, so this is
    still a normalized penalty magnitude in ``[0,1]`` rather than a signed term.
    """

    model_config = ConfigDict(extra="forbid")

    semantic_fit: float = Field(
        ...,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description="Blended dense/sparse semantic fit sub-score in [0,1].",
    )
    career_trajectory: float = Field(
        ...,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description="Career trajectory/arc alignment sub-score in [0,1].",
    )
    behavioral: float = Field(
        ...,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description="Freshness-weighted behavioral-signal sub-score in [0,1].",
    )
    hard_filter_pass: float = Field(
        ...,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description="Soft must-have satisfaction ratio in [0,1].",
    )
    disqualifying_penalty: float = Field(
        ...,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description="Soft red-flag penalty magnitude in [0,1].",
    )


class Explanation(BaseModel):
    """Recruiter-readable rationale for a candidate's ranking.

    ``summary`` is the plain-language explanation surfaced to recruiters and is
    constrained to a non-empty string of at most 1000 characters. The list
    fields default to empty collections; their semantic guarantees (e.g. that
    ``unmet_must_haves`` contains only unsatisfied MUST_HAVE requirements) are
    enforced by the explanation generator that populates them.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SUMMARY_CHARS,
        description="Non-empty recruiter-facing summary, at most 1000 characters.",
    )
    driving_signals: list[str] = Field(
        default_factory=list,
        description="Signals that drove the candidate's final score.",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Gaps between the candidate and the requirements.",
    )
    unmet_must_haves: list[str] = Field(
        default_factory=list,
        description="Unsatisfied MUST_HAVE requirements only.",
    )

    @field_validator("summary")
    @classmethod
    def _summary_not_blank(cls, value: str) -> str:
        """Reject whitespace-only summaries (a blank summary is not recruiter-readable)."""

        if not value.strip():
            raise ValueError("summary must be a non-empty, non-whitespace string")
        return value


class RankingResult(BaseModel):
    """A single candidate's ranking outcome for a given job.

    Bundles the final score, assigned rank, per-signal breakdown, recruiter
    explanation, and confidence. ``rank`` is a positive integer; the orchestrator
    is responsible for ensuring ranks are unique and contiguous from 1 across the
    full result set (this model only enforces the per-result positivity bound).
    """

    model_config = ConfigDict(extra="forbid")

    job_id: UUID = Field(..., description="Identifier of the job being ranked against.")
    candidate_id: UUID = Field(..., description="Identifier of the ranked candidate.")
    final_score: float = Field(
        ...,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description="Final post-rerank score in [0,1] used to determine rank.",
    )
    rank: int = Field(
        ...,
        ge=1,
        description="Positive integer rank (unique/contiguous from 1 across results).",
    )
    breakdown: SignalBreakdown = Field(..., description="Per-tier sub-score breakdown.")
    explanation: Explanation = Field(..., description="Recruiter-facing rationale.")
    confidence: float = Field(
        ...,
        ge=SCORE_MIN,
        le=SCORE_MAX,
        description="Confidence in [0,1] from signal coverage and score margin.",
    )


__all__ = [
    "SignalBreakdown",
    "Explanation",
    "RankingResult",
    "SCORE_MIN",
    "SCORE_MAX",
    "MAX_SUMMARY_CHARS",
]

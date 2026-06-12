"""Request / response Pydantic schemas for the ICRS ranking API (Task 17.1).

These models define the HTTP contract for the asynchronous ranking endpoint.
They are intentionally decoupled from the internal pipeline models: the request
carries *raw*, as-submitted candidate payloads (the same shape as
:class:`~icrs.models.candidate.RawCandidate`) and the response exposes exactly
the recruiter-facing shortlist fields mandated by Requirement 5.1 — a final
score, the per-signal breakdown, the explanation, and a confidence per result —
plus the run-level resilience flags surfaced by
:class:`~icrs.pipeline.orchestrator.RankingRun` (Requirement 9).

Nothing here calls an LLM/embedding provider; these are pure data models.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from icrs.models.job import JobType
from icrs.pipeline.orchestrator import RankingRun


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class CandidatePayload(BaseModel):
    """A single candidate exactly as submitted (RawCandidate-shaped).

    Mirrors :class:`~icrs.models.candidate.RawCandidate`'s public fields so the
    API layer can build a ``RawCandidate`` without leaking internal model
    construction details to clients. ``structured_fields`` is an open map of
    source-provided data, ``free_text`` is free-form prose, and
    ``external_handles`` maps platform names to handles.
    """

    model_config = ConfigDict(extra="forbid")

    structured_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Open key/value map of source-provided structured data.",
    )
    free_text: str = Field(
        default="",
        description="Free-text prose (summaries, role descriptions).",
    )
    external_handles: dict[str, str] = Field(
        default_factory=dict,
        description="External platform handles keyed by source (e.g. 'github').",
    )


class RankRequest(BaseModel):
    """A request to rank a candidate pool against a job description.

    ``raw_jd`` and a non-empty ``candidates`` list are required for a ranking to
    be produced; an empty/whitespace JD or an empty pool is rejected by the
    orchestrator with HTTP 400 (Requirement 2.6).
    """

    model_config = ConfigDict(extra="forbid")

    raw_jd: str = Field(
        ...,
        description="The raw job-description text to decompose and rank against.",
    )
    job_type: JobType = Field(
        ...,
        description="Role family driving weight-profile selection.",
    )
    title: str | None = Field(
        default=None,
        description="Optional job title (stored distinct from the role intent).",
    )
    candidates: list[CandidatePayload] = Field(
        default_factory=list,
        description="The candidate pool to rank (RawCandidate-shaped payloads).",
    )


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #
class BreakdownSchema(BaseModel):
    """Per-tier sub-score breakdown contributing to a candidate's final score."""

    model_config = ConfigDict(extra="forbid")

    semantic_fit: float = Field(..., ge=0.0, le=1.0)
    career_trajectory: float = Field(..., ge=0.0, le=1.0)
    behavioral: float = Field(..., ge=0.0, le=1.0)
    hard_filter_pass: float = Field(..., ge=0.0, le=1.0)
    disqualifying_penalty: float = Field(..., ge=0.0, le=1.0)


class ExplanationSchema(BaseModel):
    """Recruiter-facing rationale for a candidate's ranking."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    driving_signals: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    unmet_must_haves: list[str] = Field(default_factory=list)


class RankedCandidate(BaseModel):
    """A single shortlist entry exposing every Requirement 5.1 output field."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: UUID = Field(..., description="Identifier of the ranked candidate.")
    rank: int = Field(..., ge=1, description="Unique, contiguous rank from 1.")
    final_score: float = Field(..., ge=0.0, le=1.0, description="Final score in [0,1].")
    breakdown: BreakdownSchema = Field(..., description="Per-tier sub-score breakdown.")
    explanation: ExplanationSchema = Field(..., description="Recruiter-facing rationale.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in [0,1].")


class RankResponse(BaseModel):
    """The ranked shortlist plus run-level resilience flags.

    ``results`` is the ordered shortlist (one entry per hard-filter survivor).
    The remaining fields surface the honest degradation flags from
    :class:`~icrs.pipeline.orchestrator.RankingRun` (Requirement 9):
    ``reranked`` is ``False`` when the LLM reranker failed and the run fell back
    to composite ordering; ``excluded_candidate_ids`` lists candidates dropped
    before ranking; ``explanation_unavailable_ids`` lists candidates whose
    explanation could not be generated.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[RankedCandidate] = Field(default_factory=list)
    reranked: bool = Field(
        default=True,
        description="False when the run fell back to composite ordering (Req 9.4).",
    )
    excluded_candidate_ids: list[str] = Field(
        default_factory=list,
        description="Candidates dropped before ranking (Req 9.2/9.3, 4.7).",
    )
    explanation_unavailable_ids: list[str] = Field(
        default_factory=list,
        description="Candidates whose explanation could not be generated (Req 9.5).",
    )

    @classmethod
    def from_run(cls, run: RankingRun) -> "RankResponse":
        """Build the API response from an orchestrator :class:`RankingRun`."""

        results = [
            RankedCandidate(
                candidate_id=r.candidate_id,
                rank=r.rank,
                final_score=r.final_score,
                breakdown=BreakdownSchema(
                    semantic_fit=r.breakdown.semantic_fit,
                    career_trajectory=r.breakdown.career_trajectory,
                    behavioral=r.breakdown.behavioral,
                    hard_filter_pass=r.breakdown.hard_filter_pass,
                    disqualifying_penalty=r.breakdown.disqualifying_penalty,
                ),
                explanation=ExplanationSchema(
                    summary=r.explanation.summary,
                    driving_signals=list(r.explanation.driving_signals),
                    gaps=list(r.explanation.gaps),
                    unmet_must_haves=list(r.explanation.unmet_must_haves),
                ),
                confidence=r.confidence,
            )
            for r in run.results
        ]
        return cls(
            results=results,
            reranked=run.reranked,
            excluded_candidate_ids=[str(cid) for cid in run.excluded_candidate_ids],
            explanation_unavailable_ids=[
                str(cid) for cid in run.explanation_unavailable_ids
            ],
        )


class DecomposeJDRequest(BaseModel):
    """Request schema for decomposing/analyzing a job description."""

    model_config = ConfigDict(extra="forbid")

    raw_jd: str = Field(
        ...,
        description="The raw job description to analyze.",
    )


class DecomposeJDResponse(BaseModel):
    """Response schema for job description analysis/insights."""

    model_config = ConfigDict(extra="forbid")

    role_intent: str = Field(
        ...,
        description="The underlying role intent derived from the job description.",
    )
    must_have: list[str] = Field(
        default_factory=list,
        description="Must-have requirements extracted from the JD.",
    )
    nice_to_have: list[str] = Field(
        default_factory=list,
        description="Nice-to-have requirements extracted from the JD.",
    )
    behavioral_signals: list[str] = Field(
        default_factory=list,
        description="Behavioral expectations and culture signals extracted from the JD.",
    )


__all__ = [
    "CandidatePayload",
    "RankRequest",
    "BreakdownSchema",
    "ExplanationSchema",
    "RankedCandidate",
    "RankResponse",
    "DecomposeJDRequest",
    "DecomposeJDResponse",
]

"""Ranked shortlist assembly — the recruiter-facing output object (Task 19.1).

The orchestrator emits a :class:`~icrs.pipeline.orchestrator.RankingRun`: the
ranked :class:`~icrs.models.ranking.RankingResult` list plus the resilience
flags (whether reranking was applied, which candidates' explanations are
unavailable, and which candidates were excluded before ranking). This module
maps that run into a :class:`RankedShortlist` — the single object the Streamlit
UI (Task 19.2) consumes.

The shortlist is deliberately *honest* about degradation rather than hiding it
(Requirements 9.4 / 9.5):

    - 5.1: every entry carries a final score in ``[0,1]``, the four per-signal
      sub-scores (semantic-fit, career-trajectory, behavioral, hard-filter) each
      in ``[0,1]``, a non-empty recruiter summary of at most 1000 characters,
      and a confidence in ``[0,1]``. These invariants are already enforced by
      :class:`RankingResult` / :class:`Explanation`; this layer maps them through
      and re-asserts the summary bound defensively.
    - 9.4: ``reranked`` is surfaced verbatim — ``False`` means the LLM reranker
      failed and the ordering is the composite-score fallback, and a human
      readable notice says so.
    - 9.5: each entry exposes ``explanation_available``; it is ``False`` when the
      explanation could not be generated (detected via the run's
      ``explanation_unavailable_ids`` or the orchestrator's non-fabricated
      ``EXPLANATION_UNAVAILABLE_SUMMARY`` sentinel), and a notice reports the
      count. No explanation content is ever fabricated or hidden.

This module imports from the models and pipeline layers but never mutates them.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from icrs.models.ranking import (
    MAX_SUMMARY_CHARS,
    Explanation,
    RankingResult,
    SignalBreakdown,
)
from icrs.pipeline.orchestrator import EXPLANATION_UNAVAILABLE_SUMMARY, RankingRun


class ShortlistEntry(BaseModel):
    """A single recruiter-facing shortlist row.

    Mirrors a :class:`RankingResult` for one candidate, adding the per-entry
    :attr:`explanation_available` honesty flag (Requirement 9.5). Every numeric
    field carries the same ``[0,1]`` bounds the ranking models enforce so the UI
    can rely on them without re-validating.
    """

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(
        ...,
        ge=1,
        description="Positive integer rank (unique/contiguous from 1 across the shortlist).",
    )
    candidate_id: UUID = Field(..., description="Identifier of the ranked candidate.")
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final score in [0,1] used to order the shortlist.",
    )
    breakdown: SignalBreakdown = Field(
        ...,
        description="Per-signal sub-scores (semantic-fit, trajectory, behavioral, hard-filter), each in [0,1].",
    )
    explanation: Explanation = Field(
        ...,
        description="Recruiter-facing rationale (summary, driving signals, gaps, unmet must-haves).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in [0,1] from signal coverage and score margin.",
    )
    explanation_available: bool = Field(
        ...,
        description=(
            "False when the explanation could not be generated (the summary is the "
            "non-fabricated 'unavailable' sentinel); surfaced honestly per Requirement 9.5."
        ),
    )

    @field_validator("explanation")
    @classmethod
    def _summary_within_bounds(cls, value: Explanation) -> Explanation:
        """Defensively re-assert the non-empty, <=1000-char summary bound (Requirement 5.1)."""

        summary = value.summary
        if not summary.strip():
            raise ValueError("explanation summary must be non-empty")
        if len(summary) > MAX_SUMMARY_CHARS:
            raise ValueError(
                f"explanation summary must be at most {MAX_SUMMARY_CHARS} characters"
            )
        return value

    @classmethod
    def from_result(
        cls, result: RankingResult, *, explanation_available: bool
    ) -> "ShortlistEntry":
        """Build an entry from a :class:`RankingResult` and its availability flag."""

        return cls(
            rank=result.rank,
            candidate_id=result.candidate_id,
            final_score=result.final_score,
            breakdown=result.breakdown,
            explanation=result.explanation,
            confidence=result.confidence,
            explanation_available=explanation_available,
        )


class RankedShortlist(BaseModel):
    """The recruiter-facing ranked shortlist plus honest run-level degradation flags.

    Produced by :func:`assemble_shortlist` (or :meth:`from_run`) from a
    :class:`RankingRun`. The UI renders :attr:`entries` in order and shows
    :attr:`notices` so a recruiter always sees when the result was degraded
    (Requirements 9.4 / 9.5).
    """

    model_config = ConfigDict(extra="forbid")

    job_id: UUID | None = Field(
        None,
        description="Identifier of the job ranked against (None only when the shortlist is empty).",
    )
    entries: list[ShortlistEntry] = Field(
        default_factory=list,
        description="Ranked shortlist entries, ordered by ascending rank (1..N).",
    )
    reranked: bool = Field(
        ...,
        description=(
            "True when LLM contextual reranking was applied; False when the reranker "
            "failed and the ordering is the composite-score fallback (Requirement 9.4)."
        ),
    )
    explanation_unavailable_ids: list[UUID] = Field(
        default_factory=list,
        description="Candidate ids whose explanation could not be generated (Requirement 9.5).",
    )
    excluded_candidate_ids: list[UUID] = Field(
        default_factory=list,
        description="Candidate ids excluded before ranking (embedding/sub-score failures).",
    )
    notices: list[str] = Field(
        default_factory=list,
        description="Human-readable summaries of any degradations applied to this run.",
    )

    @property
    def fully_reranked(self) -> bool:
        """Whether LLM reranking was applied (no un-reranked fallback)."""

        return self.reranked

    @property
    def all_explanations_available(self) -> bool:
        """Whether every entry carries a generated (non-sentinel) explanation."""

        return not self.explanation_unavailable_ids

    @classmethod
    def from_run(cls, run: RankingRun) -> "RankedShortlist":
        """Assemble a :class:`RankedShortlist` from a :class:`RankingRun`."""

        return assemble_shortlist(run)


def _is_explanation_available(
    result: RankingResult, unavailable_ids: set[UUID]
) -> bool:
    """Decide whether ``result``'s explanation was actually generated.

    An explanation is treated as unavailable when the candidate is listed in the
    run's ``explanation_unavailable_ids`` *or* when the summary is the
    orchestrator's non-fabricated unavailable sentinel — either signal honestly
    marks the entry as lacking a real explanation (Requirement 9.5).
    """

    if result.candidate_id in unavailable_ids:
        return False
    if result.explanation.summary == EXPLANATION_UNAVAILABLE_SUMMARY:
        return False
    return True


def _build_notices(
    *,
    reranked: bool,
    explanation_unavailable_count: int,
    excluded_count: int,
) -> list[str]:
    """Build human-readable degradation notices surfaced alongside the shortlist."""

    notices: list[str] = []
    if not reranked:
        notices.append(
            "Results were not LLM-reranked due to a reranker error; "
            "ordering fell back to composite scores."
        )
    if explanation_unavailable_count > 0:
        plural = "s" if explanation_unavailable_count != 1 else ""
        notices.append(
            f"Explanation unavailable for {explanation_unavailable_count} candidate{plural}."
        )
    if excluded_count > 0:
        plural = "s" if excluded_count != 1 else ""
        notices.append(
            f"{excluded_count} candidate{plural} were excluded before ranking."
        )
    return notices


def assemble_shortlist(run: RankingRun) -> RankedShortlist:
    """Assemble a recruiter-facing :class:`RankedShortlist` from a :class:`RankingRun`.

    Maps each :class:`RankingResult` into a :class:`ShortlistEntry` (preserving
    the orchestrator's rank ordering), derives the per-entry
    :attr:`~ShortlistEntry.explanation_available` flag, and surfaces the
    run-level honesty flags and notices:

        - ``reranked`` is carried through verbatim (Requirement 9.4);
        - ids whose explanation was the unavailable sentinel are reported in
          ``explanation_unavailable_ids`` and flagged per entry (Requirement 9.5);
        - candidates dropped before ranking are reported in
          ``excluded_candidate_ids``.

    All per-result numeric invariants (scores/confidence in ``[0,1]``, non-empty
    summary of at most 1000 characters) are already enforced by the ranking
    models; this function maps them through and re-asserts the summary bound
    defensively (Requirement 5.1).

    Args:
        run: the structured ranking run emitted by the orchestrator.

    Returns:
        A :class:`RankedShortlist` ready for the UI to render.
    """

    unavailable_ids: set[UUID] = {
        cid for cid in run.explanation_unavailable_ids if isinstance(cid, UUID)
    }

    entries = [
        ShortlistEntry.from_result(
            result,
            explanation_available=_is_explanation_available(result, unavailable_ids),
        )
        for result in run.results
    ]
    # Order defensively by rank so the UI always renders 1..N top-down even if a
    # caller hands us an unordered result list.
    entries.sort(key=lambda e: e.rank)

    job_id = run.results[0].job_id if run.results else None
    excluded_ids = list(run.excluded_candidate_ids)
    unavailable_list = list(run.explanation_unavailable_ids)

    notices = _build_notices(
        reranked=run.reranked,
        explanation_unavailable_count=len(unavailable_list),
        excluded_count=len(excluded_ids),
    )

    return RankedShortlist(
        job_id=job_id,
        entries=entries,
        reranked=run.reranked,
        explanation_unavailable_ids=unavailable_list,
        excluded_candidate_ids=excluded_ids,
        notices=notices,
    )


__all__ = [
    "RankedShortlist",
    "ShortlistEntry",
    "assemble_shortlist",
]

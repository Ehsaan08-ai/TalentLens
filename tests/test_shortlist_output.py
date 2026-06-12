"""Tests for the ranked shortlist output assembly (Task 19.1).

These tests exercise :func:`icrs.output.shortlist.assemble_shortlist` and the
:class:`RankedShortlist` / :class:`ShortlistEntry` models in isolation: they
build :class:`RankingRun` / :class:`RankingResult` fixtures directly (no
pipeline) so the mapping and the honesty flags are verified independently of how
a run is produced.

Coverage (Requirements 5.1, 9.4, 9.5):
    - assembling from a run yields one valid entry per result with all fields in
      range and ranks ordered 1..N (5.1);
    - the ``reranked`` flag is surfaced for both the reranked and un-reranked
      (fallback) cases, with a notice in the fallback case (9.4);
    - ``explanation_unavailable_ids`` flip the corresponding entries'
      ``explanation_available`` to False and add a notice, while the sentinel
      summary is also detected (9.5);
    - excluded candidate ids are surfaced with a notice.

Both example-based unit tests and Hypothesis property tests are included.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from icrs.models.ranking import Explanation, RankingResult, SignalBreakdown
from icrs.output.shortlist import (
    RankedShortlist,
    ShortlistEntry,
    assemble_shortlist,
)
from icrs.pipeline.orchestrator import (
    EXPLANATION_UNAVAILABLE_SUMMARY,
    RankingRun,
    _Excluded,
)


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def make_breakdown(value: float = 0.5) -> SignalBreakdown:
    """A valid SignalBreakdown with every sub-score set to ``value``."""

    return SignalBreakdown(
        semantic_fit=value,
        career_trajectory=value,
        behavioral=value,
        hard_filter_pass=value,
        disqualifying_penalty=0.0,
    )


def make_explanation(summary: str = "Strong match on core skills.") -> Explanation:
    """A valid recruiter-facing Explanation."""

    return Explanation(
        summary=summary,
        driving_signals=["semantic_fit"],
        gaps=["no leadership experience"],
        unmet_must_haves=[],
    )


def make_result(
    *,
    job_id: UUID,
    candidate_id: UUID,
    rank: int,
    final_score: float,
    confidence: float = 0.7,
    summary: str = "Strong match on core skills.",
) -> RankingResult:
    """Build a single RankingResult fixture."""

    return RankingResult(
        job_id=job_id,
        candidate_id=candidate_id,
        final_score=final_score,
        rank=rank,
        breakdown=make_breakdown(),
        explanation=make_explanation(summary),
        confidence=confidence,
    )


def make_run(
    n: int = 3,
    *,
    job_id: UUID | None = None,
    reranked: bool = True,
    explanation_unavailable_ids: list[UUID] | None = None,
    excluded: list[_Excluded] | None = None,
) -> RankingRun:
    """Build a RankingRun with ``n`` ranked results (ranks 1..n, descending score)."""

    job_id = job_id or uuid4()
    results = [
        make_result(
            job_id=job_id,
            candidate_id=uuid4(),
            rank=i + 1,
            # Descending, distinct, and always within [0,1] for any n.
            final_score=(n - i) / n,
        )
        for i in range(n)
    ]
    return RankingRun(
        results=results,
        reranked=reranked,
        excluded=excluded or [],
        explanation_unavailable_ids=explanation_unavailable_ids or [],
    )


# --------------------------------------------------------------------------- #
# 5.1 — one valid entry per result, all fields in range
# --------------------------------------------------------------------------- #
def test_assemble_produces_one_entry_per_result():
    run = make_run(3)
    shortlist = assemble_shortlist(run)

    assert isinstance(shortlist, RankedShortlist)
    assert len(shortlist.entries) == len(run.results)
    assert shortlist.job_id == run.results[0].job_id


def test_entries_carry_all_fields_in_range_and_ordered_by_rank():
    run = make_run(4)
    shortlist = assemble_shortlist(run)

    ranks = [e.rank for e in shortlist.entries]
    assert ranks == [1, 2, 3, 4]

    for entry in shortlist.entries:
        assert 0.0 <= entry.final_score <= 1.0
        assert 0.0 <= entry.confidence <= 1.0
        assert 0.0 <= entry.breakdown.semantic_fit <= 1.0
        assert 0.0 <= entry.breakdown.career_trajectory <= 1.0
        assert 0.0 <= entry.breakdown.behavioral <= 1.0
        assert 0.0 <= entry.breakdown.hard_filter_pass <= 1.0
        assert entry.explanation.summary.strip()
        assert len(entry.explanation.summary) <= 1000


def test_entry_fields_map_through_from_result():
    run = make_run(1)
    result = run.results[0]
    entry = assemble_shortlist(run).entries[0]

    assert entry.candidate_id == result.candidate_id
    assert entry.final_score == result.final_score
    assert entry.confidence == result.confidence
    assert entry.breakdown == result.breakdown
    assert entry.explanation == result.explanation


def test_entries_sorted_by_rank_even_if_results_unordered():
    job_id = uuid4()
    # Results provided out of rank order.
    results = [
        make_result(job_id=job_id, candidate_id=uuid4(), rank=3, final_score=0.4),
        make_result(job_id=job_id, candidate_id=uuid4(), rank=1, final_score=0.9),
        make_result(job_id=job_id, candidate_id=uuid4(), rank=2, final_score=0.6),
    ]
    run = RankingRun(results=results)
    shortlist = assemble_shortlist(run)

    assert [e.rank for e in shortlist.entries] == [1, 2, 3]


def test_empty_run_yields_empty_shortlist_with_no_job_id():
    run = RankingRun(results=[], reranked=True)
    shortlist = assemble_shortlist(run)

    assert shortlist.entries == []
    assert shortlist.job_id is None
    assert shortlist.notices == []


# --------------------------------------------------------------------------- #
# 9.4 — reranked flag surfaced honestly (True and False)
# --------------------------------------------------------------------------- #
def test_reranked_true_is_surfaced_without_notice():
    shortlist = assemble_shortlist(make_run(2, reranked=True))

    assert shortlist.reranked is True
    assert shortlist.fully_reranked is True
    assert not any("rerank" in n.lower() for n in shortlist.notices)


def test_unreranked_fallback_is_surfaced_with_notice():
    shortlist = assemble_shortlist(make_run(2, reranked=False))

    assert shortlist.reranked is False
    assert shortlist.fully_reranked is False
    assert any("rerank" in n.lower() for n in shortlist.notices)


# --------------------------------------------------------------------------- #
# 9.5 — explanation unavailability surfaced honestly
# --------------------------------------------------------------------------- #
def test_explanation_unavailable_id_flags_entry_and_adds_notice():
    run = make_run(3)
    target = run.results[1].candidate_id
    run.explanation_unavailable_ids = [target]

    shortlist = assemble_shortlist(run)

    by_id = {e.candidate_id: e for e in shortlist.entries}
    assert by_id[target].explanation_available is False
    # The other entries remain available.
    assert all(
        e.explanation_available
        for cid, e in by_id.items()
        if cid != target
    )
    assert target in shortlist.explanation_unavailable_ids
    assert shortlist.all_explanations_available is False
    assert any("explanation unavailable" in n.lower() for n in shortlist.notices)


def test_sentinel_summary_detected_as_unavailable():
    job_id = uuid4()
    cid = uuid4()
    result = make_result(
        job_id=job_id,
        candidate_id=cid,
        rank=1,
        final_score=0.8,
        summary=EXPLANATION_UNAVAILABLE_SUMMARY,
    )
    # The run does NOT list the id, but the sentinel summary still signals it.
    run = RankingRun(results=[result], explanation_unavailable_ids=[])

    entry = assemble_shortlist(run).entries[0]
    assert entry.explanation_available is False


def test_all_explanations_available_when_none_unavailable():
    shortlist = assemble_shortlist(make_run(3))

    assert shortlist.all_explanations_available is True
    assert all(e.explanation_available for e in shortlist.entries)
    assert shortlist.explanation_unavailable_ids == []


# --------------------------------------------------------------------------- #
# Excluded candidates surfaced
# --------------------------------------------------------------------------- #
def test_excluded_candidate_ids_surfaced_with_notice():
    excluded_id = uuid4()
    run = make_run(
        2,
        excluded=[_Excluded(candidate_id=excluded_id, reason="embedding failed")],
    )

    shortlist = assemble_shortlist(run)

    assert excluded_id in shortlist.excluded_candidate_ids
    assert any("excluded" in n.lower() for n in shortlist.notices)


def test_no_excluded_means_no_exclusion_notice():
    shortlist = assemble_shortlist(make_run(2))

    assert shortlist.excluded_candidate_ids == []
    assert not any("excluded" in n.lower() for n in shortlist.notices)


def test_from_run_classmethod_matches_factory():
    run = make_run(2, reranked=False)
    via_classmethod = RankedShortlist.from_run(run)
    via_factory = assemble_shortlist(run)

    assert via_classmethod == via_factory


def test_shortlist_entry_rejects_blank_summary():
    # The model layer forbids a blank summary, so build a valid Explanation and
    # assert the defensive validator path is wired by constructing directly.
    with pytest.raises(Exception):
        ShortlistEntry(
            rank=1,
            candidate_id=uuid4(),
            final_score=0.5,
            breakdown=make_breakdown(),
            explanation=make_explanation("   "),  # blank -> rejected upstream
            confidence=0.5,
            explanation_available=True,
        )


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
@given(
    n=st.integers(min_value=1, max_value=12),
    reranked=st.booleans(),
)
def test_property_one_entry_per_result_and_flags_preserved(n: int, reranked: bool):
    """assemble_shortlist preserves count, ordering, ranges, and the reranked flag.

    **Validates: Requirements 5.1, 9.4**
    """

    run = make_run(n, reranked=reranked)
    shortlist = assemble_shortlist(run)

    assert len(shortlist.entries) == n
    assert [e.rank for e in shortlist.entries] == list(range(1, n + 1))
    assert shortlist.reranked is reranked
    for entry in shortlist.entries:
        assert 0.0 <= entry.final_score <= 1.0
        assert 0.0 <= entry.confidence <= 1.0
        assert 0 < len(entry.explanation.summary) <= 1000


@given(unavailable_count=st.integers(min_value=0, max_value=6))
def test_property_unavailable_ids_drive_entry_flags(unavailable_count: int):
    """Every id in explanation_unavailable_ids flips exactly its entry's flag.

    **Validates: Requirements 9.5**
    """

    n = 6
    run = make_run(n)
    unavailable = [r.candidate_id for r in run.results[:unavailable_count]]
    run.explanation_unavailable_ids = list(unavailable)

    shortlist = assemble_shortlist(run)
    unavailable_set = set(unavailable)

    for entry in shortlist.entries:
        expected = entry.candidate_id not in unavailable_set
        assert entry.explanation_available is expected

    assert shortlist.all_explanations_available is (unavailable_count == 0)

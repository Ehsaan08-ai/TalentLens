"""Unit tests for the ranking output models (icrs.models.ranking).

These cover the validation contract from the design's "Model: RankingResult"
and Requirements 5.1 / 2.4:
    - SignalBreakdown sub-scores constrained to [0,1]
    - final_score and confidence constrained to [0,1]
    - rank is a positive integer
    - summary is a non-empty string of at most 1000 characters
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from icrs.models.ranking import (
    MAX_SUMMARY_CHARS,
    Explanation,
    RankingResult,
    SignalBreakdown,
)


def _valid_breakdown() -> SignalBreakdown:
    return SignalBreakdown(
        semantic_fit=0.8,
        career_trajectory=0.6,
        behavioral=0.5,
        hard_filter_pass=1.0,
        disqualifying_penalty=0.0,
    )


def _valid_explanation() -> Explanation:
    return Explanation(
        summary="Strong semantic fit with relevant leadership trajectory.",
        driving_signals=["semantic_fit", "career_trajectory"],
        gaps=["No Kubernetes experience"],
        unmet_must_haves=["Kubernetes"],
    )


def _valid_result(**overrides) -> RankingResult:
    base = dict(
        job_id=uuid4(),
        candidate_id=uuid4(),
        final_score=0.75,
        rank=1,
        breakdown=_valid_breakdown(),
        explanation=_valid_explanation(),
        confidence=0.9,
    )
    base.update(overrides)
    return RankingResult(**base)


# --- SignalBreakdown ---------------------------------------------------------


def test_signal_breakdown_accepts_in_range_subscores():
    bd = _valid_breakdown()
    assert bd.semantic_fit == 0.8
    assert bd.disqualifying_penalty == 0.0


@pytest.mark.parametrize(
    "field",
    [
        "semantic_fit",
        "career_trajectory",
        "behavioral",
        "hard_filter_pass",
        "disqualifying_penalty",
    ],
)
@pytest.mark.parametrize("bad_value", [-0.01, 1.01, 2.0, -1.0])
def test_signal_breakdown_rejects_out_of_range_subscore(field, bad_value):
    kwargs = dict(
        semantic_fit=0.5,
        career_trajectory=0.5,
        behavioral=0.5,
        hard_filter_pass=0.5,
        disqualifying_penalty=0.0,
    )
    kwargs[field] = bad_value
    with pytest.raises(ValidationError):
        SignalBreakdown(**kwargs)


def test_signal_breakdown_accepts_boundary_values():
    SignalBreakdown(
        semantic_fit=0.0,
        career_trajectory=1.0,
        behavioral=0.0,
        hard_filter_pass=1.0,
        disqualifying_penalty=1.0,
    )


# --- Explanation -------------------------------------------------------------


def test_explanation_defaults_list_fields_to_empty():
    exp = Explanation(summary="Adequate fit.")
    assert exp.driving_signals == []
    assert exp.gaps == []
    assert exp.unmet_must_haves == []


def test_explanation_rejects_empty_summary():
    with pytest.raises(ValidationError):
        Explanation(summary="")


def test_explanation_rejects_whitespace_only_summary():
    with pytest.raises(ValidationError):
        Explanation(summary="   \n\t ")


def test_explanation_accepts_summary_at_max_length():
    exp = Explanation(summary="x" * MAX_SUMMARY_CHARS)
    assert len(exp.summary) == MAX_SUMMARY_CHARS


def test_explanation_rejects_summary_over_max_length():
    with pytest.raises(ValidationError):
        Explanation(summary="x" * (MAX_SUMMARY_CHARS + 1))


# --- RankingResult -----------------------------------------------------------


def test_ranking_result_valid_construction():
    result = _valid_result()
    assert result.rank == 1
    assert 0.0 <= result.final_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.breakdown, SignalBreakdown)
    assert isinstance(result.explanation, Explanation)


@pytest.mark.parametrize("bad_score", [-0.01, 1.01, 5.0, -3.0])
def test_ranking_result_rejects_out_of_range_final_score(bad_score):
    with pytest.raises(ValidationError):
        _valid_result(final_score=bad_score)


@pytest.mark.parametrize("bad_conf", [-0.01, 1.01, 2.5])
def test_ranking_result_rejects_out_of_range_confidence(bad_conf):
    with pytest.raises(ValidationError):
        _valid_result(confidence=bad_conf)


@pytest.mark.parametrize("bad_rank", [0, -1, -100])
def test_ranking_result_rejects_non_positive_rank(bad_rank):
    with pytest.raises(ValidationError):
        _valid_result(rank=bad_rank)


def test_ranking_result_rejects_non_integer_rank():
    with pytest.raises(ValidationError):
        _valid_result(rank=1.5)


def test_ranking_result_boundary_scores_accepted():
    _valid_result(final_score=0.0, confidence=0.0)
    _valid_result(final_score=1.0, confidence=1.0)

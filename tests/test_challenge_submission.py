"""Tests for hackathon submission CSV export helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from icrs.output.challenge_submission import (
    CHALLENGE_ROW_COUNT,
    ChallengeSubmissionError,
    rank_response_to_submission_rows,
    write_submission_csv,
)

sys.path.append(str(Path("data/India_runs_data_and_ai_challenge").resolve()))
from validate_submission import validate_submission  # noqa: E402


def _rank_response(n: int = CHALLENGE_ROW_COUNT) -> tuple[dict, dict[str, str]]:
    results = []
    mapping: dict[str, str] = {}
    for i in range(n):
        internal_id = f"00000000-0000-0000-0000-{i + 1:012d}"
        source_id = f"CAND_{i + 1:07d}"
        mapping[internal_id] = source_id
        results.append(
            {
                "candidate_id": internal_id,
                "rank": i + 1,
                "final_score": 1.0 - (i * 0.001),
                "breakdown": {},
                "explanation": {"summary": f"Reason for {source_id}"},
                "confidence": 0.8,
            }
        )
    return {"results": results}, mapping


def test_submission_rows_use_challenge_header_order_and_ids() -> None:
    response, mapping = _rank_response()

    rows = rank_response_to_submission_rows(response, mapping)

    assert len(rows) == CHALLENGE_ROW_COUNT
    assert rows[0] == {
        "candidate_id": "CAND_0000001",
        "rank": "1",
        "score": "1.0000",
        "reasoning": "Reason for CAND_0000001",
    }


def test_equal_scores_are_tiebroken_by_candidate_id() -> None:
    response, mapping = _rank_response(100)
    response["results"][0]["final_score"] = 0.9
    response["results"][1]["final_score"] = 0.9
    mapping[response["results"][0]["candidate_id"]] = "CAND_0000002"
    mapping[response["results"][1]["candidate_id"]] = "CAND_0000001"

    rows = rank_response_to_submission_rows(response, mapping)

    tied = [row for row in rows if row["score"] == "0.9000"]
    assert [row["candidate_id"] for row in tied] == ["CAND_0000001", "CAND_0000002"]


def test_write_submission_csv_passes_provided_validator(tmp_path: Path) -> None:
    response, mapping = _rank_response()
    path = tmp_path / "team_talentlens.csv"

    write_submission_csv(path, response, mapping)

    assert validate_submission(path) == []


def test_missing_source_id_is_rejected() -> None:
    response, mapping = _rank_response()
    mapping.pop(response["results"][0]["candidate_id"])

    with pytest.raises(ChallengeSubmissionError, match="missing source candidate id"):
        rank_response_to_submission_rows(response, mapping)


def test_requires_one_hundred_ranked_candidates() -> None:
    response, mapping = _rank_response(99)

    with pytest.raises(ChallengeSubmissionError, match="at least 100"):
        rank_response_to_submission_rows(response, mapping)

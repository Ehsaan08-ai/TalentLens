"""Challenge submission CSV export helpers.

The hackathon validator expects a CSV with exactly these columns:

    candidate_id,rank,score,reasoning

The ranking API intentionally uses internal UUIDs, while the challenge dataset
uses source ids such as ``CAND_0004989``. This module bridges that final mile by
mapping ranked API UUIDs back to source candidate ids and emitting a validator
compatible top-100 CSV.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping

REQUIRED_HEADER = ("candidate_id", "rank", "score", "reasoning")
CHALLENGE_ROW_COUNT = 100
_CANDIDATE_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


class ChallengeSubmissionError(ValueError):
    """Raised when ranked output cannot be converted into challenge CSV format."""


def _result_value(result: Any, key: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(key)
    return getattr(result, key)


def _explanation_summary(result: Any) -> str:
    explanation = _result_value(result, "explanation") or {}
    if isinstance(explanation, Mapping):
        summary = explanation.get("summary", "")
    else:
        summary = getattr(explanation, "summary", "")
    summary = str(summary or "").strip()
    return summary or "Ranked by TalentLens hybrid semantic and recruiter-fit scoring."


def _resolve_source_id(result: Any, uuid_to_source_id: Mapping[str, str]) -> str:
    internal_id = str(_result_value(result, "candidate_id"))
    source_id = str(uuid_to_source_id.get(internal_id, "")).strip()
    if not source_id:
        raise ChallengeSubmissionError(
            f"missing source candidate id for ranked candidate {internal_id}"
        )
    if not _CANDIDATE_ID_RE.fullmatch(source_id):
        raise ChallengeSubmissionError(
            f"source candidate id must match CAND_XXXXXXX, got {source_id!r}"
        )
    return source_id


def rank_response_to_submission_rows(
    rank_response: Mapping[str, Any],
    uuid_to_source_id: Mapping[str, str],
    *,
    row_count: int = CHALLENGE_ROW_COUNT,
) -> list[dict[str, str]]:
    """Convert a RankResponse-shaped dict into challenge submission rows.

    Rows are ordered by descending score and source candidate id ascending for
    equal scores, matching the provided validator's tie-break rule. Ranks are
    then assigned contiguously from 1.
    """

    results = list(rank_response.get("results") or [])
    if len(results) < row_count:
        raise ChallengeSubmissionError(
            f"challenge submission requires at least {row_count} ranked candidates; "
            f"got {len(results)}"
        )

    prepared: list[tuple[str, float, str]] = []
    for result in results:
        source_id = _resolve_source_id(result, uuid_to_source_id)
        try:
            score = float(_result_value(result, "final_score"))
        except (TypeError, ValueError) as exc:
            raise ChallengeSubmissionError(
                f"candidate {source_id} has a non-numeric final_score"
            ) from exc
        if not 0.0 <= score <= 1.0:
            raise ChallengeSubmissionError(
                f"candidate {source_id} score must be in [0,1], got {score}"
            )
        prepared.append((source_id, score, _explanation_summary(result)))

    prepared.sort(key=lambda item: (-item[1], item[0]))
    top_rows = prepared[:row_count]

    return [
        {
            "candidate_id": source_id,
            "rank": str(rank),
            "score": f"{score:.4f}",
            "reasoning": reasoning,
        }
        for rank, (source_id, score, reasoning) in enumerate(top_rows, start=1)
    ]


def write_submission_csv(
    path: str | Path,
    rank_response: Mapping[str, Any],
    uuid_to_source_id: Mapping[str, str],
    *,
    row_count: int = CHALLENGE_ROW_COUNT,
) -> Path:
    """Write a validator-compatible challenge submission CSV."""

    output_path = Path(path)
    rows = rank_response_to_submission_rows(
        rank_response, uuid_to_source_id, row_count=row_count
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


__all__ = [
    "CHALLENGE_ROW_COUNT",
    "REQUIRED_HEADER",
    "ChallengeSubmissionError",
    "rank_response_to_submission_rows",
    "write_submission_csv",
]

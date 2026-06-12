"""Unit tests for the pure helpers of the Streamlit PoC dashboard (Task 19.2).

These tests exercise ONLY the importable, pure helpers — payload building from
recruiter inputs, the ``RankResponse`` → display-row transform, score /
confidence formatting, and run-level notice computation — using plain sample
dicts. They never import ``streamlit`` or launch a server, proving the module
imports and the helpers run without those dependencies (Requirements 5.1, 5.2).
"""

from __future__ import annotations

import json

import pytest

from icrs.ui.dashboard import (
    BREAKDOWN_FIELDS,
    JOB_TYPE_VALUES,
    CandidatePoolError,
    build_rank_payload,
    compute_notices,
    confidence_label,
    format_confidence,
    format_score,
    parse_candidate_pool,
    transform_response_to_rows,
)


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #
def _sample_candidate_pool() -> list[dict]:
    return [
        {
            "structured_fields": {"current_title": "Backend Engineer"},
            "free_text": "Builds data pipelines.",
            "external_handles": {"github": "ira"},
        },
        {
            "structured_fields": {"current_title": "Operations Manager"},
            "free_text": "Leads support teams.",
            "external_handles": {},
        },
    ]


def _sample_response() -> dict:
    """A RankResponse-shaped dict mirroring the API's output contract."""

    return {
        "results": [
            {
                "candidate_id": "11111111-1111-1111-1111-111111111111",
                "rank": 2,
                "final_score": 0.612345,
                "breakdown": {
                    "semantic_fit": 0.5,
                    "career_trajectory": 0.4,
                    "behavioral": 0.3,
                    "hard_filter_pass": 1.0,
                    "disqualifying_penalty": 0.0,
                },
                "explanation": {
                    "summary": "Solid backend fit with adjacent ML exposure.",
                    "driving_signals": ["Spark", "Airflow"],
                    "gaps": ["Limited production ML"],
                    "unmet_must_haves": ["3y ML in production"],
                },
                "confidence": 0.81,
            },
            {
                "candidate_id": "22222222-2222-2222-2222-222222222222",
                "rank": 1,
                "final_score": 0.7,
                "breakdown": {
                    "semantic_fit": 0.8,
                    "career_trajectory": 0.7,
                    "behavioral": 0.6,
                    "hard_filter_pass": 1.0,
                    "disqualifying_penalty": 0.1,
                },
                "explanation": {
                    "summary": "Strong semantic alignment.",
                    "driving_signals": ["Domain match"],
                    "gaps": [],
                    "unmet_must_haves": [],
                },
                "confidence": 0.2,
            },
        ],
        "reranked": True,
        "excluded_candidate_ids": [],
        "explanation_unavailable_ids": [],
    }


# --------------------------------------------------------------------------- #
# parse_candidate_pool
# --------------------------------------------------------------------------- #
def test_parse_candidate_pool_accepts_full_objects():
    rows = parse_candidate_pool(json.dumps(_sample_candidate_pool()))
    assert len(rows) == 2
    assert rows[0]["structured_fields"] == {"current_title": "Backend Engineer"}
    assert rows[0]["free_text"] == "Builds data pipelines."
    assert rows[0]["external_handles"] == {"github": "ira"}


def test_parse_candidate_pool_fills_defaults_for_sparse_objects():
    rows = parse_candidate_pool(json.dumps([{"free_text": "only prose"}]))
    assert rows == [
        {"structured_fields": {}, "free_text": "only prose", "external_handles": {}}
    ]


def test_parse_candidate_pool_rejects_empty_text():
    with pytest.raises(CandidatePoolError):
        parse_candidate_pool("   ")


def test_parse_candidate_pool_rejects_invalid_json():
    with pytest.raises(CandidatePoolError):
        parse_candidate_pool("{not json")


def test_parse_candidate_pool_rejects_non_array():
    with pytest.raises(CandidatePoolError):
        parse_candidate_pool(json.dumps({"structured_fields": {}}))


def test_parse_candidate_pool_rejects_empty_array():
    with pytest.raises(CandidatePoolError):
        parse_candidate_pool("[]")


def test_parse_candidate_pool_converts_generic_json_objects():
    """Generic JSON objects (not RawCandidate-shaped) are now converted via
    _generic_dict_to_raw_candidate instead of being rejected (Bug 2 fix)."""
    result = parse_candidate_pool(json.dumps([{"unexpected": 1}]))
    assert len(result) == 1
    # The generic converter puts unknown keys into structured_fields
    assert result[0]["structured_fields"] == {"unexpected": 1}
    assert result[0]["free_text"] == ""
    assert result[0]["external_handles"] == {}


def test_parse_candidate_pool_rejects_wrong_field_types():
    with pytest.raises(CandidatePoolError):
        parse_candidate_pool(json.dumps([{"free_text": 123}]))
    with pytest.raises(CandidatePoolError):
        parse_candidate_pool(json.dumps([{"structured_fields": []}]))
    with pytest.raises(CandidatePoolError):
        parse_candidate_pool(json.dumps([{"external_handles": {"github": 5}}]))


def test_parse_candidate_pool_jsonl_standard():
    lines = (
        '{"structured_fields": {"role": "Engineer"}, "free_text": "prose 1", "external_handles": {}}\n'
        '{"structured_fields": {"role": "Designer"}, "free_text": "prose 2", "external_handles": {}}'
    )
    rows = parse_candidate_pool(lines, filename="pool.jsonl")
    assert len(rows) == 2
    assert rows[0]["free_text"] == "prose 1"
    assert rows[1]["structured_fields"] == {"role": "Designer"}


def test_parse_candidate_pool_jsonl_redrob():
    lines = (
        '{"candidate_id": "CAND1", "profile": {"summary": "summary 1"}, "skills": ["Python"]}\n'
        '{"candidate_id": "CAND2", "profile": {"summary": "summary 2"}, "skills": ["Go"]}'
    )
    rows = parse_candidate_pool(lines, filename="pool.jsonl")
    assert len(rows) == 2
    assert "summary 1" in rows[0]["free_text"]
    assert rows[0]["structured_fields"]["skills"] == ["Python"]
    # Check that protected proxies are not mapped
    assert "anonymized_name" not in json.dumps(rows)


def test_parse_candidate_pool_csv_standard():
    csv_text = (
        "structured_fields,free_text,external_handles\n"
        '"{""current_title"": ""Engineer""}","prose 1","{}"\n'
        '"{}","prose 2","{}"'
    )
    rows = parse_candidate_pool(csv_text, filename="pool.csv")
    assert len(rows) == 2
    assert rows[0]["free_text"] == "prose 1"
    assert rows[0]["structured_fields"] == {"current_title": "Engineer"}


def test_parse_candidate_pool_csv_redrob():
    csv_text = (
        "candidate_id,skills,certifications,education,profile\n"
        'CAND1,"Python, Spark",AWS Architect,"[]","{""summary"": ""prose 1""}"\n'
        'CAND2,"Go","","[]","{""summary"": ""prose 2""}"'
    )
    rows = parse_candidate_pool(csv_text, filename="pool.csv")
    assert len(rows) == 2
    assert rows[0]["structured_fields"]["skills"] == ["Python", "Spark"]
    assert rows[0]["structured_fields"]["certifications"] == ["AWS Architect"]
    assert "prose 1" in rows[0]["free_text"]


def test_parse_candidate_pool_csv_generic_flat():
    csv_text = (
        "id,name,email,skills,summary,experience,country\n"
        "CAND1,John Doe,john@example.com,\"Python, AWS\",Great developer,5 years,USA\n"
        "CAND2,Jane Doe,jane@example.com,\"Go\",Awesome engineer,10 years,Canada"
    )
    rows = parse_candidate_pool(csv_text, filename="pool.csv")
    assert len(rows) == 2
    
    # Check generic flat mappings
    assert rows[0]["free_text"] == "Great developer"
    assert rows[0]["structured_fields"]["skills"] == ["Python", "AWS"]
    assert rows[0]["structured_fields"]["experience"] == "5 years"
    
    # Check protected proxy exclusion
    blob = json.dumps(rows)
    assert "John Doe" not in blob
    assert "john@example.com" not in blob
    assert "USA" not in blob
    assert "country" not in blob


# --------------------------------------------------------------------------- #
# build_rank_payload
# --------------------------------------------------------------------------- #
def test_build_rank_payload_produces_rank_request_shape():
    payload = build_rank_payload(
        "  Senior backend engineer  ",
        "TECHNICAL",
        _sample_candidate_pool(),
        title="  Backend Eng  ",
    )
    assert payload["raw_jd"] == "Senior backend engineer"  # trimmed
    assert payload["job_type"] == "TECHNICAL"
    assert payload["title"] == "Backend Eng"  # trimmed
    assert len(payload["candidates"]) == 2
    assert set(payload["candidates"][0]) == {
        "structured_fields",
        "free_text",
        "external_handles",
    }


def test_build_rank_payload_omits_blank_title():
    payload = build_rank_payload("JD text", "SALES", _sample_candidate_pool(), title="   ")
    assert "title" not in payload


def test_build_rank_payload_rejects_empty_jd():
    with pytest.raises(CandidatePoolError):
        build_rank_payload("   ", "TECHNICAL", _sample_candidate_pool())


def test_build_rank_payload_rejects_empty_pool():
    with pytest.raises(CandidatePoolError):
        build_rank_payload("JD", "TECHNICAL", [])


def test_build_rank_payload_rejects_unknown_job_type():
    with pytest.raises(CandidatePoolError):
        build_rank_payload("JD", "INVALID", _sample_candidate_pool())


def test_all_job_type_values_are_accepted():
    for jt in JOB_TYPE_VALUES:
        payload = build_rank_payload("JD", jt, _sample_candidate_pool())
        assert payload["job_type"] == jt


# --------------------------------------------------------------------------- #
# format_score / confidence
# --------------------------------------------------------------------------- #
def test_format_score_rounds_to_two_decimals():
    assert format_score(0.612345) == "0.61"
    assert format_score(0.005) in {"0.01", "0.00"}  # banker's rounding tolerated
    assert format_score(1) == "1.00"


def test_format_score_handles_none():
    assert format_score(None) == "—"


def test_confidence_label_bands():
    assert confidence_label(0.9) == "High"
    assert confidence_label(0.5) == "Moderate"
    assert confidence_label(0.1) == "Low"
    assert confidence_label(None) == "Unknown"


def test_format_confidence_combines_band_and_value():
    assert format_confidence(0.78) == "High (0.78)"
    assert format_confidence(None) == "Unknown"


# --------------------------------------------------------------------------- #
# transform_response_to_rows
# --------------------------------------------------------------------------- #
def test_transform_orders_rows_by_rank():
    rows = transform_response_to_rows(_sample_response())
    assert [r["rank"] for r in rows] == [1, 2]
    assert rows[0]["candidate_id"] == "22222222-2222-2222-2222-222222222222"


def test_transform_formats_scores_and_confidence_without_false_precision():
    rows = transform_response_to_rows(_sample_response())
    rank2 = next(r for r in rows if r["rank"] == 2)
    assert rank2["final_score_display"] == "0.61"
    assert rank2["confidence_display"] == "High (0.81)"
    rank1 = next(r for r in rows if r["rank"] == 1)
    assert rank1["confidence_label"] == "Low"


def test_transform_carries_explanation_fields_and_breakdown():
    rows = transform_response_to_rows(_sample_response())
    rank2 = next(r for r in rows if r["rank"] == 2)
    assert rank2["summary"] == "Solid backend fit with adjacent ML exposure."
    assert rank2["driving_signals"] == ["Spark", "Airflow"]
    assert rank2["gaps"] == ["Limited production ML"]
    assert rank2["unmet_must_haves"] == ["3y ML in production"]
    assert set(rank2["breakdown"]) == set(BREAKDOWN_FIELDS)
    assert rank2["breakdown"]["hard_filter_pass"] == 1.0


def test_transform_marks_explanation_unavailable():
    response = _sample_response()
    response["explanation_unavailable_ids"] = [
        "11111111-1111-1111-1111-111111111111"
    ]
    rows = transform_response_to_rows(response)
    rank2 = next(r for r in rows if r["rank"] == 2)
    assert rank2["explanation_available"] is False
    rank1 = next(r for r in rows if r["rank"] == 1)
    assert rank1["explanation_available"] is True


def test_transform_handles_empty_results():
    assert transform_response_to_rows({"results": []}) == []
    assert transform_response_to_rows({}) == []


# --------------------------------------------------------------------------- #
# compute_notices
# --------------------------------------------------------------------------- #
def test_compute_notices_clean_run_has_none():
    assert compute_notices(_sample_response()) == []


def test_compute_notices_flags_not_reranked():
    response = _sample_response()
    response["reranked"] = False
    notices = compute_notices(response)
    assert any("NOT LLM-reranked" in n for n in notices)


def test_compute_notices_flags_unavailable_explanations():
    response = _sample_response()
    response["explanation_unavailable_ids"] = ["a", "b"]
    notices = compute_notices(response)
    assert any("Explanation unavailable for 2 candidates" in n for n in notices)


def test_compute_notices_flags_excluded_candidates():
    response = _sample_response()
    response["excluded_candidate_ids"] = ["x"]
    notices = compute_notices(response)
    assert any("1 candidate was excluded" in n for n in notices)


def test_compute_notices_missing_reranked_key_defaults_to_no_notice():
    # An absent 'reranked' key should not be treated as a degradation.
    assert compute_notices({"results": []}) == []


def test_transform_resolves_candidate_names():
    # Parse a candidate pool to generate client UUIDs
    csv_text = (
        "id,name,skills,summary\n"
        "CAND1,Ira Vora,Python,Good developer\n"
        "CAND2,Anil Dev,Go,Great developer"
    )
    candidates = parse_candidate_pool(csv_text, filename="pool.csv")
    assert len(candidates) == 2

    # Map them manually as done in the UI button trigger
    import uuid
    uuid1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "0_Ira Vora"))
    uuid2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, "1_Anil Dev"))
    uuid_to_name = {uuid1: "Ira Vora", uuid2: "Anil Dev"}

    # Mock response from backend
    response = {
        "results": [
            {
                "candidate_id": uuid1,
                "rank": 1,
                "final_score": 0.85,
                "breakdown": {
                    "semantic_fit": 0.9,
                    "career_trajectory": 0.8,
                    "behavioral": 0.7,
                    "hard_filter_pass": 1.0,
                    "disqualifying_penalty": 0.0
                },
                "explanation": {
                    "summary": "Great match",
                    "driving_signals": ["Python"],
                    "gaps": [],
                    "unmet_must_haves": []
                },
                "confidence": 0.75
            },
            {
                "candidate_id": uuid2,
                "rank": 2,
                "final_score": 0.75,
                "breakdown": {
                    "semantic_fit": 0.8,
                    "career_trajectory": 0.7,
                    "behavioral": 0.6,
                    "hard_filter_pass": 1.0,
                    "disqualifying_penalty": 0.0
                },
                "explanation": {
                    "summary": "Good match",
                    "driving_signals": ["Go"],
                    "gaps": [],
                    "unmet_must_haves": []
                },
                "confidence": 0.65
            }
        ]
    }

    # Transform response without candidates mapping -> remains UUID
    rows_no_mapping = transform_response_to_rows(response)
    assert rows_no_mapping[0]["candidate_id"] == uuid1
    assert rows_no_mapping[1]["candidate_id"] == uuid2

    # Transform response with candidates mapping -> resolves to display names
    rows_with_mapping = transform_response_to_rows(response, uuid_to_name)
    assert rows_with_mapping[0]["candidate_id"] == "Ira Vora"
    assert rows_with_mapping[1]["candidate_id"] == "Anil Dev"

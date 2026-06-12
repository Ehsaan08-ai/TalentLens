"""Tests for the Redrob -> RawCandidate dataset adapter (icrs.ingest.redrob_adapter).

These verify the adapter resolves the two real-world blockers — JSONL/object
container shapes and the Redrob field schema — and that it produces input the
ICRS normalizer accepts, while excluding protected-proxy fields (Requirement
7.1).
"""

from __future__ import annotations

import json

import pytest

from icrs.ingest.redrob_adapter import (
    PROTECTED_PROXY_SOURCE_FIELDS,
    convert_pool,
    load_redrob_records,
    redrob_to_raw_candidate,
)
from icrs.models.candidate import RawCandidate
from icrs.pipeline.enricher import CandidateEnricher


def _redrob_record() -> dict:
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Ira Vora",
            "headline": "Backend Engineer | SQL, Spark, Cloud",
            "summary": "6.9 years building data pipelines and backend systems.",
            "location": "Toronto",
            "country": "Canada",
            "years_of_experience": 6.9,
            "current_title": "Backend Engineer",
            "current_company": "Mindtree",
            "current_company_size": "10001+",
            "current_industry": "IT Services",
        },
        "career_history": [
            {
                "company": "Mindtree",
                "title": "Backend Engineer",
                "start_date": "2024-03-08",
                "end_date": None,
                "duration_months": 27,
                "is_current": True,
                "industry": "IT Services",
                "company_size": "10001+",
                "description": "Streaming pipelines on Kafka and Spark Streaming.",
            },
            {
                "company": "Dunder Mifflin",
                "title": "Analytics Engineer",
                "start_date": "2019-07-03",
                "end_date": "2024-01-08",
                "duration_months": 55,
                "is_current": False,
                "industry": "Paper",
                "company_size": "201-500",
                "description": "Airflow + Spark batch pipelines feeding Snowflake.",
            },
        ],
        "education": [
            {
                "institution": "Lovely Professional University",
                "degree": "B.E.",
                "field_of_study": "Computer Science",
                "start_year": 2017,
                "end_year": 2020,
                "grade": "8.24 CGPA",
                "tier": "tier_3",
            }
        ],
        "skills": [
            {"name": "Spark", "proficiency": "advanced", "endorsements": 9},
            {"name": "Airflow", "proficiency": "advanced", "endorsements": 5},
        ],
        "certifications": [
            {"name": "AWS Solutions Architect", "issuer": "AWS", "year": 2022}
        ],
        "redrob_signals": {"github_activity_score": 42},
    }


# --------------------------------------------------------------------------- #
# Field mapping
# --------------------------------------------------------------------------- #
def test_maps_to_raw_candidate_shape():
    out = redrob_to_raw_candidate(_redrob_record())
    assert set(out) == {"structured_fields", "free_text", "external_handles"}

    sf = out["structured_fields"]
    assert [r["title"] for r in sf["roles"]] == ["Backend Engineer", "Analytics Engineer"]
    assert sf["roles"][0]["company"] == "Mindtree"
    assert sf["roles"][0]["start"] == "2024-03-08"
    assert "end" not in sf["roles"][0]  # ongoing role -> no end mapped
    assert sf["roles"][1]["end"] == "2024-01-08"
    assert sf["skills"] == ["Spark", "Airflow"]
    assert sf["certifications"] == ["AWS Solutions Architect"]
    assert sf["education"][0]["field_of_study"] == "Computer Science"
    assert sf["source_candidate_id"] == "CAND_0000001"


def test_free_text_includes_summary_and_role_descriptions():
    out = redrob_to_raw_candidate(_redrob_record())
    text = out["free_text"]
    assert "Backend Engineer | SQL, Spark, Cloud" in text
    assert "building data pipelines" in text
    assert "Kafka and Spark Streaming" in text


# --------------------------------------------------------------------------- #
# Fairness: protected proxies excluded (Requirement 7.1)
# --------------------------------------------------------------------------- #
def test_protected_proxy_fields_are_not_mapped():
    out = redrob_to_raw_candidate(_redrob_record())
    blob = json.dumps(out)
    # Name, location, and country must not appear anywhere in the scoring input.
    assert "Ira Vora" not in blob
    assert "Toronto" not in blob
    assert "Canada" not in blob
    assert "anonymized_name" not in blob


def test_protected_proxy_source_fields_documented():
    for field in ("anonymized_name", "location", "country"):
        assert field in PROTECTED_PROXY_SOURCE_FIELDS


# --------------------------------------------------------------------------- #
# Output is accepted by the real ICRS normalizer (end-to-end shape check)
# --------------------------------------------------------------------------- #
def test_converted_candidate_normalizes_cleanly():
    out = redrob_to_raw_candidate(_redrob_record())
    raw = RawCandidate(**out)
    profile = CandidateEnricher().normalize(raw)

    assert [r.title for r in profile.roles] == ["Backend Engineer", "Analytics Engineer"]
    assert "Spark" in profile.explicit_skills
    assert profile.education[0].field_of_study == "Computer Science"
    # Tenure summed from role dates is a positive whole-month count.
    assert profile.total_tenure_months > 0


# --------------------------------------------------------------------------- #
# Loader: array, single object, wrapper, and JSONL
# --------------------------------------------------------------------------- #
def test_convert_pool_returns_json_array():
    pool = convert_pool([_redrob_record(), _redrob_record()])
    assert isinstance(pool, list)
    assert len(pool) == 2


def test_load_json_array(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([_redrob_record()]), encoding="utf-8")
    records = load_redrob_records(path)
    assert len(records) == 1
    assert records[0]["candidate_id"] == "CAND_0000001"


def test_load_single_object(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(json.dumps(_redrob_record()), encoding="utf-8")
    assert len(load_redrob_records(path)) == 1


def test_load_candidates_wrapper(tmp_path):
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"candidates": [_redrob_record()]}), encoding="utf-8")
    assert len(load_redrob_records(path)) == 1


def test_load_jsonl_and_limit(tmp_path):
    path = tmp_path / "candidates.jsonl"
    lines = "\n".join(json.dumps(_redrob_record()) for _ in range(5))
    path.write_text(lines + "\n", encoding="utf-8")

    assert len(load_redrob_records(path)) == 5
    assert len(load_redrob_records(path, limit=2)) == 2


def test_load_jsonl_rejects_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_redrob_record()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_redrob_records(path)

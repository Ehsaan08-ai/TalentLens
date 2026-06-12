"""Unit tests for the Task 2.1 job-description data models and their validation.

Covers RequirementCategory/Tier/SeniorityBand/JobType enums and the
Requirement / RequirementVector / JobDescription validation rules:
    - non-empty raw_text and role_intent (Requirements 1.7, 1.1)
    - at least one MUST_HAVE requirement (Requirements 1.5, 1.8)
    - per-category weight normalization to 1.0 with DISQUALIFYING excluded
      (Requirement 1.4)
    - empty-by-default implicit_expectations / culture_signals (Requirement 1.3)
"""

from __future__ import annotations

import math
from uuid import uuid4

import pytest
from pydantic import ValidationError

from icrs.models import (
    WEIGHT_SUM_TOLERANCE,
    JobDescription,
    JobType,
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)


def _req(
    text: str,
    category: RequirementCategory,
    weight: float = 0.0,
    tier: RequirementTier = RequirementTier.STRUCTURAL,
) -> Requirement:
    return Requirement(text=text, category=category, tier=tier, weight=weight)


# ----- Requirement -----


def test_requirement_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        _req("   ", RequirementCategory.MUST_HAVE)


def test_requirement_weight_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        _req("x", RequirementCategory.MUST_HAVE, weight=1.5)
    with pytest.raises(ValidationError):
        _req("x", RequirementCategory.MUST_HAVE, weight=-0.1)


def test_requirement_embedding_optional_defaults_none() -> None:
    r = _req("Python", RequirementCategory.MUST_HAVE)
    assert r.embedding is None


def test_disqualifying_is_not_weighted() -> None:
    assert _req("felony", RequirementCategory.DISQUALIFYING).is_weighted is False
    assert _req("Python", RequirementCategory.MUST_HAVE).is_weighted is True
    assert _req("AWS", RequirementCategory.NICE_TO_HAVE).is_weighted is True


# ----- RequirementVector -----


def _vector(**overrides) -> RequirementVector:
    defaults = dict(
        role_intent="Build and operate the data platform",
        seniority_band=SeniorityBand.SENIOR,
        requirements=[_req("Python", RequirementCategory.MUST_HAVE, weight=1.0)],
    )
    defaults.update(overrides)
    return RequirementVector(**defaults)


def test_vector_requires_at_least_one_must_have() -> None:
    with pytest.raises(ValidationError):
        RequirementVector(
            role_intent="x",
            seniority_band=SeniorityBand.MID,
            requirements=[_req("AWS", RequirementCategory.NICE_TO_HAVE, weight=1.0)],
        )


def test_vector_rejects_empty_role_intent() -> None:
    with pytest.raises(ValidationError):
        _vector(role_intent="   ")


def test_vector_collections_default_empty() -> None:
    v = _vector()
    assert v.implicit_expectations == []
    assert v.culture_signals == []


def test_must_have_weights_normalized_to_one() -> None:
    v = _vector(
        requirements=[
            _req("Python", RequirementCategory.MUST_HAVE, weight=0.3),
            _req("SQL", RequirementCategory.MUST_HAVE, weight=0.1),
        ]
    )
    total = sum(r.weight for r in v.must_haves)
    assert math.isclose(total, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE)
    # Proportions preserved: 3:1 -> 0.75 / 0.25
    assert math.isclose(v.must_haves[0].weight, 0.75, abs_tol=WEIGHT_SUM_TOLERANCE)
    assert math.isclose(v.must_haves[1].weight, 0.25, abs_tol=WEIGHT_SUM_TOLERANCE)


def test_each_category_normalized_independently() -> None:
    v = _vector(
        requirements=[
            _req("Python", RequirementCategory.MUST_HAVE, weight=0.2),
            _req("SQL", RequirementCategory.MUST_HAVE, weight=0.2),
            _req("AWS", RequirementCategory.NICE_TO_HAVE, weight=0.5),
            _req("k8s", RequirementCategory.NICE_TO_HAVE, weight=0.5),
        ]
    )
    assert math.isclose(sum(r.weight for r in v.must_haves), 1.0, abs_tol=WEIGHT_SUM_TOLERANCE)
    assert math.isclose(
        sum(r.weight for r in v.nice_to_haves), 1.0, abs_tol=WEIGHT_SUM_TOLERANCE
    )


def test_zero_weights_distributed_equally() -> None:
    v = _vector(
        requirements=[
            _req("Python", RequirementCategory.MUST_HAVE, weight=0.0),
            _req("SQL", RequirementCategory.MUST_HAVE, weight=0.0),
        ]
    )
    assert all(math.isclose(r.weight, 0.5, abs_tol=WEIGHT_SUM_TOLERANCE) for r in v.must_haves)


def test_disqualifying_excluded_from_weighting() -> None:
    v = _vector(
        requirements=[
            _req("Python", RequirementCategory.MUST_HAVE, weight=1.0),
            _req("expired security clearance", RequirementCategory.DISQUALIFYING, weight=0.9),
        ]
    )
    weighted = v.weighted_requirements()
    assert all(r.category is not RequirementCategory.DISQUALIFYING for r in weighted)
    assert len(v.disqualifiers) == 1
    # MUST_HAVE still normalizes to 1.0 on its own, ignoring the disqualifier.
    assert math.isclose(sum(r.weight for r in v.must_haves), 1.0, abs_tol=WEIGHT_SUM_TOLERANCE)


# ----- JobDescription -----


def test_jobdescription_rejects_empty_raw_text() -> None:
    with pytest.raises(ValidationError):
        JobDescription(raw_text="   ", job_type=JobType.TECHNICAL)


def test_jobdescription_parsed_defaults_none() -> None:
    jd = JobDescription(raw_text="Senior data engineer ...", job_type=JobType.TECHNICAL)
    assert jd.parsed is None
    assert jd.title == ""


def test_jobdescription_accepts_parsed_vector() -> None:
    jd = JobDescription(
        raw_text="Senior data engineer wanted",
        title="Senior Data Engineer",
        job_type=JobType.TECHNICAL,
        parsed=_vector(job_id=uuid4()),
    )
    assert jd.parsed is not None
    assert jd.parsed.must_haves

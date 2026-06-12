"""Unit tests for the Task 9.1 weight-profile registry, selection, and validation.

Covers Requirements 4.2 (select-by-job-type with default fallback), 4.3 (four
fusion weights in [0,1] summing to 1.0 within ±0.001; w5 independent), and 4.8
(reject a mis-summed profile, fall back to default, and record the indication).
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from icrs.models.job import JobType
from icrs.scoring.weights import (
    DEFAULT_WEIGHT_PROFILE,
    WEIGHT_PROFILES,
    WEIGHT_SUM_TOLERANCE,
    WeightProfile,
    WeightProfileSelection,
    get_weight_profile,
    select_weight_profile,
)

# Expected design table: (w1, w2, w3, w4, w5) per job type plus the default.
EXPECTED_DEFAULT = (0.40, 0.20, 0.15, 0.25, 0.30)
EXPECTED_BY_JOB_TYPE = {
    JobType.TECHNICAL: (0.35, 0.15, 0.30, 0.20, 0.30),
    JobType.LEADERSHIP: (0.30, 0.35, 0.10, 0.25, 0.35),
    JobType.GENERALIST: (0.45, 0.20, 0.10, 0.25, 0.25),
    JobType.SALES: (0.35, 0.30, 0.10, 0.25, 0.30),
}


def _tuple(p: WeightProfile) -> tuple[float, float, float, float, float]:
    return (p.w1, p.w2, p.w3, p.w4, p.w5)


# ----- WeightProfile model --------------------------------------------------


def test_default_profile_matches_design_table() -> None:
    assert _tuple(DEFAULT_WEIGHT_PROFILE) == EXPECTED_DEFAULT


@pytest.mark.parametrize("job_type", list(JobType))
def test_registry_values_match_design_table(job_type: JobType) -> None:
    assert _tuple(WEIGHT_PROFILES[job_type]) == EXPECTED_BY_JOB_TYPE[job_type]


def test_every_registry_profile_and_default_sum_to_one() -> None:
    for profile in (DEFAULT_WEIGHT_PROFILE, *WEIGHT_PROFILES.values()):
        assert profile.sums_to_one()
        assert math.isclose(profile.fusion_sum, 1.0, abs_tol=WEIGHT_SUM_TOLERANCE)


def test_w5_is_independent_of_fusion_sum() -> None:
    # w5 (penalty coefficient) is excluded from the sum-to-one constraint.
    p = WeightProfile(w1=0.40, w2=0.20, w3=0.15, w4=0.25, w5=0.30)
    assert p.sums_to_one()
    assert p.fusion_sum == pytest.approx(1.0)
    # A different w5 does not affect validity.
    p2 = WeightProfile(w1=0.40, w2=0.20, w3=0.15, w4=0.25, w5=0.99)
    assert p2.sums_to_one()


def test_descriptive_aliases() -> None:
    p = WEIGHT_PROFILES[JobType.TECHNICAL]
    assert p.semantic == p.w1
    assert p.trajectory == p.w2
    assert p.behavioral == p.w3
    assert p.hard_filter == p.w4
    assert p.penalty == p.w5


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_component_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        WeightProfile(w1=bad, w2=0.2, w3=0.2, w4=0.2, w5=0.3)


def test_profile_is_immutable() -> None:
    with pytest.raises(ValidationError):
        DEFAULT_WEIGHT_PROFILE.w1 = 0.99  # type: ignore[misc]


# ----- Selection per job type (Requirement 4.2) -----------------------------


@pytest.mark.parametrize("job_type", list(JobType))
def test_select_returns_configured_profile_per_job_type(job_type: JobType) -> None:
    result = select_weight_profile(job_type)
    assert isinstance(result, WeightProfileSelection)
    assert _tuple(result.profile) == EXPECTED_BY_JOB_TYPE[job_type]
    assert result.requested_job_type is job_type
    assert result.fell_back is False
    assert result.rejected is False
    assert result.reason is None


def test_get_weight_profile_convenience_wrapper() -> None:
    assert get_weight_profile(JobType.SALES) is WEIGHT_PROFILES[JobType.SALES]


# ----- Fallback for unknown / None type (Requirement 4.2) -------------------


def test_none_job_type_falls_back_to_default_with_indication() -> None:
    result = select_weight_profile(None)
    assert result.profile is DEFAULT_WEIGHT_PROFILE
    assert result.requested_job_type is None
    assert result.fell_back is True
    assert result.rejected is False
    assert result.reason is not None
    assert "default" in result.reason.lower()


def test_unknown_job_type_falls_back_to_default() -> None:
    # An empty/partial registry models a job type with no configured profile.
    partial = {JobType.SALES: WEIGHT_PROFILES[JobType.SALES]}
    result = select_weight_profile(JobType.TECHNICAL, registry=partial)
    assert result.profile is DEFAULT_WEIGHT_PROFILE
    assert result.fell_back is True
    assert result.rejected is False
    assert "TECHNICAL" in (result.reason or "")


# ----- Validation tolerance (Requirement 4.3) -------------------------------


@pytest.mark.parametrize("fusion_sum", [0.9995, 1.0, 1.0005])
def test_within_tolerance_profiles_accepted(fusion_sum: float) -> None:
    # Put the entire deviation on w1 so the fusion sum is exactly `fusion_sum`.
    w1 = fusion_sum - (0.20 + 0.15 + 0.25)
    profile = WeightProfile(w1=w1, w2=0.20, w3=0.15, w4=0.25, w5=0.30)
    assert profile.sums_to_one()
    result = select_weight_profile(JobType.TECHNICAL, registry={JobType.TECHNICAL: profile})
    assert result.rejected is False
    assert result.fell_back is False
    assert result.profile is profile


@pytest.mark.parametrize("fusion_sum", [0.998, 1.002, 0.90, 1.10])
def test_outside_tolerance_profiles_rejected(fusion_sum: float) -> None:
    w1 = fusion_sum - (0.20 + 0.15 + 0.25)
    profile = WeightProfile(w1=w1, w2=0.20, w3=0.15, w4=0.25, w5=0.30)
    assert profile.sums_to_one() is False


# ----- Reject-and-fallback-with-indication (Requirement 4.8) ----------------


def test_invalid_profile_rejected_falls_back_and_records_indication() -> None:
    # Components are individually valid [0,1] but their fusion sum is 1.10.
    invalid = WeightProfile(w1=0.50, w2=0.20, w3=0.15, w4=0.25, w5=0.30)
    assert invalid.sums_to_one() is False

    result = select_weight_profile(
        JobType.GENERALIST, registry={JobType.GENERALIST: invalid}
    )
    assert result.profile is DEFAULT_WEIGHT_PROFILE
    assert result.rejected is True
    assert result.fell_back is True
    assert result.requested_job_type is JobType.GENERALIST
    assert result.reason is not None
    assert "reject" in result.reason.lower()
    assert "GENERALIST" in result.reason


def test_rejection_uses_default_resolved_profile() -> None:
    invalid = WeightProfile(w1=0.10, w2=0.10, w3=0.10, w4=0.10, w5=0.30)
    result = select_weight_profile(JobType.SALES, registry={JobType.SALES: invalid})
    assert _tuple(result.profile) == EXPECTED_DEFAULT
    assert result.profile.sums_to_one()

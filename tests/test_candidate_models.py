"""Unit and property tests for the candidate profile models (Task 2.2).

Covers construction, the per-tier ``signal_availability`` range invariant, and
the "absent structured field is not-present (None), never defaulted" rule
(Requirements 3.1, 3.2, 3.5).
"""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from icrs.models.candidate import (
    BehavioralSignal,
    Education,
    EnrichedProfile,
    NormalizedProfile,
    RawCandidate,
    Role,
)
from icrs.models.enums import DepthBreadth, SignalTier, TrajectoryArc


# --------------------------------------------------------------------------- #
# Shared enums
# --------------------------------------------------------------------------- #


def test_signal_tier_members_are_string_valued():
    assert SignalTier.STRUCTURAL == "STRUCTURAL"
    assert SignalTier.SEMANTIC == "SEMANTIC"
    assert SignalTier.BEHAVIORAL == "BEHAVIORAL"
    assert {t.value for t in SignalTier} == {"STRUCTURAL", "SEMANTIC", "BEHAVIORAL"}


def test_trajectory_and_depth_enum_members():
    assert {a.value for a in TrajectoryArc} == {
        "ACCELERATING",
        "STEADY",
        "LATERAL",
        "DECLINING",
    }
    assert {d.value for d in DepthBreadth} == {"SPECIALIST", "BALANCED", "GENERALIST"}


# --------------------------------------------------------------------------- #
# Role / Education — not-present vs defaulted
# --------------------------------------------------------------------------- #


def test_role_absent_scalar_fields_are_none_not_defaulted():
    role = Role(title="Engineer", company="Acme")
    # Absent structured fields must be None ("not-present"), not a sentinel.
    assert role.start is None
    assert role.end is None
    assert role.prestige_tier is None


def test_role_rejects_empty_title_or_company():
    with pytest.raises(ValidationError):
        Role(title="   ", company="Acme")
    with pytest.raises(ValidationError):
        Role(title="Engineer", company="")


def test_role_prestige_tier_must_be_positive():
    with pytest.raises(ValidationError):
        Role(title="Engineer", company="Acme", prestige_tier=0)


def test_education_all_fields_optional_and_default_none():
    edu = Education()
    assert edu.institution is None
    assert edu.degree is None
    assert edu.field_of_study is None
    assert edu.start is None
    assert edu.end is None


# --------------------------------------------------------------------------- #
# RawCandidate / NormalizedProfile
# --------------------------------------------------------------------------- #


def test_raw_candidate_defaults():
    raw = RawCandidate()
    assert raw.structured_fields == {}
    assert raw.free_text == ""
    assert raw.external_handles == {}
    assert raw.id is not None


def test_normalized_profile_tenure_non_negative():
    NormalizedProfile(total_tenure_months=0)
    NormalizedProfile(total_tenure_months=42)
    with pytest.raises(ValidationError):
        NormalizedProfile(total_tenure_months=-1)


def test_normalized_profile_holds_roles_and_education():
    profile = NormalizedProfile(
        roles=[Role(title="Dev", company="Acme", start=date(2020, 1, 1))],
        education=[Education(degree="BSc")],
        certifications=["AWS SA"],
        explicit_skills=["python"],
        total_tenure_months=24,
    )
    assert profile.roles[0].title == "Dev"
    assert profile.education[0].degree == "BSc"


# --------------------------------------------------------------------------- #
# BehavioralSignal
# --------------------------------------------------------------------------- #


def test_behavioral_signal_recency_non_negative():
    BehavioralSignal(source="github", metric="commits", value=1.0, recency_days=0)
    with pytest.raises(ValidationError):
        BehavioralSignal(source="github", metric="commits", value=1.0, recency_days=-5)


def test_behavioral_signal_requires_non_empty_source_and_metric():
    with pytest.raises(ValidationError):
        BehavioralSignal(source="", metric="commits", value=1.0, recency_days=1)
    with pytest.raises(ValidationError):
        BehavioralSignal(source="github", metric="  ", value=1.0, recency_days=1)


# --------------------------------------------------------------------------- #
# EnrichedProfile — signal_availability invariant & embedding not-present
# --------------------------------------------------------------------------- #


def _base() -> NormalizedProfile:
    return NormalizedProfile(total_tenure_months=12)


def test_enriched_profile_minimal_defaults():
    e = EnrichedProfile(base=_base())
    assert e.inferred_responsibilities == []
    assert e.implicit_skills == []
    assert e.trajectory_arc is None
    assert e.depth_breadth is None
    assert e.behavioral_signals == []
    assert e.signal_availability == {}
    # Embedding is produced by a later stage; absent => None, never zero-vector.
    assert e.embedding is None


def test_enriched_profile_accepts_valid_per_tier_availability():
    e = EnrichedProfile(
        base=_base(),
        trajectory_arc=TrajectoryArc.ACCELERATING,
        depth_breadth=DepthBreadth.BALANCED,
        signal_availability={
            SignalTier.STRUCTURAL: 1.0,
            SignalTier.SEMANTIC: 0.5,
            SignalTier.BEHAVIORAL: 0.0,
        },
    )
    assert e.signal_availability[SignalTier.BEHAVIORAL] == 0.0


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
def test_enriched_profile_rejects_out_of_range_availability(bad: float):
    with pytest.raises(ValidationError):
        EnrichedProfile(
            base=_base(),
            signal_availability={SignalTier.STRUCTURAL: bad},
        )


def test_enriched_profile_string_tier_keys_coerced_to_enum():
    # Per-tier map keyed by the enum's string value is coerced to SignalTier.
    e = EnrichedProfile(base=_base(), signal_availability={"SEMANTIC": 0.25})
    assert e.signal_availability[SignalTier.SEMANTIC] == 0.25


@given(
    coverage=st.fixed_dictionaries(
        {
            SignalTier.STRUCTURAL: st.floats(min_value=0.0, max_value=1.0),
            SignalTier.SEMANTIC: st.floats(min_value=0.0, max_value=1.0),
            SignalTier.BEHAVIORAL: st.floats(min_value=0.0, max_value=1.0),
        }
    )
)
def test_signal_availability_accepts_any_unit_interval_values(coverage):
    """Any per-tier coverage within [0,1] is accepted and preserved."""
    e = EnrichedProfile(base=_base(), signal_availability=coverage)
    for tier, value in coverage.items():
        assert 0.0 <= e.signal_availability[tier] <= 1.0
        assert e.signal_availability[tier] == value


@given(bad=st.floats().filter(lambda x: not (0.0 <= x <= 1.0)))
def test_signal_availability_rejects_values_outside_unit_interval(bad):
    """Any per-tier coverage outside [0,1] (incl. NaN/inf) is rejected."""
    with pytest.raises(ValidationError):
        EnrichedProfile(base=_base(), signal_availability={SignalTier.STRUCTURAL: bad})

"""Tests for the Task 10.2 trajectory / behavioral / hard-filter-pass / penalty sub-scores.

Covers Requirements 4.5, 4.7, 7.2:
    - each sub-score stays within the inclusive range [0,1]
    - behavioral neutral prior (0.5) is applied when behavioral availability == 0
    - hard-filter-pass ratio: 0, partial, 1.0, and no-must-haves -> 1.0
    - penalty clamping over soft-flag counts
    - trajectory arc mapping monotonicity (ACCELERATING >= DECLINING)
    - the exclude + indication path when a required sub-score is uncomputable

Both example-based unit tests and Hypothesis property tests are included; the
property tests assert the universal [0,1] output bound the composite score
depends on (supports Property 1, Requirement 4.1).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from icrs.models.candidate import (
    BehavioralSignal,
    EnrichedProfile,
    Education,
    NormalizedProfile,
    Role,
)
from icrs.models.enums import DepthBreadth, SignalTier, TrajectoryArc
from icrs.models.job import (
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)
from icrs.scoring.subscores import (
    NEUTRAL_PRIOR,
    RequiredSubScore,
    SubScoreMixin,
    SubScoreUnavailable,
    SubScoreUnavailableIndication,
    behavioral_signal_score,
    career_trajectory_score,
    default_must_have_match,
    depth_breadth_alignment,
    detect_soft_flags,
    disqualifying_flag_penalty,
    hard_filter_pass_score,
    map_arc_to_score,
    seniority_alignment,
)


# ===== helpers ==============================================================


def _req(text: str, category: RequirementCategory) -> Requirement:
    return Requirement(
        text=text,
        category=category,
        tier=RequirementTier.STRUCTURAL,
        weight=1.0,
    )


def _reqs(
    *,
    seniority: SeniorityBand = SeniorityBand.SENIOR,
    must_haves: list[str] | None = None,
    nice: list[str] | None = None,
) -> RequirementVector:
    requirements: list[Requirement] = []
    for text in must_haves or ["Python"]:
        requirements.append(_req(text, RequirementCategory.MUST_HAVE))
    for text in nice or []:
        requirements.append(_req(text, RequirementCategory.NICE_TO_HAVE))
    return RequirementVector(
        role_intent="Build the backend platform",
        seniority_band=seniority,
        requirements=requirements,
    )


def _cand(
    *,
    roles: list[Role] | None = None,
    skills: list[str] | None = None,
    certs: list[str] | None = None,
    implicit_skills: list[str] | None = None,
    responsibilities: list[str] | None = None,
    education: list[Education] | None = None,
    tenure_months: int = 0,
    arc: TrajectoryArc | None = None,
    depth: DepthBreadth | None = None,
    behavioral: list[BehavioralSignal] | None = None,
    availability: dict[SignalTier, float] | None = None,
) -> EnrichedProfile:
    base = NormalizedProfile(
        roles=roles or [],
        education=education or [],
        certifications=certs or [],
        explicit_skills=skills or [],
        total_tenure_months=tenure_months,
    )
    return EnrichedProfile(
        base=base,
        inferred_responsibilities=responsibilities or [],
        implicit_skills=implicit_skills or [],
        trajectory_arc=arc,
        depth_breadth=depth,
        behavioral_signals=behavioral or [],
        signal_availability=availability or {},
    )


# ===== trajectory arc mapping monotonicity ==================================


def test_arc_mapping_is_monotonic_accelerating_ge_declining() -> None:
    accel = map_arc_to_score(TrajectoryArc.ACCELERATING)
    steady = map_arc_to_score(TrajectoryArc.STEADY)
    lateral = map_arc_to_score(TrajectoryArc.LATERAL)
    declining = map_arc_to_score(TrajectoryArc.DECLINING)
    assert accel is not None and declining is not None
    assert accel >= steady >= lateral >= declining
    assert accel >= declining


def test_arc_mapping_none_for_absent_arc() -> None:
    assert map_arc_to_score(None) is None


def test_arc_scores_all_in_unit_interval() -> None:
    for arc in TrajectoryArc:
        score = map_arc_to_score(arc)
        assert score is not None
        assert 0.0 <= score <= 1.0


# ===== seniority + depth alignment ==========================================


def test_seniority_alignment_exact_match_is_one() -> None:
    cand = _cand(roles=[Role(title="Senior Engineer", company="Acme")])
    assert seniority_alignment(cand, SeniorityBand.SENIOR) == pytest.approx(1.0)


def test_seniority_alignment_decreases_with_distance() -> None:
    junior = _cand(roles=[Role(title="Junior Engineer", company="Acme")])
    # Junior candidate against an EXECUTIVE role -> large distance -> low score.
    far = seniority_alignment(junior, SeniorityBand.EXECUTIVE)
    near = seniority_alignment(junior, SeniorityBand.MID)
    assert far is not None and near is not None
    assert near > far
    assert 0.0 <= far <= 1.0


def test_seniority_alignment_none_without_roles() -> None:
    assert seniority_alignment(_cand(), SeniorityBand.SENIOR) is None


def test_depth_breadth_alignment_match_and_none() -> None:
    # SENIOR prefers BALANCED.
    assert depth_breadth_alignment(DepthBreadth.BALANCED, SeniorityBand.SENIOR) == pytest.approx(1.0)
    # GENERALIST is two steps from SPECIALIST (preferred for JUNIOR) -> 0.0.
    assert depth_breadth_alignment(DepthBreadth.GENERALIST, SeniorityBand.JUNIOR) == pytest.approx(0.0)
    assert depth_breadth_alignment(None, SeniorityBand.SENIOR) is None


# ===== career trajectory score ==============================================


def test_career_trajectory_score_in_unit_interval() -> None:
    cand = _cand(
        roles=[Role(title="Senior Engineer", company="Acme")],
        arc=TrajectoryArc.ACCELERATING,
        depth=DepthBreadth.BALANCED,
    )
    score = career_trajectory_score(cand, _reqs(seniority=SeniorityBand.SENIOR))
    assert 0.0 <= score <= 1.0


def test_career_trajectory_accelerating_ge_declining_same_evidence() -> None:
    reqs = _reqs(seniority=SeniorityBand.SENIOR)
    accel = _cand(
        roles=[Role(title="Senior Engineer", company="Acme")],
        arc=TrajectoryArc.ACCELERATING,
        depth=DepthBreadth.BALANCED,
    )
    declining = _cand(
        roles=[Role(title="Senior Engineer", company="Acme")],
        arc=TrajectoryArc.DECLINING,
        depth=DepthBreadth.BALANCED,
    )
    assert career_trajectory_score(accel, reqs) >= career_trajectory_score(declining, reqs)


def test_career_trajectory_partial_evidence_still_computes() -> None:
    # Only an arc is known (no roles, no depth) -> still computable from arc.
    cand = _cand(arc=TrajectoryArc.STEADY)
    score = career_trajectory_score(cand, _reqs())
    assert score == pytest.approx(map_arc_to_score(TrajectoryArc.STEADY))


def test_career_trajectory_uncomputable_raises_with_indication() -> None:
    # No arc, no roles, no depth/breadth -> genuinely cannot compute.
    bare = _cand()
    with pytest.raises(SubScoreUnavailable) as exc_info:
        career_trajectory_score(bare, _reqs())
    assert exc_info.value.sub_score is RequiredSubScore.TRAJECTORY
    indication = exc_info.value.indication
    assert isinstance(indication, SubScoreUnavailableIndication)
    assert indication.sub_score is RequiredSubScore.TRAJECTORY
    assert indication.reason


# ===== behavioral signal score ==============================================


def test_behavioral_neutral_prior_when_availability_zero() -> None:
    cand = _cand(availability={SignalTier.BEHAVIORAL: 0.0})
    assert behavioral_signal_score(cand) == pytest.approx(NEUTRAL_PRIOR)


def test_behavioral_neutral_prior_when_tier_absent_entirely() -> None:
    # No behavioral entry recorded at all -> treated as availability 0.
    assert behavioral_signal_score(_cand()) == pytest.approx(NEUTRAL_PRIOR)


def test_behavioral_neutral_prior_does_not_raise_for_absent_data() -> None:
    # Absent data is the "neutral prior" path, never the "cannot compute" path.
    cand = _cand(availability={SignalTier.BEHAVIORAL: 0.0})
    # Should not raise SubScoreUnavailable.
    assert behavioral_signal_score(cand) == pytest.approx(NEUTRAL_PRIOR)


def test_behavioral_score_with_fresh_signals_in_unit_interval() -> None:
    cand = _cand(
        behavioral=[
            BehavioralSignal(
                source="github",
                metric="commit_frequency",
                value=50.0,
                recency_days=0,
            ),
            BehavioralSignal(
                source="publications",
                metric="papers",
                value=3.0,
                recency_days=30,
            ),
        ],
        availability={SignalTier.BEHAVIORAL: 0.66},
    )
    score = behavioral_signal_score(cand)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_behavioral_recent_outweighs_stale_same_value() -> None:
    fresh = _cand(
        behavioral=[
            BehavioralSignal(source="github", metric="commits", value=100.0, recency_days=0)
        ],
        availability={SignalTier.BEHAVIORAL: 0.33},
    )
    stale = _cand(
        behavioral=[
            BehavioralSignal(
                source="github", metric="commits", value=100.0, recency_days=3650
            )
        ],
        availability={SignalTier.BEHAVIORAL: 0.33},
    )
    assert behavioral_signal_score(fresh) > behavioral_signal_score(stale)


# ===== hard-filter-pass score ===============================================


def test_hard_filter_pass_full_satisfaction() -> None:
    reqs = _reqs(must_haves=["Python", "Kubernetes"])
    cand = _cand(skills=["Python", "Kubernetes"])
    assert hard_filter_pass_score(reqs, cand) == pytest.approx(1.0)


def test_hard_filter_pass_partial_satisfaction() -> None:
    reqs = _reqs(must_haves=["Python", "Kubernetes"])
    cand = _cand(skills=["Python"])
    assert hard_filter_pass_score(reqs, cand) == pytest.approx(0.5)


def test_hard_filter_pass_zero_satisfaction() -> None:
    reqs = _reqs(must_haves=["Python", "Kubernetes"])
    cand = _cand(skills=["Cobol"])
    assert hard_filter_pass_score(reqs, cand) == pytest.approx(0.0)


def test_hard_filter_pass_no_must_haves_is_one() -> None:
    # A vector with no MUST_HAVE requires using NICE_TO_HAVE only; build it via
    # the matcher contract directly since RequirementVector requires a MUST_HAVE.
    reqs = _reqs(must_haves=["anything"])
    # Override: simulate "no must-haves" by clearing them post hoc is invalid;
    # instead assert the documented no-must-have branch via a custom vector.
    empty_must = RequirementVector(
        role_intent="Generalist role",
        seniority_band=SeniorityBand.MID,
        requirements=[
            _req("Mandatory baseline", RequirementCategory.MUST_HAVE),
        ],
    )
    # Remove the must-have to exercise the no-must-have branch deterministically.
    object.__setattr__(empty_must, "requirements", [])
    assert hard_filter_pass_score(empty_must, _cand()) == pytest.approx(1.0)


def test_hard_filter_pass_matches_role_evidence() -> None:
    reqs = _reqs(must_haves=["PHP Developer"])
    cand = _cand(roles=[Role(title="PHP Developer", company="Acme")])
    assert hard_filter_pass_score(reqs, cand) == pytest.approx(1.0)


def test_default_must_have_match_predicate() -> None:
    cand = _cand(skills=["Python"])
    assert default_must_have_match(cand, _req("Python", RequirementCategory.MUST_HAVE)) is True
    assert default_must_have_match(cand, _req("Rust", RequirementCategory.MUST_HAVE)) is False


# ===== disqualifying flag penalty ===========================================


def test_penalty_zero_when_no_flags() -> None:
    cand = _cand(certs=["AWS Certified"], skills=["Python"])
    assert disqualifying_flag_penalty(_reqs(), cand) == pytest.approx(0.0)


def test_penalty_expired_certification_flag() -> None:
    cand = _cand(certs=["AWS Certified (expired)"])
    flags = detect_soft_flags(_reqs(), cand)
    assert any("expired" in f.lower() for f in flags)
    assert disqualifying_flag_penalty(_reqs(), cand) == pytest.approx(0.2)


def test_penalty_claim_activity_mismatch_only_with_behavioral_data() -> None:
    # Behavioral signals present, none corroborate the explicit skills -> flag.
    mismatch = _cand(
        skills=["Python"],
        behavioral=[
            BehavioralSignal(
                source="github",
                metric="commits",
                value=10.0,
                recency_days=10,
                corroborates_skill=["Cobol"],
            )
        ],
        availability={SignalTier.BEHAVIORAL: 0.33},
    )
    assert any("mismatch" in f.lower() for f in detect_soft_flags(_reqs(), mismatch))

    # No behavioral data -> absence must NOT produce a mismatch flag (fairness).
    no_activity = _cand(skills=["Python"])
    assert not any("mismatch" in f.lower() for f in detect_soft_flags(_reqs(), no_activity))


def test_penalty_clamped_to_one() -> None:
    # Six expired certs -> 0.2 * 6 = 1.2 -> clamped to 1.0.
    cand = _cand(certs=[f"Cert {i} expired" for i in range(6)])
    assert disqualifying_flag_penalty(_reqs(), cand) == pytest.approx(1.0)


def test_penalty_always_in_unit_interval() -> None:
    cand = _cand(certs=["expired one", "expired two"])
    penalty = disqualifying_flag_penalty(_reqs(), cand)
    assert 0.0 <= penalty <= 1.0


# ===== mixin surface ========================================================


class _Engine(SubScoreMixin):
    """Bare engine composing only the sub-score mixin for testing."""


def test_mixin_methods_match_functions() -> None:
    engine = _Engine()
    reqs = _reqs(seniority=SeniorityBand.SENIOR, must_haves=["Python"])
    cand = _cand(
        roles=[Role(title="Senior Engineer", company="Acme")],
        skills=["Python"],
        arc=TrajectoryArc.STEADY,
        depth=DepthBreadth.BALANCED,
    )
    assert engine.career_trajectory_score(cand, reqs) == career_trajectory_score(cand, reqs)
    assert engine.behavioral_signal_score(cand) == behavioral_signal_score(cand)
    assert engine.hard_filter_pass_score(reqs, cand) == hard_filter_pass_score(reqs, cand)
    assert engine.disqualifying_flag_penalty(reqs, cand) == disqualifying_flag_penalty(reqs, cand)


# ===== property tests: universal [0,1] bound ================================


_ARC_ST = st.sampled_from(list(TrajectoryArc))
_DEPTH_ST = st.sampled_from(list(DepthBreadth))
_BAND_ST = st.sampled_from(list(SeniorityBand))


@given(
    arc=st.one_of(st.none(), _ARC_ST),
    depth=st.one_of(st.none(), _DEPTH_ST),
    band=_BAND_ST,
    has_role=st.booleans(),
    tenure=st.integers(min_value=0, max_value=600),
)
def test_career_trajectory_always_in_unit_interval_or_unavailable(
    arc, depth, band, has_role, tenure
) -> None:
    roles = [Role(title="Engineer", company="Acme")] if has_role else []
    cand = _cand(roles=roles, tenure_months=tenure, arc=arc, depth=depth)
    reqs = _reqs(seniority=band)
    try:
        score = career_trajectory_score(cand, reqs)
    except SubScoreUnavailable as exc:
        # Only legitimate when no component could be derived.
        assert arc is None and not has_role and depth is None
        assert exc.sub_score is RequiredSubScore.TRAJECTORY
    else:
        assert 0.0 <= score <= 1.0


@given(
    availability=st.floats(min_value=0.0, max_value=1.0),
    values=st.lists(
        st.floats(min_value=-100.0, max_value=1000.0), min_size=0, max_size=6
    ),
    recencies=st.lists(st.integers(min_value=0, max_value=5000), min_size=0, max_size=6),
)
def test_behavioral_score_always_in_unit_interval(availability, values, recencies) -> None:
    n = min(len(values), len(recencies))
    signals = [
        BehavioralSignal(
            source="github", metric="m", value=values[i], recency_days=recencies[i]
        )
        for i in range(n)
    ]
    avail = {SignalTier.BEHAVIORAL: availability} if (signals or availability) else {}
    cand = _cand(behavioral=signals, availability=avail)
    score = behavioral_signal_score(cand)
    assert 0.0 <= score <= 1.0


@given(
    must_have_count=st.integers(min_value=1, max_value=6),
    skills=st.lists(st.sampled_from(["python", "rust", "go", "java"]), max_size=4),
)
def test_hard_filter_pass_always_in_unit_interval(must_have_count, skills) -> None:
    reqs = _reqs(must_haves=[f"req{i}" for i in range(must_have_count)])
    cand = _cand(skills=skills)
    score = hard_filter_pass_score(reqs, cand)
    assert 0.0 <= score <= 1.0


@given(expired_count=st.integers(min_value=0, max_value=12))
def test_penalty_always_in_unit_interval(expired_count) -> None:
    cand = _cand(certs=[f"cert {i} expired" for i in range(expired_count)])
    penalty = disqualifying_flag_penalty(_reqs(), cand)
    assert 0.0 <= penalty <= 1.0

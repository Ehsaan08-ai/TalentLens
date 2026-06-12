"""Tests for Task 15.2 fairness and missing-signal guarantees.

Covers ``icrs/scoring/fairness.py`` against Requirement 7:

    - 7.1: ``strip_protected_proxies`` removes exactly the Protected_Proxy fields
      and preserves job-relevant ones (including in nested structures).
    - 7.2: ``apply_neutral_prior_for_missing_tiers`` substitutes the neutral
      prior (0.5) for availability-0 tiers (not 0) and keeps present-tier scores.
    - 7.4: ``is_proxy_invariant`` returns True for a proxy-blind score function
      and False for one that improperly reads a proxy.
    - 7.5: ``score_fully_sparse_candidate`` scores a fully-sparse candidate with
      neutral-prior sub-scores and includes it at the minimum confidence rather
      than excluding it.

Both example-based unit tests and a few Hypothesis property tests are included.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from icrs.models.enums import SignalTier
from icrs.scoring.confidence import compute_confidence
from icrs.scoring.fairness import (
    PROTECTED_PROXY_FIELDS,
    FullySparseScore,
    apply_neutral_prior_for_missing_tiers,
    fully_sparse_signal_availability,
    is_fully_sparse,
    is_protected_proxy_field,
    is_proxy_invariant,
    neutral_prior_signal_bundle,
    score_fully_sparse_candidate,
    strip_protected_proxies,
)
from icrs.scoring.subscores import NEUTRAL_PRIOR
from icrs.scoring.weights import DEFAULT_WEIGHT_PROFILE, WeightProfile

# ===========================================================================
# 1. Protected_Proxy exclusion (Requirement 7.1)
# ===========================================================================

# A representative structured candidate input mixing proxy and job-relevant keys.
_JOB_RELEVANT_KEYS = {
    "skills",
    "roles",
    "total_tenure_months",
    "certifications",
    "degree",
    "field_of_study",
}


def _sample_input() -> dict:
    return {
        # --- Protected_Proxy fields (must be removed) ---
        "name": "Jordan Smith",
        "first_name": "Jordan",
        "last_name": "Smith",
        "gender": "F",
        "date_of_birth": "1990-04-01",
        "graduation_year": 2012,
        "age": 34,
        "photo_url": "https://example.com/p.jpg",
        "location": "Berlin",
        "address": "10 Main St",
        "nationality": "German",
        # --- job-relevant fields (must be preserved) ---
        "skills": ["python", "distributed systems"],
        "roles": [{"title": "Senior Engineer", "company": "Acme"}],
        "total_tenure_months": 96,
        "certifications": ["AWS SA"],
        "degree": "BSc",
        "field_of_study": "Computer Science",
    }


def test_strip_removes_exactly_protected_fields_and_preserves_relevant() -> None:
    cleaned = strip_protected_proxies(_sample_input())

    # Every job-relevant key is preserved with its value intact.
    assert set(cleaned) == _JOB_RELEVANT_KEYS
    assert cleaned["skills"] == ["python", "distributed systems"]
    assert cleaned["total_tenure_months"] == 96
    assert cleaned["degree"] == "BSc"

    # No Protected_Proxy key survived.
    for key in (
        "name",
        "first_name",
        "last_name",
        "gender",
        "date_of_birth",
        "graduation_year",
        "age",
        "photo_url",
        "location",
        "address",
        "nationality",
    ):
        assert key not in cleaned


def test_strip_does_not_mutate_input() -> None:
    original = _sample_input()
    snapshot = dict(original)
    strip_protected_proxies(original)
    assert original == snapshot  # input untouched


def test_strip_is_recursive_over_nested_dicts_and_lists() -> None:
    data = {
        "skills": ["go"],
        "contact": {"email": "a@b.c", "address": "secret", "city": "Oslo"},
        "references": [
            {"company": "Acme", "name": "Reviewer One"},
            {"company": "Globex", "gender": "M"},
        ],
    }
    cleaned = strip_protected_proxies(data)
    assert cleaned["skills"] == ["go"]
    # Nested proxy keys removed, nested job-relevant keys kept.
    assert cleaned["contact"] == {"email": "a@b.c"}
    assert cleaned["references"] == [{"company": "Acme"}, {"company": "Globex"}]


def test_strip_matches_case_and_separator_insensitively() -> None:
    data = {"Date Of Birth": "x", "Full-Name": "y", "Skills": ["k"]}
    cleaned = strip_protected_proxies(data)
    assert "Date Of Birth" not in cleaned
    assert "Full-Name" not in cleaned
    assert cleaned["Skills"] == ["k"]  # non-proxy key preserved verbatim


def test_is_protected_proxy_field_normalization() -> None:
    assert is_protected_proxy_field("date_of_birth")
    assert is_protected_proxy_field("Date Of Birth")
    assert is_protected_proxy_field("DATE-OF-BIRTH")
    assert is_protected_proxy_field("graduation_year")
    assert not is_protected_proxy_field("skills")
    assert not is_protected_proxy_field("total_tenure_months")


def test_protected_proxy_fields_cover_required_families() -> None:
    # The five families named in the design plus the closely-related sensitive ones.
    for field in (
        "name",
        "gender",
        "date_of_birth",
        "graduation_year",
        "photo",
        "location",
        "address",
    ):
        assert field in PROTECTED_PROXY_FIELDS


# ===========================================================================
# 2. Neutral prior for missing tiers (Requirement 7.2)
# ===========================================================================


def test_neutral_prior_applied_for_zero_availability_tier_not_zero() -> None:
    subscores = {
        SignalTier.STRUCTURAL: 0.8,
        SignalTier.SEMANTIC: 0.6,
        SignalTier.BEHAVIORAL: 0.0,  # a real 0 sub-score, but tier IS available
    }
    availability = {
        SignalTier.STRUCTURAL: 1.0,
        SignalTier.SEMANTIC: 1.0,
        SignalTier.BEHAVIORAL: 0.0,  # no data -> neutral prior
    }
    adjusted = apply_neutral_prior_for_missing_tiers(subscores, availability)
    # Missing tier gets the neutral prior (0.5), NOT 0.
    assert adjusted[SignalTier.BEHAVIORAL] == pytest.approx(NEUTRAL_PRIOR)
    assert adjusted[SignalTier.BEHAVIORAL] != 0.0
    # Present tiers keep their real sub-scores.
    assert adjusted[SignalTier.STRUCTURAL] == pytest.approx(0.8)
    assert adjusted[SignalTier.SEMANTIC] == pytest.approx(0.6)


def test_neutral_prior_for_tier_missing_from_availability_map() -> None:
    subscores = {SignalTier.STRUCTURAL: 0.9, SignalTier.SEMANTIC: 0.4}
    # SEMANTIC absent from the availability map -> treated as 0 coverage.
    availability = {SignalTier.STRUCTURAL: 1.0}
    adjusted = apply_neutral_prior_for_missing_tiers(subscores, availability)
    assert adjusted[SignalTier.STRUCTURAL] == pytest.approx(0.9)
    assert adjusted[SignalTier.SEMANTIC] == pytest.approx(NEUTRAL_PRIOR)


def test_neutral_prior_leaves_fully_covered_subscores_untouched() -> None:
    subscores = {
        SignalTier.STRUCTURAL: 0.1,
        SignalTier.SEMANTIC: 0.2,
        SignalTier.BEHAVIORAL: 0.3,
    }
    availability = {t: 0.5 for t in subscores}
    assert apply_neutral_prior_for_missing_tiers(subscores, availability) == subscores


def test_missing_tier_reduces_confidence_without_lowering_score() -> None:
    # Documented rule: a missing tier reduces confidence, not rank. The neutral
    # prior keeps the sub-score at the midpoint (rank-neutral) while the all-zero
    # coverage on that tier lowers confidence relative to a fully-covered peer.
    full_cov = {t: 1.0 for t in (SignalTier.STRUCTURAL, SignalTier.SEMANTIC, SignalTier.BEHAVIORAL)}
    missing_behavioral = {SignalTier.STRUCTURAL: 1.0, SignalTier.SEMANTIC: 1.0}
    conf_full = compute_confidence(full_cov, 0.5, [0.7])
    conf_missing = compute_confidence(missing_behavioral, 0.5, [0.7])
    assert conf_missing < conf_full


# ===========================================================================
# 3. Counterfactual proxy invariance (Requirement 7.4)
# ===========================================================================


def _proxy_blind_score(data) -> float:
    # Reads ONLY job-relevant signals; ignores any proxy attribute.
    skills = data.get("skills", [])
    tenure = data.get("total_tenure_months", 0)
    return min(1.0, 0.1 * len(skills) + tenure / 1000.0)


def _proxy_reading_score(data) -> float:
    # Improperly lets a proxy attribute (location) move the score.
    base = _proxy_blind_score(data)
    if data.get("location") == "Berlin":
        base += 0.2
    return base


_PROXY_PERTURBATIONS = [
    {"name": "Someone Else", "gender": "M"},
    {"location": "Tokyo"},
    {"date_of_birth": "1975-01-01", "graduation_year": 1997},
    {"photo_url": "https://example.com/other.png"},
]


def test_is_proxy_invariant_true_for_proxy_blind_scorer() -> None:
    base = _sample_input()
    assert is_proxy_invariant(_proxy_blind_score, base, _PROXY_PERTURBATIONS) is True


def test_is_proxy_invariant_false_for_proxy_reading_scorer() -> None:
    # base_input has location "Berlin"; a perturbation changes it to "Tokyo",
    # which moves the proxy-reading score -> not invariant.
    base = _sample_input()
    assert is_proxy_invariant(_proxy_reading_score, base, _PROXY_PERTURBATIONS) is False


def test_is_proxy_invariant_trivially_true_with_no_perturbations() -> None:
    assert is_proxy_invariant(_proxy_reading_score, _sample_input(), []) is True


@given(
    skills=st.lists(st.text(min_size=1, max_size=5), max_size=6),
    tenure=st.integers(min_value=0, max_value=600),
    new_location=st.text(min_size=1, max_size=8),
    new_name=st.text(min_size=1, max_size=8),
)
def test_property_proxy_blind_scorer_always_invariant(
    skills, tenure, new_location, new_name
) -> None:
    base = {"skills": skills, "total_tenure_months": tenure, "location": "X", "name": "Y"}
    perturbations = [{"location": new_location}, {"name": new_name}]
    assert is_proxy_invariant(_proxy_blind_score, base, perturbations) is True


# ===========================================================================
# 4. Fully-sparse candidate handling (Requirement 7.5)
# ===========================================================================


def test_fully_sparse_availability_and_detection() -> None:
    avail = fully_sparse_signal_availability()
    assert avail == {
        SignalTier.STRUCTURAL: 0.0,
        SignalTier.SEMANTIC: 0.0,
        SignalTier.BEHAVIORAL: 0.0,
    }
    assert is_fully_sparse(avail) is True
    assert is_fully_sparse({}) is True  # empty map -> all tiers 0
    assert is_fully_sparse({SignalTier.STRUCTURAL: 0.3}) is False


def test_neutral_prior_bundle_uses_neutral_prior_for_all_subscores() -> None:
    bundle = neutral_prior_signal_bundle()
    assert bundle.semantic_fit == pytest.approx(NEUTRAL_PRIOR)
    assert bundle.career_trajectory == pytest.approx(NEUTRAL_PRIOR)
    assert bundle.behavioral == pytest.approx(NEUTRAL_PRIOR)
    assert bundle.hard_filter_pass == pytest.approx(NEUTRAL_PRIOR)
    assert bundle.disqualifying_penalty == pytest.approx(0.0)


def test_fully_sparse_candidate_is_scored_with_neutral_priors_and_included() -> None:
    result = score_fully_sparse_candidate(DEFAULT_WEIGHT_PROFILE, neighbor_scores=[0.4])
    assert isinstance(result, FullySparseScore)

    # All sub-scores are the neutral prior -> the candidate is scored, not excluded.
    assert result.bundle.semantic_fit == pytest.approx(NEUTRAL_PRIOR)
    assert result.bundle.behavioral == pytest.approx(NEUTRAL_PRIOR)

    # Composite is the neutral-prior fusion: with penalty 0 and w1..w4 summing to
    # 1.0, fusing 0.5 across the board yields 0.5.
    assert result.composite_score == pytest.approx(0.5)

    # Availability recorded as all-zero (unknown), never excluded.
    assert is_fully_sparse(result.signal_availability)

    # Confidence is in range and below that of a fully-covered peer with the same
    # margin (coverage contributes its floor of 0).
    assert 0.0 <= result.confidence <= 1.0
    covered_conf = compute_confidence(
        {t: 1.0 for t in (SignalTier.STRUCTURAL, SignalTier.SEMANTIC, SignalTier.BEHAVIORAL)},
        result.composite_score,
        [0.4],
    )
    assert result.confidence < covered_conf


def test_fully_sparse_candidate_confidence_is_minimum_when_tied() -> None:
    # A tied neighbour gives margin 0; coverage is 0 -> confidence is exactly 0,
    # the global minimum. The candidate is still returned (included), not excluded.
    result = score_fully_sparse_candidate(
        DEFAULT_WEIGHT_PROFILE,
        final_score=0.5,
        neighbor_scores=[0.5],
    )
    assert result.confidence == pytest.approx(0.0)
    assert result.composite_score == pytest.approx(0.5)


def test_fully_sparse_candidate_lone_uses_margin_floor_from_coverage() -> None:
    # A lone fully-sparse candidate: maximal margin (1.0) but zero coverage, so
    # confidence == MARGIN_WEIGHT * 1.0; coverage term is at its 0 floor.
    result = score_fully_sparse_candidate(DEFAULT_WEIGHT_PROFILE)
    # coverage 0 -> confidence = 0.4 * normalize(1.0) = 0.4
    assert result.confidence == pytest.approx(0.4)


@given(
    w1=st.floats(min_value=0.0, max_value=1.0),
    split=st.floats(min_value=0.0, max_value=1.0),
)
def test_property_fully_sparse_composite_equals_neutral_prior(w1, split) -> None:
    # For any valid weight profile whose fusion weights sum to 1.0, fusing the
    # neutral prior across all positive sub-scores (penalty 0) yields exactly the
    # neutral prior.
    remaining = 1.0 - w1
    w2 = remaining * split
    rest = remaining - w2
    w3 = rest * split
    w4 = rest - w3
    weights = WeightProfile(w1=w1, w2=w2, w3=w3, w4=w4, w5=0.3)
    result = score_fully_sparse_candidate(weights)
    assert result.composite_score == pytest.approx(NEUTRAL_PRIOR, abs=1e-6)

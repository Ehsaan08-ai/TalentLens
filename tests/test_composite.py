"""Unit tests for the Task 11.1 weighted composite fusion with clamping.

Covers Requirements 4.1 and 4.5:
    - the exact composite formula on known inputs
    - each sub-score normalized/clamped to [0,1] *before* weighting
      (out-of-range inputs are clamped by the SignalBundle)
    - the disqualifying penalty reduces the composite score
    - the result is always within [0,1] even for extreme inputs
    - all-1.0 positive sub-scores with zero penalty under a sum-to-one profile
      yields ~1.0 (the w1+w2+w3+w4 weighting fills the unit interval)
    - SignalBundle <-> SignalBreakdown interconversion

Both example-based unit tests and Hypothesis property tests are included; the
property tests assert the universal [0,1] output bound (supports Property 1,
Requirement 4.1).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from icrs.models.ranking import SignalBreakdown
from icrs.scoring.composite import (
    CompositeMixin,
    SignalBundle,
    build_signal_bundle,
    composite,
)
from icrs.scoring.weights import (
    DEFAULT_WEIGHT_PROFILE,
    WeightProfile,
)


# ===== helpers ==============================================================


def _bundle(
    *,
    semantic: float = 0.0,
    trajectory: float = 0.0,
    behavioral: float = 0.0,
    hard_filter_pass: float = 0.0,
    penalty: float = 0.0,
) -> SignalBundle:
    return SignalBundle(
        semantic_fit=semantic,
        career_trajectory=trajectory,
        behavioral=behavioral,
        hard_filter_pass=hard_filter_pass,
        disqualifying_penalty=penalty,
    )


# A simple sum-to-one weight profile with no penalty for exact-formula tests.
_EQUAL_NO_PENALTY = WeightProfile(w1=0.25, w2=0.25, w3=0.25, w4=0.25, w5=0.0)


# ===== exact formula on known inputs ========================================


def test_composite_exact_formula_known_inputs() -> None:
    # FinalScore = .35*.8 + .15*.6 + .30*.4 + .20*.5 - .30*.0
    #            = .28 + .09 + .12 + .10 - 0 = .59
    weights = WeightProfile(w1=0.35, w2=0.15, w3=0.30, w4=0.20, w5=0.30)
    bundle = _bundle(
        semantic=0.8, trajectory=0.6, behavioral=0.4, hard_filter_pass=0.5, penalty=0.0
    )
    assert composite(bundle, weights) == pytest.approx(0.59)


def test_composite_exact_formula_with_penalty_subtracted() -> None:
    # raw = .25*.8 + .25*.8 + .25*.8 + .25*.8 - .5*.4
    #     = .8 - .2 = .6
    weights = WeightProfile(w1=0.25, w2=0.25, w3=0.25, w4=0.25, w5=0.5)
    bundle = _bundle(
        semantic=0.8, trajectory=0.8, behavioral=0.8, hard_filter_pass=0.8, penalty=0.4
    )
    assert composite(bundle, weights) == pytest.approx(0.6)


def test_composite_all_zero_is_zero() -> None:
    assert composite(_bundle(), DEFAULT_WEIGHT_PROFILE) == pytest.approx(0.0)


# ===== sub-score clamping BEFORE weighting (Requirement 4.5) ================


def test_subscores_clamped_to_unit_interval_on_construction() -> None:
    # Out-of-range inputs are clamped to [0,1] by the bundle itself.
    bundle = SignalBundle(
        semantic_fit=1.7,
        career_trajectory=-0.5,
        behavioral=2.0,
        hard_filter_pass=-3.0,
        disqualifying_penalty=9.0,
    )
    assert bundle.semantic_fit == 1.0
    assert bundle.career_trajectory == 0.0
    assert bundle.behavioral == 1.0
    assert bundle.hard_filter_pass == 0.0
    assert bundle.disqualifying_penalty == 1.0


def test_out_of_range_subscores_clamped_before_weighting() -> None:
    # A semantic_fit of 5.0 must be treated as 1.0 *before* weighting, so the
    # contribution is w1*1.0, not w1*5.0.
    weights = WeightProfile(w1=0.4, w2=0.2, w3=0.15, w4=0.25, w5=0.3)
    inflated = composite(_bundle(semantic=5.0), weights)
    clamped = composite(_bundle(semantic=1.0), weights)
    assert inflated == pytest.approx(clamped)
    assert inflated == pytest.approx(0.4)


# ===== penalty reduces the score ============================================


def test_penalty_reduces_composite_score() -> None:
    weights = WeightProfile(w1=0.4, w2=0.2, w3=0.15, w4=0.25, w5=0.3)
    base = _bundle(
        semantic=0.6, trajectory=0.6, behavioral=0.6, hard_filter_pass=0.6, penalty=0.0
    )
    flagged = _bundle(
        semantic=0.6, trajectory=0.6, behavioral=0.6, hard_filter_pass=0.6, penalty=0.5
    )
    assert composite(flagged, weights) < composite(base, weights)


def test_penalty_monotonically_non_increasing() -> None:
    weights = WeightProfile(w1=0.4, w2=0.2, w3=0.15, w4=0.25, w5=0.3)
    scores = [
        composite(_bundle(semantic=0.9, trajectory=0.9, behavioral=0.9,
                          hard_filter_pass=0.9, penalty=p), weights)
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    for earlier, later in zip(scores, scores[1:]):
        assert later <= earlier


# ===== result always within [0,1] even with extreme inputs =================


def test_extreme_all_max_with_full_penalty_clamps_to_unit_interval() -> None:
    # All positive sub-scores 1.0 and penalty 1.0 under the default profile.
    weights = DEFAULT_WEIGHT_PROFILE
    score = composite(
        _bundle(semantic=1.0, trajectory=1.0, behavioral=1.0,
                hard_filter_pass=1.0, penalty=1.0),
        weights,
    )
    # raw = (w1+w2+w3+w4)*1 - w5*1 = 1.0 - 0.30 = 0.70, within [0,1].
    assert score == pytest.approx(0.70)
    assert 0.0 <= score <= 1.0


def test_large_penalty_clamps_at_zero() -> None:
    # A penalty term larger than the positive contribution drives raw negative;
    # the final clamp pins it at 0.0.
    weights = WeightProfile(w1=0.1, w2=0.1, w3=0.1, w4=0.1, w5=1.0)
    score = composite(
        _bundle(semantic=0.1, trajectory=0.1, behavioral=0.1,
                hard_filter_pass=0.1, penalty=1.0),
        weights,
    )
    assert score == 0.0


def test_out_of_spec_weights_still_clamp_to_one() -> None:
    # An out-of-spec profile whose fusion weights exceed 1.0 could push the raw
    # sum above 1.0; the final clamp guarantees <= 1.0 (Requirement 4.1).
    weights = WeightProfile(w1=1.0, w2=1.0, w3=1.0, w4=1.0, w5=0.0)
    score = composite(
        _bundle(semantic=1.0, trajectory=1.0, behavioral=1.0,
                hard_filter_pass=1.0, penalty=0.0),
        weights,
    )
    assert score == 1.0


# ===== sum-to-one weighting of all-1.0 sub-scores yields ~1.0 ===============


def test_all_ones_zero_penalty_sum_to_one_profile_yields_one() -> None:
    # For any profile with w1+w2+w3+w4 == 1.0 and all positive sub-scores 1.0
    # and zero penalty, the composite is exactly 1.0.
    bundle = _bundle(
        semantic=1.0, trajectory=1.0, behavioral=1.0, hard_filter_pass=1.0, penalty=0.0
    )
    for weights in (
        _EQUAL_NO_PENALTY,
        WeightProfile(w1=0.35, w2=0.15, w3=0.30, w4=0.20, w5=0.30),
        WeightProfile(w1=0.40, w2=0.20, w3=0.15, w4=0.25, w5=0.30),
    ):
        assert weights.sums_to_one()
        assert composite(bundle, weights) == pytest.approx(1.0)


# ===== SignalBundle <-> SignalBreakdown interconversion =====================


def test_from_breakdown_round_trips_through_bundle() -> None:
    breakdown = SignalBreakdown(
        semantic_fit=0.7,
        career_trajectory=0.6,
        behavioral=0.5,
        hard_filter_pass=0.4,
        disqualifying_penalty=0.2,
    )
    bundle = SignalBundle.from_breakdown(breakdown)
    assert bundle.semantic_fit == pytest.approx(0.7)
    assert bundle.career_trajectory == pytest.approx(0.6)
    assert bundle.behavioral == pytest.approx(0.5)
    assert bundle.hard_filter_pass == pytest.approx(0.4)
    assert bundle.disqualifying_penalty == pytest.approx(0.2)


def test_to_breakdown_produces_valid_signal_breakdown() -> None:
    # Even from out-of-range inputs, to_breakdown yields a valid (clamped) model.
    bundle = SignalBundle(
        semantic_fit=2.0,
        career_trajectory=-1.0,
        behavioral=0.5,
        hard_filter_pass=0.5,
        disqualifying_penalty=5.0,
    )
    breakdown = bundle.to_breakdown()
    assert isinstance(breakdown, SignalBreakdown)
    assert breakdown.semantic_fit == 1.0
    assert breakdown.career_trajectory == 0.0
    assert breakdown.disqualifying_penalty == 1.0


def test_build_signal_bundle_helper_clamps() -> None:
    bundle = build_signal_bundle(
        semantic_fit=1.5,
        career_trajectory=0.3,
        behavioral=0.3,
        hard_filter_pass=0.3,
        disqualifying_penalty=-0.2,
    )
    assert bundle.semantic_fit == 1.0
    assert bundle.disqualifying_penalty == 0.0


# ===== engine mixin =========================================================


def test_composite_mixin_delegates() -> None:
    class _Engine(CompositeMixin):
        pass

    engine = _Engine()
    bundle = _bundle(semantic=0.5, trajectory=0.5, behavioral=0.5, hard_filter_pass=0.5)
    assert engine.composite(bundle, DEFAULT_WEIGHT_PROFILE) == pytest.approx(
        composite(bundle, DEFAULT_WEIGHT_PROFILE)
    )


# ===== property: composite output is always in [0,1] (Property 1) ===========


_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_wide = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)


@given(
    semantic=_wide,
    trajectory=_wide,
    behavioral=_wide,
    hard_filter_pass=_wide,
    penalty=_wide,
    w1=_unit,
    w2=_unit,
    w3=_unit,
    w4=_unit,
    w5=_unit,
)
def test_property_composite_always_in_unit_interval(
    semantic: float,
    trajectory: float,
    behavioral: float,
    hard_filter_pass: float,
    penalty: float,
    w1: float,
    w2: float,
    w3: float,
    w4: float,
    w5: float,
) -> None:
    """Composite score is always within [0,1] for arbitrary inputs and weights."""

    bundle = SignalBundle(
        semantic_fit=semantic,
        career_trajectory=trajectory,
        behavioral=behavioral,
        hard_filter_pass=hard_filter_pass,
        disqualifying_penalty=penalty,
    )
    weights = WeightProfile(w1=w1, w2=w2, w3=w3, w4=w4, w5=w5)
    score = composite(bundle, weights)
    assert 0.0 <= score <= 1.0

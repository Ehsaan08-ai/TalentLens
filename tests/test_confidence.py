"""Tests for Task 15.1 confidence computation (icrs/scoring/confidence.py).

Covers Requirement 5.5 and the design's "Confidence computation":
    - confidence is always in [0,1]
    - monotonic in coverage: higher coverage with an equal margin -> confidence >=
    - monotonic in margin: larger margin with equal coverage -> confidence >=
    - the 0.6 / 0.4 weighting on known inputs
    - a missing tier key is treated as 0 coverage (mean over the three tiers)
    - the single-candidate (no-neighbour) edge case

Both example-based unit tests and Hypothesis property tests are included. The
property tests assert the universal [0,1] bound and the two monotonicity
directions Requirement 5.5 demands (Correctness Property 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from hypothesis import given
from hypothesis import strategies as st

from icrs.models.enums import SignalTier
from icrs.scoring.confidence import (
    COVERAGE_WEIGHT,
    MARGIN_WEIGHT,
    compute_confidence,
    compute_confidence_for,
    neighbor_scores_for,
    normalize_margin,
    score_margin_to_neighbors,
    signal_coverage,
)

# ===== lightweight duck-typed stand-ins for the wrapper =====================


@dataclass
class _Profile:
    signal_availability: dict[SignalTier, float] = field(default_factory=dict)


@dataclass
class _Cand:
    """Duck-typed ScoredCandidate: exposes final_score + profile.signal_availability."""

    id_str: str
    final_score: float
    profile: _Profile = field(default_factory=_Profile)


def _full_coverage() -> dict[SignalTier, float]:
    return {
        SignalTier.STRUCTURAL: 1.0,
        SignalTier.SEMANTIC: 1.0,
        SignalTier.BEHAVIORAL: 1.0,
    }


# ===== signal coverage ======================================================


def test_coverage_full_is_one() -> None:
    assert signal_coverage(_full_coverage()) == pytest.approx(1.0)


def test_coverage_empty_is_zero() -> None:
    assert signal_coverage({}) == pytest.approx(0.0)
    assert signal_coverage(None) == pytest.approx(0.0)


def test_coverage_missing_tier_counts_as_zero() -> None:
    # Two tiers fully present, BEHAVIORAL key absent -> mean over THREE tiers.
    avail = {SignalTier.STRUCTURAL: 1.0, SignalTier.SEMANTIC: 1.0}
    assert signal_coverage(avail) == pytest.approx(2.0 / 3.0)


def test_coverage_accepts_string_keys() -> None:
    avail = {"STRUCTURAL": 1.0, "SEMANTIC": 0.5, "BEHAVIORAL": 0.0}
    assert signal_coverage(avail) == pytest.approx(1.5 / 3.0)


# ===== margin ===============================================================


def test_margin_single_candidate_is_max_separation() -> None:
    # No neighbours -> maximal separation (1.0).
    assert score_margin_to_neighbors(0.5, []) == pytest.approx(1.0)


def test_margin_is_minimum_distance_to_neighbors() -> None:
    # Neighbours at 0.9 and 0.55; candidate at 0.6 -> min(|0.6-0.9|, |0.6-0.55|).
    assert score_margin_to_neighbors(0.6, [0.9, 0.55]) == pytest.approx(0.05)


def test_normalize_margin_clamps_to_unit_interval() -> None:
    assert normalize_margin(-0.2) == pytest.approx(0.0)
    assert normalize_margin(0.3) == pytest.approx(0.3)
    assert normalize_margin(1.5) == pytest.approx(1.0)


# ===== 0.6 / 0.4 weighting on known inputs ==================================


def test_weighting_known_inputs() -> None:
    # coverage = mean(1.0, 0.5, 0.0) = 0.5 ; neighbours give margin 0.2.
    avail = {
        SignalTier.STRUCTURAL: 1.0,
        SignalTier.SEMANTIC: 0.5,
        SignalTier.BEHAVIORAL: 0.0,
    }
    # candidate 0.6, neighbour 0.8 -> margin 0.2
    conf = compute_confidence(avail, 0.6, [0.8])
    expected = 0.6 * 0.5 + 0.4 * 0.2  # = 0.38
    assert conf == pytest.approx(expected)


def test_weighting_full_coverage_isolated_candidate() -> None:
    # Full coverage + no neighbours (margin 1.0) -> 0.6*1 + 0.4*1 = 1.0.
    assert compute_confidence(_full_coverage(), 0.7, []) == pytest.approx(1.0)


def test_weights_are_the_design_values() -> None:
    assert COVERAGE_WEIGHT == pytest.approx(0.6)
    assert MARGIN_WEIGHT == pytest.approx(0.4)


# ===== bounds ===============================================================


def test_confidence_in_unit_interval_extremes() -> None:
    # Zero coverage, zero margin (a tie with a neighbour) -> 0.0.
    assert compute_confidence({}, 0.5, [0.5]) == pytest.approx(0.0)
    # Full coverage, max margin -> 1.0.
    assert compute_confidence(_full_coverage(), 0.0, [1.0]) == pytest.approx(1.0)


# ===== monotonicity in coverage =============================================


def test_monotonic_in_coverage_equal_margin() -> None:
    # Same margin (single neighbour at distance 0.1), higher coverage.
    low = compute_confidence({SignalTier.STRUCTURAL: 0.2}, 0.5, [0.6])
    high = compute_confidence(_full_coverage(), 0.5, [0.6])
    assert high >= low


def test_monotonic_in_coverage_via_wrapper() -> None:
    # Two otherwise-identical candidates differing only in coverage, with an
    # identical margin to their (shared-shape) neighbourhood.
    low = _Cand("low", 0.5, _Profile({SignalTier.STRUCTURAL: 0.1}))
    high = _Cand("high", 0.5, _Profile(_full_coverage()))
    neighbor = _Cand("n", 0.7)
    # Each ranked against an identical neighbour layout (margin 0.2 each).
    conf_low = compute_confidence_for(low, [low, neighbor])
    conf_high = compute_confidence_for(high, [high, neighbor])
    assert conf_high >= conf_low


# ===== monotonicity in margin ===============================================


def test_monotonic_in_margin_equal_coverage() -> None:
    cov = _full_coverage()
    small = compute_confidence(cov, 0.5, [0.52])  # margin 0.02
    large = compute_confidence(cov, 0.5, [0.9])  # margin 0.40
    assert large >= small


@given(
    coverage_vals=st.lists(
        st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=3
    ),
    score=st.floats(min_value=0.0, max_value=1.0),
    small_delta=st.floats(min_value=0.0, max_value=0.5),
    extra_delta=st.floats(min_value=0.0, max_value=0.5),
)
def test_property_monotonic_in_margin(
    coverage_vals, score, small_delta, extra_delta
) -> None:
    avail = {
        SignalTier.STRUCTURAL: coverage_vals[0],
        SignalTier.SEMANTIC: coverage_vals[1],
        SignalTier.BEHAVIORAL: coverage_vals[2],
    }
    # A neighbour closer than another -> smaller margin must not yield MORE
    # confidence than the larger margin (coverage held equal).
    near = score + small_delta
    far = score + small_delta + extra_delta
    conf_small = compute_confidence(avail, score, [near])
    conf_large = compute_confidence(avail, score, [far])
    assert conf_large + 1e-9 >= conf_small


@given(
    score=st.floats(min_value=0.0, max_value=1.0),
    margin_delta=st.floats(min_value=0.0, max_value=1.0),
    low_cov=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=3, max_size=3),
    bump=st.floats(min_value=0.0, max_value=1.0),
)
def test_property_monotonic_in_coverage(score, margin_delta, low_cov, bump) -> None:
    # Build a higher-coverage map by bumping every tier (clamped at 1.0), holding
    # the margin equal -> confidence must not decrease.
    low = {
        SignalTier.STRUCTURAL: low_cov[0],
        SignalTier.SEMANTIC: low_cov[1],
        SignalTier.BEHAVIORAL: low_cov[2],
    }
    high = {t: min(1.0, v + bump) for t, v in low.items()}
    neighbor = score + margin_delta
    conf_low = compute_confidence(low, score, [neighbor])
    conf_high = compute_confidence(high, score, [neighbor])
    assert conf_high + 1e-9 >= conf_low


@given(
    cov=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=0, max_size=3),
    score=st.floats(min_value=0.0, max_value=1.0),
    neighbors=st.lists(st.floats(min_value=0.0, max_value=1.0), max_size=4),
)
def test_property_confidence_always_in_unit_interval(cov, score, neighbors) -> None:
    tiers = [SignalTier.STRUCTURAL, SignalTier.SEMANTIC, SignalTier.BEHAVIORAL]
    avail = {tiers[i]: cov[i] for i in range(len(cov))}
    conf = compute_confidence(avail, score, neighbors)
    assert 0.0 <= conf <= 1.0


# ===== single-candidate edge case via the wrapper ===========================


def test_single_candidate_wrapper_uses_max_margin() -> None:
    cand = _Cand("solo", 0.42, _Profile({SignalTier.STRUCTURAL: 0.5}))
    # Lone candidate: margin is 1.0, coverage = mean(0.5, 0, 0) = 1/6.
    conf = compute_confidence_for(cand, [cand])
    expected = 0.6 * (0.5 / 3.0) + 0.4 * 1.0
    assert conf == pytest.approx(expected)


def test_neighbor_scores_for_interior_and_ends() -> None:
    a = _Cand("a", 0.9)
    b = _Cand("b", 0.6)
    c = _Cand("c", 0.3)
    ranked = [c, a, b]  # unordered on input
    # Interior candidate b has neighbours a (0.9) and c (0.3).
    assert sorted(neighbor_scores_for(b, ranked)) == [0.3, 0.9]
    # Top candidate a has a single neighbour b (0.6).
    assert neighbor_scores_for(a, ranked) == [0.6]
    # Bottom candidate c has a single neighbour b (0.6).
    assert neighbor_scores_for(c, ranked) == [0.6]


def test_neighbor_scores_for_empty_ranking() -> None:
    assert neighbor_scores_for(_Cand("x", 0.5), []) == []

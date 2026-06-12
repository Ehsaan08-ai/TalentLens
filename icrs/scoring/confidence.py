"""Confidence computation (Task 15.1).

This module implements the design's "Confidence computation" algorithm — the
value in ``[0,1]`` that accompanies every ranking result and reflects how much
the system trusts a candidate's placement. Confidence rises with two
independent factors (design + Requirement 5.5):

    1. **Signal coverage** — how much data the candidate actually has. Computed
       as the mean of :class:`~icrs.models.candidate.EnrichedProfile.signal_availability`
       over the three signal tiers (STRUCTURAL / SEMANTIC / BEHAVIORAL). More
       populated tiers ⟹ higher coverage ⟹ higher confidence. A tier whose key
       is absent from the availability map contributes ``0`` coverage (the mean
       is always taken over the three tiers), so missing data lowers confidence
       rather than silently inflating it.
    2. **Score margin to neighbors** — how clearly separated the candidate is
       from its immediate neighbours in the ranking. A larger margin ⟹ clearer
       separation ⟹ higher confidence.

The formula (design "Confidence computation"):

    coverage   = mean(signal_availability over all three tiers)
    margin     = scoreMarginToNeighbors(cand, ranked)
    confidence = clamp(0.6 * coverage + 0.4 * normalize(margin), 0, 1)

Because both coefficients are strictly positive and ``normalize`` is monotonic
non-decreasing, the result is **monotonic** in each factor (Requirement 5.5):
with the margin held equal, higher coverage yields confidence ``>=``; with
coverage held equal, a larger margin yields confidence ``>=``.

Design choices documented here:
    - ``margin`` is the **minimum absolute difference** between this candidate's
      ``final_score`` and the ``final_score`` of each of its immediate neighbours
      in the score-sorted ranking (the single neighbour at the top/bottom ends).
      The minimum is used because confidence in a placement is limited by the
      *closest* competitor — a candidate sandwiched tightly between two others is
      a less certain placement than one with clear air on at least one side.
    - ``normalize(margin)``: final scores are already in ``[0,1]`` so any margin
      is in ``[0,1]``; it is used directly (clamped defensively). This mapping is
      the identity on ``[0,1]`` and therefore monotonic non-decreasing, which is
      all Requirement 5.5 needs.
    - **Single candidate / no neighbours**: there is no competitor to be confused
      with, so separation is maximal — ``margin`` is defined as ``1.0``. This
      keeps the function total and preserves monotonicity (a lone candidate can
      only have its confidence limited by coverage).

To avoid coupling this module to the pydantic output models or risking an import
cycle with the scoring/pipeline layers, the core function accepts plain values:

    compute_confidence(signal_availability, final_score, neighbor_scores)

A convenience wrapper extracts those values from a scored candidate and its
ranked peers:

    compute_confidence_for(cand, ranked)

where ``cand`` and each element of ``ranked`` only need to expose
``.final_score`` and ``.profile.signal_availability`` (duck-typed — e.g. the
:class:`~icrs.pipeline.reranker.ScoredCandidate` dataclass), so the wrapper never
forces an import of the reranker module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from icrs.models.enums import SignalTier

# Blend coefficients from the design's confidence formula. Both are strictly
# positive, which is what guarantees monotonicity in coverage and in margin.
COVERAGE_WEIGHT = 0.6
MARGIN_WEIGHT = 0.4

# The three signal tiers coverage is averaged over. A tier missing from the
# availability map contributes 0 to the mean (mean is always over all three).
_TIERS: tuple[SignalTier, ...] = (
    SignalTier.STRUCTURAL,
    SignalTier.SEMANTIC,
    SignalTier.BEHAVIORAL,
)

# Margin assigned to a candidate that has no neighbours (a single-candidate
# ranking): maximal separation, since there is no competitor to confuse it with.
_ISOLATED_MARGIN = 1.0


def _clamp01(value: float) -> float:
    """Clamp ``value`` to the inclusive range ``[0,1]``."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def signal_coverage(signal_availability: Mapping[Any, float] | None) -> float:
    """Mean signal coverage over the three tiers, in ``[0,1]``.

    The mean is always taken over the three :class:`SignalTier` values; a tier
    whose key is absent from ``signal_availability`` (or maps to ``None``)
    contributes ``0`` coverage for that tier. This is what makes missing data
    reduce confidence rather than be ignored. Keys may be :class:`SignalTier`
    members or their string values (e.g. ``"STRUCTURAL"``).

    Each present value is clamped to ``[0,1]`` defensively; the model layer
    already validates the range, but this keeps the function total for the
    duck-typed/plain-dict entry point.
    """

    if not signal_availability:
        return 0.0

    total = 0.0
    for tier in _TIERS:
        coverage = signal_availability.get(tier)
        if coverage is None:
            # Tolerate string-keyed maps (e.g. when not using the enum directly).
            coverage = signal_availability.get(tier.value)
        if coverage is None:
            continue
        total += _clamp01(float(coverage))
    return total / len(_TIERS)


def normalize_margin(margin: float) -> float:
    """Normalize a score ``margin`` into ``[0,1]``.

    Final scores live in ``[0,1]`` so any margin is already in ``[0,1]``; this
    mapping is the identity clamped to the unit interval. It is monotonic
    non-decreasing in ``margin``, which is all Requirement 5.5 requires of the
    margin contribution.
    """

    return _clamp01(float(margin))


def score_margin_to_neighbors(
    final_score: float, neighbor_scores: Sequence[float]
) -> float:
    """Minimum absolute score distance from ``final_score`` to its neighbours.

    ``neighbor_scores`` are the ``final_score`` values of this candidate's
    immediate neighbours in the score-sorted ranking (two in the interior, one
    at an end). When there are no neighbours (a single-candidate ranking), the
    margin is ``_ISOLATED_MARGIN`` (``1.0``) — maximal separation.

    The minimum distance is used: a placement's certainty is limited by the
    closest competitor.
    """

    if not neighbor_scores:
        return _ISOLATED_MARGIN
    return min(abs(float(final_score) - float(n)) for n in neighbor_scores)


def compute_confidence(
    signal_availability: Mapping[Any, float] | None,
    final_score: float,
    neighbor_scores: Sequence[float],
) -> float:
    """Compute a candidate's confidence in ``[0,1]`` (design + Requirement 5.5).

    Args:
        signal_availability: Per-tier coverage map; missing tiers count as 0
            coverage (mean taken over all three tiers).
        final_score: This candidate's final score in ``[0,1]``.
        neighbor_scores: ``final_score`` values of the candidate's immediate
            neighbours in the ranking. Empty ⟹ single candidate ⟹ maximal margin.

    Returns:
        ``clamp(0.6 * coverage + 0.4 * normalize(margin), 0, 1)``.

    Monotonicity guarantee (Requirement 5.5): both blend coefficients are
    strictly positive and ``normalize_margin`` is monotonic non-decreasing, so a
    candidate with strictly higher coverage and an equal margin — or a larger
    margin and equal coverage — receives a confidence value ``>=`` that of the
    otherwise-identical candidate.
    """

    coverage = signal_coverage(signal_availability)
    margin = score_margin_to_neighbors(final_score, neighbor_scores)
    confidence = COVERAGE_WEIGHT * coverage + MARGIN_WEIGHT * normalize_margin(margin)
    return _clamp01(confidence)


@runtime_checkable
class _ConfidenceCandidate(Protocol):
    """Structural type the convenience wrapper needs from a scored candidate.

    Any object exposing ``final_score`` and ``profile.signal_availability``
    satisfies this — notably :class:`~icrs.pipeline.reranker.ScoredCandidate` —
    so the wrapper never has to import a concrete class.
    """

    final_score: float
    profile: Any


def _candidate_key(cand: Any, index: int) -> Any:
    """A stable identity key for locating a candidate within ``ranked``.

    Prefers ``id_str`` (the reranker's prompt/response key), then ``id``, and
    finally falls back to the candidate's position so duplicates/objects without
    ids still resolve deterministically.
    """

    key = getattr(cand, "id_str", None)
    if key is not None:
        return ("id", key)
    key = getattr(cand, "id", None)
    if key is not None:
        return ("id", str(key))
    return ("pos", index)


def neighbor_scores_for(cand: Any, ranked: Sequence[Any]) -> list[float]:
    """Final scores of ``cand``'s immediate neighbours in the score-sorted ranking.

    ``ranked`` is the (unordered or ordered) list of scored candidates. It is
    sorted by ``final_score`` descending (deterministic id tie-break) and
    ``cand``'s immediate predecessor and successor scores are returned (one at
    each end of the list, none for a single-candidate ranking).

    If ``cand`` is not found in ``ranked`` by identity, its neighbours are
    derived from its ``final_score`` position within the sorted scores so the
    function remains total.
    """

    if not ranked:
        return []

    ordered = sorted(
        ranked,
        key=lambda c: (-float(getattr(c, "final_score", 0.0)), str(_candidate_key(c, 0))),
    )

    cand_key = _candidate_key(cand, -1)
    idx: int | None = None
    for i, c in enumerate(ordered):
        if _candidate_key(c, i) == cand_key:
            idx = i
            break

    neighbors: list[float] = []
    if idx is not None:
        if idx > 0:
            neighbors.append(float(ordered[idx - 1].final_score))
        if idx < len(ordered) - 1:
            neighbors.append(float(ordered[idx + 1].final_score))
        return neighbors

    # Not found by identity: treat by score position. The neighbours are the
    # closest score above and the closest score below the candidate's score.
    score = float(getattr(cand, "final_score", 0.0))
    above = [float(c.final_score) for c in ordered if float(c.final_score) >= score]
    below = [float(c.final_score) for c in ordered if float(c.final_score) < score]
    if above:
        neighbors.append(min(above))  # closest score at or above
    if below:
        neighbors.append(max(below))  # closest score below
    return neighbors


def compute_confidence_for(cand: Any, ranked: Sequence[Any]) -> float:
    """Compute confidence for ``cand`` given the full ``ranked`` list.

    Convenience wrapper over :func:`compute_confidence`: it extracts the
    candidate's ``final_score`` and ``profile.signal_availability`` and derives
    its neighbour scores from ``ranked``. ``cand`` and the elements of ``ranked``
    only need to expose ``final_score`` and ``profile.signal_availability``
    (duck-typed), so importing a concrete candidate class is unnecessary.
    """

    profile = getattr(cand, "profile", None)
    signal_availability = getattr(profile, "signal_availability", None)
    final_score = float(getattr(cand, "final_score", 0.0))
    neighbors = neighbor_scores_for(cand, ranked)
    return compute_confidence(signal_availability, final_score, neighbors)


class ConfidenceMixin:
    """Mixin exposing confidence computation as methods on a scoring engine.

    Lets :class:`~icrs.scoring.hard_filter.HybridScoringEngine` (or any engine)
    expose ``compute_confidence`` / ``compute_confidence_for`` without
    re-implementing the logic, mirroring the other ``*Mixin`` classes in this
    package.
    """

    @staticmethod
    def compute_confidence(
        signal_availability: Mapping[Any, float] | None,
        final_score: float,
        neighbor_scores: Sequence[float],
    ) -> float:
        return compute_confidence(signal_availability, final_score, neighbor_scores)

    @staticmethod
    def compute_confidence_for(cand: Any, ranked: Sequence[Any]) -> float:
        return compute_confidence_for(cand, ranked)


__all__ = [
    "COVERAGE_WEIGHT",
    "MARGIN_WEIGHT",
    "signal_coverage",
    "normalize_margin",
    "score_margin_to_neighbors",
    "neighbor_scores_for",
    "compute_confidence",
    "compute_confidence_for",
    "ConfidenceMixin",
]

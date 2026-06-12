"""Weighted composite score fusion with clamping (Task 11.1).

This module implements the final fusion step of the ICRS hybrid scoring engine
(design "Scoring Architecture" → "Composite model")::

    FinalScore = w1 * SemanticFitScore
               + w2 * CareerTrajectoryScore
               + w3 * BehavioralSignalScore
               + w4 * HardFilterPassScore
               - w5 * DisqualifyingFlagPenalty

It is deliberately a *new* module composed of standalone, pure functions plus an
optional :class:`CompositeMixin`, so it can be developed and tested independently
of — and without co-editing — the deterministic hard-filter gate
(``hard_filter.py``), the weight registry (``weights.py``), the semantic-fit
blend (``similarity.py``), or the trajectory/behavioral/penalty sub-scores
(``subscores.py``). It *imports* the :class:`~icrs.scoring.weights.WeightProfile`
it applies and reuses the :class:`~icrs.models.ranking.SignalBreakdown` output
model rather than redefining them.

Contract (Requirements 4.1, 4.5)
--------------------------------
- **Each sub-score is normalized/clamped to ``[0,1]`` *before* weighting**
  (Requirement 4.5). The four positive sub-scores (semantic, trajectory,
  behavioral, hard-filter-pass) and the penalty magnitude are every one clamped
  to ``[0,1]`` as the bundle is constructed, so an out-of-range input (e.g. a
  cosine artefact of ``1.2`` or a negative score) can never leak into the
  weighted sum.
- The four positive terms are weighted by ``w1..w4`` and summed; the
  **independent** penalty term ``w5 * disqualifying_penalty`` is *subtracted*
  (``w5`` is not part of the sum-to-one constraint — see ``weights.py``).
- **The final result is clamped to ``[0,1]``** (Requirement 4.1). Because the
  penalty is subtracted it can drive the raw value below ``0`` and, with
  out-of-spec weights, the positive terms could exceed ``1``; the final clamp
  guarantees the score-bounds property (Property 1) regardless of inputs.

Signals container choice
-------------------------
The task notes that :class:`~icrs.models.ranking.SignalBreakdown` is a natural
"signals" container. We nonetheless define a lightweight :class:`SignalBundle`
dataclass here and have :func:`composite` accept it, for three reasons:

1. **Decoupling.** ``composite`` should depend on a minimal, scoring-local value
   object, not on the pydantic *output* model whose role is serialization /
   API contract. Keeping the fusion input separate from the output model avoids
   coupling the scoring math to the output schema.
2. **Clamp-on-construction.** :class:`SignalBundle` clamps every component to
   ``[0,1]`` in :meth:`SignalBundle.__post_init__`, which is exactly the
   "normalize each sub-score to ``[0,1]`` *before* weighting" guarantee of
   Requirement 4.5 — encoded once, in the type, so every caller benefits.
3. **Interop, not lock-in.** :meth:`SignalBundle.from_breakdown` builds a bundle
   from a :class:`SignalBreakdown`, and :meth:`SignalBundle.to_breakdown`
   produces one for output, so the two representations interconvert freely.

Determinism: :func:`composite` is a pure function of its (clamped) inputs — no
LLM, no randomness — so identical inputs always yield identical outputs
(supports Requirement 2.3 / Property 7).
"""

from __future__ import annotations

from dataclasses import dataclass

from icrs.models.ranking import SignalBreakdown
from icrs.scoring.weights import WeightProfile


def _clamp01(value: float) -> float:
    """Clamp ``value`` to the inclusive range ``[0,1]``."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class SignalBundle:
    """The five sub-scores fused into a composite score, each clamped to ``[0,1]``.

    The four positive sub-scores correspond to the weighted terms ``w1..w4`` of
    the composite formula; ``disqualifying_penalty`` is the magnitude subtracted
    via the independent ``w5`` coefficient.

    Every field is clamped to ``[0,1]`` on construction
    (:meth:`__post_init__`), which encodes Requirement 4.5's "normalize each
    sub-score to ``[0,1]`` *before* weighting" guarantee directly in the type:
    no out-of-range value can reach the weighted sum, regardless of how the
    bundle was built.

    Attributes:
        semantic_fit: Blended dense/sparse semantic-fit sub-score (term ``w1``).
        career_trajectory: Career trajectory/arc alignment sub-score (term ``w2``).
        behavioral: Freshness-weighted behavioral sub-score (term ``w3``).
        hard_filter_pass: Soft must-have satisfaction ratio (term ``w4``).
        disqualifying_penalty: Soft red-flag penalty magnitude (term ``w5``).
    """

    semantic_fit: float
    career_trajectory: float
    behavioral: float
    hard_filter_pass: float
    disqualifying_penalty: float

    def __post_init__(self) -> None:
        # Frozen dataclass: clamp via object.__setattr__ so every component is in
        # [0,1] before any weighting occurs (Requirement 4.5).
        object.__setattr__(self, "semantic_fit", _clamp01(self.semantic_fit))
        object.__setattr__(
            self, "career_trajectory", _clamp01(self.career_trajectory)
        )
        object.__setattr__(self, "behavioral", _clamp01(self.behavioral))
        object.__setattr__(self, "hard_filter_pass", _clamp01(self.hard_filter_pass))
        object.__setattr__(
            self, "disqualifying_penalty", _clamp01(self.disqualifying_penalty)
        )

    @classmethod
    def from_breakdown(cls, breakdown: SignalBreakdown) -> "SignalBundle":
        """Build a :class:`SignalBundle` from a :class:`SignalBreakdown`.

        Lets callers fuse a previously-assembled output breakdown without
        re-deriving the sub-scores. Values are re-clamped on construction, so a
        breakdown is accepted even if it carried marginal floating-point drift.
        """

        return cls(
            semantic_fit=breakdown.semantic_fit,
            career_trajectory=breakdown.career_trajectory,
            behavioral=breakdown.behavioral,
            hard_filter_pass=breakdown.hard_filter_pass,
            disqualifying_penalty=breakdown.disqualifying_penalty,
        )

    def to_breakdown(self) -> SignalBreakdown:
        """Produce a :class:`SignalBreakdown` for output from this bundle.

        The bundle's already-clamped components are guaranteed to satisfy the
        ``[0,1]`` field bounds of :class:`SignalBreakdown`, so this never raises
        a validation error.
        """

        return SignalBreakdown(
            semantic_fit=self.semantic_fit,
            career_trajectory=self.career_trajectory,
            behavioral=self.behavioral,
            hard_filter_pass=self.hard_filter_pass,
            disqualifying_penalty=self.disqualifying_penalty,
        )


def build_signal_bundle(
    *,
    semantic_fit: float,
    career_trajectory: float,
    behavioral: float,
    hard_filter_pass: float,
    disqualifying_penalty: float,
) -> SignalBundle:
    """Construct a :class:`SignalBundle` from the five raw sub-scores.

    A thin, keyword-only convenience constructor. The returned bundle has every
    component clamped to ``[0,1]`` (Requirement 4.5).
    """

    return SignalBundle(
        semantic_fit=semantic_fit,
        career_trajectory=career_trajectory,
        behavioral=behavioral,
        hard_filter_pass=hard_filter_pass,
        disqualifying_penalty=disqualifying_penalty,
    )


def composite(signals: SignalBundle, weights: WeightProfile) -> float:
    """Fuse the sub-scores into a composite score in ``[0,1]`` (design ``composite``).

    Computes::

        raw = w1 * semantic_fit
            + w2 * career_trajectory
            + w3 * behavioral
            + w4 * hard_filter_pass
            - w5 * disqualifying_penalty

    then clamps ``raw`` to ``[0,1]``.

    The sub-scores arrive already normalized/clamped to ``[0,1]`` via
    :class:`SignalBundle` (Requirement 4.5), so the only remaining step required
    by the contract is applying the weights, subtracting the independent penalty
    term, and clamping the result (Requirement 4.1). The final clamp guarantees
    the result is a valid score in ``[0,1]`` even for extreme inputs (e.g. all
    positive sub-scores ``1.0`` with a non-trivial penalty, or a penalty large
    enough to drive the raw value negative).

    Args:
        signals: The five clamped sub-scores to fuse.
        weights: The :class:`WeightProfile` supplying ``w1..w5``.

    Returns:
        The composite final score in the inclusive range ``[0,1]``.
    """

    raw = (
        weights.w1 * signals.semantic_fit
        + weights.w2 * signals.career_trajectory
        + weights.w3 * signals.behavioral
        + weights.w4 * signals.hard_filter_pass
        - weights.w5 * signals.disqualifying_penalty
    )
    return _clamp01(raw)


class CompositeMixin:
    """Mixin exposing :func:`composite` as an engine method.

    Kept separate from :class:`~icrs.scoring.hard_filter.HybridScoringEngine` (and
    the other scoring mixins) so a later task can compose this onto the engine
    without co-editing the hard-filter, similarity, or subscore modules.
    """

    def composite(self, signals: SignalBundle, weights: WeightProfile) -> float:
        """Instance form of :func:`composite` (design ``composite``)."""

        return composite(signals, weights)


__all__ = [
    "SignalBundle",
    "build_signal_bundle",
    "composite",
    "CompositeMixin",
]

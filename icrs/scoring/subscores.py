"""Trajectory, behavioral, hard-filter-pass, and penalty sub-scores (Task 10.2).

This module implements four of the five sub-scores that the hybrid scoring
engine fuses into a composite score (design "Scoring Architecture" →
"How each sub-score is computed")::

    FinalScore = w1 * SemanticFitScore        # similarity.py (Task 10.1)
               + w2 * CareerTrajectoryScore    # this module
               + w3 * BehavioralSignalScore    # this module
               + w4 * HardFilterPassScore      # this module
               - w5 * DisqualifyingFlagPenalty # this module

It is deliberately a *new* module composed of standalone, pure functions plus an
optional :class:`SubScoreMixin`, so it can be developed and tested independently
of — and without co-editing — the deterministic hard-filter gate
(``hard_filter.py``), the weight registry (``weights.py``), or the semantic-fit
blend (``similarity.py``). Each function returns a value in the inclusive range
``[0,1]`` (the penalty is a magnitude in ``[0,1]`` that is *subtracted* during
fusion).

Required sub-scores and the exclusion path (Requirement 4.7)
------------------------------------------------------------
The design's *required* sub-scores are **semantic**, **trajectory**, and
**behavioral**. When a required sub-score genuinely *cannot be computed* for a
candidate, that candidate must be excluded from the scored results and an
indication identifying the unavailable sub-score recorded (Requirement 4.7).
This module models that path with:

    * :class:`RequiredSubScore`            — the enumerated required sub-scores.
    * :class:`SubScoreUnavailable`         — an exception raised when a required
      sub-score cannot be computed; it carries the offending
      :class:`RequiredSubScore` and a human-readable reason.
    * :class:`SubScoreUnavailableIndication` — a typed, recordable indication of
      the same (for callers that prefer a returned result object over an
      exception).

Crucially, **absent data is distinguished from "cannot compute"**:

    * *Absent data* (e.g. behavioral ``signal_availability == 0``) is handled by
      substituting the :data:`NEUTRAL_PRIOR` (``0.5``) — absence is treated as
      "unknown", never as zero, and never excludes a candidate
      (Requirement 7.2). :func:`behavioral_signal_score` therefore never raises
      for missing data.
    * *Cannot compute* means no evidence exists from which any component of a
      required sub-score could be derived (e.g. a profile with no trajectory
      arc, no roles, and no depth/breadth signature). Only that genuinely
      uncomputable case raises :class:`SubScoreUnavailable`.

Determinism: every function here is a pure function of the candidate's
enriched/structured fields and the requirement vector (no LLM, no randomness),
so identical inputs always yield identical outputs (supports Requirement 2.3).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum

from icrs.models.candidate import BehavioralSignal, EnrichedProfile, Role
from icrs.models.enums import DepthBreadth, SignalTier, TrajectoryArc
from icrs.models.job import (
    Requirement,
    RequirementCategory,
    RequirementVector,
    SeniorityBand,
)

# Reuse the freshness weighting used by Tier 3 enrichment so behavioral recency
# weighting is consistent across the pipeline (import, do not re-implement).
from icrs.pipeline.enrichment import freshness_weight

#: The default sub-score assigned when a tier has no data — absence is treated
#: as "unknown" rather than zero (Requirement 7.2, design "Key Decisions").
NEUTRAL_PRIOR: float = 0.5


# --------------------------------------------------------------------------- #
# Required-sub-score exclusion path (Requirement 4.7)
# --------------------------------------------------------------------------- #
class RequiredSubScore(str, Enum):
    """The sub-scores whose unavailability excludes a candidate (Requirement 4.7).

    Behavioral is *required* in the sense that it must contribute to the
    composite, but **absent behavioral data does not make it unavailable** — it
    is supplied via :data:`NEUTRAL_PRIOR` instead (Requirement 7.2). It appears
    here so an indication can still be recorded should the behavioral sub-score
    fail to compute for a reason other than absent data.
    """

    SEMANTIC = "SEMANTIC"
    TRAJECTORY = "TRAJECTORY"
    BEHAVIORAL = "BEHAVIORAL"


@dataclass(frozen=True)
class SubScoreUnavailableIndication:
    """A typed, recordable indication that a required sub-score was unavailable.

    Returned/recorded on the Requirement 4.7 exclusion path so the orchestrator
    can report *which* sub-score could not be computed for an excluded candidate.

    Attributes:
        sub_score: which required sub-score could not be computed.
        reason: a human-readable explanation of why it was unavailable.
    """

    sub_score: RequiredSubScore
    reason: str


class SubScoreUnavailable(Exception):
    """Raised when a *required* sub-score cannot be computed for a candidate.

    Carries the offending :class:`RequiredSubScore` and a human-readable reason
    so the caller can exclude the candidate and record an indication
    (Requirement 4.7). Use :attr:`indication` to obtain the typed, recordable
    form.
    """

    def __init__(self, sub_score: RequiredSubScore, reason: str) -> None:
        self.sub_score = sub_score
        self.reason = reason
        super().__init__(f"{sub_score.value} sub-score unavailable: {reason}")

    @property
    def indication(self) -> SubScoreUnavailableIndication:
        """The typed, recordable indication corresponding to this exception."""

        return SubScoreUnavailableIndication(self.sub_score, self.reason)


def _clamp01(value: float) -> float:
    """Clamp ``value`` to the inclusive range ``[0,1]``."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# --------------------------------------------------------------------------- #
# Career trajectory sub-score (Tier 2)
# --------------------------------------------------------------------------- #

#: Maps a career trajectory arc to a score in ``[0,1]``. ACCELERATING is the
#: strongest signal and DECLINING the weakest; STEADY/LATERAL sit between. The
#: mapping is strictly monotonic (ACCELERATING > STEADY > LATERAL > DECLINING),
#: which guarantees the ACCELERATING >= DECLINING ordering relied on by tests.
_ARC_SCORE: dict[TrajectoryArc, float] = {
    TrajectoryArc.ACCELERATING: 1.0,
    TrajectoryArc.STEADY: 0.75,
    TrajectoryArc.LATERAL: 0.5,
    TrajectoryArc.DECLINING: 0.25,
}

#: Ordinal ranking of seniority bands (low → high) used for band-alignment.
_SENIORITY_ORDINAL: dict[SeniorityBand, int] = {
    SeniorityBand.JUNIOR: 0,
    SeniorityBand.MID: 1,
    SeniorityBand.SENIOR: 2,
    SeniorityBand.STAFF: 3,
    SeniorityBand.LEAD: 4,
    SeniorityBand.EXECUTIVE: 5,
}
_MAX_SENIORITY_DISTANCE: int = max(_SENIORITY_ORDINAL.values())  # == 5

#: Ordinal ranking of the depth/breadth signature (specialist → generalist).
_DEPTH_ORDINAL: dict[DepthBreadth, int] = {
    DepthBreadth.SPECIALIST: 0,
    DepthBreadth.BALANCED: 1,
    DepthBreadth.GENERALIST: 2,
}
_MAX_DEPTH_DISTANCE: int = max(_DEPTH_ORDINAL.values())  # == 2

# Title keywords mapped to the seniority band they evidence. Checked against
# lower-cased role titles; the *highest* band evidenced across roles is used
# (a candidate's peak demonstrated seniority). Longer/more specific phrases are
# checked first so e.g. "senior manager" outranks a bare "manager".
_TITLE_BAND_KEYWORDS: tuple[tuple[str, SeniorityBand], ...] = (
    ("chief", SeniorityBand.EXECUTIVE),
    ("president", SeniorityBand.EXECUTIVE),
    ("vp", SeniorityBand.EXECUTIVE),
    ("vice president", SeniorityBand.EXECUTIVE),
    ("director", SeniorityBand.EXECUTIVE),
    ("head of", SeniorityBand.EXECUTIVE),
    ("principal", SeniorityBand.LEAD),
    ("lead", SeniorityBand.LEAD),
    ("staff", SeniorityBand.STAFF),
    ("senior", SeniorityBand.SENIOR),
    ("sr.", SeniorityBand.SENIOR),
    ("sr ", SeniorityBand.SENIOR),
    ("manager", SeniorityBand.SENIOR),
    ("mid-level", SeniorityBand.MID),
    ("intermediate", SeniorityBand.MID),
    ("junior", SeniorityBand.JUNIOR),
    ("jr.", SeniorityBand.JUNIOR),
    ("jr ", SeniorityBand.JUNIOR),
    ("associate", SeniorityBand.JUNIOR),
    ("intern", SeniorityBand.JUNIOR),
    ("trainee", SeniorityBand.JUNIOR),
    ("entry", SeniorityBand.JUNIOR),
)


def map_arc_to_score(arc: TrajectoryArc | None) -> float | None:
    """Map a trajectory arc to ``[0,1]`` (``None`` when the arc is absent).

    Returns ``None`` (not a default score) when ``arc`` is ``None`` so callers
    can distinguish "no arc evidence" from a genuine low score.
    """

    if arc is None:
        return None
    return _ARC_SCORE[arc]


def _infer_candidate_band(profile_roles: list[Role], tenure_months: int) -> SeniorityBand | None:
    """Infer the candidate's demonstrated seniority band, or ``None`` if unknown.

    The peak band evidenced by any role title is preferred. When titles carry no
    seniority keyword, the band is inferred from total tenure as a fallback.
    Returns ``None`` only when there is no role evidence at all (so band
    alignment cannot be computed).
    """

    if not profile_roles:
        return None

    best_ordinal: int | None = None
    for role in profile_roles:
        title = (role.title or "").lower()
        for keyword, band in _TITLE_BAND_KEYWORDS:
            if keyword in title:
                ordinal = _SENIORITY_ORDINAL[band]
                if best_ordinal is None or ordinal > best_ordinal:
                    best_ordinal = ordinal
                break  # first (most specific) keyword wins for this title

    if best_ordinal is not None:
        return _ordinal_to_band(best_ordinal)

    # No title keyword matched: fall back to a tenure-based estimate.
    if tenure_months <= 0:
        # Roles exist but neither titles nor tenure are informative; treat the
        # presence of roles as minimal (entry-level) evidence rather than None
        # so a candidate with roles is still scorable.
        return SeniorityBand.JUNIOR
    if tenure_months < 24:
        return SeniorityBand.JUNIOR
    if tenure_months < 60:
        return SeniorityBand.MID
    if tenure_months < 120:
        return SeniorityBand.SENIOR
    return SeniorityBand.STAFF


def _ordinal_to_band(ordinal: int) -> SeniorityBand:
    """Inverse of :data:`_SENIORITY_ORDINAL`."""

    for band, value in _SENIORITY_ORDINAL.items():
        if value == ordinal:
            return band
    raise ValueError(f"no seniority band for ordinal {ordinal}")


def seniority_alignment(cand: EnrichedProfile, required: SeniorityBand) -> float | None:
    """Alignment of the candidate's demonstrated band with the required band.

    ``1.0`` for an exact match, decreasing linearly with the ordinal distance
    between bands (``1 - distance / max_distance``). Returns ``None`` when the
    candidate's band cannot be inferred (no role evidence), so trajectory scoring
    can fall back to its other components.
    """

    cand_band = _infer_candidate_band(cand.base.roles, cand.base.total_tenure_months)
    if cand_band is None:
        return None
    distance = abs(_SENIORITY_ORDINAL[cand_band] - _SENIORITY_ORDINAL[required])
    return _clamp01(1.0 - distance / _MAX_SENIORITY_DISTANCE)


def _preferred_depth_for_band(required: SeniorityBand) -> DepthBreadth:
    """The depth/breadth signature most appropriate for a required seniority band.

    Pragmatic PoC heuristic: junior/mid roles reward focused depth (SPECIALIST),
    senior/staff roles reward a BALANCED profile, and lead/executive roles reward
    breadth (GENERALIST). Documented and deterministic.
    """

    ordinal = _SENIORITY_ORDINAL[required]
    if ordinal <= _SENIORITY_ORDINAL[SeniorityBand.MID]:
        return DepthBreadth.SPECIALIST
    if ordinal <= _SENIORITY_ORDINAL[SeniorityBand.STAFF]:
        return DepthBreadth.BALANCED
    return DepthBreadth.GENERALIST


def depth_breadth_alignment(
    depth_breadth: DepthBreadth | None, required: SeniorityBand
) -> float | None:
    """Alignment of the candidate's depth/breadth signature with the role's needs.

    ``1.0`` when the candidate's signature matches the band-preferred signature,
    decreasing linearly with the ordinal distance on the
    SPECIALIST→BALANCED→GENERALIST scale. Returns ``None`` when the candidate has
    no depth/breadth signature (so it can be omitted from the trajectory mean).
    """

    if depth_breadth is None:
        return None
    preferred = _preferred_depth_for_band(required)
    distance = abs(_DEPTH_ORDINAL[depth_breadth] - _DEPTH_ORDINAL[preferred])
    return _clamp01(1.0 - distance / _MAX_DEPTH_DISTANCE)


def career_trajectory_score(cand: EnrichedProfile, reqs: RequirementVector) -> float:
    """Compute the career-trajectory sub-score in ``[0,1]`` (design ``CareerTrajectoryScore``).

    The mean of up to three components — the arc score (mapped from the
    candidate's :class:`TrajectoryArc`), the seniority band alignment with
    ``reqs.seniority_band``, and the depth/breadth alignment — over whichever
    components can be derived. Computing the mean of *available* components keeps
    a partially-observed candidate scorable (fairness) while preserving the
    design's intent.

    Raises:
        SubScoreUnavailable: when *none* of the three components can be derived
            (no arc, no role evidence, and no depth/breadth signature). This is
            the genuine "cannot compute" case of Requirement 4.7 — distinct from
            absent data on a single component, which is simply omitted from the
            mean.
    """

    components: list[float] = []

    arc = map_arc_to_score(cand.trajectory_arc)
    if arc is not None:
        components.append(arc)

    band = seniority_alignment(cand, reqs.seniority_band)
    if band is not None:
        components.append(band)

    depth = depth_breadth_alignment(cand.depth_breadth, reqs.seniority_band)
    if depth is not None:
        components.append(depth)

    if not components:
        raise SubScoreUnavailable(
            RequiredSubScore.TRAJECTORY,
            "no trajectory evidence: candidate has no trajectory arc, no role "
            "history from which to infer a seniority band, and no depth/breadth "
            "signature",
        )

    return _clamp01(sum(components) / len(components))


# --------------------------------------------------------------------------- #
# Behavioral sub-score (Tier 3)
# --------------------------------------------------------------------------- #
def _normalize_metric(value: float) -> float:
    """Squash a raw behavioral metric value into ``[0,1]``.

    Behavioral metric magnitudes (commit counts, endorsements, ...) are
    unbounded and non-negative in practice. A saturating transform
    ``v / (1 + v)`` maps ``[0, ∞)`` smoothly into ``[0, 1)`` (``0 → 0``,
    growing monotonically and saturating toward ``1``). Negative values — which
    carry no positive evidence — are floored to ``0``.
    """

    if value <= 0.0:
        return 0.0
    return value / (1.0 + value)


def behavioral_signal_score(
    cand: EnrichedProfile,
    *,
    half_life_days: float | None = None,
) -> float:
    """Compute the behavioral sub-score in ``[0,1]`` (design ``BehavioralSignalScore``).

    When the candidate's behavioral ``signal_availability`` is ``0`` (no Tier 3
    data), returns the :data:`NEUTRAL_PRIOR` (``0.5``) rather than zero — absence
    is "unknown", not a low score (Requirement 7.2). This function therefore
    **never raises for absent data**; it is the canonical example of the
    "absent → neutral prior" path distinguished from the "cannot compute →
    exclude" path of Requirement 4.7.

    Otherwise it is the freshness-weighted mean of the normalized behavioral
    signal values: each signal's normalized value (see :func:`_normalize_metric`)
    is weighted by :func:`~icrs.pipeline.enrichment.freshness_weight` of its
    recency, the weighted values are summed and divided by the signal count, and
    the result clamped to ``[0,1]`` (mirroring the design pseudocode).

    Args:
        cand: the enriched candidate profile.
        half_life_days: optional override for the freshness half-life; defaults
            to the enrichment module's default when ``None``.
    """

    availability = cand.signal_availability.get(SignalTier.BEHAVIORAL, 0.0)
    if availability == 0.0:
        return NEUTRAL_PRIOR

    signals = cand.behavioral_signals
    if not signals:
        # Availability records coverage > 0 but no concrete signals are
        # attached: treat as unknown rather than zero (fairness).
        return NEUTRAL_PRIOR

    total = 0.0
    for signal in signals:
        if half_life_days is None:
            fresh = freshness_weight(signal.recency_days)
        else:
            fresh = freshness_weight(signal.recency_days, half_life_days=half_life_days)
        total += fresh * _normalize_metric(signal.value)

    return _clamp01(total / len(signals))


# --------------------------------------------------------------------------- #
# Hard-filter-pass sub-score (soft must-have satisfaction ratio)
# --------------------------------------------------------------------------- #
# A predicate deciding whether a candidate satisfies a single MUST_HAVE
# requirement. Pluggable so callers/tests can override the default matcher.
MustHaveMatch = Callable[[EnrichedProfile, Requirement], bool]

# Token matching for the default must-have predicate. Kept local to this module
# so it does not edit or import the hard-filter gate's internals; it mirrors the
# same deterministic token-overlap approach (Requirement 4.6 / design
# ``HardFilterPassScore``).
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
        "at", "by", "from", "as", "is", "are", "be", "been", "being", "no",
        "not", "must", "have", "has", "had", "should", "shall", "will", "would",
        "can", "could", "may", "might", "do", "does", "did", "than", "then",
        "that", "this", "these", "those", "it", "its", "any", "all", "more",
        "less", "least", "most", "experience", "years", "year", "required",
        "require", "requires", "requirement",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lower-case ``text`` and return its alphanumeric tokens (length >= 2)."""

    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _salient_tokens(text: str) -> set[str]:
    """Meaningful tokens of a requirement, with stop-words removed."""

    return {tok for tok in _tokenize(text) if tok not in _STOPWORDS}


def _candidate_evidence_phrases(cand: EnrichedProfile) -> Iterator[str]:
    """Yield each concrete piece of candidate evidence as raw text.

    Absent scalar fields (``None``) and empty collections contribute nothing, so
    a sparse profile simply yields fewer phrases — it is never credited with
    evidence it does not have. ``signal_availability`` is intentionally not
    consulted here.
    """

    base = cand.base
    for role in base.roles:
        yield role.title
        yield role.company
    for edu in base.education:
        if edu.institution:
            yield edu.institution
        if edu.degree:
            yield edu.degree
        if edu.field_of_study:
            yield edu.field_of_study
    yield from base.certifications
    yield from base.explicit_skills
    yield from cand.inferred_responsibilities
    yield from cand.implicit_skills


def default_must_have_match(cand: EnrichedProfile, requirement: Requirement) -> bool:
    """Default deterministic predicate: does ``cand`` satisfy ``requirement``?

    A MUST_HAVE is satisfied when some concrete evidence item in the candidate's
    structured/enriched data contains *all* of the requirement's salient
    (non-stop-word) tokens — the same token-overlap rule the hard-filter gate
    uses for disqualifiers. A requirement with no salient tokens is treated as
    trivially satisfied (it carries no checkable content).
    """

    salient = _salient_tokens(requirement.text)
    if not salient:
        return True

    for phrase in _candidate_evidence_phrases(cand):
        tokens = set(_tokenize(phrase))
        if tokens and salient <= tokens:
            return True
    return False


def hard_filter_pass_score(
    reqs: RequirementVector,
    cand: EnrichedProfile,
    *,
    matcher: MustHaveMatch = default_must_have_match,
) -> float:
    """Soft must-have satisfaction ratio in ``[0,1]`` (design ``HardFilterPassScore``).

    Returns the fraction of ``reqs``'s MUST_HAVE requirements that ``cand``
    satisfies (per ``matcher``). This is the *soft* pass-ratio that contributes
    to the composite score, distinct from the binary hard-filter gate (which
    removes candidates positively matching a DISQUALIFYING criterion).

    When ``reqs`` contains no MUST_HAVE requirements, returns ``1.0`` (there is
    nothing left unsatisfied). The value is always in ``[0,1]``.
    """

    must_haves = reqs.must_haves
    if not must_haves:
        return 1.0

    satisfied = sum(1 for r in must_haves if matcher(cand, r))
    return _clamp01(satisfied / len(must_haves))


# --------------------------------------------------------------------------- #
# Disqualifying-flag penalty (soft red flags)
# --------------------------------------------------------------------------- #
#: Per-flag penalty increment. ``count`` soft flags yield ``0.2 * count`` capped
#: at ``1.0`` (design ``DisqualifyingFlagPenalty``).
_PENALTY_PER_FLAG: float = 0.2


def detect_soft_flags(reqs: RequirementVector, cand: EnrichedProfile) -> list[str]:
    """Detect deterministic *soft* red flags for a candidate (pragmatic PoC set).

    Soft flags are mild negative signals — not absolute disqualifiers (those are
    gated out earlier). For the PoC two deterministic flag types are detected:

    1. **Expired credential** — any certification whose text mentions "expired"
       (case-insensitive). One flag per such certification.
    2. **Claim/activity mismatch** — the candidate *has* behavioral activity but
       *none* of it corroborates any of the candidate's explicit skills (the
       public footprint does not back the claimed skills). Counted at most once,
       and **only when behavioral signals are present** so that an absent public
       footprint never produces a flag (fairness — absence is not a red flag).

    ``reqs`` is accepted for interface symmetry and future role-specific flags;
    the PoC flag set is candidate-intrinsic.

    Returns:
        A list of human-readable flag descriptions (possibly empty).
    """

    flags: list[str] = []

    for cert in cand.base.certifications:
        if cert and "expired" in cert.lower():
            flags.append(f"expired credential: {cert}")

    behavioral = cand.behavioral_signals
    if behavioral:
        explicit_skills = {
            s.strip().lower() for s in cand.base.explicit_skills if s and s.strip()
        }
        if explicit_skills:
            corroborated = {
                skill.strip().lower()
                for signal in behavioral
                for skill in signal.corroborates_skill
                if skill and skill.strip()
            }
            if not (explicit_skills & corroborated):
                flags.append(
                    "claim/activity mismatch: behavioral activity corroborates "
                    "none of the candidate's explicit skills"
                )

    return flags


def disqualifying_flag_penalty(reqs: RequirementVector, cand: EnrichedProfile) -> float:
    """Soft red-flag penalty magnitude in ``[0,1]`` (design ``DisqualifyingFlagPenalty``).

    Computes ``clamp(0.2 * flag_count, 0, 1)`` over the soft flags detected by
    :func:`detect_soft_flags`. This is a penalty *magnitude* (subtracted during
    composite fusion), so a larger value means a stronger penalty; ``0.0`` means
    no soft flags were detected.
    """

    flag_count = len(detect_soft_flags(reqs, cand))
    return _clamp01(_PENALTY_PER_FLAG * flag_count)


# --------------------------------------------------------------------------- #
# Engine mixin
# --------------------------------------------------------------------------- #
class SubScoreMixin:
    """Mixin exposing the Task 10.2 sub-scores as engine methods.

    Kept separate from :class:`~icrs.scoring.hard_filter.HybridScoringEngine` (and
    from the semantic-fit mixin) so a later task can compose this onto the engine
    without co-editing the hard-filter or similarity modules. Each method
    delegates to the standalone functions above.
    """

    def career_trajectory_score(
        self, cand: EnrichedProfile, reqs: RequirementVector
    ) -> float:
        """Instance form of :func:`career_trajectory_score`."""

        return career_trajectory_score(cand, reqs)

    def behavioral_signal_score(
        self, cand: EnrichedProfile, *, half_life_days: float | None = None
    ) -> float:
        """Instance form of :func:`behavioral_signal_score`."""

        return behavioral_signal_score(cand, half_life_days=half_life_days)

    def hard_filter_pass_score(
        self,
        reqs: RequirementVector,
        cand: EnrichedProfile,
        *,
        matcher: MustHaveMatch = default_must_have_match,
    ) -> float:
        """Instance form of :func:`hard_filter_pass_score`."""

        return hard_filter_pass_score(reqs, cand, matcher=matcher)

    def disqualifying_flag_penalty(
        self, reqs: RequirementVector, cand: EnrichedProfile
    ) -> float:
        """Instance form of :func:`disqualifying_flag_penalty`."""

        return disqualifying_flag_penalty(reqs, cand)


__all__ = [
    "NEUTRAL_PRIOR",
    "RequiredSubScore",
    "SubScoreUnavailable",
    "SubScoreUnavailableIndication",
    "map_arc_to_score",
    "seniority_alignment",
    "depth_breadth_alignment",
    "career_trajectory_score",
    "behavioral_signal_score",
    "hard_filter_pass_score",
    "default_must_have_match",
    "detect_soft_flags",
    "disqualifying_flag_penalty",
    "MustHaveMatch",
    "SubScoreMixin",
]

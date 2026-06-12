"""Fairness and missing-signal guarantees (Task 15.2).

This module **centralizes** the fairness policy that the rest of the ICRS
pipeline relies on and exposes it as composable, well-documented helpers. It is
deliberately a *new* module of standalone, pure functions: it imports the few
shared values it needs (the :data:`~icrs.scoring.subscores.NEUTRAL_PRIOR`, the
:func:`~icrs.scoring.confidence.compute_confidence` formula, the composite
fusion, the :class:`~icrs.scoring.weights.WeightProfile`) without editing any of
those modules, so it can be developed and unit-tested independently. The
orchestrator (Tasks 16.1 / 16.2) wires these helpers into the pipeline; here we
*provide and test the guarantees*.

The four guarantees this module encodes map directly to Requirement 7:

1. **Protected_Proxy exclusion (Requirement 7.1).** Demographic / proxy
   attributes (name, gender markers, age proxies such as ``date_of_birth`` /
   ``graduation_year``, photos, and location/address) must never reach scoring
   inputs or explanations. :data:`PROTECTED_PROXY_FIELDS` enumerates the policy
   and :func:`strip_protected_proxies` removes them from any structured input
   before scoring. The JD decomposer, enricher, reranker, and explanation
   generator already build prompts from positive job-relevant whitelists; this
   helper centralizes the *negative* policy so any new structured-input path can
   route through one audited place.

2. **Neutral prior for missing tiers (Requirement 7.2).**
   :func:`apply_neutral_prior_for_missing_tiers` substitutes the
   :data:`~icrs.scoring.subscores.NEUTRAL_PRIOR` (``0.5``) for any tier whose
   ``signal_availability`` is ``0`` — absence is treated as "unknown", never as
   a zero sub-score. The documented rule (and the design) is that a missing tier
   **reduces confidence** (via the confidence module) rather than lowering a
   candidate's rank: the neutral prior keeps the sub-score rank-neutral while
   the all-zero coverage on that tier pulls confidence down.

3. **Counterfactual proxy invariance (Requirement 7.4).**
   :func:`is_proxy_invariant` checks that perturbing protected-proxy attributes
   while holding job-relevant signals constant does not change a candidate's
   score. This is the testable basis for the rank-delta-0 guarantee: if no
   candidate's score moves when only proxies change, no candidate's rank can
   move.

4. **Fully-sparse candidate inclusion (Requirement 7.5).**
   :func:`score_fully_sparse_candidate` scores a candidate whose
   ``signal_availability`` is ``0`` in *every* tier using the neutral prior for
   all sub-scores and yields the **minimum** confidence (coverage contributes
   ``0``), so the candidate is *included* at the confidence floor rather than
   excluded.

Determinism: every function here is a pure function of its inputs (no LLM, no
randomness), so identical inputs always yield identical outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from icrs.models.enums import SignalTier
from icrs.scoring.composite import SignalBundle, composite

# Reuse the canonical neutral prior and confidence formula rather than
# re-implementing them (single source of truth across the scoring layer).
from icrs.scoring.confidence import compute_confidence
from icrs.scoring.subscores import NEUTRAL_PRIOR
from icrs.scoring.weights import WeightProfile

# The three signal tiers the pipeline reasons over.
_TIERS: tuple[SignalTier, ...] = (
    SignalTier.STRUCTURAL,
    SignalTier.SEMANTIC,
    SignalTier.BEHAVIORAL,
)


# --------------------------------------------------------------------------- #
# 1. Protected_Proxy definition + exclusion (Requirement 7.1)
# --------------------------------------------------------------------------- #
#: The demographic / proxy attribute names that MUST be excluded from scoring
#: inputs and explanations (Requirement 7.1 / ``Protected_Proxy`` in the
#: glossary). Names are stored in their *normalized* form (lower-cased, with
#: spaces and hyphens collapsed to underscores) so matching is robust to common
#: key spellings. The set covers the five proxy families named in the design:
#:
#:   * **name** — personal/identity names,
#:   * **gender markers** — gender, sex, pronouns, honorifics,
#:   * **age proxies** — explicit age, dates of birth, graduation years,
#:   * **photo** — any image/portrait reference,
#:   * **location / address** — residential location used as a demographic proxy,
#:
#: plus the closely-related sensitive attributes (nationality, ethnicity, race,
#: religion, marital status) that are demographic proxies by the same rationale.
PROTECTED_PROXY_FIELDS: frozenset[str] = frozenset(
    {
        # --- name ---
        "name",
        "full_name",
        "first_name",
        "last_name",
        "given_name",
        "family_name",
        "middle_name",
        "maiden_name",
        "surname",
        "forename",
        "preferred_name",
        # --- gender markers ---
        "gender",
        "sex",
        "pronoun",
        "pronouns",
        "title",
        "honorific",
        "salutation",
        # --- age proxies ---
        "age",
        "dob",
        "date_of_birth",
        "birth_date",
        "birthdate",
        "birthday",
        "year_of_birth",
        "graduation_year",
        "grad_year",
        # --- photo ---
        "photo",
        "photo_url",
        "picture",
        "image",
        "image_url",
        "avatar",
        "headshot",
        "portrait",
        # --- location / address (as demographic proxy) ---
        "location",
        "address",
        "home_address",
        "residential_address",
        "city",
        "state",
        "province",
        "country",
        "region",
        "zip",
        "zip_code",
        "zipcode",
        "postal_code",
        "postcode",
        # --- other closely-related sensitive demographic attributes ---
        "nationality",
        "ethnicity",
        "race",
        "religion",
        "marital_status",
        "citizenship",
    }
)


def _normalize_field_name(key: Any) -> str:
    """Normalize a mapping key to the canonical form used for proxy matching.

    Lower-cases the key and collapses spaces and hyphens to underscores so that
    ``"Date Of Birth"``, ``"date-of-birth"`` and ``"date_of_birth"`` all match
    the same policy entry. Non-string keys are coerced via ``str`` first.
    """

    text = str(key).strip().lower()
    for sep in (" ", "-"):
        text = text.replace(sep, "_")
    # Collapse any runs of underscores produced by mixed separators.
    while "__" in text:
        text = text.replace("__", "_")
    return text


def is_protected_proxy_field(key: Any) -> bool:
    """Return ``True`` when ``key`` names a Protected_Proxy attribute.

    Matching is performed on the *normalized* key (see
    :func:`_normalize_field_name`) against :data:`PROTECTED_PROXY_FIELDS`, so it
    is insensitive to case and to space/hyphen/underscore spelling differences.
    """

    return _normalize_field_name(key) in PROTECTED_PROXY_FIELDS


def _strip_value(value: Any) -> Any:
    """Recursively strip proxies from nested containers.

    Dicts are filtered via :func:`strip_protected_proxies`; lists/tuples have
    each element stripped (preserving sequence type for lists); all other values
    are returned unchanged.
    """

    if isinstance(value, Mapping):
        return strip_protected_proxies(value)
    if isinstance(value, list):
        return [_strip_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_value(item) for item in value)
    return value


def strip_protected_proxies(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` with all Protected_Proxy fields removed.

    This is the single policy gate every structured scoring input (and any
    structured payload destined for an explanation) should pass through before
    use (Requirement 7.1). It:

    * removes every top-level key that names a Protected_Proxy attribute, and
    * recurses into nested mappings and lists so a proxy nested inside, e.g.,
      a ``"contact"`` sub-object is removed as well.

    The input is **not mutated** — a new ``dict`` is returned — so callers can
    keep the original record intact for audit/persistence while scoring only the
    proxy-free projection. Job-relevant fields (skills, roles, tenure,
    education qualifications, etc.) are preserved untouched.

    Args:
        data: A structured candidate/record mapping. A non-mapping argument is
            returned unchanged (defensive; lets the helper sit on a mixed path).

    Returns:
        A new dict containing only the non-proxy fields of ``data``, with nested
        containers recursively cleaned.
    """

    if not isinstance(data, Mapping):
        return data  # type: ignore[return-value]

    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if is_protected_proxy_field(key):
            continue
        cleaned[key] = _strip_value(value)
    return cleaned


# --------------------------------------------------------------------------- #
# 2. Neutral prior for missing tiers (Requirement 7.2)
# --------------------------------------------------------------------------- #
def apply_neutral_prior_for_missing_tiers(
    subscores_by_tier: Mapping[SignalTier, float],
    signal_availability: Mapping[SignalTier, float],
    neutral_prior: float = NEUTRAL_PRIOR,
) -> dict[SignalTier, float]:
    """Substitute the neutral prior for any tier whose availability is ``0``.

    For each tier present in ``subscores_by_tier``, the returned map carries:

    * the :data:`~icrs.scoring.subscores.NEUTRAL_PRIOR` (``0.5`` by default) when
      that tier's ``signal_availability`` is ``0`` (or the tier is absent from
      the availability map — both mean "no data for this tier"), so absence is
      treated as *unknown* rather than as a zero sub-score, **and**
    * the candidate's actual sub-score otherwise.

    Documented rule (design "Key Decisions" + Requirement 7.2): applying the
    neutral prior keeps the missing tier **rank-neutral** (it neither rewards nor
    penalizes the candidate relative to the midpoint). The *consequence* of a
    missing tier is a **reduced confidence**, not a lowered rank — the all-zero
    coverage on that tier lowers the mean coverage fed to
    :func:`~icrs.scoring.confidence.compute_confidence`, which is the only place
    a missing tier is allowed to affect the result. This function therefore only
    ever touches sub-scores; it never excludes a candidate.

    Args:
        subscores_by_tier: The candidate's computed sub-score per tier.
        signal_availability: Per-tier coverage in ``[0,1]``; ``0`` (or a missing
            tier) triggers the neutral-prior substitution.
        neutral_prior: The value substituted for a zero-availability tier;
            defaults to :data:`NEUTRAL_PRIOR`.

    Returns:
        A new map (same tier keys as ``subscores_by_tier``) with neutral priors
        substituted where availability is ``0``.
    """

    adjusted: dict[SignalTier, float] = {}
    for tier, subscore in subscores_by_tier.items():
        availability = signal_availability.get(tier, 0.0)
        if availability == 0.0:
            adjusted[tier] = neutral_prior
        else:
            adjusted[tier] = subscore
    return adjusted


# --------------------------------------------------------------------------- #
# 3. Counterfactual proxy invariance (Requirement 7.4)
# --------------------------------------------------------------------------- #
def is_proxy_invariant(
    score_fn: Callable[[Mapping[str, Any]], float],
    base_input: Mapping[str, Any],
    proxy_perturbations: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Check that perturbing Protected_Proxy attributes leaves the score unchanged.

    This is the counterfactual-fairness invariant underpinning the rank-delta-0
    guarantee (Requirement 7.4): if a candidate's score is invariant to changes
    in protected-proxy attributes (while all job-relevant signals are held
    constant), then no ordering — and hence no rank — can shift when only proxies
    change.

    The check evaluates ``score_fn`` on ``base_input`` to obtain a baseline, then
    for each perturbation overlays the proxy override(s) onto a copy of
    ``base_input`` and re-scores. It returns ``True`` iff every perturbed score
    equals the baseline within ``tolerance``.

    By construction the perturbations only *overlay* keys onto ``base_input``;
    callers should restrict them to Protected_Proxy keys (see
    :data:`PROTECTED_PROXY_FIELDS`) so that job-relevant signals stay constant —
    that is the "holding job-relevant signals constant" precondition of the
    requirement.

    Args:
        score_fn: A scoring callable mapping a structured input to a score.
        base_input: The candidate's structured input (the baseline).
        proxy_perturbations: An iterable of proxy override maps; each is merged
            over a copy of ``base_input`` to form a perturbed input.
        tolerance: Absolute tolerance for the score-equality comparison.

    Returns:
        ``True`` if the score is invariant across all perturbations (proxy-blind
        scorer), ``False`` if any perturbation changes the score (a scorer that
        improperly reads a proxy).
    """

    baseline = float(score_fn(base_input))
    for perturbation in proxy_perturbations:
        perturbed = dict(base_input)
        perturbed.update(perturbation)
        if abs(float(score_fn(perturbed)) - baseline) > tolerance:
            return False
    return True


# --------------------------------------------------------------------------- #
# 4. Fully-sparse candidate handling (Requirement 7.5)
# --------------------------------------------------------------------------- #
def fully_sparse_signal_availability() -> dict[SignalTier, float]:
    """The ``signal_availability`` map of a candidate with no data in any tier.

    Every tier maps to ``0.0`` — i.e. zero coverage everywhere. Useful for
    scoring/confidence of a fully-sparse candidate and for tests.
    """

    return {tier: 0.0 for tier in _TIERS}


def is_fully_sparse(signal_availability: Mapping[SignalTier, float]) -> bool:
    """Return ``True`` when ``signal_availability`` is ``0`` in every tier.

    A tier missing from the map counts as ``0`` coverage, so a candidate with an
    empty or all-zero availability map is fully sparse. Such a candidate must be
    *included* at minimum confidence (Requirement 7.5), never excluded.
    """

    return all(signal_availability.get(tier, 0.0) == 0.0 for tier in _TIERS)


def neutral_prior_signal_bundle(neutral_prior: float = NEUTRAL_PRIOR) -> SignalBundle:
    """A :class:`SignalBundle` with the neutral prior for every positive sub-score.

    All four positive sub-scores (semantic, trajectory, behavioral,
    hard-filter-pass) are set to ``neutral_prior`` and the disqualifying penalty
    is ``0.0`` (a fully-sparse candidate exhibits no soft red flags either). This
    is the sub-score bundle used to score a fully-sparse candidate so that
    absence is scored as "unknown" rather than as zero (Requirement 7.5).
    """

    return SignalBundle(
        semantic_fit=neutral_prior,
        career_trajectory=neutral_prior,
        behavioral=neutral_prior,
        hard_filter_pass=neutral_prior,
        disqualifying_penalty=0.0,
    )


@dataclass(frozen=True)
class FullySparseScore:
    """The result of scoring a fully-sparse candidate (Requirement 7.5).

    Attributes:
        composite_score: The composite score in ``[0,1]`` computed from the
            neutral-prior bundle under the supplied weights. The candidate is
            *scored and included*, never excluded.
        confidence: The candidate's confidence in ``[0,1]``. Because coverage is
            ``0`` in every tier, the coverage term contributes ``0`` — its floor
            — so this is the minimum confidence achievable for the candidate's
            margin position (and exactly ``0.0`` when it ties a neighbour).
        bundle: The neutral-prior :class:`SignalBundle` used (every positive
            sub-score equals the neutral prior).
        signal_availability: The all-zero per-tier availability map recorded for
            the candidate.
    """

    composite_score: float
    confidence: float
    bundle: SignalBundle
    signal_availability: dict[SignalTier, float]


def score_fully_sparse_candidate(
    weights: WeightProfile,
    *,
    final_score: float | None = None,
    neighbor_scores: Sequence[float] = (),
    neutral_prior: float = NEUTRAL_PRIOR,
) -> FullySparseScore:
    """Score a candidate that has ``signal_availability`` of ``0`` in every tier.

    Implements Requirement 7.5: rather than excluding a candidate with no signal
    anywhere, ICRS scores it with the :data:`NEUTRAL_PRIOR` for *every* sub-score
    and **includes** it at the minimum confidence. Concretely this helper:

    1. builds a neutral-prior :class:`SignalBundle`
       (:func:`neutral_prior_signal_bundle`),
    2. fuses it with ``weights`` via :func:`~icrs.scoring.composite.composite` to
       get the composite score, and
    3. computes confidence with
       :func:`~icrs.scoring.confidence.compute_confidence` over an **all-zero**
       availability map, so the coverage term is ``0`` (its minimum) and the
       candidate lands at the confidence floor for its margin.

    The confidence uses ``final_score`` for the margin computation when supplied
    (the post-rerank score), otherwise the composite score. ``neighbor_scores``
    are the candidate's immediate neighbours in the ranking; with a tied
    neighbour the margin is ``0`` and the confidence is exactly ``0.0`` — the
    global minimum.

    Args:
        weights: The :class:`WeightProfile` to fuse the neutral-prior bundle.
        final_score: Optional post-rerank score used for the confidence margin;
            defaults to the composite score when ``None``.
        neighbor_scores: Final scores of the candidate's immediate ranking
            neighbours (empty ⟹ lone candidate ⟹ maximal margin).
        neutral_prior: The neutral prior to use for every sub-score; defaults to
            :data:`NEUTRAL_PRIOR`.

    Returns:
        A :class:`FullySparseScore` carrying the composite score, the minimum
        confidence, the neutral-prior bundle, and the all-zero availability map.
    """

    bundle = neutral_prior_signal_bundle(neutral_prior)
    composite_score = composite(bundle, weights)
    availability = fully_sparse_signal_availability()
    score_for_margin = composite_score if final_score is None else final_score
    confidence = compute_confidence(availability, score_for_margin, neighbor_scores)
    return FullySparseScore(
        composite_score=composite_score,
        confidence=confidence,
        bundle=bundle,
        signal_availability=availability,
    )


__all__ = [
    # Protected_Proxy exclusion (Requirement 7.1).
    "PROTECTED_PROXY_FIELDS",
    "is_protected_proxy_field",
    "strip_protected_proxies",
    # Neutral prior for missing tiers (Requirement 7.2).
    "apply_neutral_prior_for_missing_tiers",
    # Counterfactual proxy invariance (Requirement 7.4).
    "is_proxy_invariant",
    # Fully-sparse candidate handling (Requirement 7.5).
    "fully_sparse_signal_availability",
    "is_fully_sparse",
    "neutral_prior_signal_bundle",
    "FullySparseScore",
    "score_fully_sparse_candidate",
]

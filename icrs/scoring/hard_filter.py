"""Deterministic hard-filter gate for the ICRS hybrid scoring engine (Task 8.1).

The hard filter is the deterministic gate that runs *before* any expensive
scoring. Its single job is to remove candidates who **positively match** a
``DISQUALIFYING`` criterion, while never penalising a candidate for *missing*
data (the fairness guarantee of Requirements 4.4 and 7.3).

Design contract (see design.md "Hard-filter gate")::

    PROCEDURE hardFilter(reqs, cand): FilterResult { passed, reasons }
        FOR each DISQUALIFYING r IN reqs.requirements:
            IF candidateMatches(cand, r): RETURN { passed: false, ... }
        RETURN { passed: true, reasons: [] }     # missing data never excludes

Key invariants enforced here:
    - A candidate is excluded **only** when it positively matches at least one
      DISQUALIFYING criterion (Requirement 4.4).
    - Absence of data — including ``signal_availability`` of 0 for a tier, an
      empty profile, or simply no evidence corresponding to a criterion — is
      **never** a positive match and therefore never excludes a candidate
      (Requirements 4.4, 7.3). This module is "fail-open" on missing data.
    - A candidate matching no DISQUALIFYING criterion is always retained.

Determinism: matching is purely a function of the requirement text and the
candidate's structured/enriched fields (no LLM, no randomness), so identical
inputs always yield identical results (supports Requirement 2.3).

The match predicate is deliberately pluggable. The default,
:func:`default_disqualifier_match`, is a pragmatic keyword/skill/cert matcher
(documented below). Callers (and tests) may pass an alternative ``matcher`` to
:func:`hard_filter` to customise or stub the match condition.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from icrs.models.candidate import EnrichedProfile
from icrs.models.job import Requirement, RequirementCategory, RequirementVector

# A small, deliberately conservative English stop-word set. Removing these from
# a criterion's salient tokens prevents a candidate from being disqualified on
# the basis of incidental connective words (e.g. matching "the" or "and").
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

# Tokens are alphanumeric runs of length >= 2. Single characters (e.g. the "c"
# from "c++") are dropped to avoid noisy one-letter matches; this is a documented
# PoC limitation, not a correctness concern for the missing-data invariant.
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


# A match predicate: given a candidate and a single DISQUALIFYING requirement,
# return True iff the candidate *positively* matches that criterion.
MatchPredicate = Callable[[EnrichedProfile, Requirement], bool]


@dataclass(frozen=True)
class FilterResult:
    """Outcome of the hard-filter gate for a single candidate.

    Attributes:
        passed: ``True`` when the candidate survives the gate (matches no
            DISQUALIFYING criterion); ``False`` when it positively matches at
            least one and is therefore excluded.
        reasons: Human-readable reasons for exclusion, one per positively
            matched DISQUALIFYING criterion. Empty when ``passed`` is ``True``.
    """

    passed: bool
    reasons: list[str] = field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and return its alphanumeric tokens (length >= 2)."""

    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _salient_tokens(text: str) -> set[str]:
    """The set of meaningful tokens of a requirement, with stop-words removed.

    An empty result means the criterion carries no salient content; such a
    criterion can never produce a positive match (fail-open).
    """

    return {tok for tok in _tokenize(text) if tok not in _STOPWORDS}


def _candidate_evidence_phrases(cand: EnrichedProfile) -> Iterator[str]:
    """Yield each concrete piece of structured/enriched evidence as raw text.

    Only fields that represent *present* evidence are yielded. Absent scalar
    fields (recorded as ``None``) and empty collections contribute nothing, so a
    sparse profile simply yields fewer (or no) phrases — it is never treated as
    matching a criterion. ``signal_availability`` is intentionally **not**
    consulted here: a tier coverage of 0 must never cause exclusion
    (Requirement 7.3).
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
    # Tier 2 inferred evidence — still concrete, candidate-specific evidence.
    yield from cand.inferred_responsibilities
    yield from cand.implicit_skills


def _evidence_token_sets(cand: EnrichedProfile) -> list[set[str]]:
    """Tokenize each evidence phrase into its own token set.

    Matching is performed *per phrase* (rather than against one merged bag of
    tokens) so that a criterion's salient tokens must all co-occur within a
    single concrete evidence item, which avoids spurious cross-field matches.
    """

    sets: list[set[str]] = []
    for phrase in _candidate_evidence_phrases(cand):
        tokens = set(_tokenize(phrase))
        if tokens:
            sets.append(tokens)
    return sets


def default_disqualifier_match(cand: EnrichedProfile, requirement: Requirement) -> bool:
    """Default deterministic predicate: does ``cand`` positively match ``requirement``?

    Pragmatic PoC rule: the candidate positively matches a DISQUALIFYING
    criterion when some concrete evidence item in the candidate's
    structured/enriched data contains *all* of the criterion's salient
    (non-stop-word) tokens. Examples:

        - criterion text "COBOL" matches a candidate whose ``explicit_skills``
          include "COBOL".
        - criterion text "PHP developer" matches a role titled "PHP Developer"
          (both salient tokens ``{php, developer}`` co-occur in that phrase).

    The rule is intentionally **fail-open**: a criterion with no salient tokens,
    or a candidate with no corresponding evidence, never produces a positive
    match. Absence or unknown data therefore never disqualifies (Requirements
    4.4, 7.3).

    This predicate is deterministic (no LLM, no randomness) and is exposed so it
    can be tested directly or overridden via the ``matcher`` argument of
    :func:`hard_filter`.
    """

    salient = _salient_tokens(requirement.text)
    if not salient:
        # No meaningful criterion content => cannot positively match.
        return False

    for evidence_tokens in _evidence_token_sets(cand):
        if salient <= evidence_tokens:
            return True
    return False


def hard_filter(
    reqs: RequirementVector,
    cand: EnrichedProfile,
    *,
    matcher: MatchPredicate = default_disqualifier_match,
) -> FilterResult:
    """Apply the deterministic hard-filter gate to a single candidate.

    Iterates the DISQUALIFYING criteria of ``reqs`` and excludes ``cand`` iff it
    positively matches at least one (per ``matcher``). A candidate matching no
    disqualifier — including a candidate with sparse data or zero signal
    availability — always passes (Requirements 4.4, 7.3).

    Args:
        reqs: The decomposed requirement vector; only its DISQUALIFYING
            requirements are consulted.
        cand: The enriched candidate profile to evaluate.
        matcher: The match predicate to use; defaults to
            :func:`default_disqualifier_match`. Override for custom or stubbed
            matching.

    Returns:
        A :class:`FilterResult`. ``passed`` is ``False`` with one reason per
        matched disqualifier when excluded; otherwise ``passed`` is ``True``
        with an empty ``reasons`` list.
    """

    reasons: list[str] = []
    for requirement in _disqualifying(reqs.requirements):
        if matcher(cand, requirement):
            reasons.append(f"disqualified: {requirement.text}")

    if reasons:
        return FilterResult(passed=False, reasons=reasons)
    return FilterResult(passed=True, reasons=[])


def _disqualifying(requirements: Iterable[Requirement]) -> Iterator[Requirement]:
    """Yield only the DISQUALIFYING requirements from ``requirements``."""

    for requirement in requirements:
        if requirement.category is RequirementCategory.DISQUALIFYING:
            yield requirement


class HardFilterMixin:
    """Mixin contributing the :meth:`hard_filter` method to a scoring engine.

    Kept separate from concrete engine state so that the dense/sparse/composite/
    rerank methods added by later tasks (10.x, 11.x, 13.x) can be composed onto
    :class:`HybridScoringEngine` without any single monolithic file needing to
    be co-edited.
    """

    def hard_filter(
        self,
        reqs: RequirementVector,
        cand: EnrichedProfile,
        *,
        matcher: MatchPredicate = default_disqualifier_match,
    ) -> FilterResult:
        """Instance-method form of :func:`hard_filter` (design ``hardFilter``)."""

        return hard_filter(reqs, cand, matcher=matcher)


class HybridScoringEngine(HardFilterMixin):
    """The ICRS hybrid scoring engine.

    Currently composes only the deterministic hard-filter gate (Task 8.1). Later
    tasks attach dense similarity, sparse BM25, composite fusion, and the LLM
    reranker — each as its own mixin/module — so this class stays a thin
    composition point rather than a file multiple tasks must edit in conflict.
    """


__all__ = [
    "FilterResult",
    "MatchPredicate",
    "default_disqualifier_match",
    "hard_filter",
    "HardFilterMixin",
    "HybridScoringEngine",
]

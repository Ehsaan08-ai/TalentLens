"""Unit tests for the Task 8.1 deterministic hard-filter gate.

Covers Requirement 4.4 (exclude only on a positive DISQUALIFYING match; retain
candidates matching no disqualifier) and Requirement 7.3 (missing data /
signal_availability == 0 must never exclude).
"""

from __future__ import annotations

from icrs.models.candidate import EnrichedProfile, NormalizedProfile, Role
from icrs.models.enums import SignalTier
from icrs.models.job import (
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)
from icrs.scoring.hard_filter import (
    FilterResult,
    HybridScoringEngine,
    default_disqualifier_match,
    hard_filter,
)


# ----- helpers --------------------------------------------------------------


def _req(text: str, category: RequirementCategory) -> Requirement:
    return Requirement(
        text=text,
        category=category,
        tier=RequirementTier.STRUCTURAL,
        weight=0.0 if category is RequirementCategory.DISQUALIFYING else 1.0,
    )


def _vector(*requirements: Requirement) -> RequirementVector:
    # A valid vector needs at least one MUST_HAVE; add one if none supplied.
    reqs = list(requirements)
    if not any(r.category is RequirementCategory.MUST_HAVE for r in reqs):
        reqs.append(_req("baseline must have", RequirementCategory.MUST_HAVE))
    return RequirementVector(
        role_intent="Build and operate the service",
        seniority_band=SeniorityBand.SENIOR,
        requirements=reqs,
    )


def _profile(
    *,
    skills: list[str] | None = None,
    certifications: list[str] | None = None,
    roles: list[Role] | None = None,
    inferred_responsibilities: list[str] | None = None,
    implicit_skills: list[str] | None = None,
    signal_availability: dict[SignalTier, float] | None = None,
) -> EnrichedProfile:
    base = NormalizedProfile(
        roles=roles or [],
        certifications=certifications or [],
        explicit_skills=skills or [],
    )
    return EnrichedProfile(
        base=base,
        inferred_responsibilities=inferred_responsibilities or [],
        implicit_skills=implicit_skills or [],
        signal_availability=signal_availability or {},
    )


# ----- Requirement 4.4: positive match excludes -----------------------------


def test_candidate_matching_disqualifier_is_excluded_with_reason() -> None:
    reqs = _vector(_req("COBOL", RequirementCategory.DISQUALIFYING))
    cand = _profile(skills=["COBOL", "Python"])

    result = hard_filter(reqs, cand)

    assert isinstance(result, FilterResult)
    assert result.passed is False
    assert result.reasons == ["disqualified: COBOL"]


def test_multiword_disqualifier_matches_role_title() -> None:
    reqs = _vector(_req("PHP developer", RequirementCategory.DISQUALIFYING))
    cand = _profile(roles=[Role(title="PHP Developer", company="Acme")])

    result = hard_filter(reqs, cand)

    assert result.passed is False
    assert "disqualified: PHP developer" in result.reasons


def test_disqualifier_matches_certification_evidence() -> None:
    reqs = _vector(_req("Notary", RequirementCategory.DISQUALIFYING))
    cand = _profile(certifications=["Certified Notary Public"])

    assert hard_filter(reqs, cand).passed is False


def test_multiple_matched_disqualifiers_each_recorded() -> None:
    reqs = _vector(
        _req("COBOL", RequirementCategory.DISQUALIFYING),
        _req("Fortran", RequirementCategory.DISQUALIFYING),
    )
    cand = _profile(skills=["COBOL", "Fortran"])

    result = hard_filter(reqs, cand)

    assert result.passed is False
    assert len(result.reasons) == 2
    assert "disqualified: COBOL" in result.reasons
    assert "disqualified: Fortran" in result.reasons


# ----- Requirement 4.4: no match retains ------------------------------------


def test_candidate_matching_no_disqualifier_passes() -> None:
    reqs = _vector(_req("COBOL", RequirementCategory.DISQUALIFYING))
    cand = _profile(skills=["Python", "Go"], roles=[Role(title="Engineer", company="Acme")])

    result = hard_filter(reqs, cand)

    assert result.passed is True
    assert result.reasons == []


def test_vector_with_no_disqualifiers_passes_everyone() -> None:
    reqs = _vector(
        _req("Python", RequirementCategory.MUST_HAVE),
        _req("Kubernetes", RequirementCategory.NICE_TO_HAVE),
    )
    # Even a candidate whose skills overlap the must/nice requirements passes:
    # only DISQUALIFYING criteria gate.
    cand = _profile(skills=["Python", "Kubernetes"])

    result = hard_filter(reqs, cand)

    assert result.passed is True
    assert result.reasons == []


def test_partial_token_overlap_does_not_match() -> None:
    # "developer" alone must not trigger a "PHP developer" disqualifier: all
    # salient tokens must co-occur in one evidence item.
    reqs = _vector(_req("PHP developer", RequirementCategory.DISQUALIFYING))
    cand = _profile(roles=[Role(title="Python Developer", company="Acme")])

    assert hard_filter(reqs, cand).passed is True


# ----- Requirement 7.3: missing data / zero availability never excludes ------


def test_empty_profile_is_never_excluded() -> None:
    reqs = _vector(_req("COBOL", RequirementCategory.DISQUALIFYING))
    cand = _profile()  # no skills, roles, certs, inferred fields

    result = hard_filter(reqs, cand)

    assert result.passed is True
    assert result.reasons == []


def test_zero_signal_availability_does_not_exclude() -> None:
    reqs = _vector(_req("COBOL", RequirementCategory.DISQUALIFYING))
    cand = _profile(
        skills=["Python"],
        signal_availability={
            SignalTier.STRUCTURAL: 0.0,
            SignalTier.SEMANTIC: 0.0,
            SignalTier.BEHAVIORAL: 0.0,
        },
    )

    result = hard_filter(reqs, cand)

    assert result.passed is True
    assert result.reasons == []


def test_criterion_with_only_stopwords_never_matches() -> None:
    # A criterion that reduces to no salient tokens cannot positively match.
    reqs = _vector(_req("must have experience", RequirementCategory.DISQUALIFYING))
    cand = _profile(skills=["experience"], inferred_responsibilities=["has experience"])

    assert hard_filter(reqs, cand).passed is True


# ----- predicate + engine surface -------------------------------------------


def test_default_predicate_directly() -> None:
    cand = _profile(skills=["COBOL"])
    dq = _req("COBOL", RequirementCategory.DISQUALIFYING)
    other = _req("Rust", RequirementCategory.DISQUALIFYING)

    assert default_disqualifier_match(cand, dq) is True
    assert default_disqualifier_match(cand, other) is False


def test_custom_matcher_override() -> None:
    reqs = _vector(_req("anything", RequirementCategory.DISQUALIFYING))
    cand = _profile(skills=["Python"])

    # A matcher that always matches excludes the candidate regardless of data.
    always = hard_filter(reqs, cand, matcher=lambda c, r: True)
    assert always.passed is False

    # A matcher that never matches retains the candidate.
    never = hard_filter(reqs, cand, matcher=lambda c, r: False)
    assert never.passed is True


def test_engine_method_matches_function() -> None:
    engine = HybridScoringEngine()
    reqs = _vector(_req("COBOL", RequirementCategory.DISQUALIFYING))
    cand = _profile(skills=["COBOL"])

    via_engine = engine.hard_filter(reqs, cand)
    via_func = hard_filter(reqs, cand)

    assert via_engine.passed == via_func.passed is False
    assert via_engine.reasons == via_func.reasons


def test_only_disqualifying_category_gates() -> None:
    # Same token as a MUST_HAVE / NICE_TO_HAVE must not gate; only DISQUALIFYING.
    reqs = _vector(
        _req("Python", RequirementCategory.MUST_HAVE),
        _req("Java", RequirementCategory.NICE_TO_HAVE),
        _req("COBOL", RequirementCategory.DISQUALIFYING),
    )
    cand = _profile(skills=["Python", "Java"])

    assert hard_filter(reqs, cand).passed is True

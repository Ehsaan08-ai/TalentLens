"""Unit tests for profile normalization (Task 4.1).

Covers canonicalization of heterogeneous raw profiles into ``NormalizedProfile``
(including tenure-in-months computation from role dates), the
"absent field is not-present (None)" rule, alias tolerance, and empty/invalid
profile rejection via ``ProfileValidationError`` (Requirements 3.1, 3.2, 3.6).
"""

from __future__ import annotations

from datetime import date

import pytest

from icrs.models.candidate import NormalizedProfile, RawCandidate
from icrs.pipeline.enricher import CandidateEnricher
from icrs.pipeline.normalization import ProfileValidationError, _months_between


@pytest.fixture()
def enricher() -> CandidateEnricher:
    return CandidateEnricher()


# --------------------------------------------------------------------------- #
# Canonicalization (Requirement 3.1)
# --------------------------------------------------------------------------- #


def test_normalize_canonical_fields(enricher: CandidateEnricher):
    raw = RawCandidate(
        structured_fields={
            "roles": [
                {
                    "title": "Software Engineer",
                    "company": "Acme",
                    "start": "2020-01-01",
                    "end": "2022-01-01",
                }
            ],
            "education": [
                {"institution": "MIT", "degree": "BSc", "field_of_study": "CS"}
            ],
            "certifications": ["AWS SA", "CKA"],
            "skills": ["Python", "Go"],
        }
    )
    profile = enricher.normalize(raw)

    assert isinstance(profile, NormalizedProfile)
    assert profile.id == raw.id
    assert len(profile.roles) == 1
    assert profile.roles[0].title == "Software Engineer"
    assert profile.roles[0].company == "Acme"
    assert profile.education[0].institution == "MIT"
    assert profile.certifications == ["AWS SA", "CKA"]
    assert profile.explicit_skills == ["Python", "Go"]
    # 2020-01 .. 2022-01 == 24 whole months.
    assert profile.total_tenure_months == 24


def test_normalize_tolerates_alias_key_names(enricher: CandidateEnricher):
    raw = RawCandidate(
        structured_fields={
            "work_experience": [
                {
                    "position": "Lead Dev",
                    "employer": "Globex",
                    "from": "2019-03",
                    "to": "2020-03",
                }
            ],
            "schools": {"university": "Stanford", "qualification": "MSc"},
            "certs": "PMP; Scrum Master",
            "technologies": "java|kotlin|sql",
        }
    )
    profile = enricher.normalize(raw)

    assert profile.roles[0].title == "Lead Dev"
    assert profile.roles[0].company == "Globex"
    assert profile.education[0].institution == "Stanford"
    assert profile.education[0].degree == "MSc"
    assert profile.certifications == ["PMP", "Scrum Master"]
    assert profile.explicit_skills == ["java", "kotlin", "sql"]
    assert profile.total_tenure_months == 12


# --------------------------------------------------------------------------- #
# Tenure-in-months computation (Requirement 3.1)
# --------------------------------------------------------------------------- #


def test_tenure_sums_multiple_roles(enricher: CandidateEnricher):
    raw = RawCandidate(
        structured_fields={
            "roles": [
                {"title": "Dev", "company": "A", "start": "2018-01", "end": "2019-01"},
                {"title": "SDE", "company": "B", "start": "2019-01", "end": "2021-07"},
            ]
        }
    )
    profile = enricher.normalize(raw)
    # 12 + 30 == 42 months.
    assert profile.total_tenure_months == 42


def test_tenure_ongoing_role_measured_to_today(monkeypatch, enricher: CandidateEnricher):
    monkeypatch.setattr(CandidateEnricher, "_today", staticmethod(lambda: date(2023, 1, 1)))
    raw = RawCandidate(
        structured_fields={
            "roles": [{"title": "Dev", "company": "A", "start": "2021-01", "end": "present"}]
        }
    )
    profile = enricher.normalize(raw)
    assert profile.total_tenure_months == 24


def test_tenure_zero_when_no_dates(enricher: CandidateEnricher):
    raw = RawCandidate(
        structured_fields={"roles": [{"title": "Dev", "company": "A"}]}
    )
    profile = enricher.normalize(raw)
    assert profile.total_tenure_months == 0


def test_months_between_partial_month_day_adjustment():
    # 15th -> 10th: final month not completed.
    assert _months_between(date(2020, 1, 15), date(2020, 7, 10)) == 5
    assert _months_between(date(2020, 1, 1), date(2020, 7, 1)) == 6
    # Reversed interval clamps to 0.
    assert _months_between(date(2021, 1, 1), date(2020, 1, 1)) == 0


# --------------------------------------------------------------------------- #
# Not-present marking (Requirement 3.2)
# --------------------------------------------------------------------------- #


def test_absent_role_dates_are_none(enricher: CandidateEnricher):
    raw = RawCandidate(structured_fields={"roles": [{"title": "Dev", "company": "A"}]})
    profile = enricher.normalize(raw)
    assert profile.roles[0].start is None
    assert profile.roles[0].end is None


def test_ongoing_end_token_is_none(enricher: CandidateEnricher):
    raw = RawCandidate(
        structured_fields={
            "roles": [{"title": "Dev", "company": "A", "start": "2021-01", "end": "current"}]
        }
    )
    profile = enricher.normalize(raw)
    assert profile.roles[0].end is None


def test_unparseable_date_treated_as_not_present(enricher: CandidateEnricher):
    raw = RawCandidate(
        structured_fields={
            "roles": [{"title": "Dev", "company": "A", "start": "sometime in spring"}]
        }
    )
    profile = enricher.normalize(raw)
    assert profile.roles[0].start is None


def test_role_missing_company_is_skipped(enricher: CandidateEnricher):
    raw = RawCandidate(
        structured_fields={
            "roles": [
                {"title": "Dev"},  # no company -> skipped
                {"title": "SDE", "company": "B"},
            ]
        }
    )
    profile = enricher.normalize(raw)
    assert len(profile.roles) == 1
    assert profile.roles[0].title == "SDE"


# --------------------------------------------------------------------------- #
# Free-text-only profile is usable (Requirement 3.1/3.6)
# --------------------------------------------------------------------------- #


def test_free_text_only_profile_normalizes_to_empty_canonical(enricher: CandidateEnricher):
    raw = RawCandidate(free_text="Seasoned engineer who led platform migrations.")
    profile = enricher.normalize(raw)
    assert profile.roles == []
    assert profile.education == []
    assert profile.certifications == []
    assert profile.explicit_skills == []
    assert profile.total_tenure_months == 0


# --------------------------------------------------------------------------- #
# Empty / invalid rejection (Requirement 3.6)
# --------------------------------------------------------------------------- #


def test_empty_profile_rejected(enricher: CandidateEnricher):
    with pytest.raises(ProfileValidationError):
        enricher.normalize(RawCandidate())


def test_whitespace_free_text_and_no_structured_rejected(enricher: CandidateEnricher):
    with pytest.raises(ProfileValidationError):
        enricher.normalize(RawCandidate(free_text="   \n\t  "))


def test_structured_junk_with_no_usable_data_rejected(enricher: CandidateEnricher):
    # Unrecognized keys, no free text -> nothing usable.
    with pytest.raises(ProfileValidationError):
        enricher.normalize(RawCandidate(structured_fields={"unknown_field": 123}))


def test_schema_invalid_roles_type_rejected(enricher: CandidateEnricher):
    with pytest.raises(ProfileValidationError):
        enricher.normalize(RawCandidate(structured_fields={"roles": 42}))


def test_schema_invalid_role_entry_type_rejected(enricher: CandidateEnricher):
    with pytest.raises(ProfileValidationError):
        enricher.normalize(
            RawCandidate(structured_fields={"roles": ["not-a-dict"]})
        )


def test_no_canonical_profile_returned_on_rejection(enricher: CandidateEnricher):
    # The error path produces no NormalizedProfile (Requirement 3.6).
    try:
        enricher.normalize(RawCandidate())
    except ProfileValidationError as exc:
        assert "empty" in str(exc).lower()
    else:  # pragma: no cover
        pytest.fail("expected ProfileValidationError")

"""Deterministic profile normalization for ICRS (Task 4.1).

This module canonicalizes a heterogeneous :class:`RawCandidate` into the
canonical :class:`NormalizedProfile` schema (roles, education, certifications,
explicit skills, and total tenure expressed in *whole months*) — Requirement
3.1 — and rejects empty or schema-invalid profiles with a typed
:class:`ProfileValidationError`, producing no canonical profile — Requirement
3.6.

Design notes
------------
* Normalization is **purely deterministic parsing** — there are no LLM calls
  here. Tier 2 (semantic) and Tier 3 (behavioral) inference live in the
  enrichment step (Task 4.2).
* Source profiles are heterogeneous, so we tolerate common alias key names for
  titles, companies, dates, skills, certifications, and education (pragmatic for
  a PoC rather than exhaustive).
* The "absent field is not-present" rule (Requirement 3.2) is honoured: a scalar
  field we cannot find (a role's start/end date, an education institution, ...)
  is left as ``None`` rather than being defaulted to a sentinel.

The logic is exposed as :class:`ProfileNormalizationMixin` so the
``CandidateEnricher`` (``icrs.pipeline.enricher``) can mix it in alongside the
enrichment behaviour added later, keeping the two concerns in separate modules.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

from icrs.models.candidate import Education, NormalizedProfile, RawCandidate, Role


class ProfileValidationError(ValueError):
    """Raised when a raw candidate profile is empty or schema-invalid.

    When this is raised the normalizer produces *no* canonical profile
    (Requirement 3.6). It subclasses ``ValueError`` so callers that catch the
    broad validation category still see it, while callers that want to handle
    profile rejection specifically can catch this type.
    """


# --------------------------------------------------------------------------- #
# Alias tables — tolerate heterogeneous source key names (pragmatic, not total).
# --------------------------------------------------------------------------- #

# Top-level container fields inside RawCandidate.structured_fields.
_ROLES_KEYS = (
    "roles",
    "experience",
    "experiences",
    "work_experience",
    "work_history",
    "employment",
    "employment_history",
    "positions",
    "jobs",
)
_EDUCATION_KEYS = (
    "education",
    "educations",
    "schools",
    "academics",
    "academic_history",
    "qualifications",
)
_CERTIFICATION_KEYS = (
    "certifications",
    "certificates",
    "certs",
    "licenses",
    "licences",
    "accreditations",
)
_SKILL_KEYS = (
    "explicit_skills",
    "skills",
    "skill_set",
    "skillset",
    "technologies",
    "tech_stack",
    "competencies",
)

# Per-role field aliases.
_ROLE_TITLE_KEYS = ("title", "role", "position", "job_title", "designation", "jobtitle")
_ROLE_COMPANY_KEYS = (
    "company",
    "employer",
    "organization",
    "organisation",
    "org",
    "company_name",
    "firm",
)
_START_KEYS = ("start", "start_date", "from", "begin", "began", "started", "start_year")
_END_KEYS = ("end", "end_date", "to", "until", "ended", "finish", "end_year")

# Per-education field aliases.
_EDU_INSTITUTION_KEYS = (
    "institution",
    "school",
    "university",
    "college",
    "institute",
)
_EDU_DEGREE_KEYS = ("degree", "qualification", "diploma", "credential")
_EDU_FIELD_KEYS = (
    "field_of_study",
    "field",
    "major",
    "area_of_study",
    "specialization",
    "specialisation",
    "subject",
)

# Strings that denote an ongoing / not-yet-ended period -> not-present (None).
_ONGOING_TOKENS = frozenset(
    {
        "",
        "present",
        "current",
        "currently",
        "now",
        "ongoing",
        "to date",
        "till date",
        "till now",
        "n/a",
        "na",
        "none",
        "-",
    }
)

# Date formats attempted in order. ISO-like forms are preferred; ambiguous
# day/month orderings are intentionally limited to avoid silent misreads.
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%Y-%m",
    "%Y/%m",
    "%m-%Y",
    "%m/%Y",
    "%b %Y",
    "%B %Y",
    "%b %d %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%B %d, %Y",
    "%Y",
)


# --------------------------------------------------------------------------- #
# Small parsing helpers
# --------------------------------------------------------------------------- #


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first value whose (case-insensitive) key is present and non-null.

    Returns ``None`` when no alias key is found, so absence flows through as
    not-present rather than as a defaulted value (Requirement 3.2).
    """

    # Build a case-insensitive view once per lookup; profiles are small.
    lowered = {str(k).strip().lower(): v for k, v in mapping.items()}
    for key in keys:
        value = lowered.get(key)
        if value is not None:
            return value
    return None


def _clean_str(value: Any) -> str | None:
    """Coerce ``value`` to a trimmed non-empty string, or ``None`` if not usable."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_date(value: Any) -> date | None:
    """Parse a heterogeneous date value into a :class:`date`, or ``None``.

    Returns ``None`` for absent values and for ongoing-period tokens such as
    "present"/"current". Unparseable scalar values are treated as not-present
    (``None``) rather than raising, so a single malformed date does not discard
    an otherwise-usable profile; structural type errors are handled by the
    caller.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, int):
        # Bare year as an integer (e.g. 2020).
        if 1900 <= value <= 2100:
            return date(value, 1, 1)
        return None

    text = str(value).strip()
    if text.lower() in _ONGOING_TOKENS:
        return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Unrecognized format -> not-present.
    return None


def _normalize_str_list(value: Any) -> list[str]:
    """Normalize a skills/certifications value into a de-duplicated string list.

    Accepts either an actual list/tuple/set of items or a single delimited
    string (comma / semicolon / pipe / newline separated). Blank entries are
    dropped and order is preserved while removing case-insensitive duplicates.
    """

    if value is None:
        return []

    items: list[Any]
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, str):
        items = _split_delimited(value)
    else:
        # A scalar non-string (e.g. a number) — stringify as a single item.
        items = [value]

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_str(item)
        if cleaned is None:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _split_delimited(text: str) -> list[str]:
    """Split a delimited string on commas, semicolons, pipes, or newlines."""

    normalized = text
    for sep in (";", "|", "\n", "\r", "\t", "/"):
        normalized = normalized.replace(sep, ",")
    return [part for part in normalized.split(",")]


def _as_entry_list(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    """Coerce a roles/education container into a list of dict entries.

    A single dict is wrapped in a one-element list. A list/tuple of dicts is
    accepted as-is (non-dict elements raise). Any other type for a recognized
    container field is a schema error.
    """

    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        entries: list[dict[str, Any]] = []
        for element in value:
            if not isinstance(element, dict):
                raise ProfileValidationError(
                    f"schema-invalid profile: each {field_name} entry must be a "
                    f"mapping, got {type(element).__name__}"
                )
            entries.append(element)
        return entries
    raise ProfileValidationError(
        f"schema-invalid profile: {field_name} must be a list or mapping, "
        f"got {type(value).__name__}"
    )


def _months_between(start: date, end: date) -> int:
    """Return the number of whole months between ``start`` and ``end`` (>= 0).

    Counts completed months: the day-of-month is used to decide whether the
    final partial month has elapsed. A negative interval (end before start)
    clamps to 0.
    """

    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


# --------------------------------------------------------------------------- #
# Normalization mixin
# --------------------------------------------------------------------------- #


class ProfileNormalizationMixin:
    """Provides deterministic ``normalize`` for the candidate enricher.

    Kept as a mixin in its own module so the enrichment behaviour (Task 4.2) can
    be composed onto ``CandidateEnricher`` without either concern editing the
    other's source.
    """

    def normalize(self, raw_profile: RawCandidate) -> NormalizedProfile:
        """Canonicalize a heterogeneous raw profile into a NormalizedProfile.

        Args:
            raw_profile: the as-submitted candidate profile.

        Returns:
            A :class:`NormalizedProfile` with canonical roles, education,
            certifications, explicit skills, and ``total_tenure_months`` summed
            from role start/end dates and expressed in whole months.

        Raises:
            ProfileValidationError: if the profile is empty (no structured
                fields, no free text, no usable data) or schema-invalid. No
                canonical profile is produced in that case (Requirement 3.6).
        """

        structured = raw_profile.structured_fields or {}
        free_text = (raw_profile.free_text or "").strip()

        # Fast path: a profile with neither structured fields nor free text is
        # empty and has no usable data (Requirement 3.6).
        if not structured and not free_text:
            raise ProfileValidationError(
                "empty profile: no structured fields and no free text"
            )

        if not isinstance(structured, dict):
            raise ProfileValidationError(
                "schema-invalid profile: structured_fields must be a mapping"
            )

        roles = self._parse_roles(structured)
        education = self._parse_education(structured)
        certifications = _normalize_str_list(
            _first_present(structured, _CERTIFICATION_KEYS)
        )
        explicit_skills = _normalize_str_list(_first_present(structured, _SKILL_KEYS))
        total_tenure_months = self._compute_total_tenure_months(roles)

        # A profile whose structured fields yielded nothing usable and that has
        # no free text to fall back on is effectively empty (Requirement 3.6).
        has_structured_data = bool(
            roles or education or certifications or explicit_skills
        )
        if not has_structured_data and not free_text:
            raise ProfileValidationError(
                "schema-invalid profile: no usable structured data and no free text"
            )

        return NormalizedProfile(
            id=raw_profile.id,
            roles=roles,
            education=education,
            certifications=certifications,
            explicit_skills=explicit_skills,
            total_tenure_months=total_tenure_months,
        )

    # ------------------------------------------------------------------ #
    # Internal parsing steps
    # ------------------------------------------------------------------ #

    def _parse_roles(self, structured: dict[str, Any]) -> list[Role]:
        """Build canonical :class:`Role` objects from the roles container.

        A role is usable only when both an identifying title and company can be
        resolved (the model requires both); entries missing either are skipped
        rather than fabricated. Start/end dates are parsed tolerantly and left
        as ``None`` (not-present) when absent or unparseable.
        """

        entries = _as_entry_list(
            _first_present(structured, _ROLES_KEYS), field_name="role"
        )
        roles: list[Role] = []
        for entry in entries:
            title = _clean_str(_first_present(entry, _ROLE_TITLE_KEYS))
            company = _clean_str(_first_present(entry, _ROLE_COMPANY_KEYS))
            if title is None or company is None:
                # Not enough to form a canonical role; skip without fabricating.
                continue
            roles.append(
                Role(
                    title=title,
                    company=company,
                    start=_parse_date(_first_present(entry, _START_KEYS)),
                    end=_parse_date(_first_present(entry, _END_KEYS)),
                )
            )
        return roles

    def _parse_education(self, structured: dict[str, Any]) -> list[Education]:
        """Build canonical :class:`Education` records from the education container.

        Every education field is optional; an entry contributes only when at
        least one field resolves. Absent fields are recorded as ``None``.
        """

        entries = _as_entry_list(
            _first_present(structured, _EDUCATION_KEYS), field_name="education"
        )
        education: list[Education] = []
        for entry in entries:
            institution = _clean_str(_first_present(entry, _EDU_INSTITUTION_KEYS))
            degree = _clean_str(_first_present(entry, _EDU_DEGREE_KEYS))
            field_of_study = _clean_str(_first_present(entry, _EDU_FIELD_KEYS))
            start = _parse_date(_first_present(entry, _START_KEYS))
            end = _parse_date(_first_present(entry, _END_KEYS))

            if not any((institution, degree, field_of_study, start, end)):
                # An empty / unrecognized education entry contributes nothing.
                continue
            education.append(
                Education(
                    institution=institution,
                    degree=degree,
                    field_of_study=field_of_study,
                    start=start,
                    end=end,
                )
            )
        return education

    def _compute_total_tenure_months(self, roles: list[Role]) -> int:
        """Sum each role's duration in whole months (Requirement 3.1).

        Only roles with a known start date contribute. An absent end date is
        treated as an ongoing role and measured to today. Durations are summed
        across roles (overlaps are not de-duplicated — pragmatic for the PoC).
        """

        today = self._today()
        total = 0
        for role in roles:
            if role.start is None:
                continue
            end = role.end if role.end is not None else today
            total += _months_between(role.start, end)
        return total

    @staticmethod
    def _today() -> date:
        """Return today's date (isolated for deterministic testing/overriding)."""

        return date.today()


__all__ = ["ProfileNormalizationMixin", "ProfileValidationError"]

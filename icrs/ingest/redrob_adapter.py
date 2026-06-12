"""Adapter: Redrob Candidate Profile Schema -> ICRS RawCandidate input.

The challenge dataset (``data/India_runs_data_and_ai_challenge``) follows the
*Redrob Candidate Profile Schema*: each candidate is an object with
``candidate_id``, ``profile``, ``career_history``, ``education``, ``skills``,
``certifications``, ``languages`` and ``redrob_signals``. ICRS, however, ingests
the generic :class:`~icrs.models.candidate.RawCandidate` shape — ``structured_fields``,
``free_text`` and ``external_handles`` — which the deterministic normalizer
(Task 4.1) then canonicalizes. This module bridges the two so a recruiter can
rank the Redrob dataset without any change to the pipeline.

Two mismatches this adapter resolves (the cause of the dashboard's
"Candidate pool must be a JSON array" / "unsupported field(s)" errors):

1. **Container shape.** ``candidates.jsonl`` is JSON Lines (one object per
   line), which is *not* a JSON array; :func:`load_redrob_records` reads both
   ``.jsonl`` and a top-level ``[...]`` ``.json`` array (and tolerates a single
   object or a ``{"candidates": [...]}`` wrapper).
2. **Field schema.** :func:`redrob_to_raw_candidate` maps the Redrob fields onto
   ``structured_fields`` keys the normalizer understands (``roles``,
   ``education``, ``skills``, ``certifications``) plus a ``free_text`` summary.

Fairness (design "Avoiding demographic / proxy bias", Requirement 7.1):
    Protected-proxy source fields — ``anonymized_name``, ``location``,
    ``country`` (and the salary expectation) — are **deliberately not mapped**
    into the scoring input. Only job-relevant signals are carried across. The
    scoring layer additionally strips protected proxies, so this is defense in
    depth, not the sole guard.

Behavioral signals note:
    The rich ``redrob_signals`` block (GitHub activity, endorsements, recruiter
    response, …) is Tier-3 behavioral data. The default ingestion path uses the
    no-op behavioral source, so these are not consumed unless a dedicated
    :class:`~icrs.pipeline.enrichment.BehavioralSignalSource` is wired in. This
    adapter therefore focuses on the structural + free-text signals; mapping the
    behavioral block is a documented follow-up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

# Source fields that are protected/demographic proxies and must never be mapped
# into the scoring input (Requirement 7.1). Listed for transparency/auditing.
PROTECTED_PROXY_SOURCE_FIELDS: frozenset[str] = frozenset(
    {
        "anonymized_name",
        "location",
        "country",
        "expected_salary_range_inr_lpa",
    }
)


def _role_from_history(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Map one ``career_history`` entry to a normalizer-friendly role dict.

    Returns ``None`` when the entry lacks both a title and a company (the
    canonical role model requires both), so an unusable entry is skipped rather
    than fabricated.
    """

    title = (entry.get("title") or "").strip()
    company = (entry.get("company") or "").strip()
    if not title or not company:
        return None
    role: dict[str, Any] = {"title": title, "company": company}
    # start_date / end_date are ISO dates; end_date may be null (ongoing role).
    if entry.get("start_date"):
        role["start"] = entry["start_date"]
    if entry.get("end_date"):
        role["end"] = entry["end_date"]
    return role


def _education_from_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Map one ``education`` entry to a normalizer-friendly education dict."""

    institution = (entry.get("institution") or "").strip()
    degree = (entry.get("degree") or "").strip()
    field = (entry.get("field_of_study") or "").strip()
    if not any((institution, degree, field)):
        return None
    edu: dict[str, Any] = {}
    if institution:
        edu["institution"] = institution
    if degree:
        edu["degree"] = degree
    if field:
        edu["field_of_study"] = field
    # start_year / end_year are integers; the normalizer parses a bare year.
    if entry.get("start_year"):
        edu["start"] = entry["start_year"]
    if entry.get("end_year"):
        edu["end"] = entry["end_year"]
    return edu


def _skill_names(skills: Iterable[Any]) -> list[str]:
    """Flatten Redrob skill objects ({name, proficiency, ...}) to a name list."""

    names: list[str] = []
    for skill in skills or []:
        if isinstance(skill, dict):
            name = (skill.get("name") or "").strip()
        else:
            name = str(skill).strip()
        if name:
            names.append(name)
    return names


def _certification_names(certs: Iterable[Any]) -> list[str]:
    """Flatten Redrob certification objects to a name list."""

    names: list[str] = []
    for cert in certs or []:
        if isinstance(cert, dict):
            name = (cert.get("name") or "").strip()
        else:
            name = str(cert).strip()
        if name:
            names.append(name)
    return names


def _build_free_text(record: dict[str, Any]) -> str:
    """Assemble the candidate's free-text prose from job-relevant fields only.

    Combines the headline, professional summary, and each role's description.
    Excludes every protected-proxy field (name, location, country).
    """

    profile = record.get("profile") or {}
    parts: list[str] = []

    headline = (profile.get("headline") or "").strip()
    if headline:
        parts.append(headline)

    summary = (profile.get("summary") or "").strip()
    if summary:
        parts.append(summary)

    for entry in record.get("career_history") or []:
        description = (entry.get("description") or "").strip()
        if description:
            title = (entry.get("title") or "").strip()
            company = (entry.get("company") or "").strip()
            prefix = " / ".join(p for p in (title, company) if p)
            parts.append(f"{prefix}: {description}" if prefix else description)

    return "\n\n".join(parts)


def redrob_to_raw_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one Redrob candidate record into a RawCandidate-shaped dict.

    The result has exactly the three keys the ICRS API / UI accept:
    ``structured_fields`` (roles, education, skills, certifications, plus a few
    job-relevant scalars), ``free_text`` (headline + summary + role
    descriptions), and ``external_handles`` (empty — the dataset carries
    activity *scores*, not handles).

    Protected-proxy fields (name, location, country, salary expectation) are not
    included (Requirement 7.1).

    Args:
        record: a single Redrob candidate object.

    Returns:
        A dict ``{structured_fields, free_text, external_handles}`` ready to be
        submitted to ``POST /rank`` or pasted into the dashboard pool.
    """

    profile = record.get("profile") or {}

    roles = [
        role
        for entry in (record.get("career_history") or [])
        if (role := _role_from_history(entry)) is not None
    ]
    education = [
        edu
        for entry in (record.get("education") or [])
        if (edu := _education_from_entry(entry)) is not None
    ]

    structured_fields: dict[str, Any] = {
        "roles": roles,
        "education": education,
        "skills": _skill_names(record.get("skills")),
        "certifications": _certification_names(record.get("certifications")),
    }

    # A few job-relevant scalars (no demographic proxies). These are ignored by
    # the normalizer's role/education/skill parsing but kept for traceability and
    # any future structural-signal use.
    current_industry = (profile.get("current_industry") or "").strip()
    if current_industry:
        structured_fields["industry"] = current_industry
    if profile.get("years_of_experience") is not None:
        structured_fields["years_of_experience"] = profile["years_of_experience"]
    # Preserve the source id inside structured_fields for traceability (the
    # RawCandidate's own id is a generated UUID and cannot hold "CAND_xxxxxxx").
    source_id = record.get("candidate_id")
    if source_id:
        structured_fields["source_candidate_id"] = source_id

    return {
        "structured_fields": structured_fields,
        "free_text": _build_free_text(record),
        "external_handles": {},
    }


def convert_pool(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert an iterable of Redrob records into a RawCandidate-shaped list.

    The returned list is exactly the JSON array the dashboard / API expect as
    the candidate pool.
    """

    return [redrob_to_raw_candidate(record) for record in records]


def load_redrob_records(
    path: str | Path, *, limit: int | None = None
) -> list[dict[str, Any]]:
    """Load Redrob candidate records from a ``.json`` array or ``.jsonl`` file.

    Accepts:
        * JSON Lines (``.jsonl``) — one candidate object per line;
        * a top-level JSON array ``[ {...}, ... ]``;
        * a single candidate object ``{...}``;
        * a ``{"candidates": [...]}`` wrapper.

    Args:
        path: path to the dataset file.
        limit: optional cap on the number of records returned (useful for a
            quick PoC run against a large ``.jsonl`` file).

    Returns:
        A list of raw Redrob record dicts (not yet converted).

    Raises:
        ValueError: if the file content is neither a JSON array/object nor JSONL.
    """

    p = Path(path)
    text = p.read_text(encoding="utf-8")

    records: list[dict[str, Any]]
    if p.suffix.lower() == ".jsonl":
        records = list(_iter_jsonl(text))
    else:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
            records = parsed["candidates"]
        elif isinstance(parsed, dict):
            records = [parsed]
        else:
            raise ValueError(
                "Unsupported dataset shape: expected a JSON array, a single "
                "object, or a {'candidates': [...]} wrapper."
            )

    if limit is not None:
        records = records[:limit]
    return records


def _iter_jsonl(text: str) -> Iterator[dict[str, Any]]:
    """Yield one parsed object per non-blank line of JSONL ``text``."""

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {lineno}: {exc.msg}") from exc


def _main(argv: list[str] | None = None) -> int:
    """CLI: convert a Redrob dataset file into a RawCandidate pool JSON array.

    Usage::

        python -m icrs.ingest.redrob_adapter INPUT OUTPUT [--limit N]

    Writes a JSON array of ``{structured_fields, free_text, external_handles}``
    objects to OUTPUT, ready to upload/paste into the dashboard or POST to
    ``/rank``.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert a Redrob candidate dataset (.json array or .jsonl) into an "
            "ICRS RawCandidate pool JSON array."
        )
    )
    parser.add_argument("input", help="Path to the Redrob .json or .jsonl file.")
    parser.add_argument("output", help="Path to write the RawCandidate JSON array.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of candidates to convert.",
    )
    args = parser.parse_args(argv)

    records = load_redrob_records(args.input, limit=args.limit)
    pool = convert_pool(records)
    Path(args.output).write_text(json.dumps(pool, indent=2), encoding="utf-8")
    print(f"Converted {len(pool)} candidate(s) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

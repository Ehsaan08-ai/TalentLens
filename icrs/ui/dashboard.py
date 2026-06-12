from __future__ import annotations

import json
from typing import Any, Callable


# Default backend base URL for the FastAPI ranking service. Configurable in the
# UI; the network call (rank_via_api) appends the /rank path.
DEFAULT_BACKEND_URL = "http://localhost:8000"

# The valid JobType enum values, surfaced as the job-type selectbox options.
# Sourced from icrs.models.job.JobType but listed explicitly so the pure
# helpers and tests do not require importing the (heavier) model module.
JOB_TYPE_VALUES: tuple[str, ...] = ("TECHNICAL", "LEADERSHIP", "GENERALIST", "SALES")

# The five per-signal sub-scores reported in each result's breakdown, in the
# order they are displayed. The first four are positive-contribution signals;
# disqualifying_penalty is a subtractive soft red-flag magnitude.
BREAKDOWN_FIELDS: tuple[str, ...] = (
    "semantic_fit",
    "career_trajectory",
    "behavioral",
    "hard_filter_pass",
    "disqualifying_penalty",
)

# Human-readable labels for the breakdown sub-scores.
BREAKDOWN_LABELS: dict[str, str] = {
    "semantic_fit": "Semantic fit",
    "career_trajectory": "Career trajectory",
    "behavioral": "Behavioral",
    "hard_filter_pass": "Hard-filter (must-have) coverage",
    "disqualifying_penalty": "Disqualifying penalty",
}

# Confidence band thresholds (inclusive lower bounds) used to attach an honest
# qualitative label to a [0,1] confidence value without implying false
# precision. A value >= HIGH is "High", >= MEDIUM is "Moderate", else "Low".
CONFIDENCE_HIGH_THRESHOLD = 0.66
CONFIDENCE_MEDIUM_THRESHOLD = 0.33


class CandidatePoolError(ValueError):
    """Raised when an uploaded/pasted candidate pool is not valid input.

    The message is recruiter-readable so the UI can surface it directly.
    """


def extract_text_from_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    
    if filename.endswith(".pdf"):
        try:
            import io
            import pypdf
            f = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error reading PDF file: {e}"
            
    elif filename.endswith(".docx"):
        try:
            import io
            import docx
            f = io.BytesIO(file_bytes)
            doc = docx.Document(f)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error reading DOCX file: {e}"
            
    else:
        # Default to UTF-8 text decoding
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error reading text file: {e}"


# --------------------------------------------------------------------------- #
# Pure helpers — payload building
# --------------------------------------------------------------------------- #
def _looks_like_csv(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("[") or stripped.startswith("{"):
        return False
    return "," in stripped or "\t" in stripped or ";" in stripped


def _looks_like_jsonl(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1 and lines[0].startswith("{") and lines[0].endswith("}"):
        return True
    return False


def _parse_json_text(text: str) -> list[dict[str, Any]]:
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    raise ValueError("Candidate pool must be a JSON array (a list of candidate objects).")


def _parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    records = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {lineno}: {exc.msg}") from exc
    return records


def _parse_csv_text(text: str) -> list[dict[str, Any]]:
    import csv
    import io

    # Auto-detect delimiter
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = ","
    if "\t" in first_line and first_line.count("\t") > first_line.count(","):
        delimiter = "\t"
    elif ";" in first_line and first_line.count(";") > first_line.count(","):
        delimiter = ";"

    f = io.StringIO(text)
    reader = csv.DictReader(f, delimiter=delimiter)
    records = []
    for row in reader:
        if not any(row.values()):
            continue
        clean_row = {str(k).strip(): (str(v).strip() if v is not None else "") for k, v in row.items() if k is not None}
        records.append(clean_row)
    return records


def _generic_dict_to_raw_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Convert an arbitrary flat key/value dictionary to RawCandidate shape.

    Preserves fairness by filtering out demographic and protected proxy fields
    (Requirement 7.1).
    """
    protected_proxies = {
        "anonymized_name", "location", "country", "expected_salary_range_inr_lpa",
        "name", "email", "gender", "race", "age", "salary", "expected_salary",
        "address", "phone", "candidate_name", "first_name", "last_name",
        "ethnicity", "nationality", "marital_status", "expected_salary_range"
    }

    structured_fields: dict[str, Any] = {}
    free_text_parts: list[str] = []
    external_handles: dict[str, str] = {}

    free_text_keys = {"free_text", "summary", "profile_summary", "description", "headline", "about", "bio"}

    for key, value in row.items():
        if not key or value is None:
            continue
        
        key_lower = key.lower().replace(" ", "_").replace("-", "_")
        if key_lower in protected_proxies:
            continue

        if key_lower in {"github", "linkedin", "twitter", "portfolio"}:
            if isinstance(value, str) and value.strip():
                external_handles[key_lower] = value.strip()
            continue

        parsed_value = value
        if isinstance(value, str) and value.strip():
            val_stripped = value.strip()
            if (val_stripped.startswith("{") and val_stripped.endswith("}")) or \
               (val_stripped.startswith("[") and val_stripped.endswith("]")):
                try:
                    parsed_value = json.loads(val_stripped)
                except Exception:
                    pass

        if key_lower in {"skills", "explicit_skills", "technologies", "tech_stack", "certifications", "certs", "languages"}:
            if isinstance(parsed_value, str):
                parsed_value = [item.strip() for item in parsed_value.split(",") if item.strip()]

        if key in free_text_keys:
            if isinstance(value, str) and value.strip():
                free_text_parts.append(value.strip())
        else:
            structured_fields[key] = parsed_value

    return {
        "structured_fields": structured_fields,
        "free_text": "\n\n".join(free_text_parts),
        "external_handles": external_handles,
    }


def parse_candidate_pool(raw_text: str, filename: str | None = None) -> list[dict[str, Any]]:
    """Parse and validate a candidate-pool JSON, JSONL, or CSV document into RawCandidate dicts.

    Supports JSON arrays, JSON Lines, and CSV layouts, handling standard layouts,
    Redrob schemas, and generic flat CSV formats natively.
    """

    text = (raw_text or "").strip()
    if not text:
        raise CandidatePoolError("Candidate pool is empty. Paste or upload a candidate pool (JSON, JSONL, or CSV).")

    ext = filename.split(".")[-1].lower() if filename else None

    records: list[dict[str, Any]] = []
    parsed_successfully = False
    is_csv = False
    error_msg = ""

    if ext == "csv" or (not ext and _looks_like_csv(text)):
        try:
            records = _parse_csv_text(text)
            parsed_successfully = True
            is_csv = True
        except Exception as exc:
            error_msg = f"Failed to parse as CSV: {exc}"

    if not parsed_successfully and (ext == "jsonl" or (not ext and _looks_like_jsonl(text))):
        try:
            records = _parse_jsonl_text(text)
            parsed_successfully = True
        except Exception as exc:
            error_msg = f"Failed to parse as JSONL: {exc}"

    if not parsed_successfully and (ext == "json" or not ext):
        try:
            records = _parse_json_text(text)
            parsed_successfully = True
        except Exception as exc:
            if not error_msg:
                error_msg = f"Failed to parse as JSON: {exc}"

    if not parsed_successfully and not ext:
        try:
            records = _parse_json_text(text)
            parsed_successfully = True
        except Exception:
            try:
                if _looks_like_jsonl(text):
                    records = _parse_jsonl_text(text)
                    parsed_successfully = True
            except Exception:
                pass
            if not parsed_successfully:
                try:
                    records = _parse_csv_text(text)
                    parsed_successfully = True
                    is_csv = True
                except Exception:
                    pass

    if not parsed_successfully:
        raise CandidatePoolError(
            f"Unable to parse candidate pool. Please ensure it is valid JSON, JSONL, or CSV.\nDetails: {error_msg}"
        )

    if not records:
        raise CandidatePoolError("Candidate pool is empty. Provide at least one candidate.")

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        candidates.append(_normalize_candidate(item, index, is_csv=is_csv))
    return candidates


def _normalize_candidate(item: Any, index: int, is_csv: bool = False) -> dict[str, Any]:
    """Validate one raw candidate object and fill empty defaults.

    Accepts only the three RawCandidate-shaped keys; unknown keys are rejected
    so a malformed pool fails fast with a clear, indexed message.
    """

    where = f"Candidate #{index + 1}"
    if not isinstance(item, dict):
        raise CandidatePoolError(f"{where} must be a JSON object, got {type(item).__name__}.")

    # If it's a CSV row, values are strings, so parse JSON strings where appropriate
    if is_csv:
        # 1. Standard keys
        for k in ["structured_fields", "external_handles"]:
            val = item.get(k)
            if isinstance(val, str) and val.strip():
                val_stripped = val.strip()
                if val_stripped.startswith("{") and val_stripped.endswith("}"):
                    try:
                        item[k] = json.loads(val_stripped)
                    except Exception:
                        pass
        # 2. Redrob keys
        for k in ["profile", "career_history", "education", "skills", "certifications", "languages", "redrob_signals"]:
            val = item.get(k)
            if isinstance(val, str) and val.strip():
                val_stripped = val.strip()
                if (val_stripped.startswith("{") and val_stripped.endswith("}")) or \
                   (val_stripped.startswith("[") and val_stripped.endswith("]")):
                    try:
                        item[k] = json.loads(val_stripped)
                    except Exception:
                        pass
        # 3. For list-like Redrob fields that are raw CSV strings, convert to list
        for k in ["skills", "certifications", "languages"]:
            val = item.get(k)
            if isinstance(val, str):
                item[k] = [s.strip() for s in val.split(",") if s.strip()]

    is_redrob = (
        isinstance(item.get("profile"), dict) or
        isinstance(item.get("career_history"), list) or
        isinstance(item.get("education"), list)
    )

    if is_redrob:
        from icrs.ingest.redrob_adapter import redrob_to_raw_candidate
        item = redrob_to_raw_candidate(item)
    elif not any(k in item for k in {"structured_fields", "free_text", "external_handles"}):
        # Convert any generic flat dict (CSV, JSON, or JSONL) to RawCandidate shape.
        # Previously guarded by `if is_csv:` — this caused JSON/JSONL pools with
        # flat/generic objects to hit the unknown-keys check and fail (Bug 2).
        item = _generic_dict_to_raw_candidate(item)

    allowed = {"structured_fields", "free_text", "external_handles"}
    unknown = set(item) - allowed
    if unknown:
        raise CandidatePoolError(
            f"{where} has unsupported field(s): {', '.join(sorted(unknown))}. "
            f"Allowed fields: structured_fields, free_text, external_handles."
        )

    structured_fields = item.get("structured_fields", {})
    if not isinstance(structured_fields, dict):
        raise CandidatePoolError(f"{where}: 'structured_fields' must be an object.")

    free_text = item.get("free_text", "")
    if not isinstance(free_text, str):
        raise CandidatePoolError(f"{where}: 'free_text' must be a string.")

    external_handles = item.get("external_handles", {})
    if not isinstance(external_handles, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in external_handles.items()
    ):
        raise CandidatePoolError(f"{where}: 'external_handles' must be an object of string→string.")

    return {
        "structured_fields": structured_fields,
        "free_text": free_text,
        "external_handles": external_handles,
    }


def build_rank_payload(
    raw_jd: str,
    job_type: str,
    candidates: list[dict[str, Any]],
    *,
    title: str | None = None,
) -> dict[str, Any]:
    """Build the ``POST /rank`` request body from recruiter inputs.

    Validates the inputs the orchestrator also rejects (empty/whitespace JD,
    empty candidate pool, unknown job type) so the UI can give immediate
    feedback before any network call (Requirement 2.6).

    Args:
        raw_jd: the job-description text (pasted or uploaded).
        job_type: one of :data:`JOB_TYPE_VALUES`.
        candidates: RawCandidate-shaped dicts (see :func:`parse_candidate_pool`).
        title: optional job title, stored distinct from the JD body.

    Returns:
        A dict matching the API's ``RankRequest`` schema.

    Raises:
        CandidatePoolError: on an empty JD, empty pool, or invalid job type.
    """

    jd = (raw_jd or "").strip()
    if not jd:
        raise CandidatePoolError("Job description is empty. Paste or upload a JD before ranking.")

    if job_type not in JOB_TYPE_VALUES:
        raise CandidatePoolError(
            f"Unknown job type {job_type!r}. Choose one of: {', '.join(JOB_TYPE_VALUES)}."
        )

    if not candidates:
        raise CandidatePoolError("Candidate pool is empty. Provide at least one candidate.")

    payload: dict[str, Any] = {
        "raw_jd": jd,
        "job_type": job_type,
        "candidates": [
            {
                "structured_fields": c.get("structured_fields", {}),
                "free_text": c.get("free_text", ""),
                "external_handles": c.get("external_handles", {}),
            }
            for c in candidates
        ],
    }
    clean_title = (title or "").strip()
    if clean_title:
        payload["title"] = clean_title
    return payload


# --------------------------------------------------------------------------- #
# Pure helpers — formatting (honest, no false precision)
# --------------------------------------------------------------------------- #
def format_score(value: float | int | None) -> str:
    """Format a [0,1] score as a two-decimal *relative* score string.

    Rounds to two decimals to avoid implying false precision. ``None`` (a
    missing value) renders as an em dash rather than a fabricated number.
    """

    if value is None:
        return "—"
    return f"{round(float(value), 2):.2f}"


def confidence_label(value: float | int | None) -> str:
    """Return an honest qualitative band ("High"/"Moderate"/"Low"/"Unknown").

    The band is derived from the [0,1] confidence using
    :data:`CONFIDENCE_HIGH_THRESHOLD` / :data:`CONFIDENCE_MEDIUM_THRESHOLD`.
    A missing confidence yields "Unknown" rather than a guessed band.
    """

    if value is None:
        return "Unknown"
    v = float(value)
    if v >= CONFIDENCE_HIGH_THRESHOLD:
        return "High"
    if v >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "Moderate"
    return "Low"


def format_confidence(value: float | int | None) -> str:
    """Format confidence as a labelled, two-decimal value (e.g. "High (0.78)").

    Presents confidence honestly: a qualitative band plus the rounded numeric
    value, never an absolute-probability claim. A missing value renders as
    "Unknown".
    """

    if value is None:
        return "Unknown"
    return f"{confidence_label(value)} ({format_score(value)})"


# --------------------------------------------------------------------------- #
# Pure helpers — response → display rows / notices
# --------------------------------------------------------------------------- #
def _as_id_set(ids: Any) -> set[str]:
    """Coerce a JSON id list into a set of strings (tolerating ``None``)."""

    if not ids:
        return set()
    return {str(cid) for cid in ids}


def transform_response_to_rows(
    response: dict[str, Any],
    uuid_to_name: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Transform a ``RankResponse`` dict into ordered, display-ready rows.

    Each row carries everything the UI renders for one candidate. Matches
    response UUIDs back to original candidate display names if mapping is supplied.

    Each row includes both ``raw_candidate_id`` (the backend UUID, suitable for
    widget keys and session-state references) and ``display_name`` (the
    human-readable label resolved from ``uuid_to_name``).
    """

    results = response.get("results") or []
    unavailable_ids = _as_id_set(response.get("explanation_unavailable_ids"))

    rows: list[dict[str, Any]] = []
    for result in results:
        candidate_id = str(result.get("candidate_id", ""))
        display_name = uuid_to_name.get(candidate_id, candidate_id) if uuid_to_name else candidate_id
        breakdown = result.get("breakdown") or {}
        explanation = result.get("explanation") or {}
        final_score = result.get("final_score")
        confidence = result.get("confidence")

        explanation_available = candidate_id not in unavailable_ids

        rows.append(
            {
                "raw_candidate_id": candidate_id,
                "display_name": display_name,
                # Keep candidate_id as display_name for backward compat
                "candidate_id": display_name,
                "rank": result.get("rank"),
                "final_score": final_score,
                "final_score_display": format_score(final_score),
                "confidence": confidence,
                "confidence_label": confidence_label(confidence),
                "confidence_display": format_confidence(confidence),
                "explanation_available": explanation_available,
                "summary": explanation.get("summary", ""),
                "driving_signals": list(explanation.get("driving_signals", []) or []),
                "gaps": list(explanation.get("gaps", []) or []),
                "unmet_must_haves": list(explanation.get("unmet_must_haves", []) or []),
                "breakdown": {
                    field: breakdown.get(field) for field in BREAKDOWN_FIELDS
                },
            }
        )

    rows.sort(key=lambda r: (r["rank"] is None, r["rank"] if r["rank"] is not None else 0))
    return rows


def compute_notices(response: dict[str, Any]) -> list[str]:
    """Compute honest run-level degradation notices from a ``RankResponse``.

    Surfaces, in order:

    * an "ordering not LLM-reranked" banner when ``reranked`` is ``False``
      (Requirement 9.4);
    * an "explanation unavailable" count when ``explanation_unavailable_ids`` is
      non-empty (Requirement 9.5);
    * an "excluded before ranking" count when ``excluded_candidate_ids`` is
      non-empty (Requirement 9.2/9.3).

    A clean run (LLM-reranked, no exclusions, all explanations present) yields
    an empty list.
    """

    notices: list[str] = []

    # reranked defaults to True when absent; only a literal False is a degradation.
    if response.get("reranked", True) is False:
        notices.append(
            "Ordering was NOT LLM-reranked (reranker unavailable); the shortlist "
            "is ordered by composite scores as a fallback."
        )

    unavailable = response.get("explanation_unavailable_ids") or []
    if unavailable:
        plural = "s" if len(unavailable) != 1 else ""
        notices.append(
            f"Explanation unavailable for {len(unavailable)} candidate{plural}; "
            f"those entries show no fabricated rationale."
        )

    excluded = response.get("excluded_candidate_ids") or []
    if excluded:
        plural = "s" if len(excluded) != 1 else ""
        verb = "were" if len(excluded) != 1 else "was"
        notices.append(
            f"{len(excluded)} candidate{plural} {verb} excluded before ranking "
            f"(e.g. failed enrichment or a hard disqualifier)."
        )

    return notices


# --------------------------------------------------------------------------- #
# Backend call (injectable; httpx imported lazily)
# --------------------------------------------------------------------------- #
def rank_via_api(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """POST ``payload`` to ``{base_url}/rank`` and return the parsed JSON.

    ``httpx`` is imported lazily so this module (and its pure helpers) import
    cleanly even when ``httpx`` is not installed. This is the default client
    callable injected into :func:`render`; a stub with the same
    ``(base_url, payload) -> dict`` signature can be injected in its place.

    Args:
        base_url: backend base URL (e.g. ``http://localhost:8000``).
        payload: a ``RankRequest``-shaped dict from :func:`build_rank_payload`.
        timeout: per-request timeout in seconds.

    Returns:
        The decoded ``RankResponse`` JSON as a dict.

    Raises:
        RuntimeError: if the backend returns a non-2xx status; the message
            includes the server-provided detail when available.
    """

    import httpx  # lazy import: not needed to import this module or test helpers

    url = f"{base_url.rstrip('/')}/rank"
    response = httpx.post(url, json=payload, timeout=timeout)
    if response.status_code >= 400:
        detail: str
        try:
            detail = str(response.json().get("detail", response.text))
        except Exception:  # noqa: BLE001 - fall back to raw text on non-JSON bodies
            detail = response.text
        raise RuntimeError(f"Ranking request failed ({response.status_code}): {detail}")
    return response.json()


# --------------------------------------------------------------------------- #
# Streamlit render (st.* imported lazily; not executed on import)
# --------------------------------------------------------------------------- #
def decompose_via_api(
    base_url: str,
    raw_jd: str,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST ``raw_jd`` to ``{base_url}/decompose-jd`` and return the parsed JSON.

    ``httpx`` is imported lazily so this module (and its pure helpers) import
    cleanly even when ``httpx`` is not installed.
    """
    import httpx

    url = f"{base_url.rstrip('/')}/decompose-jd"
    response = httpx.post(url, json={"raw_jd": raw_jd}, timeout=timeout)
    if response.status_code >= 400:
        detail: str
        try:
            detail = str(response.json().get("detail", response.text))
        except Exception:  # noqa: BLE001
            detail = response.text
        raise RuntimeError(f"Decomposition request failed ({response.status_code}): {detail}")
    return response.json()


def render(rank_client: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> None:
    """Render the Streamlit dashboard with a premium multi-screen layout.

    ``streamlit`` is imported lazily here so importing this module never
    requires Streamlit to be installed (keeping the pure helpers testable in
    isolation). The ranking transport is injectable via ``rank_client`` — a
    ``(base_url, payload) -> response_dict`` callable — defaulting to
    :func:`rank_via_api`.
    """

    import streamlit as st  # lazy import: only needed when actually rendering

    # Ensure page config is set first
    st.set_page_config(page_title="TalentLens — AI-Powered Talent Intelligence", layout="wide", page_icon="🔍")

    # Custom CSS block to match the premium Cool Violet / Deep Navy Stitch theme
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

        /* ── Global: force light-mode colours on all Streamlit elements ── */
        .stApp {
            background-color: #f8f9ff;
            font-family: 'Inter', sans-serif;
            color: #0f172a;
        }

        /* Body text, markdown, paragraphs, list items */
        .stApp p, .stApp li, .stApp span, .stApp label,
        .stApp .stMarkdown, .stApp .stMarkdown p,
        .stApp .stText, .stApp div {
            color: #1e293b !important;
        }

        /* Headings */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #0f172a !important;
        }

        /* Captions / small text */
        .stApp .stCaption, .stApp small,
        .stApp [data-testid="stCaptionContainer"] {
            color: #64748b !important;
        }

        /* Input labels */
        .stApp [data-testid="stWidgetLabel"] label,
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp .stSelectbox label, .stApp .stTextInput label,
        .stApp .stTextArea label, .stApp .stNumberInput label,
        .stApp .stFileUploader label, .stApp .stRadio label {
            color: #334155 !important;
            font-weight: 600;
        }

        /* Text inputs, text areas, number inputs */
        .stApp .stTextInput input, .stApp .stTextArea textarea,
        .stApp .stNumberInput input, .stApp .stSelectbox > div > div {
            color: #0f172a !important;
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* Selectbox dropdown text */
        .stApp [data-baseweb="select"] span,
        .stApp [data-baseweb="select"] div {
            color: #0f172a !important;
        }

        /* Info / Warning / Error boxes */
        .stApp .stAlert p, .stApp .stAlert span {
            color: #1e293b !important;
        }

        /* Tables */
        .stApp table, .stApp table th, .stApp table td {
            color: #1e293b !important;
            border-color: #e2e8f0 !important;
        }
        .stApp table th {
            background-color: #f1f5f9 !important;
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        .stApp table td {
            background-color: #ffffff !important;
        }

        /* Expanders */
        .stApp [data-testid="stExpander"] summary span,
        .stApp [data-testid="stExpander"] summary p {
            color: #0f172a !important;
            font-weight: 600;
        }
        .stApp [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
        }

        /* Metric values */
        .stApp [data-testid="stMetricValue"] {
            color: #7c3aed !important;
        }
        .stApp [data-testid="stMetricLabel"] {
            color: #475569 !important;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #94a3b8 !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background-color: #1e293b !important;
            color: #f1f5f9 !important;
            border-color: #334155 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #334155 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #cbd5e1 !important;
        }

        /* Radio buttons in sidebar */
        section[data-testid="stSidebar"] .stRadio label span {
            color: #e2e8f0 !important;
        }

        /* ── Primary button ── */
        .stApp .stButton > button[kind="primary"],
        .stApp .stButton > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            transition: all 0.2s ease;
        }
        .stApp .stButton > button[kind="primary"]:hover,
        .stApp .stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35) !important;
            transform: translateY(-1px);
        }

        /* ── Custom component styles ── */
        .premium-title {
            background: linear-gradient(135deg, #7c3aed, #0f172a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.2rem;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 16px;
        }

        .insight-section {
            margin-bottom: 20px;
        }

        .insight-title {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: #64748b !important;
            margin-bottom: 8px;
        }

        .role-intent-box {
            background-color: #eff4ff;
            border-left: 4px solid #7c3aed;
            padding: 14px;
            border-radius: 4px 12px 12px 4px;
            color: #0f172a !important;
            font-weight: 600;
            font-size: 15px;
            line-height: 1.4;
        }

        .badge-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .badge-must {
            background-color: rgba(124, 58, 237, 0.08);
            border: 1px solid rgba(124, 58, 237, 0.2);
            color: #7c3aed !important;
            padding: 6px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 13px;
        }

        .badge-nice {
            background-color: #e5eeff;
            border: 1px solid #c6d0e2;
            color: #334155 !important;
            padding: 6px 14px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 13px;
        }

        .behavioral-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(226, 232, 240, 0.5);
        }
        .behavioral-row:last-child {
            border-bottom: none;
        }

        .behavioral-label {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
            color: #0f172a !important;
            font-size: 14px;
        }

        .behavior-icon {
            width: 30px;
            height: 30px;
            border-radius: 6px;
            background-color: #eff4ff;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #7c3aed !important;
        }

        .progress-track {
            width: 90px;
            height: 6px;
            background-color: #e2e8f0;
            border-radius: 9999px;
            overflow: hidden;
        }

        .material-symbols-outlined {
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            display: inline-block;
            vertical-align: middle;
        }

        /* Spinner text */
        .stApp .stSpinner > div > span {
            color: #475569 !important;
        }

        /* File uploader */
        .stApp [data-testid="stFileUploader"] label {
            color: #334155 !important;
        }
        .stApp [data-testid="stFileUploader"] section {
            background-color: #ffffff !important;
            border: 1px dashed #cbd5e1 !important;
        }
        .stApp [data-testid="stFileUploader"] section small {
            color: #64748b !important;
        }
        .stApp [data-testid="stFileUploader"] *,
        .stApp [data-testid="stFileUploader"] span,
        .stApp [data-testid="stFileUploader"] div,
        .stApp [data-testid="stFileUploaderFileName"],
        .stApp [data-testid="stUploadedFile"] * {
            color: #1e293b !important;
        }
        /* "Browse files" button inside file uploader */
        .stApp [data-testid="stFileUploader"] button,
        .stApp [data-testid="stFileUploader"] section button,
        .stApp [data-testid="baseButton-secondary"] {
            background-color: #f1f5f9 !important;
            color: #334155 !important;
            border: 1px solid #cbd5e1 !important;
            font-weight: 600 !important;
        }
        .stApp [data-testid="stFileUploader"] button:hover,
        .stApp [data-testid="baseButton-secondary"]:hover {
            background-color: #e2e8f0 !important;
            color: #0f172a !important;
            border-color: #94a3b8 !important;
        }

        /* ── Sidebar caption (override global .stApp rules) ── */
        .stApp section[data-testid="stSidebar"] .stCaption,
        .stApp section[data-testid="stSidebar"] .stCaption p,
        .stApp section[data-testid="stSidebar"] .stCaption span,
        .stApp section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        .stApp section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        .stApp section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] span {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
        }

        /* ── Recruiter-Style Candidate Cards ── */
        .candidate-card {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02) !important;
            margin-bottom: 16px !important;
        }
        .candidate-card-header {
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            border-bottom: 2px solid #f1f5f9 !important;
            padding-bottom: 10px !important;
            margin-bottom: 14px !important;
        }
        .candidate-card-name {
            font-size: 18px !important;
            font-weight: 700 !important;
            color: #0f172a !important;
        }
        .candidate-card-match {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #7c3aed !important;
        }
        .candidate-card-grid {
            display: grid !important;
            grid-template-columns: repeat(3, 1fr) !important;
            gap: 12px !important;
            margin-bottom: 16px !important;
        }
        .candidate-card-fit-item {
            background-color: #f8fafc !important;
            padding: 10px !important;
            border-radius: 8px !important;
            border: 1px solid #f1f5f9 !important;
            text-align: center !important;
        }
        .candidate-card-fit-label {
            font-size: 11px !important;
            text-transform: uppercase !important;
            color: #64748b !important;
            font-weight: 600 !important;
            margin-bottom: 4px !important;
        }
        .candidate-card-fit-value {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #0f172a !important;
        }
        .candidate-card-section {
            margin-bottom: 12px !important;
        }
        .candidate-card-section-title {
            font-size: 12px !important;
            font-weight: 700 !important;
            color: #475569 !important;
            margin-bottom: 6px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }
        .candidate-card-signal-item {
            display: flex !important;
            align-items: flex-start !important;
            gap: 8px !important;
            font-size: 14px !important;
            color: #334155 !important;
            margin-bottom: 4px !important;
        }
        .candidate-card-icon-check {
            color: #10b981 !important;
            font-weight: bold !important;
        }
        .candidate-card-icon-warn {
            color: #f59e0b !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state variables
    if "raw_jd" not in st.session_state:
        st.session_state.raw_jd = ""
    if "job_title" not in st.session_state:
        st.session_state.job_title = ""
    if "job_type" not in st.session_state:
        st.session_state.job_type = "TECHNICAL"
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "navigation_page" not in st.session_state:
        st.session_state.navigation_page = "Job Creation & Analysis"
    if "ranking_response" not in st.session_state:
        st.session_state.ranking_response = None
    if "uuid_to_name" not in st.session_state:
        st.session_state.uuid_to_name = None
    if "selected_candidate_id" not in st.session_state:
        st.session_state.selected_candidate_id = None

    with st.sidebar:
        st.markdown(
            "<div style='padding: 4px 0 2px 0;'>"
            "<span style='font-size: 1.6rem; font-weight: 800; "
            "background: linear-gradient(135deg, #a78bfa, #f8fafc); "
            "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
            "letter-spacing: -0.02em;'>🔍 TalentLens</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("AI-powered hiring intelligence — rank, match & shortlist in seconds.")
        
        # Navigation radio for page routing.
        # Button handlers set navigation_page, and we sync it to the radio
        # widget key BEFORE the widget is instantiated (Streamlit forbids
        # writes to a widget key after the widget is created).
        pages = ["Job Creation & Analysis", "Candidate Ranking & Match", "Explainability Panel"]

        # Sync programmatic navigation → radio widget BEFORE it renders.
        target = st.session_state.get("navigation_page", pages[0])
        if target in pages:
            st.session_state.nav_radio = target

        navigation_page = st.radio(
            "Go to page:",
            options=pages,
            index=0,
            key="nav_radio"
        )

        # Sync radio selection back to navigation_page.
        st.session_state.navigation_page = navigation_page

        st.markdown("---")
        st.header("Backend Config")
        base_url = st.text_input("Ranking API base URL", value=DEFAULT_BACKEND_URL)
        timeout_seconds = st.number_input(
            "Request timeout (seconds)",
            min_value=30,
            max_value=3600,
            value=600,
            step=30,
            help=(
                "Ranking calls an LLM per candidate. On Groq's free tier, large "
                "pools pace themselves under the rate limit and can take minutes."
            ),
        )

    # --- Screen routing ------------------------------------------------------
    if st.session_state.navigation_page == "Job Creation & Analysis":
        from icrs.ui.screen_job_creation import render_job_creation
        render_job_creation(st, base_url)
    elif st.session_state.navigation_page == "Candidate Ranking & Match":
        from icrs.ui.screen_candidate_ranking import render_candidate_ranking
        render_candidate_ranking(st, base_url, timeout_seconds, rank_client)
    elif st.session_state.navigation_page == "Explainability Panel":
        from icrs.ui.screen_explainability import render_explainability
        render_explainability(st)



# re_rank_rows has been removed.
# The backend's ranking (which uses per-job-type weight profiles and the LLM
# rerank blend) is the single source of truth. Client-side re-weighting was
# silently discarding the 0.6*composite + 0.4*LLM blend and ignoring
# per-job-type weight profiles, violating the design contract.


def _render_results(
    st: Any,
    response: dict[str, Any],
    uuid_to_name: dict[str, str] | None = None,
) -> None:
    """Render the ranked shortlist, notices, and per-candidate detail sections.

    Separated from :func:`render` so the bulk of the display logic operates on
    a plain ``RankResponse`` dict via the pure helpers.
    """

    notices = compute_notices(response)
    for notice in notices:
        st.warning(notice)

    rows = transform_response_to_rows(response, uuid_to_name)
    if not rows:
        st.info("No candidates were ranked. (All may have been excluded before ranking.)")
        return

    st.subheader("Ranked shortlist")
    st.caption("Ordered by rank. Candidate metrics are shown relative to the target requirements.")

    for row in rows:
        # Calculate percentages safely
        overall_match_pct = f"{int(round(float(row['final_score']) * 100))}%" if row['final_score'] is not None else "—"
        
        semantic = row['breakdown'].get('semantic_fit')
        career = row['breakdown'].get('career_trajectory')
        behavioral = row['breakdown'].get('behavioral')
        
        skill_fit_pct = f"{int(round(float(semantic) * 100))}%" if semantic is not None else "—"
        experience_fit_pct = f"{int(round(float(career) * 100))}%" if career is not None else "—"
        behavior_fit_pct = f"{int(round(float(behavioral) * 100))}%" if behavioral is not None else "—"
        
        # Build Strengths list (driving_signals)
        strengths_html = ""
        if row['driving_signals']:
            for s in row['driving_signals']:
                strengths_html += f'<div class="candidate-card-signal-item"><span class="candidate-card-icon-check">✓</span> {s}</div>'
        else:
            strengths_html = '<div class="candidate-card-signal-item" style="color: #64748b; font-style: italic;">No specific strengths highlighted</div>'
            
        # Build Gaps list (unmet_must_haves + gaps)
        gaps_list = list(row['unmet_must_haves']) + list(row['gaps'])
        gaps_html = ""
        if gaps_list:
            for g in gaps_list:
                gaps_html += f'<div class="candidate-card-signal-item"><span class="candidate-card-icon-warn">⚠</span> {g}</div>'
        else:
            gaps_html = '<div class="candidate-card-signal-item" style="color: #64748b; font-style: italic;">No significant gaps identified</div>'

        st.markdown(
            f"""
            <div class="candidate-card">
                <div class="candidate-card-header">
                    <span class="candidate-card-name">#{row['rank']} {row['candidate_id']}</span>
                    <span class="candidate-card-match">Overall Match: {overall_match_pct}</span>
                </div>
                
                <div class="candidate-card-grid">
                    <div class="candidate-card-fit-item">
                        <div class="candidate-card-fit-label">Skill Fit</div>
                        <div class="candidate-card-fit-value">{skill_fit_pct}</div>
                    </div>
                    <div class="candidate-card-fit-item">
                        <div class="candidate-card-fit-label">Experience Fit</div>
                        <div class="candidate-card-fit-value">{experience_fit_pct}</div>
                    </div>
                    <div class="candidate-card-fit-item">
                        <div class="candidate-card-fit-label">Behavior Fit</div>
                        <div class="candidate-card-fit-value">{behavior_fit_pct}</div>
                    </div>
                </div>
                
                <div class="candidate-card-section">
                    <div class="candidate-card-section-title">Top Strengths:</div>
                    {strengths_html}
                </div>
                
                <div class="candidate-card-section">
                    <div class="candidate-card-section-title">Potential Gaps:</div>
                    {gaps_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(f"View Details — {row['display_name']}", key=f"inspect_btn_{row['rank']}", use_container_width=True):
            st.session_state.selected_candidate_id = row['raw_candidate_id']
            st.session_state.navigation_page = "Explainability Panel"
            st.rerun()
        
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)


def main() -> None:
    """Entry point used by ``streamlit run icrs/ui/dashboard.py``."""

    render()


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_BACKEND_URL",
    "JOB_TYPE_VALUES",
    "BREAKDOWN_FIELDS",
    "BREAKDOWN_LABELS",
    "CandidatePoolError",
    "parse_candidate_pool",
    "build_rank_payload",
    "format_score",
    "confidence_label",
    "format_confidence",
    "transform_response_to_rows",
    "compute_notices",
    "rank_via_api",
    "decompose_via_api",
    "render",
    "main",
    # re_rank_rows removed: backend ranking is the single source of truth.
]

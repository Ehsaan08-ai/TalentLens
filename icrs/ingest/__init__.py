"""Dataset adapters that map external candidate schemas onto ICRS inputs.

These adapters convert third-party candidate datasets into the ICRS
``RawCandidate`` shape (``structured_fields`` / ``free_text`` /
``external_handles``) that the ranking pipeline and the API/UX consume. They are
deliberately separate from the pipeline so new source schemas can be supported
without touching the scoring or orchestration layers.
"""

from icrs.ingest.redrob_adapter import (
    PROTECTED_PROXY_SOURCE_FIELDS,
    convert_pool,
    load_redrob_records,
    redrob_to_raw_candidate,
)

__all__ = [
    "PROTECTED_PROXY_SOURCE_FIELDS",
    "convert_pool",
    "load_redrob_records",
    "redrob_to_raw_candidate",
]

"""Output layer for ICRS (Layer 5).

This package turns the orchestrator's :class:`~icrs.pipeline.orchestrator.RankingRun`
into a recruiter-facing, UI-consumable shortlist object. It owns no scoring or
ranking logic — it maps an already-produced run into the shortlist contract and
honestly surfaces any degraded behaviour (un-reranked fallback, unavailable
explanations, excluded candidates) per Requirements 5.1, 9.4, and 9.5.
"""

from __future__ import annotations

from icrs.output.shortlist import (
    RankedShortlist,
    ShortlistEntry,
    assemble_shortlist,
)

__all__ = [
    "RankedShortlist",
    "ShortlistEntry",
    "assemble_shortlist",
]

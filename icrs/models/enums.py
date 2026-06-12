"""Shared enumerations for ICRS data models.

These enums are intentionally placed in a standalone module so that models from
different tasks can reference them without creating import cycles. In particular
``SignalTier`` is shared between the requirement models (Task 2.1 — a
``Requirement`` carries a ``tier``) and the candidate models (Task 2.2 — an
``EnrichedProfile`` records ``signal_availability`` per tier).

All enums subclass ``str`` so their members serialize to their string value
(useful for JSON payloads, persistence, and prompt construction) while still
being comparable to plain strings.
"""

from __future__ import annotations

from enum import Enum


class SignalTier(str, Enum):
    """The three signal tiers ICRS reasons over.

    - ``STRUCTURAL`` (Tier 1): extracted directly from structured profile fields.
    - ``SEMANTIC`` (Tier 2): inferred from free text via embeddings + LLM.
    - ``BEHAVIORAL`` (Tier 3): derived from external platform activity.

    Shared by requirement classification (``Requirement.tier``) and per-tier
    candidate coverage (``EnrichedProfile.signal_availability``).
    """

    STRUCTURAL = "STRUCTURAL"
    SEMANTIC = "SEMANTIC"
    BEHAVIORAL = "BEHAVIORAL"


class TrajectoryArc(str, Enum):
    """Tier 2 career trajectory arc inferred from a candidate's role history."""

    ACCELERATING = "ACCELERATING"
    STEADY = "STEADY"
    LATERAL = "LATERAL"
    DECLINING = "DECLINING"


class DepthBreadth(str, Enum):
    """Tier 2 depth-versus-breadth signature (specialist vs generalist)."""

    SPECIALIST = "SPECIALIST"
    BALANCED = "BALANCED"
    GENERALIST = "GENERALIST"


__all__ = ["SignalTier", "TrajectoryArc", "DepthBreadth"]

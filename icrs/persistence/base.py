"""Abstract async persistence contract for ICRS (Task 6.1).

The pipeline persists three kinds of records as part of Phase 1 ingestion:

    - a :class:`~icrs.models.job.JobDescription` together with its parsed
      :class:`~icrs.models.job.RequirementVector`,
    - candidate profiles (:class:`~icrs.models.candidate.EnrichedProfile`), and
    - one dense embedding per candidate (pgvector is the system-of-record at PoC
      scale — see the design's data-layer section).

This module defines the *interface* only. Two concrete implementations satisfy
it:

    - :class:`~icrs.persistence.postgres.PostgresRankingStore` — the real
      PostgreSQL + pgvector system-of-record (SQLAlchemy async + asyncpg).
    - :class:`~icrs.persistence.memory.InMemoryRankingStore` — a dependency-free
      async store used by tests and by the Phase 1 ingestion path (Task 6.2)
      without requiring a running database.

Both implementations enforce the same invariant from Requirement 2.2: a stored
embedding's dimensionality must match the configured embedding model's
dimensionality. Violations raise :class:`DimensionalityError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from icrs.models.candidate import EnrichedProfile
from icrs.models.job import JobDescription
from icrs.providers.base import Vector


class PersistenceError(RuntimeError):
    """Base class for persistence-layer errors."""


class DimensionalityError(PersistenceError, ValueError):
    """Raised when an embedding's dimensionality does not match the configured model.

    Inherits from :class:`ValueError` as well so callers that catch the standard
    exception for bad values still handle it.
    """

    def __init__(self, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"embedding dimensionality {actual} does not match the configured "
            f"model dimensionality {expected}"
        )


def validate_dimensionality(vector: Vector, expected_dim: int) -> Vector:
    """Return ``vector`` unchanged if its length matches ``expected_dim``.

    Raises :class:`DimensionalityError` otherwise. Centralized here so every
    store implementation validates identically (Requirement 2.2).
    """

    if len(vector) != expected_dim:
        raise DimensionalityError(expected=expected_dim, actual=len(vector))
    return vector


class RankingStore(ABC):
    """Async system-of-record for jobs, candidate profiles, and embeddings.

    All methods are asynchronous so the same contract serves the async
    PostgreSQL implementation and the in-memory test/PoC implementation. The
    pipeline depends only on this interface, never on a concrete backend.

    ``embedding_dim`` is the dimensionality every stored embedding must match;
    it is supplied at construction time (from :class:`icrs.config.Settings`) so
    the store can reject embeddings produced by a mismatched model.
    """

    #: The configured embedding model dimensionality every embedding must match.
    embedding_dim: int

    # ----- Jobs + parsed RequirementVector -----

    @abstractmethod
    async def save_job(self, job: JobDescription) -> None:
        """Insert or update a :class:`JobDescription` (including ``parsed``)."""

    @abstractmethod
    async def get_job(self, job_id: UUID) -> JobDescription | None:
        """Return the stored job for ``job_id``, or ``None`` if absent."""

    # ----- Candidate profiles -----

    @abstractmethod
    async def save_candidate(
        self, profile: EnrichedProfile, *, job_id: UUID | None = None
    ) -> None:
        """Insert or update a candidate ``profile``.

        When ``job_id`` is supplied the candidate is associated with that job so
        it is returned by :meth:`list_candidates_for_job`. If the profile carries
        an ``embedding`` it is validated and stored as well.
        """

    @abstractmethod
    async def get_candidate(self, candidate_id: UUID) -> EnrichedProfile | None:
        """Return the stored candidate profile, or ``None`` if absent."""

    @abstractmethod
    async def list_candidates_for_job(self, job_id: UUID) -> list[EnrichedProfile]:
        """Return all candidate profiles associated with ``job_id``."""

    # ----- Per-candidate embeddings -----

    @abstractmethod
    async def upsert_embedding(self, candidate_id: UUID, vector: Vector) -> None:
        """Store ``vector`` as ``candidate_id``'s embedding.

        Raises :class:`DimensionalityError` when ``len(vector)`` does not equal
        the configured embedding dimensionality (Requirement 2.2).
        """

    @abstractmethod
    async def get_embedding(self, candidate_id: UUID) -> Vector | None:
        """Return the stored embedding for ``candidate_id``, or ``None``."""

    # ----- Ranking run responses caching (Cache-Aside DB layer) -----

    @abstractmethod
    async def save_ranking_run(self, run_id: str, document: dict[str, Any]) -> None:
        """Insert or update a ranking run response document keyed by run_id."""

    @abstractmethod
    async def get_ranking_run(self, run_id: str) -> dict[str, Any] | None:
        """Return the stored ranking run response document for run_id, or ``None``."""


__all__ = [
    "PersistenceError",
    "DimensionalityError",
    "validate_dimensionality",
    "RankingStore",
]

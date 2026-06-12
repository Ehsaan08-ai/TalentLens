"""In-memory async implementation of :class:`RankingStore` (Task 6.1).

This store keeps everything in plain dictionaries behind an ``asyncio`` lock. It
has no external dependencies, so it is used by the unit tests and by the Phase 1
ingestion path (Task 6.2) without requiring a running PostgreSQL server. It
implements the exact same async contract as the pgvector system-of-record, so
code written against :class:`RankingStore` behaves identically with either.

Profiles and jobs are deep-copied on the way in and out so callers cannot mutate
stored state by holding a reference — this mirrors the round-trip isolation a
real database provides (objects are re-materialized from rows on read).
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from icrs.config import get_settings
from icrs.models.candidate import EnrichedProfile
from icrs.models.job import JobDescription
from icrs.persistence.base import RankingStore, validate_dimensionality
from icrs.providers.base import Vector


class InMemoryRankingStore(RankingStore):
    """A dependency-free, async, dictionary-backed :class:`RankingStore`.

    ``embedding_dim`` defaults to the configured embedding model dimensionality
    (``Settings.embedding_dim``) so dimensionality validation matches the rest
    of the system; it can be overridden for tests.
    """

    def __init__(self, embedding_dim: int | None = None) -> None:
        self.embedding_dim = (
            embedding_dim if embedding_dim is not None else get_settings().embedding_dim
        )
        self._jobs: dict[UUID, JobDescription] = {}
        self._candidates: dict[UUID, EnrichedProfile] = {}
        self._embeddings: dict[UUID, Vector] = {}
        # candidate_id -> job_id association for list_candidates_for_job.
        self._candidate_job: dict[UUID, UUID] = {}
        self._ranking_runs: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    # ----- Jobs + parsed RequirementVector -----

    async def save_job(self, job: JobDescription) -> None:
        async with self._lock:
            # Deep copy via pydantic so later mutations of the caller's object do
            # not leak into stored state.
            self._jobs[job.id] = job.model_copy(deep=True)

    async def get_job(self, job_id: UUID) -> JobDescription | None:
        async with self._lock:
            stored = self._jobs.get(job_id)
            return stored.model_copy(deep=True) if stored is not None else None

    # ----- Candidate profiles -----

    async def save_candidate(
        self, profile: EnrichedProfile, *, job_id: UUID | None = None
    ) -> None:
        # Validate any embedding the profile carries before taking the lock so a
        # bad vector is rejected without partially mutating state.
        if profile.embedding is not None:
            validate_dimensionality(profile.embedding, self.embedding_dim)

        async with self._lock:
            self._candidates[profile.id] = profile.model_copy(deep=True)
            if job_id is not None:
                self._candidate_job[profile.id] = job_id
            if profile.embedding is not None:
                self._embeddings[profile.id] = list(profile.embedding)

    async def get_candidate(self, candidate_id: UUID) -> EnrichedProfile | None:
        async with self._lock:
            stored = self._candidates.get(candidate_id)
            return stored.model_copy(deep=True) if stored is not None else None

    async def list_candidates_for_job(self, job_id: UUID) -> list[EnrichedProfile]:
        async with self._lock:
            return [
                self._candidates[cid].model_copy(deep=True)
                for cid, jid in self._candidate_job.items()
                if jid == job_id and cid in self._candidates
            ]

    # ----- Per-candidate embeddings -----

    async def upsert_embedding(self, candidate_id: UUID, vector: Vector) -> None:
        validate_dimensionality(vector, self.embedding_dim)
        async with self._lock:
            self._embeddings[candidate_id] = list(vector)

    async def get_embedding(self, candidate_id: UUID) -> Vector | None:
        async with self._lock:
            stored = self._embeddings.get(candidate_id)
            return list(stored) if stored is not None else None

    # ----- Ranking run responses caching -----

    async def save_ranking_run(self, run_id: str, document: dict[str, Any]) -> None:
        import copy
        async with self._lock:
            self._ranking_runs[run_id] = copy.deepcopy(document)

    async def get_ranking_run(self, run_id: str) -> dict[str, Any] | None:
        import copy
        async with self._lock:
            stored = self._ranking_runs.get(run_id)
            return copy.deepcopy(stored) if stored is not None else None


__all__ = ["InMemoryRankingStore"]

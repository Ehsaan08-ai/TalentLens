"""Phase 1 end-to-end ingestion path for ICRS (Task 6.2).

This module wires the Phase 1 pipeline components together and persists their
output. Given a raw job description and a pool of raw candidates it runs, in
order:

    decompose -> (per candidate) normalize -> enrich -> embed

and persists the results through the async :class:`~icrs.persistence.base.RankingStore`:

    1. Build a :class:`~icrs.models.job.JobDescription`, decompose its raw text
       into a :class:`~icrs.models.job.RequirementVector` (associating the parsed
       vector with the job via ``job_id``), and persist the job with
       :meth:`RankingStore.save_job` (Requirements 2.1, 2.2).
    2. For each :class:`~icrs.models.candidate.RawCandidate`: normalize ->
       enrich -> embed, attach the embedding to the
       :class:`~icrs.models.candidate.EnrichedProfile`, persist the candidate
       associated with the job via :meth:`RankingStore.save_candidate`, and store
       one embedding per candidate with :meth:`RankingStore.upsert_embedding`
       (Requirements 2.1, 2.2).
    3. Optionally embed the requirement vector so it can later be retrieved for
       dense candidate/requirement comparison (Requirement 6.4).

Dependency injection
--------------------
Every collaborator (``decomposer``, ``enricher``, ``embedder``, ``store``) is
injected, so the ingestion path depends only on the component contracts and is
fully testable with stub providers and the dependency-free
:class:`~icrs.persistence.memory.InMemoryRankingStore` — no live database or LLM
is required.

Async handling
--------------
The store is asynchronous; the decomposer, enricher, and embedder are synchronous
(pure-Python / provider-backed) and are therefore called directly. ``ingest`` is
an ``async`` coroutine so it can ``await`` the store between synchronous compute
steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol
from uuid import UUID

from icrs.models.candidate import EnrichedProfile, RawCandidate
from icrs.models.job import JobDescription, JobType, RequirementVector
from icrs.persistence.base import RankingStore
from icrs.providers.base import Vector


# --------------------------------------------------------------------------- #
# Structural collaborator protocols
# --------------------------------------------------------------------------- #
# These describe only the methods the ingestion path uses, so any object
# satisfying the shape (the real components or a test stub) can be injected.
class _Decomposer(Protocol):
    def decompose(self, raw_jd: str) -> RequirementVector: ...


class _Enricher(Protocol):
    def normalize(self, raw_profile: RawCandidate): ...

    def enrich(self, profile) -> EnrichedProfile: ...


class _Embedder(Protocol):
    def embed(self, profile: EnrichedProfile) -> Vector: ...

    def embed_requirement(self, reqs: RequirementVector) -> Vector: ...


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IngestionResult:
    """Summary of a Phase 1 ingestion run.

    Attributes:
        job_id: the id of the persisted :class:`JobDescription`.
        requirement_vector: the parsed requirement vector persisted with the job.
        candidate_ids: ids of the candidates ingested (in pool order).
        embedded_count: number of candidates for which an embedding was stored —
            equal to ``len(candidate_ids)`` on a fully successful run (one
            embedding per candidate, Requirement 2.2).
        requirement_embedding: the requirement vector's embedding when computed
            for later retrieval, else ``None``.
    """

    job_id: UUID
    requirement_vector: RequirementVector
    candidate_ids: list[UUID] = field(default_factory=list)
    embedded_count: int = 0
    requirement_embedding: Vector | None = None

    @property
    def candidate_count(self) -> int:
        """Number of candidates ingested in this run."""

        return len(self.candidate_ids)


# --------------------------------------------------------------------------- #
# Phase 1 ingestion path
# --------------------------------------------------------------------------- #
async def ingest(
    raw_jd: str,
    title: str,
    job_type: JobType,
    candidate_pool: Iterable[RawCandidate],
    *,
    decomposer: _Decomposer,
    enricher: _Enricher,
    embedder: _Embedder,
    store: RankingStore,
    embed_requirement: bool = True,
) -> IngestionResult:
    """Run and persist the Phase 1 ingestion pipeline for one JD + candidate pool.

    Composes the injected Phase 1 components and persists their output through the
    async store (Requirements 2.1, 2.2):

        1. decompose the JD, persist the job + its parsed RequirementVector;
        2. normalize -> enrich -> embed each candidate, persisting the profile
           (associated with the job) and exactly one embedding per candidate;
        3. optionally embed the requirement vector for later retrieval.

    Args:
        raw_jd: the raw job-description text (must be non-empty; the decomposer
            rejects empty/whitespace-only input).
        title: the job title (stored distinct from the decomposed role intent).
        job_type: the role family driving later weight-profile selection.
        candidate_pool: the raw candidates to ingest.
        decomposer: produces a :class:`RequirementVector` from ``raw_jd``.
        enricher: ``normalize`` then ``enrich`` each raw candidate.
        embedder: ``embed`` enriched profiles and ``embed_requirement`` the JD.
        store: the async persistence backend (system-of-record).
        embed_requirement: when ``True`` (default) the requirement vector is
            embedded and returned for later retrieval.

    Returns:
        An :class:`IngestionResult` summarizing the persisted job id and the
        number of candidates ingested with embeddings.
    """

    # ----- 1. Build the job, decompose, persist (Requirements 2.1, 2.2) ----- #
    job = JobDescription(raw_text=raw_jd, title=title, job_type=job_type)

    requirement_vector = decomposer.decompose(raw_jd)
    # Associate the parsed vector with this job (the decomposer assigns its own
    # default job_id; rebind it to the owning job).
    requirement_vector = requirement_vector.model_copy(update={"job_id": job.id})
    job.parsed = requirement_vector

    await store.save_job(job)

    # ----- 2. Ingest each candidate: normalize -> enrich -> embed ----- #
    candidate_ids: list[UUID] = []
    embedded_count = 0

    for raw_candidate in candidate_pool:
        normalized = enricher.normalize(raw_candidate)
        enriched = enricher.enrich(normalized)
        vector = embedder.embed(enriched)

        # Attach the embedding so the persisted profile carries it, then persist
        # both the job-associated profile and one embedding per candidate.
        enriched.embedding = vector
        await store.save_candidate(enriched, job_id=job.id)
        await store.upsert_embedding(enriched.id, vector)

        candidate_ids.append(enriched.id)
        embedded_count += 1

    # ----- 3. Optionally embed the requirement vector for later retrieval ----- #
    requirement_embedding: Vector | None = None
    if embed_requirement:
        requirement_embedding = embedder.embed_requirement(requirement_vector)

    return IngestionResult(
        job_id=job.id,
        requirement_vector=requirement_vector,
        candidate_ids=candidate_ids,
        embedded_count=embedded_count,
        requirement_embedding=requirement_embedding,
    )


__all__ = ["ingest", "IngestionResult"]

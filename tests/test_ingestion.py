"""Integration tests for the Phase 1 ingestion path (Task 6.2).

Exercises :func:`icrs.pipeline.ingestion.ingest` end-to-end by composing the
real Phase 1 components (``JDDecomposer``, ``CandidateEnricher``,
``EmbeddingGenerator``) over stubbed providers and the dependency-free
:class:`InMemoryRankingStore` — no live database or LLM is required.

Verifies (Requirements 2.1, 2.2):
    - the job and its parsed RequirementVector are persisted;
    - each candidate is persisted and associated with the job;
    - one embedding per candidate is stored with the correct dimensionality.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence

import pytest

from icrs.models.candidate import RawCandidate
from icrs.models.job import JobType, RequirementCategory
from icrs.persistence import InMemoryRankingStore
from icrs.pipeline.embedding import EmbeddingGenerator
from icrs.pipeline.enricher import CandidateEnricher
from icrs.pipeline.ingestion import IngestionResult, ingest
from icrs.pipeline.jd_decomposer import JDDecomposer
from icrs.providers.base import (
    EmbeddingProvider,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    Vector,
)

# A small embedding dimensionality keeps the tests fast and decoupled from the
# configured production model size.
DIM = 8


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class FixedLLM(LLMProvider):
    """An LLM provider that always returns the same scripted JSON payload."""

    def __init__(self, payload: dict, *, model: str = "stub-llm") -> None:
        self._text = json.dumps(payload)
        self._model = model
        self.calls: list[list[LLMMessage]] = []

    @property
    def model_id(self) -> str:
        return self._model

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens=None,
        response_format=None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(text=self._text, model=self._model)


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider producing ``DIM``-length vectors.

    Each text maps to a stable pseudo-random vector derived from its content
    hash, so embeddings are deterministic across runs and distinct texts get
    distinct vectors. ``max_input_tokens`` is large so the generator uses its
    single-embedding (non-chunked) path.
    """

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    @property
    def model_id(self) -> str:
        return "stub-embedding"

    @property
    def dimensionality(self) -> int:
        return self._dim

    @property
    def max_input_tokens(self) -> int:
        return 100_000

    def embed(self, text: str) -> Vector:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Build a non-zero deterministic vector from the digest bytes.
        return [((digest[i % len(digest)] + i + 1) / 255.0) for i in range(self._dim)]

    def embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed(t) for t in texts]


def _decompose_payload() -> dict:
    return {
        "role_intent": "Build and operate the payments platform end to end",
        "seniority_band": "SENIOR",
        "requirements": [
            {
                "text": "5+ years backend engineering experience",
                "category": "MUST_HAVE",
                "tier": "STRUCTURAL",
                "weight": 0.6,
            },
            {
                "text": "Experience with distributed systems",
                "category": "MUST_HAVE",
                "tier": "SEMANTIC",
                "weight": 0.4,
            },
            {
                "text": "Familiarity with Kubernetes",
                "category": "NICE_TO_HAVE",
                "tier": "STRUCTURAL",
                "weight": 1.0,
            },
            {
                "text": "Active non-compete with a direct competitor",
                "category": "DISQUALIFYING",
                "tier": "STRUCTURAL",
                "weight": 0.0,
            },
        ],
        "implicit_expectations": ["ownership"],
        "culture_signals": ["fast-paced"],
    }


def _semantic_payload() -> dict:
    return {
        "inferred_responsibilities": ["led platform migration"],
        "implicit_skills": ["distributed systems"],
        "trajectory_arc": "ACCELERATING",
        "depth_breadth": "SPECIALIST",
    }


def _raw_candidate(title: str, company: str) -> RawCandidate:
    return RawCandidate(
        structured_fields={
            "roles": [
                {
                    "title": title,
                    "company": company,
                    "start": "2018-01",
                    "end": "2023-01",
                }
            ],
            "skills": ["Python", "Postgres", "Kubernetes"],
            "certifications": ["AWS SA"],
        },
        free_text=f"{title} who built and scaled backend services at {company}.",
    )


@pytest.fixture()
def components():
    """Build the real Phase 1 components over stub providers + in-memory store."""

    decomposer = JDDecomposer(provider=FixedLLM(_decompose_payload()))
    enricher = CandidateEnricher(llm_provider=FixedLLM(_semantic_payload()))
    embedder = EmbeddingGenerator(StubEmbeddingProvider(DIM))
    store = InMemoryRankingStore(embedding_dim=DIM)
    return decomposer, enricher, embedder, store


@pytest.fixture()
def pool() -> list[RawCandidate]:
    return [
        _raw_candidate("Backend Engineer", "Acme"),
        _raw_candidate("Senior Backend Engineer", "Globex"),
        _raw_candidate("Staff Engineer", "Initech"),
    ]


# --------------------------------------------------------------------------- #
# Job + parsed RequirementVector persistence (Requirement 2.1, 2.2)
# --------------------------------------------------------------------------- #
async def test_ingest_persists_job_and_parsed_requirement_vector(components, pool):
    decomposer, enricher, embedder, store = components

    result = await ingest(
        "We are hiring a senior backend engineer for payments.",
        "Senior Backend Engineer",
        JobType.TECHNICAL,
        pool,
        decomposer=decomposer,
        enricher=enricher,
        embedder=embedder,
        store=store,
    )

    assert isinstance(result, IngestionResult)

    stored_job = await store.get_job(result.job_id)
    assert stored_job is not None
    assert stored_job.title == "Senior Backend Engineer"
    assert stored_job.job_type is JobType.TECHNICAL
    # Parsed RequirementVector persisted and associated with the job.
    assert stored_job.parsed is not None
    assert stored_job.parsed.job_id == result.job_id
    assert stored_job.parsed.role_intent
    assert len(stored_job.parsed.must_haves) == 2
    assert len(stored_job.parsed.disqualifiers) == 1
    # Every must-have is classified MUST_HAVE.
    assert all(
        r.category is RequirementCategory.MUST_HAVE
        for r in stored_job.parsed.must_haves
    )


# --------------------------------------------------------------------------- #
# Candidate persistence + job association (Requirement 2.1)
# --------------------------------------------------------------------------- #
async def test_ingest_persists_each_candidate_associated_with_job(components, pool):
    decomposer, enricher, embedder, store = components

    result = await ingest(
        "Hiring a senior backend engineer.",
        "Senior Backend Engineer",
        JobType.TECHNICAL,
        pool,
        decomposer=decomposer,
        enricher=enricher,
        embedder=embedder,
        store=store,
    )

    assert result.candidate_count == len(pool)

    listed = await store.list_candidates_for_job(result.job_id)
    assert len(listed) == len(pool)
    assert {c.id for c in listed} == set(result.candidate_ids)

    # Each candidate is individually retrievable.
    for cid in result.candidate_ids:
        assert await store.get_candidate(cid) is not None


# --------------------------------------------------------------------------- #
# One embedding per candidate, correct dimensionality (Requirement 2.2)
# --------------------------------------------------------------------------- #
async def test_ingest_stores_one_embedding_per_candidate_with_correct_dim(
    components, pool
):
    decomposer, enricher, embedder, store = components

    result = await ingest(
        "Hiring a senior backend engineer.",
        "Senior Backend Engineer",
        JobType.TECHNICAL,
        pool,
        decomposer=decomposer,
        enricher=enricher,
        embedder=embedder,
        store=store,
    )

    assert result.embedded_count == len(pool)

    for cid in result.candidate_ids:
        vector = await store.get_embedding(cid)
        assert vector is not None
        assert len(vector) == DIM
        # The embedding generator returns a unit-normalized vector.
        norm = math.sqrt(sum(v * v for v in vector))
        assert norm == pytest.approx(1.0, abs=1e-3)
        # The persisted profile also carries the embedding.
        profile = await store.get_candidate(cid)
        assert profile is not None
        assert profile.embedding is not None
        assert len(profile.embedding) == DIM


# --------------------------------------------------------------------------- #
# Requirement vector embedding for later retrieval (Requirement 6.4)
# --------------------------------------------------------------------------- #
async def test_ingest_embeds_requirement_vector_with_matching_dim(components, pool):
    decomposer, enricher, embedder, store = components

    result = await ingest(
        "Hiring a senior backend engineer.",
        "Senior Backend Engineer",
        JobType.TECHNICAL,
        pool,
        decomposer=decomposer,
        enricher=enricher,
        embedder=embedder,
        store=store,
    )

    assert result.requirement_embedding is not None
    # Candidate and requirement vectors share dimensionality for cosine comparison.
    assert len(result.requirement_embedding) == DIM


async def test_ingest_can_skip_requirement_embedding(components, pool):
    decomposer, enricher, embedder, store = components

    result = await ingest(
        "Hiring a senior backend engineer.",
        "Senior Backend Engineer",
        JobType.TECHNICAL,
        pool,
        decomposer=decomposer,
        enricher=enricher,
        embedder=embedder,
        store=store,
        embed_requirement=False,
    )

    assert result.requirement_embedding is None


# --------------------------------------------------------------------------- #
# Empty pool: job still ingested, no candidates/embeddings
# --------------------------------------------------------------------------- #
async def test_ingest_with_empty_pool_persists_job_only(components):
    decomposer, enricher, embedder, store = components

    result = await ingest(
        "Hiring a senior backend engineer.",
        "Senior Backend Engineer",
        JobType.TECHNICAL,
        [],
        decomposer=decomposer,
        enricher=enricher,
        embedder=embedder,
        store=store,
    )

    assert result.candidate_count == 0
    assert result.embedded_count == 0
    assert await store.get_job(result.job_id) is not None
    assert await store.list_candidates_for_job(result.job_id) == []

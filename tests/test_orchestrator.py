"""Integration tests for the ranking orchestrator (Task 16.1).

These tests exercise :class:`icrs.pipeline.orchestrator.RankingOrchestrator`
end-to-end by composing **stub LLM providers** (decompose / enrich / rerank /
explain) and a **stub EmbeddingProvider** with the **real scoring components**
(hard filter, sub-scores, composite fusion, reranker, confidence). No live API
or database is touched.

They verify the orchestration contract (Requirements 2.1, 2.2, 2.4, 2.5, 2.6,
5.3, 5.6):

    - exactly one RankingResult per hard-filter survivor;
    - disqualified candidates are absent from the results;
    - ranks are unique and contiguous from 1 to N;
    - ordering is monotonic — a higher final score earns a lower rank number;
    - each result carries an explanation and a confidence in [0,1];
    - an empty pool or empty/whitespace JD is rejected and no ranking produced.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

import pytest

from icrs.models.candidate import RawCandidate
from icrs.models.job import JobType
from icrs.pipeline.embedding import EmbeddingGenerator
from icrs.pipeline.enricher import CandidateEnricher
from icrs.pipeline.explanation import ExplanationGenerator
from icrs.pipeline.jd_decomposer import JDDecomposer
from icrs.pipeline.orchestrator import (
    InvalidRankingInputError,
    RankingOrchestrator,
)
from icrs.pipeline.reranker import Reranker
from icrs.providers.base import (
    EmbeddingProvider,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    Vector,
)

# Small embedding dimensionality keeps the tests fast and model-agnostic.
DIM = 8


# --------------------------------------------------------------------------- #
# Stub providers
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


class StubRerankLLM(LLMProvider):
    """Reranker stub: scores every candidate id found in the prompt equally.

    Returning an equal LLM score for each candidate keeps the rerank blend a
    monotonic function of the composite score (the deterministic ordering the
    orchestrator relies on), while still exercising the real reranker code path.
    """

    def __init__(self, *, score: int = 50) -> None:
        self._score = score
        self.calls: list[list[LLMMessage]] = []

    @property
    def model_id(self) -> str:
        return "stub-rerank-llm"

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens=None,
        response_format=None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        user = messages[-1].content
        ids = re.findall(r"id: ([0-9a-fA-F-]{36})", user)
        rankings = [{"id": cid, "score": self._score} for cid in ids]
        return LLMResponse(
            text=json.dumps({"rankings": rankings}), model=self.model_id
        )


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider producing stable DIM-length vectors."""

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
        return [((digest[i % len(digest)] + i + 1) / 255.0) for i in range(self._dim)]

    def embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        return [self.embed(t) for t in texts]


# --------------------------------------------------------------------------- #
# Scripted payloads
# --------------------------------------------------------------------------- #
def _decompose_payload() -> dict:
    """A JD with two MUST_HAVEs, one NICE_TO_HAVE, and one DISQUALIFYING gate.

    Requirement texts are single salient tokens so the deterministic
    token-overlap matcher can resolve them against a candidate's explicit skills.
    """

    return {
        "role_intent": "Build and operate the backend payments platform",
        "seniority_band": "SENIOR",
        "requirements": [
            {
                "text": "Python",
                "category": "MUST_HAVE",
                "tier": "STRUCTURAL",
                "weight": 0.5,
            },
            {
                "text": "Postgres",
                "category": "MUST_HAVE",
                "tier": "STRUCTURAL",
                "weight": 0.5,
            },
            {
                "text": "Kubernetes",
                "category": "NICE_TO_HAVE",
                "tier": "STRUCTURAL",
                "weight": 1.0,
            },
            {
                "text": "COBOL",
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


def _explain_payload() -> dict:
    return {
        "summary": "Strong backend match with relevant platform experience.",
        "driving_signals": ["Coverage of the role's must-have requirements"],
    }


def _raw_candidate(title: str, company: str, skills: list[str]) -> RawCandidate:
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
            "skills": skills,
        },
        free_text=f"{title} who built and scaled backend services at {company}.",
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def orchestrator() -> RankingOrchestrator:
    """An orchestrator wiring stub providers with the real scoring stack."""

    decomposer = JDDecomposer(provider=FixedLLM(_decompose_payload()))
    enricher = CandidateEnricher(llm_provider=FixedLLM(_semantic_payload()))
    embedder = EmbeddingGenerator(StubEmbeddingProvider(DIM))
    reranker = Reranker(provider=StubRerankLLM(), k=10)
    explainer = ExplanationGenerator(provider=FixedLLM(_explain_payload()))
    return RankingOrchestrator(
        decomposer=decomposer,
        enricher=enricher,
        embedder=embedder,
        reranker=reranker,
        explainer=explainer,
    )


@pytest.fixture()
def pool() -> list[RawCandidate]:
    """Two qualified candidates of differing strength + one disqualified."""

    strong = _raw_candidate(
        "Senior Backend Engineer", "Globex", ["Python", "Postgres", "Kubernetes"]
    )
    weak = _raw_candidate("Backend Engineer", "Acme", ["Python"])
    disqualified = _raw_candidate(
        "Mainframe Engineer", "Initech", ["COBOL", "Python"]
    )
    return [strong, weak, disqualified]


_JD_TEXT = "We are hiring a senior backend engineer for the payments platform."


# --------------------------------------------------------------------------- #
# Exactly one result per survivor; disqualified candidates absent (Req 2.4)
# --------------------------------------------------------------------------- #
async def test_one_result_per_survivor_disqualified_absent(orchestrator, pool):
    results = await orchestrator.rank_candidates(_JD_TEXT, pool, JobType.GENERALIST)

    # The COBOL candidate (index 2) is gated out; two survivors remain.
    assert len(results) == 2

    disqualified_id = pool[2].id
    result_ids = {r.candidate_id for r in results}
    assert disqualified_id not in result_ids
    # Both survivors are present exactly once.
    assert result_ids == {pool[0].id, pool[1].id}
    assert len(result_ids) == len(results)


# --------------------------------------------------------------------------- #
# Ranks unique and contiguous from 1 (Requirement 2.5)
# --------------------------------------------------------------------------- #
async def test_ranks_unique_and_contiguous_from_one(orchestrator, pool):
    results = await orchestrator.rank_candidates(_JD_TEXT, pool, JobType.GENERALIST)

    ranks = sorted(r.rank for r in results)
    assert ranks == list(range(1, len(results) + 1))
    assert len(set(ranks)) == len(ranks)


# --------------------------------------------------------------------------- #
# Monotonic ordering: higher final score -> lower rank number (Req 5.3)
# --------------------------------------------------------------------------- #
async def test_monotonic_ordering_by_final_score(orchestrator, pool):
    results = await orchestrator.rank_candidates(_JD_TEXT, pool, JobType.GENERALIST)

    by_rank = sorted(results, key=lambda r: r.rank)
    scores = [r.final_score for r in by_rank]
    # Final scores are non-increasing as rank increases.
    assert scores == sorted(scores, reverse=True)
    # The rank-1 candidate has the maximum final score.
    assert by_rank[0].final_score == max(r.final_score for r in results)


# --------------------------------------------------------------------------- #
# Every result carries an explanation and confidence in [0,1] (Req 2.4)
# --------------------------------------------------------------------------- #
async def test_each_result_has_explanation_and_valid_confidence(orchestrator, pool):
    results = await orchestrator.rank_candidates(_JD_TEXT, pool, JobType.GENERALIST)

    for r in results:
        assert 0.0 <= r.final_score <= 1.0
        assert 0.0 <= r.confidence <= 1.0
        assert r.explanation is not None
        assert r.explanation.summary.strip()  # non-empty recruiter summary
        assert r.explanation.driving_signals  # at least one driving signal
        # The breakdown sub-scores are all valid normalized values.
        for sub in (
            r.breakdown.semantic_fit,
            r.breakdown.career_trajectory,
            r.breakdown.behavioral,
            r.breakdown.hard_filter_pass,
            r.breakdown.disqualifying_penalty,
        ):
            assert 0.0 <= sub <= 1.0


# --------------------------------------------------------------------------- #
# Determinism: identical inputs yield identical ordering (Req 5.6)
# --------------------------------------------------------------------------- #
async def test_repeated_runs_are_deterministic(orchestrator, pool):
    first = await orchestrator.rank_candidates(_JD_TEXT, pool, JobType.GENERALIST)
    second = await orchestrator.rank_candidates(_JD_TEXT, pool, JobType.GENERALIST)

    assert [(r.candidate_id, r.rank) for r in first] == [
        (r.candidate_id, r.rank) for r in second
    ]


# --------------------------------------------------------------------------- #
# Requirement 2.6 — empty pool / empty JD rejection
# --------------------------------------------------------------------------- #
async def test_empty_pool_rejected(orchestrator):
    with pytest.raises(InvalidRankingInputError):
        await orchestrator.rank_candidates(_JD_TEXT, [], JobType.GENERALIST)


async def test_empty_jd_rejected(orchestrator, pool):
    with pytest.raises(InvalidRankingInputError):
        await orchestrator.rank_candidates("   ", pool, JobType.GENERALIST)


async def test_job_id_propagated_to_results(orchestrator, pool):
    results = await orchestrator.rank_candidates(_JD_TEXT, pool, JobType.GENERALIST)

    # Every result is stamped with the same RequirementVector job id.
    job_ids = {r.job_id for r in results}
    assert len(job_ids) == 1

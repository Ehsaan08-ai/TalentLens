"""Resilience / graceful-degradation tests for the ranking orchestrator (Task 16.2).

These failure-injection tests exercise the Task 16.2 error-handling behaviour of
:class:`icrs.pipeline.orchestrator.RankingOrchestrator` by composing **stub LLM
providers** and **failure-injecting stubs** with the **real scoring components**
(hard filter, sub-scores, composite fusion, confidence). No live API, network, or
``sleep`` is exercised — the retry backoff is injected as a recording no-op.

They verify Requirement 9:

    - 9.1: a behavioral-signal source that fails/times out does not abort the
      ranking; the candidate proceeds with behavioral availability 0 and the
      neutral prior (0.5) behavioral sub-score.
    - 9.2 / 9.3: per-candidate embedding is retried with backoff; a transient
      failure that later succeeds keeps the candidate; a candidate that cannot be
      embedded after the retries is excluded while the rest are ranked.
    - 9.4: a reranker failure falls back to composite-score ordering and flags the
      run un-reranked.
    - 9.5: an explanation failure for a candidate marks that candidate's
      explanation unavailable without fabricating content.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

import pytest

from icrs.models.candidate import NormalizedProfile, RawCandidate
from icrs.models.enums import SignalTier
from icrs.models.job import JobType
from icrs.pipeline.embedding import EmbeddingGenerator
from icrs.pipeline.enricher import CandidateEnricher
from icrs.pipeline.enrichment import BehavioralSignalSource
from icrs.pipeline.explanation import ExplanationGenerator
from icrs.pipeline.jd_decomposer import JDDecomposer
from icrs.pipeline.orchestrator import (
    EXPLANATION_UNAVAILABLE_SUMMARY,
    RankingOrchestrator,
    RankingRun,
)
from icrs.pipeline.reranker import RerankError, Reranker
from icrs.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    LLMResponse,
    Vector,
)

DIM = 8


# --------------------------------------------------------------------------- #
# Stub providers (mirroring tests/test_orchestrator.py)
# --------------------------------------------------------------------------- #
class FixedLLM(LLMProvider):
    """An LLM provider that always returns the same scripted JSON payload."""

    def __init__(self, payload: dict, *, model: str = "stub-llm") -> None:
        self._text = json.dumps(payload)
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def complete(
        self, messages, *, temperature=0.0, max_tokens=None, response_format=None
    ) -> LLMResponse:
        return LLMResponse(text=self._text, model=self._model)


class StubRerankLLM(LLMProvider):
    """Reranker stub: scores every candidate id found in the prompt equally."""

    def __init__(self, *, score: int = 50) -> None:
        self._score = score

    @property
    def model_id(self) -> str:
        return "stub-rerank-llm"

    def complete(
        self, messages, *, temperature=0.0, max_tokens=None, response_format=None
    ) -> LLMResponse:
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
# Failure-injecting stubs
# --------------------------------------------------------------------------- #
class _ScriptedEmbedder:
    """An embedder stub that can fail per candidate to exercise 9.2 / 9.3.

    ``always_fail_ids`` always raise on :meth:`embed`; ``transient_fail`` maps a
    candidate id to the number of leading attempts that should raise before the
    call succeeds. Attempt counts are recorded so tests can assert the retry
    budget was honoured.
    """

    def __init__(
        self,
        *,
        always_fail_ids: set | None = None,
        transient_fail: dict | None = None,
        dim: int = DIM,
    ) -> None:
        self.dim = dim
        self.always_fail_ids = set(always_fail_ids or set())
        self.transient_fail = dict(transient_fail or {})
        self.attempts: dict = {}

    def embed_requirement(self, reqs) -> Vector:
        return self._vector("requirement")

    def embed(self, profile) -> Vector:
        pid = profile.id
        self.attempts[pid] = self.attempts.get(pid, 0) + 1
        if pid in self.always_fail_ids:
            raise RuntimeError("embedding service error (permanent)")
        remaining = self.transient_fail.get(pid, 0)
        if remaining > 0:
            self.transient_fail[pid] = remaining - 1
            raise RuntimeError("embedding service error (transient)")
        return self._vector(str(pid))

    def _vector(self, seed: str) -> Vector:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        return [((digest[i % len(digest)] + i + 1) / 255.0) for i in range(self.dim)]


class _RaisingReranker:
    """A reranker stub whose :meth:`rerank` always raises (exercises 9.4)."""

    def rerank(self, topK, reqs):
        raise RerankError("simulated reranker failure")


class _RaisingBehavioralSource(BehavioralSignalSource):
    """A Tier 3 source that always raises, simulating an unavailable fetch (9.1)."""

    def fetch(self, profile: NormalizedProfile):
        raise RuntimeError("behavioral platform unavailable")


class _SelectiveFailingExplainer:
    """Wraps a real explainer but raises for a chosen set of candidate ids (9.5)."""

    def __init__(self, inner: ExplanationGenerator, fail_ids: set) -> None:
        self._inner = inner
        self._fail_ids = set(fail_ids)

    def explain(self, profile, reqs, breakdown=None):
        if profile.id in self._fail_ids:
            raise RuntimeError("explanation service failure")
        return self._inner.explain(profile, reqs, breakdown)


# --------------------------------------------------------------------------- #
# Scripted payloads (single-token requirement texts for the overlap matcher)
# --------------------------------------------------------------------------- #
def _decompose_payload() -> dict:
    return {
        "role_intent": "Build and operate the backend payments platform",
        "seniority_band": "SENIOR",
        "requirements": [
            {"text": "Python", "category": "MUST_HAVE", "tier": "STRUCTURAL", "weight": 0.5},
            {"text": "Postgres", "category": "MUST_HAVE", "tier": "STRUCTURAL", "weight": 0.5},
            {"text": "Kubernetes", "category": "NICE_TO_HAVE", "tier": "STRUCTURAL", "weight": 1.0},
            {"text": "COBOL", "category": "DISQUALIFYING", "tier": "STRUCTURAL", "weight": 0.0},
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
                {"title": title, "company": company, "start": "2018-01", "end": "2023-01"}
            ],
            "skills": skills,
        },
        free_text=f"{title} who built and scaled backend services at {company}.",
    )


_JD_TEXT = "We are hiring a senior backend engineer for the payments platform."


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _enricher(behavioral_source: BehavioralSignalSource | None = None) -> CandidateEnricher:
    return CandidateEnricher(
        llm_provider=FixedLLM(_semantic_payload()),
        behavioral_source=behavioral_source,
    )


def _real_embedder() -> EmbeddingGenerator:
    return EmbeddingGenerator(StubEmbeddingProvider(DIM))


def _build_orchestrator(
    *,
    enricher: CandidateEnricher | None = None,
    embedder=None,
    reranker=None,
    explainer=None,
    sleep=None,
    embed_max_attempts: int = 3,
) -> RankingOrchestrator:
    return RankingOrchestrator(
        decomposer=JDDecomposer(provider=FixedLLM(_decompose_payload())),
        enricher=enricher or _enricher(),
        embedder=embedder or _real_embedder(),
        reranker=reranker or Reranker(provider=StubRerankLLM(), k=10),
        explainer=explainer or ExplanationGenerator(provider=FixedLLM(_explain_payload())),
        sleep=sleep if sleep is not None else (lambda _seconds: None),
        embed_max_attempts=embed_max_attempts,
    )


def _qualified_pool() -> list[RawCandidate]:
    """Three qualified candidates of differing strength (no disqualifier)."""

    strong = _raw_candidate(
        "Senior Backend Engineer", "Globex", ["Python", "Postgres", "Kubernetes"]
    )
    mid = _raw_candidate("Backend Engineer", "Acme", ["Python", "Postgres"])
    weak = _raw_candidate("Backend Engineer", "Hooli", ["Python"])
    return [strong, mid, weak]


# --------------------------------------------------------------------------- #
# 9.1 — behavioral fetch failure => candidate proceeds with neutral prior
# --------------------------------------------------------------------------- #
async def test_behavioral_fetch_failure_proceeds_with_neutral_prior():
    enricher = _enricher(behavioral_source=_RaisingBehavioralSource())
    orchestrator = _build_orchestrator(enricher=enricher)
    pool = _qualified_pool()

    results = await orchestrator.rank_candidates(_JD_TEXT, pool, JobType.GENERALIST)

    # The ranking is produced (the failing behavioral fetch did not abort it).
    assert len(results) == len(pool)
    # Every candidate's behavioral tier fell back to the neutral prior (0.5).
    for r in results:
        assert r.breakdown.behavioral == pytest.approx(0.5)


async def test_raising_behavioral_source_yields_zero_availability():
    """The enricher records behavioral availability 0 when the fetch fails (9.1)."""

    enricher = _enricher(behavioral_source=_RaisingBehavioralSource())
    raw = _raw_candidate("Backend Engineer", "Acme", ["Python"])

    enriched = enricher.enrich(enricher.normalize(raw))

    assert enriched.signal_availability[SignalTier.BEHAVIORAL] == 0.0
    assert enriched.behavioral_signals == []


# --------------------------------------------------------------------------- #
# 9.2 / 9.3 — embedding exclusion and transient-retry inclusion
# --------------------------------------------------------------------------- #
async def test_unembeddable_candidate_excluded_others_ranked():
    pool = _qualified_pool()
    failing_id = pool[1].id  # the middle candidate can never be embedded
    embedder = _ScriptedEmbedder(always_fail_ids={failing_id})
    recorded_sleeps: list[float] = []
    orchestrator = _build_orchestrator(
        embedder=embedder, sleep=recorded_sleeps.append
    )

    run: RankingRun = await orchestrator.rank_candidates_run(
        _JD_TEXT, pool, JobType.GENERALIST
    )

    result_ids = {r.candidate_id for r in run.results}
    # The un-embeddable candidate is excluded; the other two are ranked.
    assert failing_id not in result_ids
    assert result_ids == {pool[0].id, pool[2].id}
    assert failing_id in run.excluded_candidate_ids
    # It was retried the full budget (3 attempts) before exclusion.
    assert embedder.attempts[failing_id] == 3
    # Backoff was slept between the failed attempts (2 gaps for 3 attempts).
    assert len(recorded_sleeps) == 2
    # Ranks remain unique and contiguous over the surviving candidates.
    assert sorted(r.rank for r in run.results) == [1, 2]


async def test_transient_embedding_failure_then_success_includes_candidate():
    pool = _qualified_pool()
    transient_id = pool[1].id  # fails twice, succeeds on the third attempt
    embedder = _ScriptedEmbedder(transient_fail={transient_id: 2})
    orchestrator = _build_orchestrator(embedder=embedder, embed_max_attempts=3)

    run = await orchestrator.rank_candidates_run(_JD_TEXT, pool, JobType.GENERALIST)

    result_ids = {r.candidate_id for r in run.results}
    # The transiently-failing candidate is retried and ultimately included.
    assert transient_id in result_ids
    assert len(run.results) == len(pool)
    assert run.excluded == []
    assert embedder.attempts[transient_id] == 3


# --------------------------------------------------------------------------- #
# 9.4 — reranker failure => composite-ordering fallback, flagged un-reranked
# --------------------------------------------------------------------------- #
async def test_reranker_failure_falls_back_to_composite_ordering():
    pool = _qualified_pool()
    orchestrator = _build_orchestrator(reranker=_RaisingReranker())

    run = await orchestrator.rank_candidates_run(_JD_TEXT, pool, JobType.GENERALIST)

    # The run completed but is honestly flagged un-reranked (9.4).
    assert run.reranked is False
    assert len(run.results) == len(pool)
    # Ordering is still well-formed: unique/contiguous ranks, scores non-increasing.
    ranks = sorted(r.rank for r in run.results)
    assert ranks == list(range(1, len(pool) + 1))
    by_rank = sorted(run.results, key=lambda r: r.rank)
    scores = [r.final_score for r in by_rank]
    assert scores == sorted(scores, reverse=True)


async def test_successful_rerank_flagged_reranked_true():
    """Control: the success path reports ``reranked == True``."""

    orchestrator = _build_orchestrator()
    run = await orchestrator.rank_candidates_run(
        _JD_TEXT, _qualified_pool(), JobType.GENERALIST
    )
    assert run.reranked is True
    assert run.excluded == []
    assert run.explanation_unavailable_ids == []


# --------------------------------------------------------------------------- #
# 9.5 — explanation failure => marked unavailable, no fabricated content
# --------------------------------------------------------------------------- #
async def test_explanation_failure_marked_unavailable_without_fabrication():
    pool = _qualified_pool()
    inner = ExplanationGenerator(provider=FixedLLM(_explain_payload()))
    failing_id = pool[0].id
    explainer = _SelectiveFailingExplainer(inner, fail_ids={failing_id})
    orchestrator = _build_orchestrator(explainer=explainer)

    run = await orchestrator.rank_candidates_run(_JD_TEXT, pool, JobType.GENERALIST)

    # Every survivor is still emitted with a score and rank.
    assert len(run.results) == len(pool)
    assert failing_id in run.explanation_unavailable_ids

    failed = next(r for r in run.results if r.candidate_id == failing_id)
    # The explanation is the honest sentinel — no fabricated rationale/signals/gaps.
    assert failed.explanation.summary == EXPLANATION_UNAVAILABLE_SUMMARY
    assert failed.explanation.driving_signals == []
    assert failed.explanation.gaps == []
    assert failed.explanation.unmet_must_haves == []

    # Other candidates keep their real, generated explanations.
    for r in run.results:
        if r.candidate_id != failing_id:
            assert r.explanation.summary != EXPLANATION_UNAVAILABLE_SUMMARY
            assert r.explanation.driving_signals

"""API-layer tests for the asynchronous ICRS ranking endpoint (Task 17.1).

These tests exercise :func:`icrs.api.app.create_app` through Starlette's
``TestClient`` with a **stub-backed orchestrator** injected via dependency
injection — composing stub LLM providers (decompose / enrich / rerank / explain)
and a stub embedding provider with the **real scoring stack**. No live LLM,
embedding, or database call is ever made.

They verify the HTTP contract:

    - ``POST /rank`` returns 200 with one shortlist entry per hard-filter
      survivor, each populated with the Requirement 5.1 fields (final score,
      per-signal breakdown, recruiter summary, confidence) plus a unique rank;
    - the run-level resilience flags (Requirement 9) are present on the response;
    - an empty candidate pool or an empty / whitespace JD is rejected with HTTP
      400 (Requirement 2.6);
    - ``GET /health`` returns ``ok``.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

import pytest
from starlette.testclient import TestClient

from icrs.api.app import create_app
from icrs.pipeline.embedding import EmbeddingGenerator
from icrs.pipeline.enricher import CandidateEnricher
from icrs.pipeline.explanation import ExplanationGenerator
from icrs.pipeline.jd_decomposer import JDDecomposer
from icrs.pipeline.orchestrator import RankingOrchestrator
from icrs.pipeline.reranker import Reranker
from icrs.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    LLMResponse,
    Vector,
)

DIM = 8


# --------------------------------------------------------------------------- #
# Stub providers (no network) — mirror the orchestrator integration tests
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
        self, messages, *, temperature: float = 0.0, max_tokens=None, response_format=None
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
        self, messages, *, temperature: float = 0.0, max_tokens=None, response_format=None
    ) -> LLMResponse:
        user = messages[-1].content
        ids = re.findall(r"id: ([0-9a-fA-F-]{36})", user)
        rankings = [{"id": cid, "score": self._score} for cid in ids]
        return LLMResponse(text=json.dumps({"rankings": rankings}), model=self.model_id)


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


def _candidate_payload(title: str, company: str, skills: list[str]) -> dict:
    """A RawCandidate-shaped request payload."""

    return {
        "structured_fields": {
            "roles": [
                {"title": title, "company": company, "start": "2018-01", "end": "2023-01"}
            ],
            "skills": skills,
        },
        "free_text": f"{title} who built and scaled backend services at {company}.",
        "external_handles": {},
    }


_JD_TEXT = "We are hiring a senior backend engineer for the payments platform."


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def stub_orchestrator() -> RankingOrchestrator:
    """An orchestrator wiring stub providers with the real scoring stack."""

    return RankingOrchestrator(
        decomposer=JDDecomposer(provider=FixedLLM(_decompose_payload())),
        enricher=CandidateEnricher(llm_provider=FixedLLM(_semantic_payload())),
        embedder=EmbeddingGenerator(StubEmbeddingProvider(DIM)),
        reranker=Reranker(provider=StubRerankLLM(), k=10),
        explainer=ExplanationGenerator(provider=FixedLLM(_explain_payload())),
    )


@pytest.fixture()
def client(stub_orchestrator: RankingOrchestrator) -> TestClient:
    """A TestClient over an app with the stub orchestrator injected."""

    app = create_app(orchestrator=stub_orchestrator)
    return TestClient(app)


def _rank_body(candidates: list[dict], *, raw_jd: str = _JD_TEXT) -> dict:
    return {"raw_jd": raw_jd, "job_type": "GENERALIST", "title": "Backend Engineer", "candidates": candidates}


@pytest.fixture()
def pool_payload() -> list[dict]:
    """Two qualified candidates of differing strength + one disqualified (COBOL)."""

    return [
        _candidate_payload("Senior Backend Engineer", "Globex", ["Python", "Postgres", "Kubernetes"]),
        _candidate_payload("Backend Engineer", "Acme", ["Python"]),
        _candidate_payload("Mainframe Engineer", "Initech", ["COBOL", "Python"]),
    ]


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #
def test_health_returns_ok(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# POST /rank — 200 with one item per survivor + all required fields (Req 5.1)
# --------------------------------------------------------------------------- #
def test_rank_returns_shortlist_with_all_fields(client: TestClient, pool_payload):
    response = client.post("/rank", json=_rank_body(pool_payload))
    assert response.status_code == 200

    body = response.json()
    results = body["results"]

    # The COBOL candidate is gated out; two survivors remain, one entry each.
    assert len(results) == 2

    ranks = sorted(r["rank"] for r in results)
    assert ranks == [1, 2]  # unique + contiguous from 1

    for r in results:
        # Requirement 5.1: every result carries score, breakdown, summary, confidence.
        assert 0.0 <= r["final_score"] <= 1.0
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["candidate_id"]
        breakdown = r["breakdown"]
        for key in (
            "semantic_fit",
            "career_trajectory",
            "behavioral",
            "hard_filter_pass",
            "disqualifying_penalty",
        ):
            assert 0.0 <= breakdown[key] <= 1.0
        explanation = r["explanation"]
        assert explanation["summary"].strip()
        assert isinstance(explanation["driving_signals"], list)
        assert explanation["driving_signals"]
        assert isinstance(explanation["gaps"], list)
        assert isinstance(explanation["unmet_must_haves"], list)

    # Ordering: higher final score earns a lower rank number (Req 5.3).
    by_rank = sorted(results, key=lambda r: r["rank"])
    scores = [r["final_score"] for r in by_rank]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# POST /rank — run-level resilience flags present (Requirement 9)
# --------------------------------------------------------------------------- #
def test_rank_exposes_run_level_flags(client: TestClient, pool_payload):
    response = client.post("/rank", json=_rank_body(pool_payload))
    assert response.status_code == 200

    body = response.json()
    assert body["reranked"] is True
    assert isinstance(body["excluded_candidate_ids"], list)
    assert isinstance(body["explanation_unavailable_ids"], list)
    # On the success path no candidate is excluded or explanation-unavailable.
    assert body["excluded_candidate_ids"] == []
    assert body["explanation_unavailable_ids"] == []


# --------------------------------------------------------------------------- #
# POST /rank — Requirement 2.6: empty pool / empty JD -> HTTP 400
# --------------------------------------------------------------------------- #
def test_rank_empty_pool_returns_400(client: TestClient):
    response = client.post("/rank", json=_rank_body([]))
    assert response.status_code == 400


def test_rank_empty_jd_returns_400(client: TestClient, pool_payload):
    response = client.post("/rank", json=_rank_body(pool_payload, raw_jd="   "))
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Determinism: identical requests yield identical ordering (Req 5.6)
# --------------------------------------------------------------------------- #
def test_rank_is_deterministic(client: TestClient, pool_payload):
    # candidate_id is a system-generated id minted per submitted profile, so it
    # is intentionally NOT stable across two independent requests. The ranking
    # *outcome* (ordering, ranks, scores) must be deterministic for identical
    # input content (Requirement 5.6), so determinism is asserted on those.
    first = client.post("/rank", json=_rank_body(pool_payload)).json()["results"]
    second = client.post("/rank", json=_rank_body(pool_payload)).json()["results"]

    def _ordered_signature(results: list[dict]) -> list[tuple]:
        return [
            (
                r["rank"],
                r["final_score"],
                tuple(sorted(r["breakdown"].items())),
                r["explanation"]["summary"],
            )
            for r in sorted(results, key=lambda x: x["rank"])
        ]

    assert _ordered_signature(first) == _ordered_signature(second)


# --------------------------------------------------------------------------- #
# POST /decompose-jd
# --------------------------------------------------------------------------- #
def test_decompose_jd_returns_success(client: TestClient):
    response = client.post("/decompose-jd", json={"raw_jd": "We need python and postgres"})
    assert response.status_code == 200

    body = response.json()
    assert body["role_intent"] == "Build and operate the backend payments platform"
    assert "Python" in body["must_have"]
    assert "Postgres" in body["must_have"]
    assert "Kubernetes" in body["nice_to_have"]
    assert "ownership" in body["behavioral_signals"]
    assert "fast-paced" in body["behavioral_signals"]


def test_decompose_jd_empty_returns_400(client: TestClient):
    response = client.post("/decompose-jd", json={"raw_jd": "  "})
    assert response.status_code == 400


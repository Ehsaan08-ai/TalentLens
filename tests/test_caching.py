import json
import hashlib
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from icrs.api.app import create_app
from icrs.api.schemas import RankResponse
from icrs.pipeline.orchestrator import RankingOrchestrator, RankingRun
from icrs.persistence.memory import InMemoryRankingStore


class MockRedis:
    def __init__(self, should_fail=False):
        self.store = {}
        self.get_calls = 0
        self.set_calls = 0
        self.should_fail = should_fail

    async def ping(self):
        if self.should_fail:
            raise RuntimeError("Redis connection error")
        return True

    async def get(self, key):
        self.get_calls += 1
        if self.should_fail:
            raise RuntimeError("Redis get error")
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.set_calls += 1
        if self.should_fail:
            raise RuntimeError("Redis set error")
        self.store[key] = value

    async def close(self):
        pass


@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock(spec=RankingOrchestrator)
    # Mock rank_candidates_run
    run = RankingRun(results=[], reranked=True, excluded=[], explanation_unavailable_ids=[])
    orchestrator.rank_candidates_run = AsyncMock(return_value=run)
    return orchestrator


@pytest.fixture
def test_app(mock_orchestrator):
    app = create_app(orchestrator=mock_orchestrator)
    # Manually configure state to avoid automatic lifespan in non-async tests
    app.state.redis_client = MockRedis()
    app.state.postgres_store = InMemoryRankingStore()
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


def _payload():
    return {
        "raw_jd": "We need a Python developer.",
        "job_type": "GENERALIST",
        "title": "Developer",
        "candidates": [
            {
                "structured_fields": {"client_assigned_id": "82610b93-4e7a-4833-be5e-d71a6c3cc066"},
                "free_text": "Experienced in Python.",
                "external_handles": {},
            }
        ]
    }


def _payload_hash(payload):
    payload_dict = {
        "raw_jd": payload["raw_jd"],
        "job_type": payload["job_type"],
        "title": payload["title"],
        "candidates": [
            {
                "structured_fields": c["structured_fields"],
                "free_text": c["free_text"],
                "external_handles": c["external_handles"],
            }
            for c in payload["candidates"]
        ]
    }
    payload_bytes = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest()


def test_cache_miss_runs_orchestrator_and_populates_caches(client, test_app, mock_orchestrator):
    payload = _payload()
    h = _payload_hash(payload)
    redis_key = f"icrs:rank:{h}"

    # Verify both caches are empty initially
    assert test_app.state.redis_client.get_calls == 0
    assert h not in test_app.state.postgres_store._ranking_runs

    # Make request
    response = client.post("/rank", json=payload)
    assert response.status_code == 200

    # Verify orchestrator was called
    mock_orchestrator.rank_candidates_run.assert_called_once()

    # Verify both caches were populated
    assert test_app.state.redis_client.get_calls == 1
    assert test_app.state.redis_client.set_calls == 1
    assert redis_key in test_app.state.redis_client.store
    assert h in test_app.state.postgres_store._ranking_runs


def test_redis_cache_hit_returns_cached_response_directly(client, test_app, mock_orchestrator):
    payload = _payload()
    h = _payload_hash(payload)
    redis_key = f"icrs:rank:{h}"

    # Populate Redis cache manually
    cached_payload = {
        "results": [
            {
                "candidate_id": "82610b93-4e7a-4833-be5e-d71a6c3cc066",
                "rank": 1,
                "final_score": 0.95,
                "breakdown": {
                    "semantic_fit": 0.9,
                    "career_trajectory": 0.9,
                    "behavioral": 0.9,
                    "hard_filter_pass": 1.0,
                    "disqualifying_penalty": 1.0,
                },
                "explanation": {
                    "summary": "Excellent cached candidate.",
                    "driving_signals": ["Fast progress"],
                    "gaps": [],
                    "unmet_must_haves": [],
                },
                "confidence": 0.9,
            }
        ],
        "reranked": True,
        "excluded_candidate_ids": [],
        "explanation_unavailable_ids": [],
    }
    test_app.state.redis_client.store[redis_key] = json.dumps(cached_payload)

    # Make request
    response = client.post("/rank", json=payload)
    assert response.status_code == 200
    body = response.json()

    # Verify response matches cached payload
    assert body["results"][0]["explanation"]["summary"] == "Excellent cached candidate."

    # Verify orchestrator was NEVER called
    mock_orchestrator.rank_candidates_run.assert_not_called()
    assert test_app.state.redis_client.get_calls == 1
    assert test_app.state.redis_client.set_calls == 0


def test_database_hit_redis_miss_populates_redis_and_returns(client, test_app, mock_orchestrator):
    payload = _payload()
    h = _payload_hash(payload)
    redis_key = f"icrs:rank:{h}"

    # Populate DB but leave Redis empty
    db_payload = {
        "results": [
            {
                "candidate_id": "82610b93-4e7a-4833-be5e-d71a6c3cc066",
                "rank": 1,
                "final_score": 0.88,
                "breakdown": {
                    "semantic_fit": 0.8,
                    "career_trajectory": 0.8,
                    "behavioral": 0.8,
                    "hard_filter_pass": 1.0,
                    "disqualifying_penalty": 1.0,
                },
                "explanation": {
                    "summary": "Excellent DB cached candidate.",
                    "driving_signals": ["Steady progress"],
                    "gaps": [],
                    "unmet_must_haves": [],
                },
                "confidence": 0.8,
            }
        ],
        "reranked": True,
        "excluded_candidate_ids": [],
        "explanation_unavailable_ids": [],
    }
    test_app.state.postgres_store._ranking_runs[h] = db_payload

    # Make request
    response = client.post("/rank", json=payload)
    assert response.status_code == 200
    body = response.json()

    # Verify response matches DB payload
    assert body["results"][0]["explanation"]["summary"] == "Excellent DB cached candidate."

    # Verify orchestrator was NEVER called
    mock_orchestrator.rank_candidates_run.assert_not_called()
    # Verify Redis is populated from DB
    assert test_app.state.redis_client.get_calls == 1
    assert test_app.state.redis_client.set_calls == 1
    assert redis_key in test_app.state.redis_client.store


def test_graceful_degradation_when_caches_throw_errors(client, test_app, mock_orchestrator):
    payload = _payload()

    # Configure Redis to raise errors
    test_app.state.redis_client = MockRedis(should_fail=True)

    # Configure DB store to raise errors by mocking get_ranking_run to throw
    test_app.state.postgres_store.get_ranking_run = AsyncMock(side_effect=RuntimeError("DB error"))
    test_app.state.postgres_store.save_ranking_run = AsyncMock(side_effect=RuntimeError("DB error"))

    # Make request - should not crash
    response = client.post("/rank", json=payload)
    assert response.status_code == 200

    # Verify orchestrator was called successfully
    mock_orchestrator.rank_candidates_run.assert_called_once()

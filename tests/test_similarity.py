"""Tests for the Task 10.1 dense / sparse similarity and semantic-fit blend.

Covers Requirement 4.6:
    - dense cosine similarity mapped to [0,1] (identical -> 1.0, opposite -> 0.0)
    - BM25 sparse similarity normalized to [0,1]
    - the 0.7 * dense + 0.3 * sparse semantic-fit blend
    - dense top-N retrieval via the VectorStore abstraction (stubbed)

Both example-based unit tests and Hypothesis property tests are included; the
property tests assert the universal [0,1] output bound that the whole composite
score depends on (supports Property 1, Requirement 4.1).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from icrs.models.candidate import EnrichedProfile, NormalizedProfile, Role
from icrs.models.job import (
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)
from icrs.providers.base import Vector, VectorMatch, VectorRecord, VectorStore
from icrs.scoring.similarity import (
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    SemanticFitMixin,
    candidate_text_corpus,
    cosine_similarity,
    dense_retrieve,
    dense_similarity,
    requirement_query_terms,
    semantic_fit,
    semantic_fit_from_inputs,
    sparse_similarity,
)


# ===== dense cosine similarity ==============================================


def test_identical_vectors_map_to_one() -> None:
    v = [1.0, 2.0, 3.0, 4.0]
    assert dense_similarity(v, v) == pytest.approx(1.0)


def test_parallel_vectors_map_to_one() -> None:
    # Same direction, different magnitude -> cosine 1 -> 1.0.
    assert dense_similarity([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)


def test_opposite_vectors_map_to_zero() -> None:
    assert dense_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(0.0)


def test_orthogonal_vectors_map_to_half() -> None:
    assert dense_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.5)


def test_zero_vector_yields_neutral_half() -> None:
    # Cosine undefined for a zero-magnitude vector -> neutral 0.5.
    assert dense_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == pytest.approx(0.5)
    assert dense_similarity([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.5)


def test_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        dense_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


def test_cosine_similarity_raw_range() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


@given(
    st.lists(st.floats(min_value=-1e3, max_value=1e3), min_size=1, max_size=12),
    st.lists(st.floats(min_value=-1e3, max_value=1e3), min_size=1, max_size=12),
)
def test_dense_similarity_always_in_unit_interval(a: list[float], b: list[float]) -> None:
    # Equalize lengths so the call is well-formed.
    n = min(len(a), len(b))
    score = dense_similarity(a[:n], b[:n])
    assert 0.0 <= score <= 1.0


# ===== sparse BM25 similarity ===============================================


def test_sparse_strong_match_high_score() -> None:
    corpus = [
        "senior python backend engineer",
        "frontend react developer",
        "data analyst sql",
    ]
    score = sparse_similarity("python backend engineer", corpus)
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_sparse_no_lexical_overlap_is_zero() -> None:
    corpus = ["frontend react developer", "graphic designer"]
    score = sparse_similarity("kubernetes rust systems", corpus)
    assert score == pytest.approx(0.0)


def test_sparse_empty_corpus_is_zero() -> None:
    assert sparse_similarity("python engineer", []) == 0.0


def test_sparse_empty_query_is_zero() -> None:
    assert sparse_similarity("", ["python engineer"]) == 0.0


def test_sparse_corpus_of_empty_docs_is_zero() -> None:
    assert sparse_similarity("python", ["", "   "]) == 0.0


def test_sparse_accepts_term_list_query() -> None:
    corpus = ["python backend engineer", "java developer"]
    score = sparse_similarity(["python", "engineer"], corpus)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_sparse_duplicate_terms_normalization() -> None:
    # A query with duplicate terms should normalize to exactly 1.0 against a doc equal to itself.
    # We check it with a mock/approx comparison to ensure it doesn't just hit the 1.0 clamp
    # from a ratio > 1.0.
    from rank_bm25 import BM25Plus
    from icrs.scoring.similarity import _bm25_self_score
    query_tokens = ["python", "python", "backend"]
    corpus = [["python", "python", "backend"]]
    bm25 = BM25Plus(corpus)
    best_real = bm25.get_scores(query_tokens)[0]
    self_score = _bm25_self_score(bm25, query_tokens)
    assert best_real == pytest.approx(self_score)

    # Also verify the high-level function returns exactly 1.0.
    assert sparse_similarity("python python backend", ["python python backend"]) == pytest.approx(1.0)


@given(
    st.lists(
        st.text(alphabet="abcdefghij ", min_size=0, max_size=20),
        min_size=0,
        max_size=6,
    ),
    st.text(alphabet="abcdefghij ", min_size=0, max_size=20),
)
def test_sparse_similarity_always_in_unit_interval(corpus: list[str], query: str) -> None:
    score = sparse_similarity(query, corpus)
    assert 0.0 <= score <= 1.0


# ===== semantic-fit blend ===================================================


def test_blend_weights_are_point_seven_point_three() -> None:
    assert DENSE_WEIGHT == pytest.approx(0.7)
    assert SPARSE_WEIGHT == pytest.approx(0.3)
    assert DENSE_WEIGHT + SPARSE_WEIGHT == pytest.approx(1.0)


def test_blend_exact_combination() -> None:
    assert semantic_fit(1.0, 0.0) == pytest.approx(0.7)
    assert semantic_fit(0.0, 1.0) == pytest.approx(0.3)
    assert semantic_fit(0.8, 0.5) == pytest.approx(0.7 * 0.8 + 0.3 * 0.5)


def test_blend_clamps_out_of_range_inputs() -> None:
    assert semantic_fit(2.0, 2.0) == pytest.approx(1.0)
    assert semantic_fit(-1.0, -1.0) == pytest.approx(0.0)


def test_blend_endpoints() -> None:
    assert semantic_fit(0.0, 0.0) == pytest.approx(0.0)
    assert semantic_fit(1.0, 1.0) == pytest.approx(1.0)


@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_blend_is_convex_combination_in_unit_interval(dense: float, sparse: float) -> None:
    result = semantic_fit(dense, sparse)
    assert 0.0 <= result <= 1.0
    # Convex combination lies between its inputs.
    assert min(dense, sparse) - 1e-9 <= result <= max(dense, sparse) + 1e-9


def test_semantic_fit_from_inputs_end_to_end() -> None:
    req_vec = [1.0, 0.0, 0.0]
    cand_vec = [1.0, 0.0, 0.0]  # identical -> dense 1.0
    corpus = ["python backend engineer"]
    result = semantic_fit_from_inputs(req_vec, cand_vec, "python engineer", corpus)
    assert 0.0 <= result <= 1.0
    # Dense is 1.0; with a positive sparse component the blend exceeds 0.7.
    assert result >= 0.7


# ===== corpus / query helpers ===============================================


def _enriched(
    *,
    roles: list[Role] | None = None,
    skills: list[str] | None = None,
    implicit_skills: list[str] | None = None,
    responsibilities: list[str] | None = None,
) -> EnrichedProfile:
    base = NormalizedProfile(roles=roles or [], explicit_skills=skills or [])
    return EnrichedProfile(
        base=base,
        implicit_skills=implicit_skills or [],
        inferred_responsibilities=responsibilities or [],
    )


def test_candidate_text_corpus_collects_evidence() -> None:
    cand = _enriched(
        roles=[Role(title="Senior Engineer", company="Acme")],
        skills=["Python", "Go"],
        implicit_skills=["Distributed Systems"],
        responsibilities=["Led migration to microservices"],
    )
    corpus = candidate_text_corpus(cand)
    assert "Senior Engineer Acme" in corpus
    assert "Python" in corpus
    assert "Distributed Systems" in corpus
    assert "Led migration to microservices" in corpus


def test_candidate_text_corpus_empty_for_bare_profile() -> None:
    assert candidate_text_corpus(_enriched()) == []


def _req(text: str, category: RequirementCategory) -> Requirement:
    return Requirement(text=text, category=category, tier=RequirementTier.STRUCTURAL, weight=1.0)


def test_requirement_query_terms_excludes_disqualifiers() -> None:
    reqs = RequirementVector(
        role_intent="Build the payments platform",
        seniority_band=SeniorityBand.SENIOR,
        requirements=[
            _req("Python", RequirementCategory.MUST_HAVE),
            _req("Kubernetes", RequirementCategory.NICE_TO_HAVE),
            _req("COBOL", RequirementCategory.DISQUALIFYING),
        ],
    )
    terms = requirement_query_terms(reqs)
    assert "Build the payments platform" in terms
    assert "Python" in terms
    assert "Kubernetes" in terms
    assert "COBOL" not in terms


# ===== dense ANN retrieval via the VectorStore abstraction ==================


class StubVectorStore(VectorStore):
    """In-memory stub VectorStore for testing dense retrieval wiring.

    Ranks stored records by cosine similarity to the query and returns the
    top-N. Depends on nothing external — exercises the retrieval adapter without
    a real Qdrant / pgvector backend.
    """

    def __init__(self) -> None:
        self._collections: dict[str, list[VectorRecord]] = {}
        self.search_calls: list[dict[str, Any]] = []

    async def ensure_collection(self, name: str, dimensionality: int) -> None:
        self._collections.setdefault(name, [])

    async def upsert(self, collection: str, records) -> None:
        bucket = self._collections.setdefault(collection, [])
        by_id = {r.id: r for r in bucket}
        for rec in records:
            by_id[rec.id] = rec
        self._collections[collection] = list(by_id.values())

    async def search(
        self,
        collection: str,
        query: Vector,
        *,
        top_n: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        self.search_calls.append(
            {"collection": collection, "top_n": top_n, "filters": filters}
        )
        records = self._collections.get(collection, [])
        if filters:
            records = [
                r
                for r in records
                if all(r.payload.get(k) == v for k, v in filters.items())
            ]
        scored = [
            VectorMatch(id=r.id, score=cosine_similarity(query, r.vector), payload=r.payload)
            for r in records
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:top_n]


async def _seed(store: StubVectorStore) -> None:
    await store.ensure_collection("candidates", 3)
    await store.upsert(
        "candidates",
        [
            VectorRecord(id="a", vector=[1.0, 0.0, 0.0], payload={"role": "backend"}),
            VectorRecord(id="b", vector=[0.0, 1.0, 0.0], payload={"role": "frontend"}),
            VectorRecord(id="c", vector=[0.9, 0.1, 0.0], payload={"role": "backend"}),
            VectorRecord(id="d", vector=[-1.0, 0.0, 0.0], payload={"role": "backend"}),
        ],
    )


async def test_dense_retrieve_returns_top_n_nearest() -> None:
    store = StubVectorStore()
    await _seed(store)

    matches = await dense_retrieve(store, "candidates", [1.0, 0.0, 0.0], top_n=2)

    assert [m.id for m in matches] == ["a", "c"]
    # Nearest first.
    assert matches[0].score >= matches[1].score


async def test_dense_retrieve_respects_top_n_bound() -> None:
    store = StubVectorStore()
    await _seed(store)

    matches = await dense_retrieve(store, "candidates", [1.0, 0.0, 0.0], top_n=10)
    assert len(matches) == 4  # only four records exist


async def test_dense_retrieve_forwards_filters() -> None:
    store = StubVectorStore()
    await _seed(store)

    matches = await dense_retrieve(
        store, "candidates", [1.0, 0.0, 0.0], top_n=5, filters={"role": "backend"}
    )
    assert {m.id for m in matches} == {"a", "c", "d"}
    assert store.search_calls[-1]["filters"] == {"role": "backend"}


async def test_dense_retrieve_rejects_non_positive_top_n() -> None:
    store = StubVectorStore()
    await _seed(store)
    with pytest.raises(ValueError):
        await dense_retrieve(store, "candidates", [1.0, 0.0, 0.0], top_n=0)


# ===== SemanticFitMixin surface =============================================


class _Engine(SemanticFitMixin):
    """Bare engine composing only the semantic-fit mixin for testing."""


def test_mixin_dense_and_blend_match_functions() -> None:
    engine = _Engine()
    v1, v2 = [1.0, 0.0], [0.0, 1.0]
    assert engine.dense_score(v1, v2) == dense_similarity(v1, v2)

    corpus = ["python engineer", "java developer"]
    assert engine.sparse_score("python engineer", corpus) == sparse_similarity(
        "python engineer", corpus
    )


def test_mixin_sparse_score_accepts_structured_inputs() -> None:
    engine = _Engine()
    reqs = RequirementVector(
        role_intent="Build backend services",
        seniority_band=SeniorityBand.SENIOR,
        requirements=[_req("python backend", RequirementCategory.MUST_HAVE)],
    )
    cand = _enriched(skills=["Python"], roles=[Role(title="Backend Engineer", company="Acme")])

    score = engine.sparse_score(reqs, cand)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_mixin_semantic_fit_score_in_range() -> None:
    engine = _Engine()
    score = engine.semantic_fit_score(
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], "python", ["python engineer"]
    )
    assert 0.0 <= score <= 1.0


async def test_mixin_dense_retrieve_delegates() -> None:
    engine = _Engine()
    store = StubVectorStore()
    await _seed(store)
    matches = await engine.dense_retrieve(store, "candidates", [1.0, 0.0, 0.0], top_n=1)
    assert matches[0].id == "a"

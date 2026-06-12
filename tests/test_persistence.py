"""Unit + property tests for the ICRS persistence layer (Task 6.1).

These tests exercise the async :class:`RankingStore` contract through the
dependency-free :class:`InMemoryRankingStore`, so they require **no** running
PostgreSQL server. They cover round-trip save/get for jobs (including the parsed
RequirementVector), candidate profiles, and per-candidate embeddings; embedding
dimensionality validation against the configured model (Requirement 2.2); and
listing candidates associated with a job.

A small set of structural checks also confirm the PostgreSQL + pgvector
implementation imports cleanly and fails closed (with a clear error) when its
optional dependency stack is unavailable.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from hypothesis import given
from hypothesis import strategies as st

from icrs.models.candidate import EnrichedProfile, NormalizedProfile, Role
from icrs.models.enums import SignalTier, TrajectoryArc
from icrs.models.job import (
    JobDescription,
    JobType,
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)
from icrs.persistence import (
    POSTGRES_AVAILABLE,
    DimensionalityError,
    InMemoryRankingStore,
    PostgresRankingStore,
    RankingStore,
)

# A small embedding dimensionality keeps the tests fast and independent of the
# configured production model size.
DIM = 8


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _requirement_vector(job_id=None) -> RequirementVector:
    return RequirementVector(
        job_id=job_id or uuid4(),
        role_intent="Build and operate the payments platform",
        seniority_band=SeniorityBand.SENIOR,
        requirements=[
            Requirement(
                text="5+ years backend engineering",
                category=RequirementCategory.MUST_HAVE,
                tier=RequirementTier.STRUCTURAL,
                weight=1.0,
            ),
            Requirement(
                text="No active non-compete with a competitor",
                category=RequirementCategory.DISQUALIFYING,
                tier=RequirementTier.STRUCTURAL,
            ),
        ],
        implicit_expectations=["ownership"],
        culture_signals=["fast-paced"],
    )


def _job(parsed: bool = True) -> JobDescription:
    job = JobDescription(
        raw_text="We are hiring a senior backend engineer for payments.",
        title="Senior Backend Engineer",
        job_type=JobType.TECHNICAL,
    )
    if parsed:
        job.parsed = _requirement_vector(job_id=job.id)
    return job


def _enriched(embedding: list[float] | None = None) -> EnrichedProfile:
    return EnrichedProfile(
        base=NormalizedProfile(
            roles=[Role(title="Backend Engineer", company="Acme")],
            explicit_skills=["python", "postgres"],
            total_tenure_months=72,
        ),
        inferred_responsibilities=["led service migration"],
        trajectory_arc=TrajectoryArc.ACCELERATING,
        signal_availability={
            SignalTier.STRUCTURAL: 1.0,
            SignalTier.SEMANTIC: 0.5,
            SignalTier.BEHAVIORAL: 0.0,
        },
        embedding=embedding,
    )


@pytest.fixture()
def store() -> InMemoryRankingStore:
    return InMemoryRankingStore(embedding_dim=DIM)


# --------------------------------------------------------------------------- #
# Job round-trip (incl. parsed RequirementVector)
# --------------------------------------------------------------------------- #


async def test_save_and_get_job_roundtrip(store: RankingStore):
    job = _job(parsed=True)
    await store.save_job(job)

    fetched = await store.get_job(job.id)
    assert fetched is not None
    assert fetched.id == job.id
    assert fetched.raw_text == job.raw_text
    assert fetched.title == job.title
    assert fetched.job_type is JobType.TECHNICAL
    # Parsed RequirementVector survives the round-trip intact.
    assert fetched.parsed is not None
    assert fetched.parsed.role_intent == job.parsed.role_intent
    assert fetched.parsed.seniority_band is SeniorityBand.SENIOR
    assert len(fetched.parsed.must_haves) == 1
    assert len(fetched.parsed.disqualifiers) == 1


async def test_get_job_missing_returns_none(store: RankingStore):
    assert await store.get_job(uuid4()) is None


async def test_saved_job_is_isolated_from_later_mutation(store: RankingStore):
    job = _job(parsed=False)
    await store.save_job(job)
    # Mutating the caller's object after saving must not change stored state.
    job.title = "Mutated Title"

    fetched = await store.get_job(job.id)
    assert fetched is not None
    assert fetched.title == "Senior Backend Engineer"


# --------------------------------------------------------------------------- #
# Candidate round-trip + listing
# --------------------------------------------------------------------------- #


async def test_save_and_get_candidate_roundtrip(store: RankingStore):
    profile = _enriched()
    await store.save_candidate(profile)

    fetched = await store.get_candidate(profile.id)
    assert fetched is not None
    assert fetched.id == profile.id
    assert fetched.base.explicit_skills == ["python", "postgres"]
    assert fetched.trajectory_arc is TrajectoryArc.ACCELERATING
    assert fetched.signal_availability[SignalTier.BEHAVIORAL] == 0.0


async def test_get_candidate_missing_returns_none(store: RankingStore):
    assert await store.get_candidate(uuid4()) is None


async def test_list_candidates_for_job_returns_only_associated(store: RankingStore):
    job_a, job_b = uuid4(), uuid4()
    c1, c2, c3 = _enriched(), _enriched(), _enriched()
    await store.save_candidate(c1, job_id=job_a)
    await store.save_candidate(c2, job_id=job_a)
    await store.save_candidate(c3, job_id=job_b)
    # A candidate saved with no job association is excluded from both listings.
    await store.save_candidate(_enriched())

    listed_a = await store.list_candidates_for_job(job_a)
    listed_b = await store.list_candidates_for_job(job_b)

    assert {c.id for c in listed_a} == {c1.id, c2.id}
    assert {c.id for c in listed_b} == {c3.id}


async def test_list_candidates_for_unknown_job_is_empty(store: RankingStore):
    assert await store.list_candidates_for_job(uuid4()) == []


async def test_saving_candidate_with_embedding_stores_it(store: RankingStore):
    vector = [float(i) / DIM for i in range(DIM)]
    profile = _enriched(embedding=vector)
    await store.save_candidate(profile, job_id=uuid4())

    assert await store.get_embedding(profile.id) == vector


# --------------------------------------------------------------------------- #
# Embedding round-trip + dimensionality validation (Requirement 2.2)
# --------------------------------------------------------------------------- #


async def test_upsert_and_get_embedding_roundtrip(store: RankingStore):
    cid = uuid4()
    vector = [0.1 * i for i in range(DIM)]
    await store.upsert_embedding(cid, vector)

    assert await store.get_embedding(cid) == vector


async def test_get_embedding_missing_returns_none(store: RankingStore):
    assert await store.get_embedding(uuid4()) is None


async def test_upsert_embedding_overwrites_previous(store: RankingStore):
    cid = uuid4()
    await store.upsert_embedding(cid, [0.0] * DIM)
    await store.upsert_embedding(cid, [1.0] * DIM)
    assert await store.get_embedding(cid) == [1.0] * DIM


@pytest.mark.parametrize("wrong_len", [DIM - 1, DIM + 1, 0, 1, 2 * DIM])
async def test_upsert_embedding_rejects_wrong_dimensionality(
    store: RankingStore, wrong_len: int
):
    with pytest.raises(DimensionalityError) as exc:
        await store.upsert_embedding(uuid4(), [0.0] * wrong_len)
    assert exc.value.expected == DIM
    assert exc.value.actual == wrong_len


async def test_save_candidate_rejects_wrong_dimensionality_embedding(store: RankingStore):
    bad = _enriched(embedding=[0.0] * (DIM + 3))
    with pytest.raises(DimensionalityError):
        await store.save_candidate(bad)
    # A rejected save must not leave partial state behind.
    assert await store.get_candidate(bad.id) is None
    assert await store.get_embedding(bad.id) is None


async def test_store_uses_configured_embedding_dim_by_default():
    # When no override is given, the store adopts the configured model dim.
    from icrs.config import get_settings

    default_store = InMemoryRankingStore()
    assert default_store.embedding_dim == get_settings().embedding_dim


# --------------------------------------------------------------------------- #
# Property: any correctly-sized embedding round-trips; any wrong size is rejected
# --------------------------------------------------------------------------- #


@given(
    vector=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=DIM,
        max_size=DIM,
    )
)
async def test_property_correct_dim_embedding_roundtrips(vector):
    store = InMemoryRankingStore(embedding_dim=DIM)
    cid = uuid4()
    await store.upsert_embedding(cid, vector)
    assert await store.get_embedding(cid) == vector


@given(size=st.integers(min_value=0, max_value=64).filter(lambda n: n != DIM))
async def test_property_wrong_dim_embedding_always_rejected(size):
    store = InMemoryRankingStore(embedding_dim=DIM)
    with pytest.raises(DimensionalityError):
        await store.upsert_embedding(uuid4(), [0.0] * size)


# --------------------------------------------------------------------------- #
# PostgreSQL + pgvector implementation structural checks
# --------------------------------------------------------------------------- #


def test_postgres_store_importable_and_guards_missing_stack():
    """The pgvector store must import cleanly and fail closed without its stack."""
    if POSTGRES_AVAILABLE:
        # Stack present: the class is constructible (no live DB call on init is
        # asserted here beyond engine creation, which is lazy).
        assert PostgresRankingStore is not None
    else:
        # Stack absent (this environment): constructing must raise a clear error,
        # never an opaque ImportError at call time.
        with pytest.raises(RuntimeError):
            PostgresRankingStore()


def test_inmemory_is_a_ranking_store():
    assert issubclass(InMemoryRankingStore, RankingStore)
    assert issubclass(PostgresRankingStore, RankingStore)

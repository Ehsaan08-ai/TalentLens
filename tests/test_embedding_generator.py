"""Unit tests for the Embedding Generator (Task 5.1).

These tests use a deterministic stub :class:`EmbeddingProvider` so the generator
is exercised without any concrete model. They cover:

* the short-profile single-embed path (unit norm + correct dimensionality),
* the long-profile chunked path (chunk count aligns to sections, not fixed size),
* weighted aggregation (recent >= older role weights; unit-norm result),
* requirement embedding sharing the provider's dimensionality.

Requirements: 6.1, 6.2, 6.3, 6.4.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta

import pytest

from icrs.models.candidate import (
    EnrichedProfile,
    NormalizedProfile,
    Role,
)
from icrs.models.enums import DepthBreadth, TrajectoryArc
from icrs.models.job import (
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)
from icrs.pipeline.embedding import Chunk, EmbeddingGenerator, default_token_count
from icrs.providers.base import EmbeddingProvider, Vector

DIM = 16


class StubEmbeddingProvider(EmbeddingProvider):
    """Deterministic, dependency-free embedding provider for tests.

    Each text is mapped to a fixed pseudo-random unit-ish vector derived from a
    hash of its content, so embeddings are reproducible and distinct per text
    without any ML dependency.
    """

    def __init__(self, *, dim: int = DIM, max_tokens: int = 64) -> None:
        self._dim = dim
        self._max_tokens = max_tokens
        self.embed_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "stub-embedder"

    @property
    def dimensionality(self) -> int:
        return self._dim

    @property
    def max_input_tokens(self) -> int:
        return self._max_tokens

    def _vector_for(self, text: str) -> Vector:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Build a deterministic vector in roughly [-1, 1] from the digest bytes.
        return [
            ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self._dim)
        ]

    def embed(self, text: str) -> Vector:
        self.embed_calls.append(text)
        return self._vector_for(text)

    def embed_batch(self, texts):
        texts = list(texts)
        self.batch_calls.append(texts)
        return [self._vector_for(t) for t in texts]


def _l2_norm(vector: Vector) -> float:
    return math.sqrt(sum(v * v for v in vector))


def _make_profile(roles: list[Role], *, skills: list[str] | None = None) -> EnrichedProfile:
    base = NormalizedProfile(
        roles=roles,
        explicit_skills=skills or [],
        total_tenure_months=24,
    )
    return EnrichedProfile(
        base=base,
        inferred_responsibilities=["led a team", "owned delivery"],
        implicit_skills=["mentoring"],
        trajectory_arc=TrajectoryArc.ACCELERATING,
        depth_breadth=DepthBreadth.BALANCED,
    )


def _make_requirements() -> RequirementVector:
    return RequirementVector(
        role_intent="Build and scale backend services",
        seniority_band=SeniorityBand.SENIOR,
        requirements=[
            Requirement(
                text="5+ years Python",
                category=RequirementCategory.MUST_HAVE,
                tier=RequirementTier.STRUCTURAL,
                weight=1.0,
            ),
            Requirement(
                text="Distributed systems experience",
                category=RequirementCategory.NICE_TO_HAVE,
                tier=RequirementTier.SEMANTIC,
                weight=1.0,
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# Requirement 6.1 — short profile single-embed path
# --------------------------------------------------------------------------- #


def test_short_profile_single_embed_unit_norm_and_dim():
    provider = StubEmbeddingProvider(max_tokens=10_000)
    gen = EmbeddingGenerator(provider)
    profile = _make_profile(
        [Role(title="Engineer", company="Acme", start=date(2020, 1, 1))],
        skills=["python"],
    )

    vector = gen.embed(profile)

    assert len(vector) == provider.dimensionality
    assert _l2_norm(vector) == pytest.approx(1.0, abs=1e-3)
    # Short path uses a single embed() call and never the batch path.
    assert len(provider.embed_calls) == 1
    assert provider.batch_calls == []


# --------------------------------------------------------------------------- #
# Requirement 6.2 — long profile chunked path aligns to sections
# --------------------------------------------------------------------------- #


def test_long_profile_uses_section_aligned_chunks():
    # Force chunking by setting a tiny token budget.
    provider = StubEmbeddingProvider(max_tokens=1)
    gen = EmbeddingGenerator(provider)

    roles = [
        Role(title="Junior Dev", company="OldCo", start=date(2010, 1, 1), end=date(2013, 1, 1)),
        Role(title="Senior Dev", company="MidCo", start=date(2013, 2, 1), end=date(2018, 1, 1)),
        Role(title="Staff Eng", company="NewCo", start=date(2018, 2, 1)),
    ]
    profile = _make_profile(roles, skills=["python", "go"])

    vector = gen.embed(profile)

    # Expected sections: 3 roles + education(0) + skills(1) + summary(1) = 5.
    expected_sections = len(gen._profile_sections(profile))
    assert expected_sections == 5
    # The chunked path embeds exactly one block per section (not fixed-size).
    assert len(provider.batch_calls) == 1
    assert len(provider.batch_calls[0]) == expected_sections
    assert len(vector) == provider.dimensionality
    assert _l2_norm(vector) == pytest.approx(1.0, abs=1e-3)


def test_chunk_count_changes_with_profile_structure_not_fixed_size():
    provider = StubEmbeddingProvider(max_tokens=1)
    gen = EmbeddingGenerator(provider)

    one_role = _make_profile([Role(title="A", company="X", start=date(2020, 1, 1))])
    three_roles = _make_profile(
        [
            Role(title="A", company="X", start=date(2016, 1, 1), end=date(2018, 1, 1)),
            Role(title="B", company="Y", start=date(2018, 2, 1), end=date(2020, 1, 1)),
            Role(title="C", company="Z", start=date(2020, 2, 1)),
        ]
    )

    # More roles -> more role chunks: chunk count tracks structure, not length.
    assert len(gen._profile_sections(three_roles)) - len(
        gen._profile_sections(one_role)
    ) == 2


# --------------------------------------------------------------------------- #
# Requirement 6.3 — weighted aggregation: recent >= older; unit norm
# --------------------------------------------------------------------------- #


def test_recent_role_weight_at_least_older_role_weight():
    provider = StubEmbeddingProvider(max_tokens=1)
    gen = EmbeddingGenerator(provider)

    old_role = Role(title="Old", company="X", start=date(2005, 1, 1), end=date(2007, 1, 1))
    recent_role = Role(title="Recent", company="Y", start=date(2021, 1, 1), end=date(2023, 1, 1))
    profile = _make_profile([old_role, recent_role])

    sections = gen._profile_sections(profile)
    weights = gen.chunk_weights(sections)

    # Index 0 is the old role, index 1 is the recent role.
    assert sections[0].kind == "role"
    assert sections[1].kind == "role"
    assert weights[1] >= weights[0]
    # All weights are non-negative.
    assert all(w >= 0.0 for w in weights)


def test_weighted_mean_result_is_unit_norm():
    provider = StubEmbeddingProvider(max_tokens=1)
    gen = EmbeddingGenerator(provider)
    profile = _make_profile(
        [
            Role(title="A", company="X", start=date(2015, 1, 1), end=date(2018, 1, 1)),
            Role(title="B", company="Y", start=date(2018, 2, 1)),
        ],
        skills=["python"],
    )

    vector = gen.embed(profile)

    assert _l2_norm(vector) == pytest.approx(1.0, abs=1e-3)


def test_ongoing_role_weighted_at_least_as_old_ended_role():
    provider = StubEmbeddingProvider()
    gen = EmbeddingGenerator(provider)
    today = date.today()

    ended = Chunk(kind="role", text="old", reference_date=today - timedelta(days=4000))
    ongoing = Chunk(kind="role", text="cur", reference_date=today)
    weights = gen.chunk_weights([ended, ongoing])

    assert weights[1] >= weights[0]


# --------------------------------------------------------------------------- #
# Requirement 6.4 — candidate and requirement vectors share dimensionality
# --------------------------------------------------------------------------- #


def test_requirement_embedding_shares_dimensionality_with_profile():
    provider = StubEmbeddingProvider(max_tokens=10_000)
    gen = EmbeddingGenerator(provider)

    profile = _make_profile([Role(title="Eng", company="Acme", start=date(2020, 1, 1))])
    reqs = _make_requirements()

    cand_vec = gen.embed(profile)
    req_vec = gen.embed_requirement(reqs)

    assert len(cand_vec) == provider.dimensionality
    assert len(req_vec) == provider.dimensionality
    assert len(cand_vec) == len(req_vec)
    assert _l2_norm(req_vec) == pytest.approx(1.0, abs=1e-3)


def test_embed_requirement_chunked_path_unit_norm():
    provider = StubEmbeddingProvider(max_tokens=1)
    gen = EmbeddingGenerator(provider)
    reqs = _make_requirements()

    req_vec = gen.embed_requirement(reqs)

    # role_intent + 2 requirements = 3 chunks via the batch path.
    assert len(provider.batch_calls) == 1
    assert len(provider.batch_calls[0]) == 3
    assert _l2_norm(req_vec) == pytest.approx(1.0, abs=1e-3)


def test_embedRequirement_alias_matches_embed_requirement():
    provider = StubEmbeddingProvider(max_tokens=10_000)
    gen = EmbeddingGenerator(provider)
    reqs = _make_requirements()

    assert gen.embedRequirement(reqs) == gen.embed_requirement(reqs)


# --------------------------------------------------------------------------- #
# Token-count heuristic (injectable)
# --------------------------------------------------------------------------- #


def test_default_token_count_heuristic():
    assert default_token_count("") == 0
    assert default_token_count("abcd") == 1
    assert default_token_count("a" * 9) == 3


def test_injected_token_counter_controls_chunking():
    provider = StubEmbeddingProvider(max_tokens=100)
    # A token counter that always reports a huge count forces the chunked path
    # regardless of actual text length.
    gen = EmbeddingGenerator(provider, token_counter=lambda _t: 10_000)
    profile = _make_profile([Role(title="Eng", company="Acme", start=date(2020, 1, 1))])

    gen.embed(profile)

    assert len(provider.batch_calls) == 1
    assert provider.embed_calls == []

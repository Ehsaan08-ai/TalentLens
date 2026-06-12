"""Embedding generation with section-aligned chunking (Task 5.1).

The :class:`EmbeddingGenerator` turns an
:class:`~icrs.models.candidate.EnrichedProfile` or a
:class:`~icrs.models.job.RequirementVector` into a single dense vector that is
directly comparable by cosine similarity, because both are produced by the *same*
injected :class:`~icrs.providers.base.EmbeddingProvider` (same model, same
dimensionality) — Requirement 6.4.

Two embedding paths (mirroring the design's ``embed`` algorithm):

* **Short input** — when the serialized text fits within the provider's
  ``max_input_tokens`` it is embedded in one call and L2-normalized to a unit
  vector (Requirement 6.1).
* **Long input** — otherwise the text is split into *section-aligned* chunks
  (one role or one section block per chunk, never fixed-size character/token
  splits — Requirement 6.2). Each chunk is embedded, the chunk vectors are
  combined by a **recency/relevance-weighted mean** in which a more recent or
  more role-relevant chunk receives a weight greater than or equal to that of a
  less recent / less relevant one (Requirement 6.3), and the aggregate is
  L2-normalized to a unit vector.

This module depends only on the abstract :class:`EmbeddingProvider` interface; no
concrete model (sentence-transformers, OpenAI, ...) is imported here, so the
generator can be unit-tested with a deterministic stub provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Callable, Sequence

from icrs.models.candidate import EnrichedProfile, Role
from icrs.models.job import RequirementVector
from icrs.providers.base import EmbeddingProvider, Vector

# L2 norm tolerance for the unit-vector guarantee (Requirements 6.1, 6.3).
NORM_TOLERANCE = 1e-3

# Reference for the average length (in characters) of one token. Used only by
# the default, injectable token-count heuristic; the PoC does not need an exact
# tokenizer to decide when chunking is required.
_CHARS_PER_TOKEN = 4

# Relevance priors per section kind. Relevance defaults to uniform for the PoC
# (every block is equally job-relevant a priori); recency differentiates role
# chunks. Kept as named constants so the weighting policy is explicit and the
# "recent/relevant >= older/less-relevant" invariant is easy to reason about.
_ROLE_RELEVANCE = 1.0
_SECTION_RELEVANCE = 1.0


def default_token_count(text: str) -> int:
    """Estimate the token count of ``text`` with a cheap char-based heuristic.

    This is intentionally approximate (≈ four characters per token) — it only
    has to decide *whether* an input is large enough to require section-aligned
    chunking, not to reproduce a specific tokenizer. It is injectable/overridable
    via the :class:`EmbeddingGenerator` constructor so a real tokenizer can be
    substituted later.
    """

    if not text:
        return 0
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Chunk:
    """A section-aligned unit of text to embed.

    Each chunk corresponds to exactly one role or one section block (never a
    fixed-size slice). ``reference_date`` drives recency weighting for role
    chunks (more recent => higher weight); section chunks without a date are
    treated as current. ``relevance`` is the (uniform, for the PoC) role-relevance
    prior.
    """

    kind: str  # "role" | "education" | "skills" | "summary" | "requirement" | ...
    text: str
    reference_date: date | None = None
    relevance: float = 1.0


class EmbeddingGenerator:
    """Produces unit-normalized embeddings for profiles and requirement vectors.

    The generator depends only on the abstract
    :class:`~icrs.providers.base.EmbeddingProvider`; the concrete model is
    injected so candidate and requirement vectors always share model and
    dimensionality (Requirement 6.4).
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        """Construct the generator.

        Args:
            provider: the embedding provider supplying ``embed``/``embed_batch``,
                ``dimensionality``, and ``max_input_tokens``.
            token_counter: optional override for the token-count heuristic used
                to decide when chunking is required. Defaults to
                :func:`default_token_count`.
        """

        self._provider = provider
        self._count_tokens = token_counter or default_token_count

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def embed(self, profile: EnrichedProfile) -> Vector:
        """Embed an enriched profile into a single unit-normalized vector.

        When the serialized profile fits within the provider's token limit it is
        embedded in one call (Requirement 6.1). Otherwise it is split into
        section-aligned chunks and aggregated by a recency/relevance-weighted
        mean (Requirements 6.2, 6.3). The returned vector always has the
        provider's dimensionality and unit L2 norm (±0.001).
        """

        sections = self._profile_sections(profile)
        return self._embed_sections(sections)

    def embed_requirement(self, reqs: RequirementVector) -> Vector:
        """Embed a requirement vector into a single unit-normalized vector.

        Uses the *same* provider (model + dimensionality) as :meth:`embed`, so
        the resulting requirement vector is directly cosine-comparable with any
        candidate vector (Requirement 6.4). Short requirement text is embedded in
        one call; long text is chunked per requirement/section block and
        aggregated by a weighted mean.
        """

        sections = self._requirement_sections(reqs)
        return self._embed_sections(sections)

    # Design/spec traceability alias (the design names this ``embedRequirement``).
    embedRequirement = embed_requirement

    # ------------------------------------------------------------------ #
    # Core embed-or-chunk-and-aggregate logic
    # ------------------------------------------------------------------ #

    def _embed_sections(self, sections: list[Chunk]) -> Vector:
        """Embed pre-built section chunks, single-shot or chunked-and-aggregated."""

        if not sections:
            # Defensive: never call the provider with nothing. An empty input
            # yields a zero vector of the provider's dimensionality.
            return [0.0] * self._provider.dimensionality

        full_text = self._join_sections(sections)

        # Short path: one block fits within the limit -> single embedding.
        if self._count_tokens(full_text) <= self._provider.max_input_tokens:
            return self._l2_normalize(self._provider.embed(full_text))

        # Long path: section-aligned chunks (one role/section block per chunk),
        # never fixed-size splits (Requirement 6.2).
        texts = [chunk.text for chunk in sections]
        vectors = self._provider.embed_batch(texts)
        weights = self.chunk_weights(sections)
        aggregate = self._weighted_mean(vectors, weights)
        return self._l2_normalize(aggregate)

    # ------------------------------------------------------------------ #
    # Section construction (role/section-aligned, never fixed-size)
    # ------------------------------------------------------------------ #

    def _profile_sections(self, profile: EnrichedProfile) -> list[Chunk]:
        """Build section-aligned chunks for an enriched profile.

        Produces one chunk per role plus separate chunks for the education,
        skills, and summary sections. Empty sections are omitted so the chunk
        count reflects the profile's actual structure (not a fixed number).
        """

        base = profile.base
        chunks: list[Chunk] = []

        # One chunk per role (the primary, recency-weighted blocks).
        for role in base.roles:
            chunks.append(
                Chunk(
                    kind="role",
                    text=self._serialize_role(role),
                    reference_date=self._role_reference_date(role),
                    relevance=_ROLE_RELEVANCE,
                )
            )

        # Education section (one combined block).
        if base.education:
            edu_lines = []
            for edu in base.education:
                parts = [
                    p
                    for p in (edu.degree, edu.field_of_study, edu.institution)
                    if p
                ]
                if parts:
                    edu_lines.append(" - ".join(parts))
            if edu_lines:
                chunks.append(
                    Chunk(
                        kind="education",
                        text="Education: " + "; ".join(edu_lines),
                        relevance=_SECTION_RELEVANCE,
                    )
                )

        # Skills section: explicit + implicit skills + certifications.
        skill_terms = [
            *base.explicit_skills,
            *profile.implicit_skills,
            *base.certifications,
        ]
        if skill_terms:
            chunks.append(
                Chunk(
                    kind="skills",
                    text="Skills: " + ", ".join(skill_terms),
                    relevance=_SECTION_RELEVANCE,
                )
            )

        # Summary section: inferred Tier-2 signals.
        summary_parts: list[str] = []
        if profile.inferred_responsibilities:
            summary_parts.append(
                "Responsibilities: " + "; ".join(profile.inferred_responsibilities)
            )
        if profile.trajectory_arc is not None:
            summary_parts.append(f"Trajectory: {profile.trajectory_arc.value}")
        if profile.depth_breadth is not None:
            summary_parts.append(f"Profile: {profile.depth_breadth.value}")
        if summary_parts:
            chunks.append(
                Chunk(
                    kind="summary",
                    text=" | ".join(summary_parts),
                    relevance=_SECTION_RELEVANCE,
                )
            )

        return chunks

    def _requirement_sections(self, reqs: RequirementVector) -> list[Chunk]:
        """Build section-aligned chunks for a requirement vector.

        Produces a role-intent chunk, one chunk per requirement, and blocks for
        implicit expectations and culture signals. Relevance is uniform; there is
        no recency dimension for requirements.
        """

        chunks: list[Chunk] = [
            Chunk(
                kind="role_intent",
                text=f"Role intent ({reqs.seniority_band.value}): {reqs.role_intent}",
                relevance=_SECTION_RELEVANCE,
            )
        ]

        for requirement in reqs.requirements:
            chunks.append(
                Chunk(
                    kind="requirement",
                    text=f"[{requirement.category.value}] {requirement.text}",
                    relevance=_SECTION_RELEVANCE,
                )
            )

        if reqs.implicit_expectations:
            chunks.append(
                Chunk(
                    kind="implicit_expectations",
                    text="Implicit expectations: "
                    + "; ".join(reqs.implicit_expectations),
                    relevance=_SECTION_RELEVANCE,
                )
            )

        if reqs.culture_signals:
            chunks.append(
                Chunk(
                    kind="culture_signals",
                    text="Culture signals: " + "; ".join(reqs.culture_signals),
                    relevance=_SECTION_RELEVANCE,
                )
            )

        return chunks

    @staticmethod
    def _serialize_role(role: Role) -> str:
        """Render a single role as a self-contained text block."""

        start = role.start.isoformat() if role.start else "?"
        end = role.end.isoformat() if role.end else "present"
        return f"{role.title} at {role.company} ({start} - {end})"

    def _role_reference_date(self, role: Role) -> date:
        """Pick the date used for a role's recency weighting.

        Prefers the end date; an ongoing role (no end date) is treated as the
        most recent (today). Falls back to the start date, then to today when no
        date is present so the chunk is still weighted as current rather than
        being silently dropped.
        """

        if role.end is not None:
            return role.end
        # Ongoing role, or no end recorded -> as recent as today.
        if role.start is not None and role.end is None:
            return self._today()
        if role.start is not None:
            return role.start
        return self._today()

    # ------------------------------------------------------------------ #
    # Weighting
    # ------------------------------------------------------------------ #

    def chunk_weights(self, chunks: Sequence[Chunk]) -> list[float]:
        """Return the non-negative aggregation weight for each chunk.

        A chunk's weight is its relevance prior scaled by a recency factor that
        is monotonically non-increasing with the age of the chunk's reference
        date. With uniform relevance this guarantees that a more recent chunk
        receives a weight greater than or equal to an older one (Requirement
        6.3). Chunks without a reference date (non-role sections) are treated as
        current (recency factor 1.0).
        """

        today = self._today()
        return [self._chunk_weight(chunk, today) for chunk in chunks]

    @staticmethod
    def _chunk_weight(chunk: Chunk, today: date) -> float:
        """Compute a single chunk's non-negative weight (relevance × recency)."""

        if chunk.reference_date is None:
            recency = 1.0
        else:
            age_days = max((today - chunk.reference_date).days, 0)
            age_years = age_days / 365.25
            # 1/(1+age) is in (0,1], strictly decreasing in age => monotone
            # non-increasing weight with age (recent >= older).
            recency = 1.0 / (1.0 + age_years)
        return max(chunk.relevance, 0.0) * recency

    # ------------------------------------------------------------------ #
    # Vector math
    # ------------------------------------------------------------------ #

    def _weighted_mean(
        self, vectors: Sequence[Vector], weights: Sequence[float]
    ) -> Vector:
        """Compute the weighted mean of equal-length vectors.

        Falls back to a uniform mean when all weights are zero so a degenerate
        weighting never collapses the aggregate to a zero vector.
        """

        if not vectors:
            return [0.0] * self._provider.dimensionality
        if len(vectors) != len(weights):
            raise ValueError("vectors and weights must have equal length")

        dim = len(vectors[0])
        total_weight = sum(weights)
        if total_weight <= 0.0:
            # Degenerate: treat as a uniform mean.
            weights = [1.0] * len(vectors)
            total_weight = float(len(vectors))

        acc = [0.0] * dim
        for vector, weight in zip(vectors, weights):
            if len(vector) != dim:
                raise ValueError("all chunk vectors must share dimensionality")
            for i, value in enumerate(vector):
                acc[i] += value * weight

        return [value / total_weight for value in acc]

    @staticmethod
    def _l2_normalize(vector: Vector) -> Vector:
        """Return ``vector`` scaled to unit L2 norm.

        A zero vector cannot be normalized and is returned unchanged (its norm
        is already well-defined as 0); all non-degenerate vectors come back with
        an L2 norm of 1.0 within ``NORM_TOLERANCE`` (Requirements 6.1, 6.3).
        """

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return list(vector)
        return [value / norm for value in vector]

    @staticmethod
    def _join_sections(sections: Sequence[Chunk]) -> str:
        """Join section chunks into a single canonical, signal-ordered text."""

        return "\n\n".join(chunk.text for chunk in sections)

    @staticmethod
    def _today() -> date:
        """Return today's date (isolated for deterministic testing/overriding)."""

        return date.today()


__all__ = [
    "NORM_TOLERANCE",
    "Chunk",
    "EmbeddingGenerator",
    "default_token_count",
]

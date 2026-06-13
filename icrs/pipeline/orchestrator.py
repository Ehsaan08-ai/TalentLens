"""Ranking orchestrator — full pipeline wiring + rank assignment (Task 16.1).

This module implements :class:`RankingOrchestrator`, the Layer-6 component that
coordinates every prior pipeline stage into a single end-to-end ranking run
(design "Component 6: Ranking Orchestrator" and "Main ranking algorithm"). It
composes the already-implemented stages — JD decomposition, candidate
enrichment, embedding, the deterministic hard-filter gate, composite scoring,
LLM contextual reranking, and explanation generation — and emits exactly one
:class:`~icrs.models.ranking.RankingResult` per surviving candidate.

Behavioral contract (Requirements 2.1, 2.2, 2.4, 2.5, 2.6, 5.3, 5.6):
    - 2.1: stages run in the order decompose → enrich → hard filter → composite
      score → rerank → explain, returning a list of ranking results.
    - 2.2: each candidate completes normalize → enrich → embed before it enters
      composite scoring.
    - 2.4: exactly one result per hard-filter survivor, each carrying a final
      score in ``[0,1]``, a unique integer rank, a signal breakdown, an
      explanation, and a confidence in ``[0,1]``.
    - 2.5 / 5.3 / 5.6: ranks are unique and contiguous from ``1`` to ``N``; a
      strictly higher final score earns a numerically lower rank; equal final
      scores are broken deterministically (by candidate id) so identical inputs
      always yield identical ordering.
    - 2.6: an empty / whitespace-only JD or an empty candidate pool is rejected
      with a typed :class:`InvalidRankingInputError` and no ranking is produced.

Scope (success path only): the resilience / graceful-degradation behaviour
(behavioral-fetch timeout, embedding retry/exclusion, reranker fallback,
explanation-unavailable) is Task 16.2. This module is deliberately structured
with one private method per stage (``_decompose`` … ``_explain_and_assemble``)
and clear stage boundaries so 16.2 can wrap each stage with error handling
without restructuring the success-path wiring.

Resilience (Task 16.2, Requirement 9): each stage is wrapped so a single
failure degrades gracefully instead of aborting the whole ranking —

    - 9.1 (behavioral fetch): a behavioral source that times out (10s) or is
      otherwise unavailable does not abort enrichment; the candidate proceeds
      with behavioral availability 0 and the neutral prior. This guarantee lives
      in the :class:`~icrs.pipeline.enricher.CandidateEnricher` (per the
      requirement's allocation to the Candidate_Enricher); the orchestrator
      relies on and is verified against it.
    - 9.2 / 9.3 (embedding): per-candidate embedding is retried up to 3 times
      with increasing backoff, a cached embedding is reused instead of retrying
      when available, and a candidate that still cannot be embedded is *excluded*
      while every successfully-embedded candidate proceeds.
    - 9.4 (reranker): a reranker failure falls back to composite-score ordering
      and the run is flagged un-reranked.
    - 9.5 (explanation): an explanation failure for a candidate marks that
      candidate's explanation unavailable without fabricating any content.

Result shape & back-compat: :meth:`rank_candidates` keeps returning
``list[RankingResult]`` for existing callers. The structured
:meth:`rank_candidates_run` returns a :class:`RankingRun` that additionally
surfaces the un-reranked flag, the explanation-unavailable candidate ids, and
the excluded candidates so the output layer (Task 19.1) can present them
honestly.

Dependency injection: every collaborator (decomposer, enricher, embedder,
reranker, explainer, and an optional store) is supplied via the constructor, so
the orchestrator depends only on the component contracts and is fully testable
with stub providers and real scoring components.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from icrs.models.candidate import EnrichedProfile, NormalizedProfile, RawCandidate
from icrs.models.job import JobType, RequirementVector
from icrs.models.ranking import Explanation, RankingResult, SignalBreakdown
from icrs.pipeline.reranker import Reranker, ScoredCandidate
from icrs.providers.base import Vector
from icrs.scoring.composite import build_signal_bundle, composite
from icrs.scoring.confidence import compute_confidence_for
from icrs.scoring.hard_filter import hard_filter
from icrs.scoring.similarity import (
    candidate_text_corpus,
    requirement_query_terms,
    semantic_fit_from_inputs,
)
from icrs.scoring.weights import WeightProfile, select_weight_profile

# Default retry budget and backoff for per-candidate embedding (Requirement 9.2):
# up to 3 attempts with an increasing (exponential) backoff between them.
DEFAULT_EMBED_MAX_ATTEMPTS = 3
DEFAULT_EMBED_BACKOFF_BASE_SECONDS = 0.5

# Honest, non-fabricated summary used when explanation generation fails for a
# candidate (Requirement 9.5). It states unavailability rather than inventing
# any rationale, driving signals, or gaps.
EXPLANATION_UNAVAILABLE_SUMMARY = (
    "Explanation unavailable: the explanation service could not generate a "
    "rationale for this candidate. The score and signal breakdown remain valid."
)

EXPLANATION_SKIPPED_SUMMARY = (
    "Explanation skipped for this candidate to keep large-pool ranking fast. "
    "The score, rank, confidence, and signal breakdown remain valid."
)


# --------------------------------------------------------------------------- #
# Typed errors
# --------------------------------------------------------------------------- #
class OrchestratorError(Exception):
    """Base class for ranking-orchestration failures."""


class InvalidRankingInputError(OrchestratorError):
    """Raised when the JD is empty/whitespace or the candidate pool is empty.

    Implements Requirement 2.6: the orchestrator rejects an invalid request with
    an error indicating the invalid input and produces no ranking.
    """


class EmbeddingUnavailableError(OrchestratorError):
    """Raised internally when a candidate's embedding cannot be produced.

    Signals that, after the configured retries (Requirement 9.2) and with no
    cached embedding to reuse, a single candidate could not be embedded. The
    orchestrator catches this per candidate and *excludes only that candidate*
    while every successfully-embedded candidate proceeds (Requirement 9.3); it is
    never propagated out of :meth:`RankingOrchestrator.rank_candidates`.
    """


# --------------------------------------------------------------------------- #
# Structural collaborator protocols (only the methods the orchestrator uses)
# --------------------------------------------------------------------------- #
class _Decomposer(Protocol):
    def decompose(self, raw_jd: str) -> RequirementVector: ...


class _Enricher(Protocol):
    def normalize(self, raw_profile: RawCandidate) -> NormalizedProfile: ...

    def enrich(self, profile: NormalizedProfile) -> EnrichedProfile: ...


class _Embedder(Protocol):
    def embed(self, profile: EnrichedProfile) -> Vector: ...

    def embed_requirement(self, reqs: RequirementVector) -> Vector: ...


class _Explainer(Protocol):
    def explain(
        self,
        profile: EnrichedProfile,
        reqs: RequirementVector,
        breakdown: SignalBreakdown | None = ...,
    ) -> Explanation: ...


# --------------------------------------------------------------------------- #
# Internal stage result containers
# --------------------------------------------------------------------------- #
@dataclass
class _Excluded:
    """A candidate excluded during scoring with the reason it was dropped.

    Recorded (rather than silently swallowed) so the orchestrator can surface
    *which* required sub-score could not be computed (Requirement 4.7). It is not
    part of the emitted ranking.
    """

    candidate_id: object
    reason: str


@dataclass
class _ScoringOutcome:
    """The result of composite-scoring every hard-filter survivor."""

    scored: list[ScoredCandidate] = field(default_factory=list)
    excluded: list[_Excluded] = field(default_factory=list)


@dataclass
class _EnrichmentOutcome:
    """The result of normalize → enrich → resilient-embed over the pool.

    ``enriched`` are the candidates that completed all three sub-stages (and have
    an embedding attached); ``embed_excluded`` records candidates dropped because
    their embedding could not be produced after retries with no cache to reuse
    (Requirements 9.2 / 9.3).
    """

    enriched: list[EnrichedProfile] = field(default_factory=list)
    embed_excluded: list[_Excluded] = field(default_factory=list)


@dataclass
class RankingRun:
    """A full ranking run plus the resilience flags the output layer surfaces.

    :meth:`RankingOrchestrator.rank_candidates` returns just ``results`` for
    back-compat; :meth:`RankingOrchestrator.rank_candidates_run` returns this
    richer container so the shortlist output (Task 19.1) can honestly report
    degraded behaviour (Requirement 9):

    Attributes:
        results: the ranked :class:`RankingResult` list (one per survivor).
        reranked: ``True`` when LLM contextual reranking was applied; ``False``
            when the reranker failed and the orchestrator fell back to
            composite-score ordering (Requirement 9.4 — flagged un-reranked).
        excluded: candidates dropped before ranking — those that could not be
            embedded (Requirements 9.2 / 9.3) or for which a required sub-score
            could not be computed (Requirement 4.7) — each with its reason.
        explanation_unavailable_ids: ids of candidates whose explanation could
            not be generated and were marked unavailable without fabrication
            (Requirement 9.5).
    """

    results: list[RankingResult] = field(default_factory=list)
    reranked: bool = True
    excluded: list[_Excluded] = field(default_factory=list)
    explanation_unavailable_ids: list[object] = field(default_factory=list)

    @property
    def excluded_candidate_ids(self) -> list[object]:
        """The ids of all candidates excluded before ranking."""

        return [item.candidate_id for item in self.excluded]

    @property
    def all_explanations_available(self) -> bool:
        """Whether every emitted result carries a generated (non-sentinel) explanation."""

        return not self.explanation_unavailable_ids


class RankingOrchestrator:
    """Coordinate the full ICRS pipeline and assemble the ranked output.

    The orchestrator owns no scoring logic of its own — it sequences the injected
    stages and the (deterministic) scoring functions, then assigns ranks and
    assembles :class:`RankingResult` objects. Each stage is a discrete private
    method with a clear boundary so the resilience layer (Task 16.2) can wrap
    individual stages without touching the wiring here.

    Args:
        decomposer: produces a :class:`RequirementVector` from raw JD text.
        enricher: ``normalize`` then ``enrich`` each raw candidate.
        embedder: ``embed`` enriched profiles and ``embed_requirement`` the JD.
        reranker: the LLM contextual reranker (owns the configured ``K`` bound
            and the composite/LLM score blend).
        explainer: produces a recruiter-facing :class:`Explanation` per result.
        store: optional persistence backend; accepted for symmetry and future
            use (Task 16.2 / 17). Unused on the success path.
        embed_max_attempts: maximum per-candidate embedding attempts before the
            candidate is excluded (Requirement 9.2). Defaults to 3.
        embed_backoff_base_seconds: base of the exponential backoff slept between
            embedding attempts (attempt ``i`` waits ``base * 2**i`` seconds).
        sleep: the callable used to wait between embedding retries; defaults to
            :func:`time.sleep`. Inject a no-op in tests so retries never block.
    """

    def __init__(
        self,
        *,
        decomposer: _Decomposer,
        enricher: _Enricher,
        embedder: _Embedder,
        reranker: Reranker,
        explainer: _Explainer,
        store: object | None = None,
        embed_max_attempts: int = DEFAULT_EMBED_MAX_ATTEMPTS,
        embed_backoff_base_seconds: float = DEFAULT_EMBED_BACKOFF_BASE_SECONDS,
        explain_top_n: int | None = 10,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._decomposer = decomposer
        self._enricher = enricher
        self._embedder = embedder
        self._reranker = reranker
        self._explainer = explainer
        self._store = store
        # Embedding resilience (Requirements 9.2 / 9.3).
        self._embed_max_attempts = max(1, int(embed_max_attempts))
        self._embed_backoff_base_seconds = max(0.0, float(embed_backoff_base_seconds))
        self._sleep: Callable[[float], None] = sleep or time.sleep
        self._explain_top_n = None if explain_top_n is None else max(0, int(explain_top_n))
        # Content-hash keyed embedding cache: a cached vector is reused instead
        # of retrying a failed embedding call (Requirement 9.2).
        self._embedding_cache: dict[str, Vector] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def rank_candidates(
        self,
        raw_jd: str,
        pool: Sequence[RawCandidate],
        job_type: JobType | None,
    ) -> list[RankingResult]:
        """Rank ``pool`` against ``raw_jd`` and return the ranked shortlist.

        Runs the stages in order (Requirement 2.1): decompose → (per candidate)
        normalize → enrich → embed (Requirement 2.2) → hard filter → composite
        score → rerank → assign ranks → explain + confidence. Emits exactly one
        :class:`RankingResult` per hard-filter survivor (Requirement 2.4) with
        unique, contiguous ranks ``1..N`` ordered by descending final score and a
        deterministic tie-break (Requirements 2.5, 5.3, 5.6).

        This is the back-compatible entry point returning just the result list.
        Use :meth:`rank_candidates_run` to additionally obtain the resilience
        flags (un-reranked, explanation-unavailable, excluded candidates).

        Raises:
            InvalidRankingInputError: ``raw_jd`` is empty/whitespace or ``pool``
                is empty (Requirement 2.6).
        """

        run = await self.rank_candidates_run(raw_jd, pool, job_type)
        return run.results

    def decompose_jd(self, raw_jd: str) -> RequirementVector:
        """Decompose a raw JD into a :class:`RequirementVector`.

        Public API surface so the API layer can invoke JD decomposition
        without reaching into the private ``_decomposer`` attribute,
        preserving the layering contract.

        Args:
            raw_jd: the raw job-description text (must be non-empty).

        Returns:
            The parsed :class:`RequirementVector`.

        Raises:
            InvalidRankingInputError: ``raw_jd`` is empty or whitespace-only.
        """

        if raw_jd is None or not raw_jd.strip():
            raise InvalidRankingInputError(
                "raw_jd must contain at least one non-whitespace character"
            )
        return self._decomposer.decompose(raw_jd)

    async def rank_candidates_run(
        self,
        raw_jd: str,
        pool: Sequence[RawCandidate],
        job_type: JobType | None,
    ) -> RankingRun:
        """Rank ``pool`` against ``raw_jd`` and return a structured :class:`RankingRun`.

        Identical pipeline to :meth:`rank_candidates`, but each stage is wrapped
        with the Task 16.2 resilience behaviour and the result carries the
        degradation flags (Requirement 9):

            - candidates that cannot be embedded after retries (no cache) are
              excluded; the rest proceed (9.2 / 9.3);
            - a reranker failure falls back to composite ordering and sets
              ``reranked = False`` (9.4);
            - an explanation failure marks that candidate's explanation
              unavailable without fabrication and records its id (9.5).

        Raises:
            InvalidRankingInputError: ``raw_jd`` is empty/whitespace or ``pool``
                is empty (Requirement 2.6).
        """

        # ----- Requirement 2.6: reject invalid input before any work ----- #
        self._validate_inputs(raw_jd, pool)

        # ----- Stage 1: decompose JD, embed requirements, pick weights ----- #
        reqs = self._decompose(raw_jd)
        req_vec = self._embedder.embed_requirement(reqs)
        weights = self._select_weights(job_type)

        # ----- Stage 2: normalize -> enrich -> resilient embed each candidate -- #
        enrichment = self._enrich_pool(pool)

        # ----- Stage 3: deterministic hard-filter gate ----- #
        survivors = self._hard_filter(reqs, enrichment.enriched)

        # ----- Stage 4: composite scoring over survivors ----- #
        outcome = self._score_survivors(reqs, req_vec, weights, survivors)

        # ----- Stage 5: LLM contextual rerank (top-K); fall back on failure --- #
        reranked = self._rerank(outcome.scored, reqs)

        # ----- Stage 6: rank assignment over ALL survivors ----- #
        ordered = self._assign_order(outcome.scored)

        # ----- Stage 7: explanations + confidence -> RankingResult ----- #
        results, explanation_unavailable_ids = self._explain_and_assemble(
            ordered, reqs
        )

        return RankingRun(
            results=results,
            reranked=reranked,
            excluded=[*enrichment.embed_excluded, *outcome.excluded],
            explanation_unavailable_ids=explanation_unavailable_ids,
        )

    # ------------------------------------------------------------------ #
    # Stage 0: input validation (Requirement 2.6)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_inputs(raw_jd: str, pool: Sequence[RawCandidate]) -> None:
        """Reject an empty/whitespace JD or an empty candidate pool (2.6)."""

        if raw_jd is None or not raw_jd.strip():
            raise InvalidRankingInputError(
                "raw_jd must contain at least one non-whitespace character"
            )
        if pool is None or len(pool) == 0:
            raise InvalidRankingInputError(
                "candidate pool must contain at least one candidate"
            )

    # ------------------------------------------------------------------ #
    # Stage 1: decomposition + weight selection
    # ------------------------------------------------------------------ #
    def _decompose(self, raw_jd: str) -> RequirementVector:
        """Decompose the raw JD into a :class:`RequirementVector`."""

        return self._decomposer.decompose(raw_jd)

    @staticmethod
    def _select_weights(job_type: JobType | None) -> WeightProfile:
        """Select the weight profile for ``job_type`` (default fallback)."""

        return select_weight_profile(job_type).profile

    # ------------------------------------------------------------------ #
    # Stage 2: per-candidate normalize -> enrich -> embed (Requirement 2.2)
    #          with embedding retry / cache reuse / exclusion (9.2 / 9.3)
    # ------------------------------------------------------------------ #
    def _enrich_pool(self, pool: Sequence[RawCandidate]) -> _EnrichmentOutcome:
        """Normalize, enrich, and resiliently embed every candidate in ``pool``.

        Each candidate completes normalize → enrich → embed before it can enter
        composite scoring (Requirement 2.2). Embedding is wrapped with the Task
        16.2 resilience policy (:meth:`_embed_with_resilience`): up to
        :attr:`_embed_max_attempts` attempts with increasing backoff, reusing a
        cached embedding when available instead of retrying (Requirement 9.2). A
        candidate whose embedding still cannot be produced — and for which no
        cached embedding exists — is *excluded* and recorded, while every
        successfully-embedded candidate proceeds (Requirement 9.3).

        Behavioral-fetch resilience (Requirement 9.1) is handled inside
        :meth:`CandidateEnricher.enrich` (a timed-out / unavailable behavioral
        source yields availability 0 + neutral prior), so enrichment itself does
        not abort here for a flaky behavioral source.
        """

        outcome = _EnrichmentOutcome()
        for raw in pool:
            normalized = self._enricher.normalize(raw)
            profile = self._enricher.enrich(normalized)
            try:
                profile.embedding = self._embed_with_resilience(profile)
            except EmbeddingUnavailableError as exc:
                # Requirement 9.3: exclude only this candidate; others proceed.
                outcome.embed_excluded.append(_Excluded(profile.id, str(exc)))
                continue
            outcome.enriched.append(profile)
        return outcome

    def _embed_with_resilience(self, profile: EnrichedProfile) -> Vector:
        """Embed ``profile`` with retry, backoff, and cache reuse (9.2 / 9.3).

        Resolution order:

        1. **Reuse an already-attached embedding.** If ``profile.embedding`` is
           already set (e.g. loaded from persistence) it is returned directly —
           no service call.
        2. **Reuse a cached embedding.** A content-hash cache keyed by the
           enriched profile (excluding the embedding field) is consulted; a hit
           is returned without calling the embedder (Requirement 9.2 — reuse a
           cached embedding instead of retrying).
        3. **Call the embedder with bounded retries.** Up to
           :attr:`_embed_max_attempts` attempts; between failed attempts the
           backoff ``base * 2**attempt`` is slept via the injected
           :attr:`_sleep`, and the cache is re-checked first so a concurrently
           cached vector short-circuits further retries (Requirement 9.2).

        Raises:
            EmbeddingUnavailableError: all attempts failed and no cached/attached
                embedding was available (Requirement 9.3 — caller excludes the
                candidate).
        """

        if profile.embedding is not None:
            return profile.embedding

        cache_key = self._embedding_cache_key(profile)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self._embed_max_attempts):
            try:
                vector = self._embedder.embed(profile)
            except Exception as exc:  # embedding service error (Requirement 9.2)
                last_error = exc
                # Reuse a cached embedding instead of retrying when one exists.
                cached = self._embedding_cache.get(cache_key)
                if cached is not None:
                    return cached
                if attempt < self._embed_max_attempts - 1:
                    self._sleep(self._backoff_seconds(attempt))
                continue
            self._embedding_cache[cache_key] = vector
            return vector

        raise EmbeddingUnavailableError(
            f"embedding could not be produced for candidate {profile.id} after "
            f"{self._embed_max_attempts} attempt(s) and no cached embedding was "
            f"available: {last_error}"
        )

    def _backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff (in seconds) before the next embedding retry."""

        return self._embed_backoff_base_seconds * (2 ** attempt)

    @staticmethod
    def _embedding_cache_key(profile: EnrichedProfile) -> str:
        """Stable content hash of the enriched profile (excluding its embedding)."""

        payload = profile.model_dump_json(
            exclude={"embedding": True, "id": True, "base": {"id": True}}
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # Stage 3: deterministic hard-filter gate
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hard_filter(
        reqs: RequirementVector, enriched: Sequence[EnrichedProfile]
    ) -> list[EnrichedProfile]:
        """Keep only candidates that match no DISQUALIFYING criterion."""

        return [e for e in enriched if hard_filter(reqs, e).passed]

    # ------------------------------------------------------------------ #
    # Stage 4: composite scoring over survivors
    # ------------------------------------------------------------------ #
    def _score_survivors(
        self,
        reqs: RequirementVector,
        req_vec: Vector,
        weights: WeightProfile,
        survivors: Sequence[EnrichedProfile],
    ) -> _ScoringOutcome:
        """Composite-score each survivor into a :class:`ScoredCandidate`.

        A candidate for which a *required* sub-score genuinely cannot be computed
        (e.g. no trajectory evidence at all) is excluded and recorded rather than
        scored (Requirement 4.7); absent data on a single tier is handled by the
        sub-scores themselves (neutral prior) and never excludes.
        """

        outcome = _ScoringOutcome()
        # Imported lazily to avoid a package-init import cycle: ``subscores``
        # imports ``icrs.pipeline.enrichment`` (which runs ``icrs.pipeline``'s
        # __init__, importing this module), so a top-level import here would
        # reference a partially-initialized ``subscores``.
        from icrs.scoring.subscores import SubScoreUnavailable

        for e in survivors:
            try:
                scored = self._score_one(reqs, req_vec, weights, e)
            except SubScoreUnavailable as exc:
                outcome.excluded.append(_Excluded(e.id, str(exc)))
                continue
            outcome.scored.append(scored)
        return outcome

    @staticmethod
    def _score_one(
        reqs: RequirementVector,
        req_vec: Vector,
        weights: WeightProfile,
        e: EnrichedProfile,
    ) -> ScoredCandidate:
        """Compute the five sub-scores, fuse them, and wrap in a ScoredCandidate."""

        # Lazy import (see ``_score_survivors``) to avoid the package-init cycle.
        from icrs.scoring.subscores import (
            behavioral_signal_score,
            career_trajectory_score,
            disqualifying_flag_penalty,
            hard_filter_pass_score,
        )

        semantic = semantic_fit_from_inputs(
            req_vec,
            e.embedding or [],
            requirement_query_terms(reqs),
            candidate_text_corpus(e),
        )
        trajectory = career_trajectory_score(e, reqs)  # may raise (Req 4.7)
        behavioral = behavioral_signal_score(e)
        hard_pass = hard_filter_pass_score(reqs, e)
        penalty = disqualifying_flag_penalty(reqs, e)

        bundle = build_signal_bundle(
            semantic_fit=semantic,
            career_trajectory=trajectory,
            behavioral=behavioral,
            hard_filter_pass=hard_pass,
            disqualifying_penalty=penalty,
        )
        composite_score = composite(bundle, weights)
        return ScoredCandidate(
            id=e.id,
            profile=e,
            composite_score=composite_score,
            breakdown=bundle.to_breakdown(),
        )

    # ------------------------------------------------------------------ #
    # Stage 5: rerank (top-K) — blends each selected candidate's final_score
    #          with a composite-ordering fallback on failure (Requirement 9.4)
    # ------------------------------------------------------------------ #
    def _rerank(
        self, scored: list[ScoredCandidate], reqs: RequirementVector
    ) -> bool:
        """Rerank the top-K survivors in place, falling back on reranker failure.

        On the success path the reranker selects and rescores the ``K`` highest
        composite-scored candidates, mutating their ``final_score`` with the
        composite/LLM blend. Candidates outside the top-K keep
        ``final_score == composite_score`` (set on construction), so a single
        final-score ordering spans all survivors.

        If the reranker raises (Requirement 9.4), the orchestrator does **not**
        abort: every candidate's ``final_score`` is reset to its deterministic
        ``composite_score`` so the subsequent ordering is the pure composite
        ranking, and ``False`` is returned to flag the run as un-reranked.

        Returns:
            ``True`` when reranking was applied, ``False`` when the orchestrator
            fell back to composite-score ordering.
        """

        if not scored:
            return True
        try:
            # The reranker mutates the selected ScoredCandidate objects in place;
            # the returned shortlist is the reranked subset (unused here — every
            # survivor's final_score is now set on the shared objects).
            self._reranker.rerank(scored, reqs)
        except Exception:
            # Requirement 9.4: fall back to composite-score ordering. Reset every
            # final_score to the composite score in case the reranker mutated a
            # subset before failing, so the ordering is purely composite-based.
            for c in scored:
                c.final_score = c.composite_score
            return False
        return True

    # ------------------------------------------------------------------ #
    # Stage 6: rank assignment over ALL survivors (Requirements 2.5/5.3/5.6)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _assign_order(scored: Sequence[ScoredCandidate]) -> list[ScoredCandidate]:
        """Order all survivors by descending final score with a deterministic tie-break.

        Sorting by ``(-final_score, id_str)`` guarantees a strictly higher final
        score sorts ahead (Requirement 5.3) and equal scores are broken
        deterministically by candidate id (Requirement 5.6), so identical inputs
        always yield identical ordering — the basis for the unique/contiguous
        ranks assigned in :meth:`_explain_and_assemble`.
        """

        return sorted(scored, key=lambda c: (-c.final_score, c.id_str))

    # ------------------------------------------------------------------ #
    # Stage 7: explanations + confidence -> RankingResult (Requirement 2.4)
    #          explanation failure -> marked unavailable (Requirement 9.5)
    # ------------------------------------------------------------------ #
    def _explain_and_assemble(
        self, ordered: list[ScoredCandidate], reqs: RequirementVector
    ) -> tuple[list[RankingResult], list[object]]:
        """Build one :class:`RankingResult` per survivor with rank 1..N.

        Ranks are the 1-based positions in the final-score ordering (unique and
        contiguous — Requirement 2.5). Confidence is computed over the full
        ranked set so it reflects the candidate's margin to its neighbours.

        Each per-candidate explanation is generated independently; if the
        explanation generator raises for a candidate (Requirement 9.5), that
        candidate's explanation is replaced with a sentinel that clearly states
        it is unavailable — **no** rationale, driving signals, or gaps are
        fabricated — and the candidate's id is recorded. The candidate still
        receives its score, rank, breakdown, and confidence.

        Returns:
            A ``(results, explanation_unavailable_ids)`` pair.
        """

        from concurrent.futures import ThreadPoolExecutor

        results: list[RankingResult] = []
        explanation_unavailable_ids: list[object] = []

        explain_limit = self._explain_top_n

        def get_single_explanation(c: ScoredCandidate) -> tuple[ScoredCandidate, Explanation, bool]:
            try:
                exp = self._explainer.explain(c.profile, reqs, c.breakdown)
                return c, exp, True
            except Exception:
                # Requirement 9.5: mark unavailable without fabricating content.
                return c, self._unavailable_explanation(), False

        if explain_limit is None:
            to_explain = ordered
            skipped_candidates: list[ScoredCandidate] = []
        elif explain_limit == 0:
            to_explain = []
            skipped_candidates = list(ordered)
        else:
            to_explain = ordered[:explain_limit]
            skipped_candidates = ordered[explain_limit:]

        skipped = {c.id_str: c for c in skipped_candidates}
        explanations_by_id: dict[str, tuple[Explanation, bool]] = {
            cid: (self._skipped_explanation(), False) for cid in skipped
        }

        # Run concurrent LLM calls only for the explainable prefix. Large pools
        # still get full ranking, but do not spend API quota on every rationale.
        max_workers = min(len(to_explain), 10) if to_explain else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit in original order
            futures = [executor.submit(get_single_explanation, c) for c in to_explain]

            for future in futures:
                c, explanation, success = future.result()
                explanations_by_id[c.id_str] = (explanation, success)

        # Resolve in final rank order (preserves ranks 1..N).
        for rank, c in enumerate(ordered, start=1):
            explanation, success = explanations_by_id[c.id_str]
            if not success:
                explanation_unavailable_ids.append(c.profile.id)
            confidence = compute_confidence_for(c, ordered)
            results.append(
                RankingResult(
                    job_id=reqs.job_id,
                    candidate_id=c.profile.id,
                    final_score=c.final_score,
                    rank=rank,
                    breakdown=c.breakdown,
                    explanation=explanation,
                    confidence=confidence,
                )
            )

        return results, explanation_unavailable_ids

    @staticmethod
    def _unavailable_explanation() -> Explanation:
        """Build the non-fabricated 'explanation unavailable' sentinel (9.5).

        The summary honestly states unavailability and the driving signals, gaps,
        and unmet-must-have lists are left empty — the orchestrator never invents
        rationale when the explanation generator fails.
        """

        return Explanation(
            summary=EXPLANATION_UNAVAILABLE_SUMMARY,
            driving_signals=[],
            gaps=[],
            unmet_must_haves=[],
        )

    @staticmethod
    def _skipped_explanation() -> Explanation:
        """Build the honest large-pool explanation-skip sentinel."""

        return Explanation(
            summary=EXPLANATION_SKIPPED_SUMMARY,
            driving_signals=[],
            gaps=[],
            unmet_must_haves=[],
        )


__all__ = [
    "RankingOrchestrator",
    "RankingRun",
    "OrchestratorError",
    "InvalidRankingInputError",
    "EmbeddingUnavailableError",
    "EXPLANATION_UNAVAILABLE_SUMMARY",
    "EXPLANATION_SKIPPED_SUMMARY",
]

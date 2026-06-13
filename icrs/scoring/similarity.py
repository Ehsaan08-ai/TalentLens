"""Dense / sparse similarity and the semantic-fit blend (Task 10.1).

This module implements the *semantic fit* sub-score of the hybrid scoring engine
(design "Scoring Architecture")::

    PROCEDURE SemanticFitScore(reqVec, candVec, reqs, cand)
        dense  ← cosineSimilarity(reqVec.aggregate_embedding, candVec)   # [0,1]
        sparse ← normalizedBM25(reqs, cand.text_corpus)                  # [0,1]
        RETURN 0.7 * dense + 0.3 * sparse   # dense-favored; sparse for recall

It is deliberately kept in its own module so it can be developed and tested
independently of the deterministic hard-filter gate (Task 8.1, in
``hard_filter.py``). The standalone functions here are pure and deterministic;
an optional :class:`SemanticFitMixin` exposes them as engine methods so a later
task can compose them onto :class:`~icrs.scoring.hard_filter.HybridScoringEngine`
without any single file needing to be co-edited.

Requirement 4.6 contract:
    - :func:`dense_similarity` — cosine similarity over two vectors, mapped from
      ``[-1,1]`` to ``[0,1]`` via ``(x + 1) / 2`` and clamped.
    - :func:`sparse_similarity` — a BM25 score (``rank_bm25.BM25Plus``)
      normalized to ``[0,1]``.
    - :func:`semantic_fit` — the ``0.7 * dense + 0.3 * sparse`` blend, in
      ``[0,1]``.
    - :func:`dense_retrieve` — top-N dense candidate retrieval via the
      :class:`~icrs.providers.base.VectorStore` abstraction (depends on the
      abstraction, never instantiates a concrete Qdrant / pgvector store).

All public scoring functions are guaranteed to return a value in the inclusive
range ``[0,1]``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from rank_bm25 import BM25Plus

from icrs.models.candidate import EnrichedProfile
from icrs.models.job import RequirementVector
from icrs.providers.base import Vector, VectorMatch, VectorStore

# ----- blend weights (design: dense-favored semantic fit) -------------------

#: Weight applied to the dense (vector cosine) component of the semantic-fit
#: blend. Dense is favoured; sparse provides lexical recall safety.
DENSE_WEIGHT: float = 0.7
#: Weight applied to the sparse (BM25) component of the semantic-fit blend.
SPARSE_WEIGHT: float = 0.3

#: Neutral value returned for a cosine that is undefined (a zero-magnitude
#: vector). ``(0 + 1) / 2 == 0.5`` — the midpoint, i.e. "no signal either way".
_NEUTRAL_COSINE_MAPPED: float = 0.5

# Alphanumeric tokens of length >= 1 (single characters are retained for sparse
# matching, unlike the hard filter, because lexical recall benefits from short
# tokens such as "go" or "r"). Lower-cased before matching.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _clamp01(value: float) -> float:
    """Clamp ``value`` to the inclusive range ``[0,1]``."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


# ----- dense cosine similarity ----------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the raw cosine similarity of ``a`` and ``b`` in ``[-1, 1]``.

    Args:
        a: First vector.
        b: Second vector, of the same dimensionality as ``a``.

    Returns:
        The cosine similarity. ``0.0`` is returned when either vector has zero
        magnitude (cosine is undefined for the zero vector; ``0.0`` maps to the
        neutral ``0.5`` after the ``[0,1]`` rescaling in :func:`dense_similarity`).

    Raises:
        ValueError: If the two vectors have differing dimensionality.
    """

    if len(a) != len(b):
        raise ValueError(
            f"cosine_similarity requires equal-length vectors; "
            f"got {len(a)} and {len(b)}"
        )
    if not a:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        # Zero-magnitude vector: cosine is undefined -> treat as no signal.
        return 0.0

    cos = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    # Guard against floating-point drift just outside [-1, 1].
    if cos > 1.0:
        return 1.0
    if cos < -1.0:
        return -1.0
    return cos


def dense_similarity(req_vec: Sequence[float], cand_vec: Sequence[float]) -> float:
    """Dense semantic similarity of a requirement and candidate vector in ``[0,1]``.

    Computes cosine similarity (in ``[-1,1]``) and maps it to ``[0,1]`` with
    ``(cos + 1) / 2``, clamped. Consequences (Requirement 4.6):

        - Identical (parallel) vectors → cosine ``1`` → ``1.0``.
        - Opposite (anti-parallel) vectors → cosine ``-1`` → ``0.0``.
        - Orthogonal vectors → cosine ``0`` → ``0.5``.
        - A zero-magnitude vector yields the neutral ``0.5``.

    Args:
        req_vec: The (aggregate) requirement embedding.
        cand_vec: The candidate embedding, of equal dimensionality.

    Returns:
        A similarity score in the inclusive range ``[0,1]``.

    Raises:
        ValueError: If the two vectors have differing dimensionality.
    """

    if len(req_vec) != len(cand_vec):
        raise ValueError(
            f"dense_similarity requires equal-length vectors; "
            f"got {len(req_vec)} and {len(cand_vec)}"
        )
    if not req_vec:
        # Two empty vectors carry no signal -> neutral midpoint.
        return _NEUTRAL_COSINE_MAPPED

    cos = cosine_similarity(req_vec, cand_vec)
    return _clamp01((cos + 1.0) / 2.0)


# ----- sparse BM25 similarity -----------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lower-case ``text`` and return its alphanumeric tokens."""

    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _as_query_tokens(query: str | Iterable[str]) -> list[str]:
    """Normalize a query (free text or pre-split terms) into BM25 query tokens."""

    if isinstance(query, str):
        return _tokenize(query)
    # An iterable of terms: tokenize each so multi-word terms are split too.
    tokens: list[str] = []
    for term in query:
        tokens.extend(_tokenize(term))
    return tokens


def _bm25_self_score(bm25: BM25Plus, query_tokens: Sequence[str]) -> float:
    """Theoretical maximum BM25+ score for ``query_tokens`` under ``bm25``'s stats.

    Computes the BM25+ score the query would receive against a *hypothetical
    document equal to the query itself*, using the model's fitted IDF, average
    document length, and ``k1`` / ``b`` / ``delta`` parameters. This is the
    principled upper bound used to normalize observed scores into ``[0,1]``.

    Crucially this is computed analytically rather than by appending the query
    to the corpus: appending the query would change the document frequencies and
    therefore the IDF, distorting the normalization (acutely so on the small
    per-candidate corpora used here).
    """

    counts: dict[str, int] = {}
    for tok in query_tokens:
        counts[tok] = counts.get(tok, 0) + 1

    doc_len = len(query_tokens)
    avgdl = bm25.avgdl or 1.0
    k1 = bm25.k1
    b = bm25.b
    delta = bm25.delta
    length_norm = 1.0 - b + b * (doc_len / avgdl)

    score = 0.0
    for term, freq in counts.items():
        idf = bm25.idf.get(term, 0.0)
        if idf <= 0.0:
            continue
        # Mirror BM25Plus.get_scores term contribution exactly.
        score += freq * idf * (delta + (freq * (k1 + 1.0)) / (k1 * length_norm + freq))
    return score


def sparse_similarity(
    query: str | Iterable[str],
    candidate_corpus: Sequence[str],
) -> float:
    """BM25 lexical similarity of ``query`` against ``candidate_corpus`` in ``[0,1]``.

    Scoring uses ``rank_bm25.BM25Plus``. BM25+ is chosen over plain BM25Okapi
    here deliberately: Okapi's IDF term ``log((N - n + 0.5) / (n + 0.5))``
    collapses to zero (or goes negative) for any term occurring in roughly half
    or more of the documents, which is the *common* case on the small
    per-candidate corpora ICRS scores against — that would silently zero out
    genuine lexical matches. BM25+ uses the strictly-positive IDF
    ``log((N + 1) / n)`` plus a ``delta`` floor on the term-frequency component,
    so present terms always contribute and recall is preserved.

    Raw BM25+ scores are unbounded above, so they are normalized into ``[0,1]``
    using a *self-score* upper bound:

        The model is fitted over the candidate corpus and the query is scored
        against every document; the best (maximum) document score is the
        observed match strength. That value is divided by the query's
        *self-score* — the score the query would achieve against a document
        identical to itself under the same corpus statistics — which is the
        maximum achievable score. The ratio is clamped to ``[0,1]``.

    This is preferred over "divide by the observed max" (which would force the
    best document to ``1.0`` even when it matches poorly): the self-score
    reflects how well the *best possible* document for this query would score, so
    a weakly matching corpus correctly yields a low value. The self-score is
    computed analytically (not by adding the query to the corpus) so the corpus
    IDF statistics — and hence the normalization — are not distorted.

    Args:
        query: The requirement text, or a collection of requirement terms.
        candidate_corpus: The candidate's text corpus as a list of documents
            (e.g. one entry per role / skill / responsibility). May be empty.

    Returns:
        A normalized lexical similarity in the inclusive range ``[0,1]``. Returns
        ``0.0`` when the query has no tokens, the corpus is empty, or the corpus
        contains no tokens (absence of lexical evidence is *not* a match).
    """

    query_tokens = _as_query_tokens(query)
    if not query_tokens:
        return 0.0

    tokenized_corpus = [_tokenize(doc) for doc in candidate_corpus]
    # Drop empty documents — BM25 cannot fit on documents of length 0 and they
    # carry no lexical evidence anyway.
    real_docs = [doc for doc in tokenized_corpus if doc]
    if not real_docs:
        return 0.0

    bm25 = BM25Plus(real_docs)
    scores = bm25.get_scores(query_tokens)
    best_real = max(scores) if len(scores) else 0.0
    if best_real <= 0.0:
        return 0.0

    self_score = _bm25_self_score(bm25, query_tokens)
    if self_score <= 0.0:
        return 0.0

    return _clamp01(best_real / self_score)


# ----- semantic-fit blend ----------------------------------------------------


def semantic_fit(dense: float, sparse: float) -> float:
    """Blend the dense and sparse sub-scores into the semantic-fit score.

    Implements ``DENSE_WEIGHT * dense + SPARSE_WEIGHT * sparse`` (``0.7`` /
    ``0.3`` per the design), clamping each input and the result to ``[0,1]`` so
    the output is always a valid sub-score (Requirement 4.6). Because the weights
    are non-negative and sum to ``1.0``, a convex combination of two ``[0,1]``
    inputs is itself in ``[0,1]``; the clamps defend only against
    out-of-contract inputs and floating-point drift.

    Args:
        dense: The dense similarity sub-score (expected in ``[0,1]``).
        sparse: The sparse similarity sub-score (expected in ``[0,1]``).

    Returns:
        The blended semantic-fit sub-score in the inclusive range ``[0,1]``.
    """

    d = _clamp01(dense)
    s = _clamp01(sparse)
    return _clamp01(DENSE_WEIGHT * d + SPARSE_WEIGHT * s)


def semantic_fit_from_inputs(
    req_vec: Sequence[float],
    cand_vec: Sequence[float],
    query: str | Iterable[str],
    candidate_corpus: Sequence[str],
) -> float:
    """Compute the semantic-fit sub-score end-to-end from raw inputs.

    Convenience wrapper that computes :func:`dense_similarity` and
    :func:`sparse_similarity` and blends them via :func:`semantic_fit`.

    Args:
        req_vec: Aggregate requirement embedding.
        cand_vec: Candidate embedding (equal dimensionality).
        query: Requirement text / terms for lexical matching.
        candidate_corpus: The candidate's text corpus (list of documents).

    Returns:
        The semantic-fit sub-score in the inclusive range ``[0,1]``.
    """

    dense = dense_similarity(req_vec, cand_vec)
    sparse = sparse_similarity(query, candidate_corpus)
    return semantic_fit(dense, sparse)


# ----- corpus / query helpers (wiring into pipeline data) -------------------


def candidate_text_corpus(cand: EnrichedProfile) -> list[str]:
    """Build a candidate's lexical text corpus from its enriched profile.

    Each concrete piece of evidence (role titles/companies, education, certs,
    explicit + implicit skills, inferred responsibilities) becomes one document
    so BM25 can match the requirement query against the *best* evidence item.
    Absent fields contribute nothing (they are simply not emitted), preserving
    the missing-data fairness guarantee — sparse candidates yield fewer
    documents, never a fabricated one.

    Args:
        cand: The enriched candidate profile.

    Returns:
        A list of non-empty text documents (possibly empty for a bare profile).
    """

    docs: list[str] = []
    base = cand.base
    for role in base.roles:
        docs.append(f"{role.title} {role.company}".strip())
    for edu in base.education:
        parts = [edu.institution, edu.degree, edu.field_of_study]
        text = " ".join(p for p in parts if p)
        if text:
            docs.append(text)
    docs.extend(base.certifications)
    docs.extend(base.explicit_skills)
    docs.extend(cand.implicit_skills)
    docs.extend(cand.inferred_responsibilities)
    return [d for d in docs if d and d.strip()]


def requirement_query_terms(reqs: RequirementVector) -> list[str]:
    """Collect the lexical query terms from a requirement vector.

    Uses the role intent plus every *weighted* requirement's text (MUST_HAVE and
    NICE_TO_HAVE); DISQUALIFYING criteria are gates handled elsewhere and are
    excluded from the lexical query.

    Args:
        reqs: The decomposed requirement vector.

    Returns:
        A list of requirement text fragments to use as the BM25 query.
    """

    terms: list[str] = [reqs.role_intent]
    for requirement in reqs.weighted_requirements():
        terms.append(requirement.text)
    return terms


# ----- dense ANN retrieval wiring -------------------------------------------


async def dense_retrieve(
    store: VectorStore,
    collection: str,
    query_vec: Vector,
    *,
    top_n: int,
    filters: dict[str, object] | None = None,
) -> list[VectorMatch]:
    """Retrieve the top-N nearest candidates for ``query_vec`` via a vector store.

    This is the dense ANN retrieval wiring point of the design's hybrid scoring
    engine. It depends only on the :class:`~icrs.providers.base.VectorStore`
    abstraction — it never instantiates a concrete Qdrant / pgvector store — so
    the retrieval source is swappable and a stub store can drive tests.

    Args:
        store: Any :class:`VectorStore` implementation (real or stub).
        collection: The collection / table to search.
        query_vec: The requirement (query) embedding.
        top_n: Maximum number of neighbours to return; must be >= 1.
        filters: Optional payload filters forwarded to the store (e.g. to
            pre-gate on hard-filter survivors).

    Returns:
        Up to ``top_n`` :class:`VectorMatch` hits, ordered as returned by the
        store (nearest first).

    Raises:
        ValueError: If ``top_n`` is less than 1.
    """

    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    return await store.search(
        collection, query_vec, top_n=top_n, filters=filters
    )


# ----- engine mixin ----------------------------------------------------------


class SemanticFitMixin:
    """Mixin exposing the dense / sparse / semantic-fit methods on an engine.

    Kept separate from :class:`~icrs.scoring.hard_filter.HybridScoringEngine` so
    a later task can compose this onto the engine without co-editing the
    hard-filter module. Each method delegates to the standalone functions above.
    """

    def dense_score(
        self, req_vec: Sequence[float], cand_vec: Sequence[float]
    ) -> float:
        """Instance form of :func:`dense_similarity` (design ``denseScore``)."""

        return dense_similarity(req_vec, cand_vec)

    def sparse_score(
        self,
        reqs_or_query: RequirementVector | str | Iterable[str],
        cand: EnrichedProfile | Sequence[str],
    ) -> float:
        """Instance form of :func:`sparse_similarity` (design ``sparseScore``).

        Accepts either the structured ``(RequirementVector, EnrichedProfile)``
        pair used by the pipeline — in which case the query terms and candidate
        corpus are derived automatically — or a raw ``(query, corpus)`` pair.
        """

        if isinstance(reqs_or_query, RequirementVector):
            query: str | Iterable[str] = requirement_query_terms(reqs_or_query)
        else:
            query = reqs_or_query

        if isinstance(cand, EnrichedProfile):
            corpus: Sequence[str] = candidate_text_corpus(cand)
        else:
            corpus = cand

        return sparse_similarity(query, corpus)

    def semantic_fit_score(
        self,
        req_vec: Sequence[float],
        cand_vec: Sequence[float],
        query: str | Iterable[str],
        candidate_corpus: Sequence[str],
    ) -> float:
        """Compute the blended semantic-fit sub-score (design ``SemanticFitScore``)."""

        return semantic_fit_from_inputs(
            req_vec, cand_vec, query, candidate_corpus
        )

    async def dense_retrieve(
        self,
        store: VectorStore,
        collection: str,
        query_vec: Vector,
        *,
        top_n: int,
        filters: dict[str, object] | None = None,
    ) -> list[VectorMatch]:
        """Instance form of :func:`dense_retrieve` (dense ANN retrieval)."""

        return await dense_retrieve(
            store, collection, query_vec, top_n=top_n, filters=filters
        )


__all__ = [
    "DENSE_WEIGHT",
    "SPARSE_WEIGHT",
    "cosine_similarity",
    "dense_similarity",
    "sparse_similarity",
    "semantic_fit",
    "semantic_fit_from_inputs",
    "candidate_text_corpus",
    "requirement_query_terms",
    "dense_retrieve",
    "SemanticFitMixin",
]

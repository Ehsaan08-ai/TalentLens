"""Pipeline stages and orchestration for ICRS.

JD decomposition, candidate enrichment, embedding generation, and the ranking
orchestrator are implemented across the pipeline tasks. This package is the
namespace they live in.
"""

from icrs.pipeline.embedding import (
    Chunk,
    EmbeddingGenerator,
    default_token_count,
)
from icrs.pipeline.enricher import CandidateEnricher
from icrs.pipeline.explanation import ExplanationGenerator
from icrs.pipeline.ingestion import IngestionResult, ingest
from icrs.pipeline.enrichment import (
    BehavioralSignalSource,
    EnrichmentError,
    EnrichmentMixin,
    NullBehavioralSignalSource,
    freshness_weight,
)
from icrs.pipeline.normalization import (
    ProfileNormalizationMixin,
    ProfileValidationError,
)
from icrs.pipeline.orchestrator import (
    EmbeddingUnavailableError,
    InvalidRankingInputError,
    OrchestratorError,
    RankingOrchestrator,
    RankingRun,
)
from icrs.pipeline.reranker import (
    RerankError,
    Reranker,
    ScoredCandidate,
    build_rerank_prompt,
    parse_llm_scores,
)

__all__ = [
    "CandidateEnricher",
    "ExplanationGenerator",
    "EmbeddingGenerator",
    "Chunk",
    "default_token_count",
    "ingest",
    "IngestionResult",
    "ProfileNormalizationMixin",
    "ProfileValidationError",
    "EnrichmentMixin",
    "EnrichmentError",
    "BehavioralSignalSource",
    "NullBehavioralSignalSource",
    "freshness_weight",
    "Reranker",
    "ScoredCandidate",
    "RerankError",
    "build_rerank_prompt",
    "parse_llm_scores",
    "RankingOrchestrator",
    "RankingRun",
    "OrchestratorError",
    "InvalidRankingInputError",
    "EmbeddingUnavailableError",
]

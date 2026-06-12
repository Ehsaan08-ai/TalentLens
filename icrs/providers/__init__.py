"""Provider interfaces and registry for ICRS.

Concrete, swappable backends (sentence-transformers embeddings, Groq + Gemini
LLMs, Qdrant / pgvector stores) are added in later tasks. This package exposes
the abstract contracts the rest of the pipeline depends on.
"""

from icrs.providers.base import (
    ConfigurableProvider,
    EmbeddingProvider,
    LLMMessage,
    LLMProvider,
    LLMProviderRegistry,
    LLMResponse,
    LLMTask,
    Vector,
    VectorMatch,
    VectorRecord,
    VectorStore,
)

__all__ = [
    "ConfigurableProvider",
    "EmbeddingProvider",
    "LLMMessage",
    "LLMProvider",
    "LLMProviderRegistry",
    "LLMResponse",
    "LLMTask",
    "Vector",
    "VectorMatch",
    "VectorRecord",
    "VectorStore",
]

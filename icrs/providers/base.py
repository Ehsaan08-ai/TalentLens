"""Abstract provider interfaces for ICRS.

These contracts keep the concrete models swappable: the pipeline depends only on
the interfaces here, never on a specific SDK. Concrete implementations live in
sibling modules (e.g. a sentence-transformers embedding provider, a Groq LLM
provider, a Gemini LLM provider, a Qdrant / pgvector store) and are selected at
runtime via configuration.

Nothing in this module imports a provider SDK, so it is safe to import without
any API keys or heavy ML dependencies installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

# A dense embedding vector. Kept as a plain list of floats at the interface
# boundary so callers are not coupled to numpy / torch tensor types.
Vector = list[float]


class LLMTask(str, Enum):
    """The distinct LLM-bearing roles in the pipeline.

    Different tasks may be routed to different concrete providers (see
    ``LLMProviderRegistry``). For example, decomposition and reranking can be
    served by a Groq-hosted model while explanation generation is served by
    Gemini — without any pipeline code knowing which backend handles which task.
    """

    DECOMPOSE = "decompose"  # Job-description decomposition -> RequirementVector
    ENRICH = "enrich"  # Tier 2 semantic enrichment of a normalized candidate profile
    RERANK = "rerank"  # Contextual reranking of the top-K candidates
    EXPLAIN = "explain"  # Recruiter-facing explanation generation


@dataclass(frozen=True)
class LLMMessage:
    """A single chat message passed to an LLM provider."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """A normalized LLM completion result, independent of any vendor SDK."""

    text: str
    model: str
    raw: Any = field(default=None, repr=False)


class EmbeddingProvider(ABC):
    """Produces dense vector embeddings for text.

    Implementations MUST embed candidate profiles and requirement vectors with
    the same model and dimensionality so the resulting vectors are directly
    comparable by cosine similarity (Requirement 6.4).
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier of the underlying embedding model (e.g. the HF repo id)."""

    @property
    @abstractmethod
    def dimensionality(self) -> int:
        """Output vector dimensionality the provider is configured to produce."""

    @property
    @abstractmethod
    def max_input_tokens(self) -> int:
        """Maximum input length (in tokens) before chunking is required."""

    @abstractmethod
    def embed(self, text: str) -> Vector:
        """Return a single embedding vector for ``text``."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        """Return one embedding vector per input text, order-preserving."""


class LLMProvider(ABC):
    """Generates text/structured completions from a chat-style prompt.

    A provider is associated with one or more :class:`LLMTask` roles via the
    registry. The interface is deliberately minimal so Groq (OpenAI-compatible)
    and Gemini implementations can satisfy it identically.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier of the underlying model (e.g. ``llama-3.3-70b-versatile``)."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> LLMResponse:
        """Return a completion for ``messages``.

        ``response_format`` may request structured output (e.g. ``"json"``) where
        the backend supports it; implementations should degrade gracefully when
        it does not.
        """


@dataclass(frozen=True)
class VectorRecord:
    """A stored vector with its identifier and arbitrary filterable payload."""

    id: str
    vector: Vector
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorMatch:
    """A nearest-neighbour search hit."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Stores and searches dense vectors with payload filtering.

    Abstracts over pgvector (system of record at PoC scale) and Qdrant (dedicated
    vector search), so the retrieval layer can be repointed without changing
    scoring code.
    """

    @abstractmethod
    async def ensure_collection(self, name: str, dimensionality: int) -> None:
        """Create the collection/table for ``name`` if it does not already exist."""

    @abstractmethod
    async def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Insert or update vector records in ``collection``."""

    @abstractmethod
    async def search(
        self,
        collection: str,
        query: Vector,
        *,
        top_n: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Return up to ``top_n`` nearest neighbours, optionally payload-filtered."""


@runtime_checkable
class ConfigurableProvider(Protocol):
    """Optional protocol for providers built from the ICRS settings object."""

    @classmethod
    def from_settings(cls, settings: Any) -> "ConfigurableProvider":
        ...


class LLMProviderRegistry:
    """Routes each :class:`LLMTask` to a concrete :class:`LLMProvider`.

    This is what makes the per-task provider substitution explicit and swappable:
    decomposition + reranking can be bound to a Groq provider while explanation is
    bound to a Gemini provider, all behind one lookup.
    """

    def __init__(self) -> None:
        self._providers: dict[LLMTask, LLMProvider] = {}

    def register(self, task: LLMTask, provider: LLMProvider) -> None:
        """Bind ``provider`` to ``task`` (overwriting any prior binding)."""

        self._providers[task] = provider

    def get(self, task: LLMTask) -> LLMProvider:
        """Return the provider bound to ``task``.

        Raises ``KeyError`` with an actionable message when no provider is
        registered for the requested task.
        """

        try:
            return self._providers[task]
        except KeyError as exc:
            raise KeyError(
                f"No LLM provider registered for task {task!r}. "
                f"Register one via LLMProviderRegistry.register()."
            ) from exc

    def is_registered(self, task: LLMTask) -> bool:
        """Return whether a provider is bound to ``task``."""

        return task in self._providers

    def tasks(self) -> list[LLMTask]:
        """Return the tasks that currently have a provider bound."""

        return list(self._providers.keys())


__all__ = [
    "Vector",
    "LLMTask",
    "LLMMessage",
    "LLMResponse",
    "EmbeddingProvider",
    "LLMProvider",
    "VectorRecord",
    "VectorMatch",
    "VectorStore",
    "ConfigurableProvider",
    "LLMProviderRegistry",
]

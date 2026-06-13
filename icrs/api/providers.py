"""Lazy, config-driven default provider wiring for the ICRS API (Task 17.1).

The API's default orchestrator is built from concrete providers selected by the
process settings (:class:`~icrs.config.Settings`): Groq serves the
``DECOMPOSE`` / ``ENRICH`` / ``RERANK`` LLM tasks, Gemini serves ``EXPLAIN``,
and a local sentence-transformers model produces embeddings — all behind the
abstract :mod:`icrs.providers.base` interfaces so the pipeline never depends on
a vendor SDK.

IMPORTANT — no real API at import time:
    Every concrete provider here imports its SDK and instantiates clients/models
    *lazily*, on the first call, never at construction or module-import time.
    :func:`build_default_orchestrator` only wires objects together; it performs
    no network call and loads no model weights. The heavy/networked work happens
    the first time a request actually exercises a provider. Tests MUST inject a
    stub-backed orchestrator into ``create_app`` and never reach this module's
    network paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from icrs.config import Settings, get_settings
from icrs.providers.base import (
    EmbeddingProvider,
    LLMMessage,
    LLMProvider,
    LLMProviderRegistry,
    LLMResponse,
    LLMTask,
    Vector,
)


class GroqLLMProvider(LLMProvider):
    """A Groq (OpenAI-compatible) chat-completion provider.

    The Groq SDK and client are created lazily on first :meth:`complete` so
    importing this module — and building the default orchestrator — never
    requires the ``groq`` package or a valid API key to be present.

    Rate-limit resilience: Groq's free tier enforces a tokens-per-minute (TPM)
    cap, and the pipeline issues one call per candidate during enrichment, so a
    moderately sized pool can briefly exceed the cap and receive HTTP 429. Rather
    than failing the whole ranking, :meth:`complete` retries a rate-limited call
    up to ``max_retries`` times, honoring the server's ``retry-after`` hint when
    present and otherwise backing off exponentially (capped at
    ``max_backoff_seconds``). The SDK's own auto-retry is disabled so this is the
    single, predictable retry layer.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        max_retries: int = 5,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_retries = max(0, int(max_retries))
        self._max_backoff_seconds = max(0.0, float(max_backoff_seconds))
        self._client = None  # built lazily

    @property
    def model_id(self) -> str:
        return self._model

    def _ensure_client(self):
        if self._client is None:
            # Lazy import + construction: no SDK/key needed until first use.
            from groq import Groq

            if not self._api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not configured; cannot call the Groq LLM "
                    "provider. Set it in the environment or inject a stub "
                    "orchestrator."
                )
            # Disable the SDK's built-in retries; we own retry/backoff here so
            # the behaviour (and the honored retry-after) is explicit and single.
            self._client = Groq(api_key=self._api_key, max_retries=0)
        return self._client

    @staticmethod
    def _retry_after_seconds(exc: Exception, attempt: int) -> float:
        """Seconds to wait before the next retry of a rate-limited call.

        Prefers the server's ``retry-after`` response header (seconds) when
        available; otherwise uses exponential backoff (``1 * 2**attempt``). The
        caller caps the result at ``max_backoff_seconds``.
        """

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after")
            if raw:
                try:
                    return max(0.0, float(raw))
                except (TypeError, ValueError):
                    pass
        return float(2 ** attempt)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> LLMResponse:
        import time

        from groq import RateLimitError

        client = self._ensure_client()
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        # Retry on 429 (TPM cap), honoring retry-after; re-raise other errors and
        # the final 429 once the retry budget is exhausted.
        for attempt in range(self._max_retries + 1):
            try:
                completion = client.chat.completions.create(**kwargs)
                break
            except RateLimitError as exc:
                if attempt >= self._max_retries:
                    raise
                delay = min(
                    self._retry_after_seconds(exc, attempt) + 0.05,
                    self._max_backoff_seconds,
                )
                time.sleep(delay)

        text = completion.choices[0].message.content or ""
        return LLMResponse(text=text, model=self._model, raw=completion)


class GeminiLLMProvider(LLMProvider):
    """A Google Gemini chat-completion provider.

    The ``google-generativeai`` SDK and model handle are created lazily on first
    :meth:`complete`, so neither the SDK nor an API key is required to import
    this module or build the default orchestrator.
    """

    def __init__(self, *, api_key: str | None, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._model_handle = None  # built lazily

    @property
    def model_id(self) -> str:
        return self._model

    def _ensure_model(self):
        if self._model_handle is None:
            import google.generativeai as genai

            if not self._api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not configured; cannot call the Gemini "
                    "LLM provider. Set it in the environment or inject a stub "
                    "orchestrator."
                )
            genai.configure(api_key=self._api_key)
            self._model_handle = genai.GenerativeModel(self._model)
        return self._model_handle

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> LLMResponse:
        model = self._ensure_model()
        # Flatten the chat into a single prompt: a leading system message becomes
        # an instruction prefix, the rest are concatenated in order.
        prompt = "\n\n".join(f"{m.role.upper()}: {m.content}" for m in messages)
        generation_config: dict = {"temperature": temperature}
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        if response_format == "json":
            generation_config["response_mime_type"] = "application/json"
        response = model.generate_content(
            prompt, generation_config=generation_config
        )
        return LLMResponse(text=response.text or "", model=self._model, raw=response)


class OpenRouterLLMProvider(LLMProvider):
    """An OpenRouter chat-completion provider using the OpenAI SDK.

    OpenRouter provides access to various models (including free ones like
    'meta-llama/llama-3.3-70b-instruct' or 'google/gemini-2.5-flash').
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        max_retries: int = 5,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_retries = max(0, int(max_retries))
        self._max_backoff_seconds = max(0.0, float(max_backoff_seconds))
        self._client = None  # built lazily

    @property
    def model_id(self) -> str:
        return self._model

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            if not self._api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not configured; cannot call the OpenRouter "
                    "provider. Set it in the environment or inject a stub."
                )
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://openrouter.ai/api/v1",
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _retry_after_seconds(exc: Exception, attempt: int) -> float:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) if response else None
        if headers is not None:
            raw = headers.get("retry-after")
            if raw:
                try:
                    return max(0.0, float(raw))
                except (TypeError, ValueError):
                    pass
        return float(2 ** attempt)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: str | None = None,
    ) -> LLMResponse:
        import time
        from openai import RateLimitError

        client = self._ensure_client()
        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self._max_retries + 1):
            try:
                extra_headers = {
                    "HTTP-Referer": "https://github.com/talentlens/icrs",
                    "X-Title": "ICRS Candidate Ranking System",
                }
                completion = client.chat.completions.create(
                    extra_headers=extra_headers,
                    **kwargs
                )
                break
            except RateLimitError as exc:
                if attempt >= self._max_retries:
                    raise
                delay = min(
                    self._retry_after_seconds(exc, attempt) + 0.05,
                    self._max_backoff_seconds,
                )
                time.sleep(delay)

        text = completion.choices[0].message.content or ""
        return LLMResponse(text=text, model=self._model, raw=completion)


class LocalEmbeddingProvider(EmbeddingProvider):
    """A local sentence-transformers embedding provider.

    The model is downloaded/loaded lazily on first :meth:`embed`/:meth:`embed_batch`,
    so constructing this provider (and the default orchestrator) loads no weights
    and touches no network.
    """

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        device: str,
        max_tokens: int,
        cache_dir: str | None = None,
    ) -> None:
        self._model_id = model
        self._dim = dim
        self._device = device
        self._max_tokens = max_tokens
        self._cache_dir = self._resolve_cache_dir(cache_dir)
        self._model = None  # loaded lazily

    @staticmethod
    def _resolve_cache_dir(cache_dir: str | None) -> Path:
        root = Path(cache_dir or ".cache/huggingface").expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[2] / root
        return root

    def _configure_cache_env(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self._cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(self._cache_dir / "hub"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(self._cache_dir / "sentence-transformers"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self._cache_dir / "transformers"))

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensionality(self) -> int:
        return self._dim

    @property
    def max_input_tokens(self) -> int:
        return self._max_tokens

    def _ensure_model(self):
        if self._model is None:
            self._configure_cache_env()
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_id,
                device=self._device,
                cache_folder=str(self._cache_dir / "sentence-transformers"),
            )
        return self._model

    def embed(self, text: str) -> Vector:
        model = self._ensure_model()
        vector = model.encode(text, normalize_embeddings=False)
        return [float(x) for x in vector]

    def embed_batch(self, texts: Sequence[str]) -> list[Vector]:
        model = self._ensure_model()
        vectors = model.encode(list(texts), normalize_embeddings=False)
        return [[float(x) for x in v] for v in vectors]


def build_default_provider_registry(settings: Settings) -> LLMProviderRegistry:
    """Build the per-task LLM provider registry from ``settings`` (lazy clients).

    Routes DECOMPOSE, ENRICH and RERANK to OpenRouter if openrouter_api_key is set,
    otherwise defaults to Groq. Gemini is bound to EXPLAIN. No network call or SDK
    import happens here — the providers build their clients lazily on first use.
    """

    if settings.openrouter_api_key:
        default_provider = OpenRouterLLMProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
            max_retries=settings.groq_max_retries,
        )
    else:
        default_provider = GroqLLMProvider(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            max_retries=settings.groq_max_retries,
        )

    gemini = GeminiLLMProvider(
        api_key=settings.google_api_key, model=settings.gemini_model
    )
    registry = LLMProviderRegistry()
    registry.register(LLMTask.DECOMPOSE, default_provider)
    registry.register(LLMTask.ENRICH, default_provider)
    registry.register(LLMTask.RERANK, default_provider)
    registry.register(LLMTask.EXPLAIN, gemini)
    return registry


def build_default_orchestrator(settings: Settings | None = None):
    """Wire the production :class:`RankingOrchestrator` from config (no I/O here).

    Constructs the JD decomposer, candidate enricher, embedding generator,
    reranker, and explanation generator from config-selected providers and
    assembles them into a :class:`~icrs.pipeline.orchestrator.RankingOrchestrator`.
    All providers build their clients/models lazily, so calling this performs no
    network request and loads no model weights — that is deferred to the first
    request that exercises each stage.
    """

    # Imported here (not at module import) to keep this module import-light and
    # to mirror the lazy-wiring contract.
    from icrs.pipeline.embedding import EmbeddingGenerator
    from icrs.pipeline.enricher import CandidateEnricher
    from icrs.pipeline.explanation import ExplanationGenerator
    from icrs.pipeline.jd_decomposer import JDDecomposer
    from icrs.pipeline.orchestrator import RankingOrchestrator
    from icrs.pipeline.reranker import Reranker

    settings = settings or get_settings()
    registry = build_default_provider_registry(settings)
    embedding_provider = LocalEmbeddingProvider(
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        device=settings.embedding_device,
        max_tokens=settings.max_input_tokens,
        cache_dir=settings.model_cache_dir,
    )

    return RankingOrchestrator(
        decomposer=JDDecomposer(registry),
        enricher=CandidateEnricher(registry, semantic_mode=settings.enrichment_mode),
        embedder=EmbeddingGenerator(embedding_provider),
        reranker=Reranker(registry, k=settings.rerank_k),
        explainer=ExplanationGenerator(registry),
        explain_top_n=settings.explain_top_n,
    )


__all__ = [
    "GroqLLMProvider",
    "GeminiLLMProvider",
    "OpenRouterLLMProvider",
    "LocalEmbeddingProvider",
    "build_default_provider_registry",
    "build_default_orchestrator",
]

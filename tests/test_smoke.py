"""Smoke tests verifying the Task 1 scaffold imports and wires together cleanly.

These exercise the package layout, the typed settings loader, the abstract
provider interfaces, and the task-keyed LLM provider registry — without any API
keys, network access, or heavy ML dependencies.
"""

from __future__ import annotations

import importlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from icrs.config import K_MAX, K_MIN, Settings, get_settings
from icrs.providers import (
    EmbeddingProvider,
    LLMMessage,
    LLMProvider,
    LLMProviderRegistry,
    LLMResponse,
    LLMTask,
    VectorStore,
)


def test_packages_import() -> None:
    """All scaffolded packages import without error."""

    for module in (
        "icrs",
        "icrs.config",
        "icrs.models",
        "icrs.providers",
        "icrs.pipeline",
        "icrs.scoring",
        "icrs.persistence",
        "icrs.ui",
    ):
        assert importlib.import_module(module) is not None


def test_default_settings_have_free_provider_defaults() -> None:
    """Defaults point at the free / local providers and carry no secrets."""

    cfg = Settings(_env_file=None)
    # Local open-source embedding model, 1024-dim, CPU.
    assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
    assert cfg.embedding_dim == 1024
    assert cfg.embedding_device == "cpu"
    # Free-tier LLM models.
    assert cfg.groq_model == "llama-3.3-70b-versatile"
    assert cfg.openrouter_model == "google/gemini-2.5-flash:free"
    assert cfg.gemini_model == "gemini-2.5-flash"
    # No secrets baked in.
    assert cfg.groq_api_key is None
    assert cfg.google_api_key is None
    assert cfg.openrouter_api_key is None
    # K bound within the allowed range.
    assert K_MIN <= cfg.rerank_k <= K_MAX


def test_get_settings_is_cached() -> None:
    """The settings accessor returns a cached singleton."""

    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()


def test_rerank_k_must_be_in_bounds() -> None:
    """Out-of-range K is rejected by the typed settings validator."""

    with pytest.raises(ValueError):
        Settings(_env_file=None, rerank_k=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, rerank_k=51)


def test_embedding_dim_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, embedding_dim=0)


# ----- A minimal in-memory fake provider exercises the abstract contracts -----


class _FakeEmbedding(EmbeddingProvider):
    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def model_id(self) -> str:
        return "fake-embed"

    @property
    def dimensionality(self) -> int:
        return self._dim

    @property
    def max_input_tokens(self) -> int:
        return 512

    def embed(self, text: str):
        return [float(len(text) % 7)] * self._dim

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class _EchoLLM(LLMProvider):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def model_id(self) -> str:
        return self._name

    def complete(self, messages, *, temperature=0.0, max_tokens=None, response_format=None):
        last = messages[-1].content if messages else ""
        return LLMResponse(text=f"{self._name}:{last}", model=self._name)


def test_embedding_provider_contract() -> None:
    emb = _FakeEmbedding(dim=8)
    vec = emb.embed("hello")
    assert len(vec) == emb.dimensionality == 8
    batch = emb.embed_batch(["a", "bb", "ccc"])
    assert len(batch) == 3 and all(len(v) == 8 for v in batch)


def test_llm_registry_routes_tasks_to_distinct_providers() -> None:
    """Decompose + rerank route to one provider, explain to another."""

    registry = LLMProviderRegistry()
    groq_like = _EchoLLM("groq-llama")
    gemini_like = _EchoLLM("gemini")

    registry.register(LLMTask.DECOMPOSE, groq_like)
    registry.register(LLMTask.RERANK, groq_like)
    registry.register(LLMTask.EXPLAIN, gemini_like)

    assert registry.get(LLMTask.DECOMPOSE).model_id == "groq-llama"
    assert registry.get(LLMTask.RERANK).model_id == "groq-llama"
    assert registry.get(LLMTask.EXPLAIN).model_id == "gemini"
    assert set(registry.tasks()) == {LLMTask.DECOMPOSE, LLMTask.RERANK, LLMTask.EXPLAIN}

    resp = registry.get(LLMTask.EXPLAIN).complete([LLMMessage(role="user", content="why?")])
    assert resp.text == "gemini:why?"


def test_llm_registry_raises_for_unregistered_task() -> None:
    registry = LLMProviderRegistry()
    assert not registry.is_registered(LLMTask.DECOMPOSE)
    with pytest.raises(KeyError):
        registry.get(LLMTask.DECOMPOSE)


def test_vector_store_is_abstract() -> None:
    """VectorStore cannot be instantiated directly (it is an interface)."""

    with pytest.raises(TypeError):
        VectorStore()  # type: ignore[abstract]


@given(st.integers(min_value=K_MIN, max_value=K_MAX))
def test_in_bounds_k_always_accepted(k: int) -> None:
    """Property: any K within [1, 50] is accepted by the settings validator."""

    cfg = Settings(_env_file=None, rerank_k=k)
    assert cfg.rerank_k == k


def test_build_default_provider_registry() -> None:
    """Registry conditionally selects Groq or OpenRouter based on config keys."""
    from icrs.api.providers import build_default_provider_registry, GroqLLMProvider, OpenRouterLLMProvider

    # Case 1: No OpenRouter key set -> use Groq provider
    cfg_groq = Settings(_env_file=None, google_api_key="g", groq_api_key="gr", openrouter_api_key=None)
    registry_groq = build_default_provider_registry(cfg_groq)
    decompose_prov = registry_groq.get(LLMTask.DECOMPOSE)
    assert isinstance(decompose_prov, GroqLLMProvider)
    assert decompose_prov.model_id == "llama-3.3-70b-versatile"

    # Case 2: OpenRouter key set -> use OpenRouter provider
    cfg_or = Settings(_env_file=None, google_api_key="g", openrouter_api_key="or", openrouter_model="some-or-model")
    registry_or = build_default_provider_registry(cfg_or)
    decompose_prov_or = registry_or.get(LLMTask.DECOMPOSE)
    assert isinstance(decompose_prov_or, OpenRouterLLMProvider)
    assert decompose_prov_or.model_id == "some-or-model"

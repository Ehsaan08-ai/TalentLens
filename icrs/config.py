"""Typed settings / configuration loader for ICRS.

Configuration is read from environment variables and an optional ``.env`` file
(see ``.env.example``). Secrets (API keys) are NEVER hardcoded — they are read
from the environment only, and default to ``None`` when absent so that the rest
of the system can be imported and unit-tested without credentials present.

The settings object is the single source of truth for:
    - LLM provider API keys (Groq, Google) and model ids
    - the embedding model id, dimensionality, and device
    - pipeline limits (max input tokens, rerank bound K)
    - persistence URLs (PostgreSQL, Qdrant)
    - the fixed random seed used by deterministic stages

Concrete providers are intentionally swappable; this module only carries the
configuration that selects and parameterizes them.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Rerank bound K is constrained to this inclusive range (Requirement 8.1).
K_MIN = 1
K_MAX = 50


class Settings(BaseSettings):
    """Process-wide configuration, populated from env vars / ``.env``.

    Field names map to environment variables via the ``ICRS_`` prefix, except
    the two API keys which use their conventional names (``GROQ_API_KEY`` and
    ``GOOGLE_API_KEY``) so they line up with the provider SDKs' expectations.
    """

    model_config = SettingsConfigDict(
        env_prefix="ICRS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ----- LLM provider API keys (no defaults; read from env, may be absent) -----
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    # ----- Redis configurations -----
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    redis_ttl: int = 86400

    # ----- LLM model selection (overridable) -----
    # Groq-hosted model used for JD decomposition + contextual reranking.
    groq_model: str = "llama-3.3-70b-versatile"
    # OpenRouter model (used if openrouter_api_key is set).
    openrouter_model: str = "google/gemini-2.5-flash:free"
    # Gemini model used for explanation generation.
    gemini_model: str = "gemini-2.5-flash"

    # ----- Embedding provider (local, free, swappable) -----
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"

    # ----- Pipeline limits -----
    max_input_tokens: int = 512
    rerank_k: int = 10

    # Groq free-tier rate-limit resilience: retry a 429'd call this many times,
    # honoring the server's retry-after, before giving up.
    groq_max_retries: int = 5

    # ----- Persistence -----
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/icrs"
    qdrant_url: str = "http://localhost:6333"

    # ----- Determinism -----
    random_seed: int = 1729

    @field_validator("embedding_dim")
    @classmethod
    def _dim_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        return value

    @field_validator("max_input_tokens")
    @classmethod
    def _tokens_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_input_tokens must be a positive integer")
        return value

    @field_validator("rerank_k")
    @classmethod
    def _k_in_bounds(cls, value: int) -> int:
        if not (K_MIN <= value <= K_MAX):
            raise ValueError(f"rerank_k must be within [{K_MIN}, {K_MAX}], got {value}")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so repeated calls are cheap and consistent within a process. Tests
    that need to override values can call ``get_settings.cache_clear()`` after
    setting environment variables.
    """

    return Settings()

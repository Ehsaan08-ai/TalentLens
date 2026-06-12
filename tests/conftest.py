"""Shared pytest fixtures and Hypothesis configuration for ICRS.

A deterministic Hypothesis profile (``icrs-deterministic``) is registered and
selected by default so the deterministic pipeline stages (hard filter, dense /
sparse similarity, composite fusion) are exercised with a fixed seed and
reproducible example generation. This supports Requirement 2.3 (identical inputs
+ fixed model versions yield identical deterministic-stage outputs).
"""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, Verbosity, settings

# Seed used by deterministic stages and Hypothesis. Mirrors ICRS_RANDOM_SEED so
# tests and runtime share the same fixed seed.
DETERMINISTIC_SEED = int(os.environ.get("ICRS_RANDOM_SEED", "1729"))


# A deterministic profile: fixed derandomized seed, generous deadline so the
# (potentially heavier) pipeline stages are not flagged as slow.
settings.register_profile(
    "icrs-deterministic",
    settings(
        derandomize=True,
        deadline=None,
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
        verbosity=Verbosity.normal,
    ),
)

# A quick profile for fast local iteration.
settings.register_profile(
    "icrs-fast",
    settings(parent=settings.get_profile("icrs-deterministic"), max_examples=25),
)

# Select the deterministic profile unless overridden via HYPOTHESIS_PROFILE.
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "icrs-deterministic"))


@pytest.fixture(scope="session")
def deterministic_seed() -> int:
    """The fixed seed shared by deterministic stages and tests."""

    return DETERMINISTIC_SEED


@pytest.fixture()
def settings_env(monkeypatch: pytest.MonkeyPatch):
    """Helper to set ICRS settings via env vars and reset the cached singleton.

    Usage::

        def test_x(settings_env):
            cfg = settings_env(ICRS_RERANK_K="5")
            assert cfg.rerank_k == 5
    """

    from icrs.config import get_settings

    def _apply(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        get_settings.cache_clear()
        return get_settings()

    get_settings.cache_clear()
    yield _apply
    get_settings.cache_clear()

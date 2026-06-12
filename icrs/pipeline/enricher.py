"""Candidate enricher for ICRS.

The :class:`CandidateEnricher` is the pipeline component responsible for turning
a heterogeneous :class:`~icrs.models.candidate.RawCandidate` into a canonical
:class:`~icrs.models.candidate.NormalizedProfile` and an
:class:`~icrs.models.candidate.EnrichedProfile`.

Two responsibilities, two tasks (kept in separate mixin modules so neither
concern edits the other's source):

* ``normalize`` (Task 4.1) — deterministic canonicalization, provided by
  :class:`~icrs.pipeline.normalization.ProfileNormalizationMixin`. No LLM calls,
  usable with no provider configured.
* ``enrich`` (Task 4.2) — three-tier signal enrichment (Tier 1 structural, Tier
  2 semantic via a batched LLM call, Tier 3 behavioral via an injectable
  source), provided by :class:`~icrs.pipeline.enrichment.EnrichmentMixin`.

Dependency injection:
    The Tier 2 LLM provider is reached only through the abstract provider
    interface — pass an :class:`~icrs.providers.base.LLMProviderRegistry`
    (resolved under :attr:`~icrs.providers.base.LLMTask.ENRICH`) or, for
    convenience/testing, an explicit :class:`~icrs.providers.base.LLMProvider`.
    The Tier 3 :class:`~icrs.pipeline.enrichment.BehavioralSignalSource` is also
    injectable and defaults to the no-op
    :class:`~icrs.pipeline.enrichment.NullBehavioralSignalSource`.

    Both are optional: a ``CandidateEnricher()`` constructed with no arguments
    still supports :meth:`normalize` (which is deterministic). The LLM provider
    is required only when :meth:`enrich` is actually called.
"""

from __future__ import annotations

from icrs.pipeline.enrichment import (
    BehavioralSignalSource,
    DEFAULT_BEHAVIORAL_FETCH_TIMEOUT_SECONDS,
    EnrichmentError,
    EnrichmentMixin,
    NullBehavioralSignalSource,
    freshness_weight,
)
from icrs.pipeline.normalization import (
    ProfileNormalizationMixin,
    ProfileValidationError,
)
from icrs.providers.base import LLMProvider, LLMProviderRegistry, LLMTask


class CandidateEnricher(ProfileNormalizationMixin, EnrichmentMixin):
    """Normalizes and enriches candidate profiles.

    Exposes :meth:`normalize` (Task 4.1, deterministic) and :meth:`enrich`
    (Task 4.2, three-tier signal enrichment).
    """

    def __init__(
        self,
        registry: LLMProviderRegistry | None = None,
        *,
        llm_provider: LLMProvider | None = None,
        behavioral_source: BehavioralSignalSource | None = None,
        behavioral_timeout_seconds: float | None = (
            DEFAULT_BEHAVIORAL_FETCH_TIMEOUT_SECONDS
        ),
        task: LLMTask = LLMTask.ENRICH,
    ) -> None:
        """Construct an enricher.

        Args:
            registry: provider registry; the Tier 2 enrichment provider is
                resolved from it via ``task`` when :meth:`enrich` is called.
            llm_provider: an explicit provider (takes precedence over
                ``registry``); convenient for tests.
            behavioral_source: the Tier 3 source; defaults to a no-op source that
                returns no signals (the PoC never calls real external APIs).
            behavioral_timeout_seconds: wall-clock budget for a single external
                behavioral fetch (Requirement 9.1). A fetch that does not complete
                within this many seconds — or that otherwise fails — is treated as
                unavailable, so the behavioral tier records availability 0 and the
                candidate proceeds on the neutral prior. Defaults to 10 seconds.
                Only enforced for a real (non-null) ``behavioral_source``; pass
                ``None`` to disable the timeout entirely.
            task: the :class:`LLMTask` used to resolve the provider from the
                registry. Defaults to :attr:`LLMTask.ENRICH`.

        Constructing with no arguments is supported and keeps :meth:`normalize`
        fully usable; only :meth:`enrich` requires an LLM provider.
        """

        self._enrich_registry = registry
        self._enrich_provider = llm_provider
        self._enrich_task = task
        self._behavioral_source: BehavioralSignalSource = (
            behavioral_source or NullBehavioralSignalSource()
        )
        self._behavioral_timeout_seconds = behavioral_timeout_seconds
        self._enrich_cache: dict = {}


__all__ = [
    "CandidateEnricher",
    "ProfileValidationError",
    "EnrichmentError",
    "BehavioralSignalSource",
    "NullBehavioralSignalSource",
    "freshness_weight",
]

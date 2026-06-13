"""Unit tests for three-tier signal enrichment (Task 4.2).

Exercises :meth:`CandidateEnricher.enrich` against a deterministic stub
``LLMProvider`` and a stub :class:`BehavioralSignalSource` (no real APIs). Covers:

    - Tier 1 structural derivation with not-present (``None``) marking (Req 3.2)
    - Tier 2 semantic inference parsed into the shared enums (Req 3.3)
    - Tier 3 freshness weighting: in [0,1] and monotonically non-increasing (Req 3.4)
    - per-tier signal_availability fractions, incl. 0 coverage when absent (Req 3.5)
    - enrichment cached by content hash (single LLM call across repeats)
    - back-compat: a no-arg enricher still normalizes; enrich() needs a provider
"""

from __future__ import annotations

import json

import pytest

from icrs.models.candidate import (
    BehavioralSignal,
    NormalizedProfile,
    Role,
)
from icrs.models.enums import DepthBreadth, SignalTier, TrajectoryArc
from icrs.pipeline.enricher import CandidateEnricher
from icrs.pipeline.enrichment import (
    EXPECTED_BEHAVIORAL_SOURCES,
    BehavioralSignalSource,
    EnrichmentError,
    NullBehavioralSignalSource,
    freshness_weight,
)
from icrs.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderRegistry,
    LLMResponse,
    LLMTask,
)


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class StubLLM(LLMProvider):
    """A scripted LLM provider: returns a queued response per ``complete`` call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    @property
    def model_id(self) -> str:
        return "stub-enrich-llm"

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens=None,
        response_format=None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("StubLLM.complete called more times than scripted")
        return LLMResponse(text=self._responses.pop(0), model=self.model_id)


class StubBehavioralSource(BehavioralSignalSource):
    """A stub Tier 3 source returning a fixed signal list and recording calls."""

    def __init__(self, signals: list[BehavioralSignal]) -> None:
        self._signals = list(signals)
        self.calls = 0

    def fetch(self, profile: NormalizedProfile) -> list[BehavioralSignal]:
        self.calls += 1
        return list(self._signals)


def _semantic_payload(**overrides) -> dict:
    payload = {
        "inferred_responsibilities": ["led platform migration", "mentored engineers"],
        "implicit_skills": ["distributed systems", "leadership"],
        "trajectory_arc": "ACCELERATING",
        "depth_breadth": "SPECIALIST",
    }
    payload.update(overrides)
    return payload


def _profile(**overrides) -> NormalizedProfile:
    base = dict(
        roles=[
            Role(title="Software Engineer", company="Acme"),
            Role(title="Senior Engineer", company="Globex"),
        ],
        education=[],
        certifications=["AWS SA"],
        explicit_skills=["Python", "Go"],
        total_tenure_months=36,
    )
    base.update(overrides)
    return NormalizedProfile(**base)


def _enricher(
    responses: list[str],
    *,
    source: BehavioralSignalSource | None = None,
) -> tuple[CandidateEnricher, StubLLM, BehavioralSignalSource]:
    stub = StubLLM(responses)
    src = source if source is not None else NullBehavioralSignalSource()
    return (
        CandidateEnricher(llm_provider=stub, behavioral_source=src),
        stub,
        src,
    )


# --------------------------------------------------------------------------- #
# Tier 1 — structural derivation with not-present marking (Req 3.2)
# --------------------------------------------------------------------------- #
def test_structural_signals_marks_absent_fields_not_present() -> None:
    enricher, _, _ = _enricher([json.dumps(_semantic_payload())])
    # roles + skills present; tenure 0, no education, no certs -> not-present.
    profile = _profile(
        education=[],
        certifications=[],
        explicit_skills=["Python"],
        total_tenure_months=0,
    )

    signals = enricher.derive_structural_signals(profile)

    assert signals["roles"] == ["Software Engineer", "Senior Engineer"]
    assert signals["explicit_skills"] == ["Python"]
    # Absent / empty fields are None ("not-present"), never defaulted.
    assert signals["tenure"] is None
    assert signals["education"] is None
    assert signals["certifications"] is None


def test_structural_signals_all_present() -> None:
    from icrs.models.candidate import Education

    enricher, _, _ = _enricher([json.dumps(_semantic_payload())])
    profile = _profile(education=[Education(institution="MIT", degree="BSc")])

    signals = enricher.derive_structural_signals(profile)
    assert all(v is not None for v in signals.values())


# --------------------------------------------------------------------------- #
# Tier 2 — semantic inference parsed into enums (Req 3.3)
# --------------------------------------------------------------------------- #
def test_semantic_signals_parsed_into_enums() -> None:
    enricher, stub, _ = _enricher([json.dumps(_semantic_payload())])

    enriched = enricher.enrich(_profile())

    assert enriched.inferred_responsibilities == [
        "led platform migration",
        "mentored engineers",
    ]
    assert enriched.implicit_skills == ["distributed systems", "leadership"]
    assert enriched.trajectory_arc is TrajectoryArc.ACCELERATING
    assert enriched.depth_breadth is DepthBreadth.SPECIALIST
    assert len(stub.calls) == 1
    # The profile is fenced as untrusted data in the prompt (security).
    user_msg = stub.calls[0][1].content
    assert "CANDIDATE_PROFILE_BEGIN" in user_msg
    system_msg = stub.calls[0][0].content
    assert "untrusted" in system_msg.lower()


def test_semantic_invalid_enum_values_become_not_present() -> None:
    payload = _semantic_payload(trajectory_arc="ROCKETING", depth_breadth="???")
    enricher, _, _ = _enricher([json.dumps(payload)])

    enriched = enricher.enrich(_profile())

    assert enriched.trajectory_arc is None
    assert enriched.depth_breadth is None
    # Lists still parse.
    assert enriched.inferred_responsibilities


def test_semantic_malformed_json_degrades_to_absent() -> None:
    enricher, _, _ = _enricher(["not json at all"])

    enriched = enricher.enrich(_profile())

    assert enriched.inferred_responsibilities == []
    assert enriched.implicit_skills == []
    assert enriched.trajectory_arc is None
    assert enriched.depth_breadth is None
    # Semantic availability is therefore 0.
    assert enriched.signal_availability[SignalTier.SEMANTIC] == 0.0


# --------------------------------------------------------------------------- #
# Tier 3 — freshness weighting (Req 3.4)
# --------------------------------------------------------------------------- #
def test_freshness_weight_in_unit_interval_and_endpoints() -> None:
    assert freshness_weight(0) == 1.0
    for days in (0, 30, 365, 1000, 10_000):
        w = freshness_weight(days)
        assert 0.0 <= w <= 1.0


def test_freshness_weight_monotonically_non_increasing() -> None:
    ages = [0, 1, 7, 30, 90, 365, 730, 1825, 5000]
    weights = [freshness_weight(a) for a in ages]
    for earlier, later in zip(weights, weights[1:]):
        assert later <= earlier


def test_freshness_weight_half_life() -> None:
    assert freshness_weight(365, half_life_days=365) == pytest.approx(0.5, abs=1e-9)
    assert freshness_weight(730, half_life_days=365) == pytest.approx(0.25, abs=1e-9)


def test_negative_recency_clamped_to_current() -> None:
    assert freshness_weight(-5) == 1.0


def test_behavioral_signals_attached_from_source() -> None:
    signals = [
        BehavioralSignal(
            source="github", metric="commits", value=120.0, recency_days=10
        ),
        BehavioralSignal(
            source="linkedin", metric="endorsements", value=30.0, recency_days=400
        ),
    ]
    enricher, _, src = _enricher(
        [json.dumps(_semantic_payload())], source=StubBehavioralSource(signals)
    )

    enriched = enricher.enrich(_profile())

    assert len(enriched.behavioral_signals) == 2
    assert {s.source for s in enriched.behavioral_signals} == {"github", "linkedin"}
    assert src.calls == 1


# --------------------------------------------------------------------------- #
# signal_availability fractions per tier (Req 3.5)
# --------------------------------------------------------------------------- #
def test_signal_availability_fractions_per_tier() -> None:
    from icrs.models.candidate import Education

    signals = [
        BehavioralSignal(source="github", metric="commits", value=1.0, recency_days=5),
        BehavioralSignal(
            source="linkedin", metric="posts", value=1.0, recency_days=20
        ),
    ]
    enricher, _, _ = _enricher(
        [json.dumps(_semantic_payload())], source=StubBehavioralSource(signals)
    )
    profile = _profile(education=[Education(institution="MIT")])

    enriched = enricher.enrich(profile)

    # All 5 structural fields present -> 1.0
    assert enriched.signal_availability[SignalTier.STRUCTURAL] == pytest.approx(1.0)
    # All 4 semantic fields present -> 1.0
    assert enriched.signal_availability[SignalTier.SEMANTIC] == pytest.approx(1.0)
    # 2 of 3 expected behavioral sources covered -> 2/3
    assert enriched.signal_availability[SignalTier.BEHAVIORAL] == pytest.approx(
        2 / len(EXPECTED_BEHAVIORAL_SOURCES)
    )


def test_behavioral_zero_coverage_when_no_signals() -> None:
    # Default null source -> no behavioral data -> 0 coverage (not fabricated).
    enricher, _, _ = _enricher([json.dumps(_semantic_payload())])

    enriched = enricher.enrich(_profile())

    assert enriched.signal_availability[SignalTier.BEHAVIORAL] == 0.0


def test_partial_structural_availability_is_a_fraction() -> None:
    enricher, _, _ = _enricher([json.dumps(_semantic_payload())])
    # roles + skills present; tenure/education/certs absent -> 2 of 5.
    profile = _profile(
        education=[],
        certifications=[],
        explicit_skills=["Python"],
        total_tenure_months=0,
    )

    enriched = enricher.enrich(profile)

    assert enriched.signal_availability[SignalTier.STRUCTURAL] == pytest.approx(2 / 5)
    assert 0.0 <= enriched.signal_availability[SignalTier.STRUCTURAL] <= 1.0


def test_all_availability_values_in_unit_interval() -> None:
    enricher, _, _ = _enricher([json.dumps(_semantic_payload())])
    enriched = enricher.enrich(_profile())
    for coverage in enriched.signal_availability.values():
        assert 0.0 <= coverage <= 1.0


# --------------------------------------------------------------------------- #
# Caching by content hash
# --------------------------------------------------------------------------- #
def test_enrichment_cached_by_content_hash() -> None:
    src = StubBehavioralSource(
        [BehavioralSignal(source="github", metric="c", value=1.0, recency_days=1)]
    )
    enricher, stub, _ = _enricher([json.dumps(_semantic_payload())], source=src)
    profile = _profile()

    first = enricher.enrich(profile)
    second = enricher.enrich(profile)  # identical content -> cache hit

    assert first.inferred_responsibilities == second.inferred_responsibilities
    assert first.signal_availability == second.signal_availability
    # Only one LLM call and one behavioral fetch despite two enrich() calls.
    assert len(stub.calls) == 1
    assert src.calls == 1


def test_cached_result_is_independent_copy() -> None:
    enricher, _, _ = _enricher([json.dumps(_semantic_payload())])
    profile = _profile()

    first = enricher.enrich(profile)
    first.inferred_responsibilities.append("mutated")
    second = enricher.enrich(profile)

    assert "mutated" not in second.inferred_responsibilities


def test_different_profiles_not_served_from_cache() -> None:
    enricher, stub, _ = _enricher(
        [
            json.dumps(_semantic_payload()),
            json.dumps(_semantic_payload(trajectory_arc="STEADY")),
        ]
    )

    enricher.enrich(_profile(total_tenure_months=12))
    second = enricher.enrich(_profile(total_tenure_months=99))

    assert second.trajectory_arc is TrajectoryArc.STEADY
    assert len(stub.calls) == 2


# --------------------------------------------------------------------------- #
# Provider wiring & back-compat
# --------------------------------------------------------------------------- #
def test_enrich_resolves_provider_from_registry() -> None:
    stub = StubLLM([json.dumps(_semantic_payload())])
    registry = LLMProviderRegistry()
    registry.register(LLMTask.ENRICH, stub)
    enricher = CandidateEnricher(registry)

    enriched = enricher.enrich(_profile())

    assert enriched.trajectory_arc is TrajectoryArc.ACCELERATING
    assert len(stub.calls) == 1


def test_local_semantic_enrichment_makes_no_llm_call() -> None:
    stub = StubLLM([json.dumps(_semantic_payload())])
    enricher = CandidateEnricher(llm_provider=stub, semantic_mode="local")

    enriched = enricher.enrich(_profile())

    assert len(stub.calls) == 0
    assert enriched.inferred_responsibilities
    assert "backend development" in enriched.implicit_skills
    assert enriched.trajectory_arc is TrajectoryArc.ACCELERATING
    assert enriched.depth_breadth is not None
    assert enriched.signal_availability[SignalTier.SEMANTIC] > 0.0


def test_invalid_semantic_mode_rejected() -> None:
    with pytest.raises(ValueError):
        CandidateEnricher(semantic_mode="expensive-but-mysterious")


def test_enrich_without_provider_raises() -> None:
    enricher = CandidateEnricher()  # no provider configured
    with pytest.raises(EnrichmentError):
        enricher.enrich(_profile())


def test_no_arg_enricher_still_normalizes() -> None:
    # Back-compat with Task 4.1: normalize works with no LLM provider.
    from icrs.models.candidate import RawCandidate

    enricher = CandidateEnricher()
    profile = enricher.normalize(
        RawCandidate(
            structured_fields={
                "roles": [{"title": "Dev", "company": "A", "start": "2020-01", "end": "2021-01"}],
                "skills": ["Python"],
            }
        )
    )
    assert profile.roles[0].title == "Dev"
    assert profile.total_tenure_months == 12

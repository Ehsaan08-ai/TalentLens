"""Unit tests for the Task 14.1 Explanation Generator.

These tests exercise :class:`icrs.pipeline.explanation.ExplanationGenerator`
against a deterministic stub :class:`LLMProvider` so no real API is ever called.
They cover Requirements 5.2, 5.4, and 7.1:

    - summary is non-empty and at most 1000 characters (incl. truncation of an
      over-long LLM summary, and a deterministic fallback for empty LLM output)
    - at least one driving signal is always produced (LLM-supplied, or derived
      from the breakdown's highest sub-scores)
    - ``unmet_must_haves`` contains only unsatisfied MUST_HAVE requirements
      (never NICE_TO_HAVE, DISQUALIFYING, or satisfied MUST_HAVEs)
    - ``gaps`` is empty only when every requirement is satisfied
    - Protected_Proxy attributes (dates / graduation years, locations) never
      appear in the prompt
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from icrs.models.candidate import (
    EnrichedProfile,
    Education,
    NormalizedProfile,
    Role,
)
from icrs.models.job import (
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)
from icrs.models.ranking import MAX_SUMMARY_CHARS, SignalBreakdown
from icrs.pipeline.explanation import ExplanationGenerator
from icrs.providers.base import (
    LLMMessage,
    LLMProvider,
    LLMProviderRegistry,
    LLMResponse,
    LLMTask,
)


# --------------------------------------------------------------------------- #
# Stub provider
# --------------------------------------------------------------------------- #
class StubLLM(LLMProvider):
    """A scripted LLM provider returning a queued response per ``complete`` call."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    @property
    def model_id(self) -> str:
        return "stub-explain-llm"

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


def _generator(responses: list[str]) -> tuple[ExplanationGenerator, StubLLM]:
    stub = StubLLM(responses)
    return ExplanationGenerator(provider=stub), stub


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _req(text: str, category: RequirementCategory) -> Requirement:
    return Requirement(
        text=text,
        category=category,
        tier=RequirementTier.STRUCTURAL,
        weight=1.0,
    )


def _reqs(
    *,
    must_haves: list[str],
    nice: list[str] | None = None,
    disqualifying: list[str] | None = None,
    seniority: SeniorityBand = SeniorityBand.SENIOR,
) -> RequirementVector:
    requirements = [_req(t, RequirementCategory.MUST_HAVE) for t in must_haves]
    for t in nice or []:
        requirements.append(_req(t, RequirementCategory.NICE_TO_HAVE))
    for t in disqualifying or []:
        requirements.append(_req(t, RequirementCategory.DISQUALIFYING))
    return RequirementVector(
        role_intent="Build and operate the data platform",
        seniority_band=seniority,
        requirements=requirements,
    )


def _profile(
    *,
    roles: list[Role] | None = None,
    education: list[Education] | None = None,
    skills: list[str] | None = None,
    tenure_months: int = 60,
) -> EnrichedProfile:
    base = NormalizedProfile(
        roles=roles or [Role(title="Senior Python Engineer", company="Acme")],
        education=education or [],
        explicit_skills=skills or ["Python", "AWS"],
        total_tenure_months=tenure_months,
    )
    return EnrichedProfile(base=base)


def _breakdown() -> SignalBreakdown:
    return SignalBreakdown(
        semantic_fit=0.9,
        career_trajectory=0.4,
        behavioral=0.5,
        hard_filter_pass=0.8,
        disqualifying_penalty=0.0,
    )


_GOOD_JSON = json.dumps(
    {
        "summary": "Strong Python engineer with relevant AWS experience.",
        "driving_signals": ["Semantic fit with the role", "Python depth"],
    }
)


# --------------------------------------------------------------------------- #
# Requirement 5.2 — summary non-empty and <= 1000 chars
# --------------------------------------------------------------------------- #
def test_summary_non_empty_and_within_limit() -> None:
    gen, stub = _generator([_GOOD_JSON])

    exp = gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    assert exp.summary.strip()
    assert len(exp.summary) <= MAX_SUMMARY_CHARS
    assert len(stub.calls) == 1


def test_overlong_llm_summary_is_truncated() -> None:
    long_summary = "x" * 5000
    gen, _ = _generator([json.dumps({"summary": long_summary, "driving_signals": ["s"]})])

    exp = gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    assert len(exp.summary) <= MAX_SUMMARY_CHARS
    assert exp.summary  # still non-empty after truncation


def test_empty_llm_summary_falls_back_to_deterministic_summary() -> None:
    gen, _ = _generator([json.dumps({"summary": "   ", "driving_signals": []})])

    exp = gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    assert exp.summary.strip()
    assert len(exp.summary) <= MAX_SUMMARY_CHARS
    # The deterministic fallback names the role intent.
    assert "data platform" in exp.summary


def test_non_json_llm_output_used_as_summary() -> None:
    gen, _ = _generator(["Just plain prose, no JSON here."])

    exp = gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    assert exp.summary == "Just plain prose, no JSON here."


# --------------------------------------------------------------------------- #
# Requirement 5.2 — at least one driving signal
# --------------------------------------------------------------------------- #
def test_driving_signals_from_llm_when_provided() -> None:
    gen, _ = _generator([_GOOD_JSON])

    exp = gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    assert exp.driving_signals == ["Semantic fit with the role", "Python depth"]


def test_driving_signals_derived_from_breakdown_when_llm_gives_none() -> None:
    gen, _ = _generator([json.dumps({"summary": "ok", "driving_signals": []})])

    exp = gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    assert len(exp.driving_signals) >= 1
    # Highest sub-score (semantic_fit=0.9) should be the top driving signal.
    assert exp.driving_signals[0] == "Semantic fit with the role"


def test_at_least_one_driving_signal_without_breakdown_or_llm_signals() -> None:
    gen, _ = _generator([json.dumps({"summary": "ok", "driving_signals": []})])

    exp = gen.explain(_profile(skills=["Go"]), _reqs(must_haves=["Python"]))

    assert len(exp.driving_signals) >= 1


# --------------------------------------------------------------------------- #
# Requirement 5.4 — unmet_must_haves contains only unsatisfied MUST_HAVEs
# --------------------------------------------------------------------------- #
def test_unmet_must_haves_only_unsatisfied_must_haves() -> None:
    gen, _ = _generator([_GOOD_JSON])
    profile = _profile(
        roles=[Role(title="Python Engineer", company="Acme")],
        skills=["Python"],
    )
    reqs = _reqs(
        must_haves=["Python", "Kubernetes"],   # Python satisfied, Kubernetes not
        nice=["GraphQL"],                       # nice-to-have, never an unmet must-have
        disqualifying=["Kubernetes restricted"],  # disqualifying, never included
    )

    exp = gen.explain(profile, reqs, _breakdown())

    # Only the unsatisfied MUST_HAVE appears.
    assert exp.unmet_must_haves == ["Kubernetes"]
    # Satisfied must-have, nice-to-have, and disqualifying are all excluded.
    assert "Python" not in exp.unmet_must_haves
    assert "GraphQL" not in exp.unmet_must_haves
    assert "Kubernetes restricted" not in exp.unmet_must_haves


def test_unmet_must_haves_empty_when_all_must_haves_satisfied() -> None:
    gen, _ = _generator([_GOOD_JSON])
    profile = _profile(skills=["Python", "AWS"])
    reqs = _reqs(must_haves=["Python", "AWS"])

    exp = gen.explain(profile, reqs, _breakdown())

    assert exp.unmet_must_haves == []


# --------------------------------------------------------------------------- #
# Requirement 5.2 — gaps empty only when no unmet requirements
# --------------------------------------------------------------------------- #
def test_gaps_empty_when_all_requirements_met() -> None:
    gen, _ = _generator([_GOOD_JSON])
    profile = _profile(skills=["Python", "AWS", "GraphQL"])
    reqs = _reqs(must_haves=["Python", "AWS"], nice=["GraphQL"])

    exp = gen.explain(profile, reqs, _breakdown())

    assert exp.gaps == []


def test_gaps_non_empty_when_a_requirement_is_unmet() -> None:
    gen, _ = _generator([_GOOD_JSON])
    profile = _profile(skills=["Python"])
    reqs = _reqs(must_haves=["Python"], nice=["GraphQL"])  # GraphQL unmet

    exp = gen.explain(profile, reqs, _breakdown())

    assert "GraphQL" in exp.gaps


# --------------------------------------------------------------------------- #
# Requirement 7.1 — Protected_Proxy attributes excluded from the prompt
# --------------------------------------------------------------------------- #
def test_protected_proxy_dates_never_appear_in_prompt() -> None:
    gen, stub = _generator([_GOOD_JSON])
    profile = _profile(
        roles=[
            Role(
                title="Senior Engineer",
                company="Acme",
                start=date(2011, 3, 1),
                end=date(2019, 7, 1),
            )
        ],
        education=[
            Education(
                institution="State University",
                degree="BSc",
                field_of_study="Computer Science",
                start=date(2007, 9, 1),
                end=date(2011, 6, 1),
            )
        ],
        skills=["Python"],
    )

    gen.explain(profile, _reqs(must_haves=["Python"]), _breakdown())

    prompt = "\n".join(m.content for m in stub.calls[0])
    # Age-proxy years from role and education dates must not leak.
    for year in ("2007", "2011", "2019"):
        assert year not in prompt
    # Institution (pedigree/location proxy) is excluded; job-relevant degree kept.
    assert "State University" not in prompt
    assert "Computer Science" in prompt
    # The job-relevant title/skill is present.
    assert "Senior Engineer" in prompt
    assert "Python" in prompt


def test_prompt_fences_candidate_signals_as_untrusted_data() -> None:
    gen, stub = _generator([_GOOD_JSON])

    gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    system = stub.calls[0][0].content
    user = stub.calls[0][1].content
    assert "untrusted" in system.lower()
    assert "CANDIDATE_SIGNALS_BEGIN" in user
    assert "CANDIDATE_SIGNALS_END" in user


# --------------------------------------------------------------------------- #
# Registry wiring (LLMTask.EXPLAIN)
# --------------------------------------------------------------------------- #
def test_resolves_provider_from_registry_by_explain_task() -> None:
    stub = StubLLM([_GOOD_JSON])
    registry = LLMProviderRegistry()
    registry.register(LLMTask.EXPLAIN, stub)
    gen = ExplanationGenerator(registry)

    exp = gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    assert exp.summary
    assert len(stub.calls) == 1


def test_requires_a_provider_or_registry() -> None:
    with pytest.raises(ValueError):
        ExplanationGenerator()


def test_llm_failure_falls_back_to_deterministic_summary() -> None:
    class FailingLLM(LLMProvider):
        @property
        def model_id(self) -> str:
            return "failing"

        def complete(self, messages, **kwargs) -> LLMResponse:
            raise RuntimeError("provider down")

    gen = ExplanationGenerator(provider=FailingLLM())

    exp = gen.explain(_profile(), _reqs(must_haves=["Python"]), _breakdown())

    assert exp.summary.strip()
    assert len(exp.driving_signals) >= 1

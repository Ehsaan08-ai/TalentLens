"""Unit tests for the Task 13.1 LLM contextual reranker.

These tests exercise :class:`icrs.pipeline.reranker.Reranker` against a
deterministic stub :class:`LLMProvider` so no real API is ever called. They cover
Requirement 8:

    - 8.1: only the K highest composite-scored candidates reach the LLM when
      survivors exceed K (the prompt carries exactly K candidates).
    - 8.5: all survivors are reranked when survivors <= K.
    - 8.2: Final_Score is the fixed 0.6 * composite + 0.4 * llm_norm blend, in [0,1].
    - 8.3: the prompt includes each candidate's signal breakdown.
    - 8.4: no Protected_Proxy attribute appears in the prompt.
    - ordering: results are sorted descending by final_score.
    - registry wiring via LLMTask.RERANK and unparseable-output -> RerankError.
"""

from __future__ import annotations

import json
import re
from uuid import UUID, uuid4

import pytest

from icrs.models.candidate import EnrichedProfile, NormalizedProfile, Role
from icrs.models.enums import DepthBreadth, TrajectoryArc
from icrs.models.job import (
    Requirement,
    RequirementCategory,
    RequirementTier,
    RequirementVector,
    SeniorityBand,
)
from icrs.models.ranking import SignalBreakdown
from icrs.pipeline.reranker import (
    DEFAULT_COMPOSITE_WEIGHT,
    DEFAULT_LLM_WEIGHT,
    PROTECTED_PROXY_ATTRIBUTES,
    RerankError,
    Reranker,
    ScoredCandidate,
    build_rerank_prompt,
)
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
    """Scripted LLM provider returning a fixed (or scoring-derived) response.

    ``score_map`` (id_str -> 0-100 score) lets a test return scores aligned to the
    candidates actually present in the prompt; alternatively a list of canned
    response strings can be supplied. Every call's messages are recorded.
    """

    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        score_map: dict[str, float] | None = None,
    ) -> None:
        self._responses = list(responses) if responses else None
        self._score_map = score_map
        self.calls: list[list[LLMMessage]] = []

    @property
    def model_id(self) -> str:
        return "stub-rerank-llm"

    def complete(
        self,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens=None,
        response_format=None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self._responses is not None:
            return LLMResponse(text=self._responses.pop(0), model=self.model_id)
        # Derive a response from the ids present in the (fenced) user prompt.
        user = messages[-1].content
        ids = re.findall(r"id: ([0-9a-fA-F-]{36})", user)
        rankings = [
            {"id": cid, "score": (self._score_map or {}).get(cid, 50)} for cid in ids
        ]
        return LLMResponse(
            text=json.dumps({"rankings": rankings}), model=self.model_id
        )

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1][-1].content


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _reqs() -> RequirementVector:
    return RequirementVector(
        role_intent="Build and operate the data platform",
        seniority_band=SeniorityBand.SENIOR,
        requirements=[
            Requirement(
                text="5+ years of Python",
                category=RequirementCategory.MUST_HAVE,
                tier=RequirementTier.STRUCTURAL,
                weight=1.0,
            ),
            Requirement(
                text="Kafka streaming experience",
                category=RequirementCategory.NICE_TO_HAVE,
                tier=RequirementTier.SEMANTIC,
                weight=1.0,
            ),
        ],
    )


def _breakdown(value: float = 0.5) -> SignalBreakdown:
    return SignalBreakdown(
        semantic_fit=value,
        career_trajectory=value,
        behavioral=value,
        hard_filter_pass=value,
        disqualifying_penalty=0.0,
    )


def _candidate(composite: float, *, cid: UUID | None = None) -> ScoredCandidate:
    cid = cid or uuid4()
    profile = EnrichedProfile(
        id=cid,
        base=NormalizedProfile(
            id=cid,
            roles=[Role(title="Senior Data Engineer", company="ACME Corp")],
            explicit_skills=["Python", "Kafka"],
        ),
        inferred_responsibilities=["led platform migration"],
        trajectory_arc=TrajectoryArc.ACCELERATING,
        depth_breadth=DepthBreadth.SPECIALIST,
    )
    return ScoredCandidate(
        id=cid,
        profile=profile,
        composite_score=composite,
        breakdown=_breakdown(round(composite, 3)),
    )


# --------------------------------------------------------------------------- #
# 8.1 — only the top-K reach the LLM when survivors > K
# --------------------------------------------------------------------------- #
def test_only_top_k_candidates_reach_the_llm() -> None:
    survivors = [_candidate(c) for c in (0.9, 0.8, 0.7, 0.6, 0.5)]
    stub = StubLLM()
    reranker = Reranker(provider=stub, k=3)

    result = reranker.rerank(survivors, _reqs())

    # Exactly one LLM call, and the prompt carries exactly K=3 candidate ids.
    assert len(stub.calls) == 1
    prompt_ids = set(re.findall(r"id: ([0-9a-fA-F-]{36})", stub.last_user_prompt))
    assert len(prompt_ids) == 3
    # The three highest composite scores were selected.
    top_ids = {c.id_str for c in sorted(survivors, key=lambda c: -c.composite_score)[:3]}
    assert prompt_ids == top_ids
    assert len(result) == 3
    assert {c.id_str for c in result} == top_ids


# --------------------------------------------------------------------------- #
# 8.5 — all survivors reranked when survivors <= K
# --------------------------------------------------------------------------- #
def test_all_survivors_reranked_when_within_k() -> None:
    survivors = [_candidate(0.6), _candidate(0.4)]
    stub = StubLLM()
    reranker = Reranker(provider=stub, k=5)

    result = reranker.rerank(survivors, _reqs())

    prompt_ids = re.findall(r"id: ([0-9a-fA-F-]{36})", stub.last_user_prompt)
    assert len(prompt_ids) == 2
    assert len(result) == 2


# --------------------------------------------------------------------------- #
# 8.2 — Final_Score is the fixed 0.6/0.4 blend, within [0,1]
# --------------------------------------------------------------------------- #
def test_final_score_is_the_fixed_blend() -> None:
    c1 = _candidate(0.8)
    c2 = _candidate(0.4)
    # LLM (0-100) -> normalized: c1 -> 0.30, c2 -> 0.90.
    score_map = {c1.id_str: 30, c2.id_str: 90}
    stub = StubLLM(score_map=score_map)
    reranker = Reranker(provider=stub, k=5)

    result = reranker.rerank([c1, c2], _reqs())

    by_id = {c.id_str: c for c in result}
    expected_c1 = 0.6 * 0.8 + 0.4 * 0.30
    expected_c2 = 0.6 * 0.4 + 0.4 * 0.90
    assert by_id[c1.id_str].final_score == pytest.approx(expected_c1, abs=1e-9)
    assert by_id[c2.id_str].final_score == pytest.approx(expected_c2, abs=1e-9)
    for c in result:
        assert 0.0 <= c.final_score <= 1.0


def test_blend_weights_are_the_design_defaults() -> None:
    assert DEFAULT_COMPOSITE_WEIGHT == pytest.approx(0.6)
    assert DEFAULT_LLM_WEIGHT == pytest.approx(0.4)
    assert DEFAULT_COMPOSITE_WEIGHT + DEFAULT_LLM_WEIGHT == pytest.approx(1.0)


def test_final_score_always_clamped_to_unit_interval() -> None:
    # Extreme inputs (max composite + max LLM) stay within [0,1].
    c = _candidate(1.0)
    stub = StubLLM(score_map={c.id_str: 100})
    reranker = Reranker(provider=stub, k=5)

    [result] = reranker.rerank([c], _reqs())

    assert result.final_score == pytest.approx(1.0)
    assert 0.0 <= result.final_score <= 1.0


# --------------------------------------------------------------------------- #
# Ordering — sorted descending by final_score
# --------------------------------------------------------------------------- #
def test_result_sorted_descending_by_final_score() -> None:
    c1 = _candidate(0.8)  # high composite, low LLM
    c2 = _candidate(0.4)  # low composite, high LLM -> should overtake
    stub = StubLLM(score_map={c1.id_str: 10, c2.id_str: 100})
    reranker = Reranker(provider=stub, k=5)

    result = reranker.rerank([c1, c2], _reqs())

    scores = [c.final_score for c in result]
    assert scores == sorted(scores, reverse=True)
    # c2 (0.6*0.4 + 0.4*1.0 = 0.64) overtakes c1 (0.6*0.8 + 0.4*0.1 = 0.52).
    assert result[0].id_str == c2.id_str


# --------------------------------------------------------------------------- #
# 8.3 — prompt includes each candidate's signal breakdown
# --------------------------------------------------------------------------- #
def test_prompt_includes_signal_breakdowns() -> None:
    survivors = [_candidate(0.7), _candidate(0.5)]
    stub = StubLLM()
    reranker = Reranker(provider=stub, k=5)

    reranker.rerank(survivors, _reqs())

    prompt = stub.last_user_prompt
    assert "signal_breakdown" in prompt
    # Every breakdown sub-score label is present for the candidates.
    for label in (
        "semantic_fit",
        "career_trajectory",
        "behavioral",
        "hard_filter_pass",
        "disqualifying_penalty",
    ):
        assert label in prompt


# --------------------------------------------------------------------------- #
# 8.4 — Protected_Proxy attributes never appear in the prompt
# --------------------------------------------------------------------------- #
def test_protected_proxy_attributes_excluded_from_prompt() -> None:
    cid = uuid4()
    profile = EnrichedProfile(
        id=cid,
        base=NormalizedProfile(
            id=cid,
            roles=[Role(title="Staff Engineer", company="Globex")],
            explicit_skills=["Python", "Go"],
        ),
        trajectory_arc=TrajectoryArc.STEADY,
        depth_breadth=DepthBreadth.BALANCED,
    )
    candidate = ScoredCandidate(
        id=cid, profile=profile, composite_score=0.7, breakdown=_breakdown()
    )
    stub = StubLLM()
    reranker = Reranker(provider=stub, k=5)

    reranker.rerank([candidate], _reqs())

    full_prompt = "\n".join(m.content for m in stub.calls[-1])
    # No demographic / proxy attribute key leaks into the candidate data block.
    # (The system prompt mentions them only to instruct the model to ignore them;
    #  the user data block must contain none of them.)
    user_prompt = stub.last_user_prompt
    lower = user_prompt.lower()
    for attr in PROTECTED_PROXY_ATTRIBUTES:
        assert attr not in lower
    # Job-relevant evidence IS present (proves we didn't strip everything).
    assert "Python" in full_prompt
    assert "Staff Engineer" in full_prompt


# --------------------------------------------------------------------------- #
# build_rerank_prompt — direct checks
# --------------------------------------------------------------------------- #
def test_build_prompt_fences_candidate_data_as_untrusted() -> None:
    candidate = _candidate(0.6)
    messages = build_rerank_prompt([candidate], _reqs())

    system = messages[0].content
    user = messages[1].content
    assert "untrusted" in system.lower()
    assert "CANDIDATE_DATA_BEGIN" in user
    assert "CANDIDATE_DATA_END" in user
    assert candidate.id_str in user


# --------------------------------------------------------------------------- #
# Registry wiring + error handling
# --------------------------------------------------------------------------- #
def test_resolves_provider_from_registry_by_rerank_task() -> None:
    stub = StubLLM()
    registry = LLMProviderRegistry()
    registry.register(LLMTask.RERANK, stub)
    reranker = Reranker(registry, k=5)

    result = reranker.rerank([_candidate(0.6)], _reqs())

    assert len(result) == 1
    assert len(stub.calls) == 1


def test_unparseable_output_raises_rerank_error() -> None:
    stub = StubLLM(responses=["this is not json"])
    reranker = Reranker(provider=stub, k=5)

    with pytest.raises(RerankError):
        reranker.rerank([_candidate(0.6)], _reqs())


def test_incomplete_coverage_raises_rerank_error() -> None:
    c1 = _candidate(0.7)
    c2 = _candidate(0.5)
    # LLM only scores one of the two selected candidates.
    stub = StubLLM(responses=[json.dumps({"rankings": [{"id": c1.id_str, "score": 80}]})])
    reranker = Reranker(provider=stub, k=5)

    with pytest.raises(RerankError):
        reranker.rerank([c1, c2], _reqs())


def test_ordering_only_response_is_parsed() -> None:
    c1 = _candidate(0.5, cid=uuid4())
    c2 = _candidate(0.5, cid=uuid4())
    # Pure ordering (best first), no scores.
    stub = StubLLM(responses=[json.dumps({"ranking": [c2.id_str, c1.id_str]})])
    reranker = Reranker(provider=stub, k=5)

    result = reranker.rerank([c1, c2], _reqs())

    # c2 ranked first by the LLM -> llm_norm 1.0; c1 -> 0.0.
    assert result[0].id_str == c2.id_str


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
def test_k_out_of_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        Reranker(provider=StubLLM(), k=0)
    with pytest.raises(ValueError):
        Reranker(provider=StubLLM(), k=51)


def test_blend_weights_must_sum_to_one_and_be_nonzero() -> None:
    with pytest.raises(ValueError):
        Reranker(provider=StubLLM(), composite_weight=0.0, llm_weight=1.0)
    with pytest.raises(ValueError):
        Reranker(provider=StubLLM(), composite_weight=0.7, llm_weight=0.4)


def test_requires_provider_or_registry() -> None:
    with pytest.raises(ValueError):
        Reranker()


def test_empty_input_returns_empty_list() -> None:
    reranker = Reranker(provider=StubLLM(), k=5)
    assert reranker.rerank([], _reqs()) == []

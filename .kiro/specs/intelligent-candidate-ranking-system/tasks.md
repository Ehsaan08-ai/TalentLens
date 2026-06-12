# Implementation Plan: Intelligent Candidate Ranking System (ICRS)

## Overview

This plan implements the ICRS PoC as a Python (FastAPI async) service following the design's five-layer pipeline and four-phase roadmap. Implementation language is Python, matching the design's stack: FastAPI, PostgreSQL + pgvector, Qdrant, `text-embedding-3-large`, GPT-4o / Claude Sonnet 4 / Gemini 1.5 Pro, Streamlit for the PoC UI, and Hypothesis + pytest for testing.

Tasks proceed incrementally: foundational data models and abstract provider interfaces first, then each pipeline layer (JD decomposition, ingestion/enrichment, embedding, hybrid scoring, reranking, explanation), wired together by the orchestrator, then surfaced through the Streamlit UI. Property-based tests (Hypothesis) validate the nine Correctness Properties from the design and are placed close to the code they validate. Each task references the specific requirements it implements.

## Tasks

- [x] 1. Set up project structure, configuration, and provider interfaces
  - Create the package layout (`icrs/models`, `icrs/providers`, `icrs/pipeline`, `icrs/scoring`, `icrs/persistence`, `icrs/ui`, `tests`)
  - Add dependencies and dev tooling (FastAPI, pydantic, async DB driver + pgvector, qdrant-client, rank-bm25, openai/anthropic/google SDK clients, streamlit, pytest, hypothesis)
  - Define abstract provider interfaces (`EmbeddingProvider`, `LLMProvider`, `VectorStore`) so concrete models are swappable, plus a typed settings/config loader for API keys, model versions, K bound, and token limits
  - Set up the test framework with a `conftest.py` and a Hypothesis profile (fixed seed for deterministic stages)
  - _Requirements: 2.1, 4.2_

- [x] 2. Define core data models and validation (Phase 1 foundation)
  - [x] 2.1 Implement RequirementVector and JobDescription models
    - Implement `Requirement` (text, category enum MUST_HAVE/NICE_TO_HAVE/DISQUALIFYING, tier enum, weight, embedding), `RequirementVector` (role_intent, seniority_band enum, requirements, implicit_expectations, culture_signals), and `JobDescription` with validation: non-empty raw_text, at least one MUST_HAVE, per-category weight sum normalized to 1.0, DISQUALIFYING excluded from weighted contribution
    - _Requirements: 1.2, 1.4, 1.5, 1.3_

  - [x] 2.2 Implement candidate profile models (Raw/Normalized/Enriched)
    - Implement `RawCandidate`, `NormalizedProfile` (roles, education, certifications, explicit_skills, total_tenure_months), `EnrichedProfile` (inferred signals, trajectory_arc, depth_breadth, behavioral_signals, signal_availability per tier, embedding), and `BehavioralSignal`
    - Enforce that `signal_availability` is per-tier in [0,1] and that absent structured fields are marked not-present rather than defaulted
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 2.3 Implement ranking output models
    - Implement `SignalBreakdown`, `Explanation` (summary, driving_signals, gaps, unmet_must_haves), and `RankingResult` (final_score, rank, breakdown, explanation, confidence) with validation: final_score in [0,1], confidence in [0,1], summary length ≤ 1000 chars
    - _Requirements: 5.1, 2.4_

  - [ ]* 2.4 Write unit tests for data model validation
    - Test rejection of empty JD, missing MUST_HAVE, mis-normalized category weights, and out-of-range scores/confidence
    - _Requirements: 1.5, 1.7, 5.1_

- [x] 3. Implement JD Decomposer (Phase 1)
  - [x] 3.1 Implement JD decomposition with schema validation and retry
    - Implement `JDDecomposer.decompose(rawJD)` using the LLM provider (GPT-4o) to produce a `RequirementVector`; reject empty/whitespace-only JD with a validation error; validate the LLM output against the schema, retry exactly once with a stricter prompt on failure, and return a parse error after a second failure; return a decomposition error when zero MUST_HAVE requirements are extracted; treat profile/JD text as data, not instructions
    - Cache decomposition by content hash
    - _Requirements: 1.1, 1.2, 1.3, 1.6, 1.7, 1.8_

  - [ ]* 3.2 Write unit tests for JD Decomposer error paths
    - Test empty-JD rejection, single-retry-then-parse-error, and zero-MUST_HAVE decomposition error using a stubbed LLM provider
    - _Requirements: 1.6, 1.7, 1.8_

- [x] 4. Implement Candidate Enricher (Phase 1)
  - [x] 4.1 Implement profile normalization
    - Implement `CandidateEnricher.normalize(rawProfile)` to canonicalize heterogeneous profiles into `NormalizedProfile` (roles, education, certifications, explicit_skills, whole-month total tenure); reject empty/invalid profiles with an error and produce no canonical profile
    - _Requirements: 3.1, 3.6_

  - [x] 4.2 Implement three-tier signal enrichment
    - Implement `CandidateEnricher.enrich(profile)`: derive Tier 1 structural signals from structured fields (marking absent fields not-present), infer Tier 2 semantic signals via batched LLM calls (inferred responsibilities, implicit skills, trajectory arc, depth/breadth), derive freshness-weighted Tier 3 behavioral signals where handles exist (freshness weight in [0,1], monotonically non-increasing with age), and record `signal_availability` per tier as the populated-field fraction
    - Cache enrichment by content hash
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

  - [ ]* 4.3 Write property test for missing-data fairness recording
    - **Property 5: Missing-data fairness** (recording side: absent signals yield availability 0, never a defaulted value)
    - **Validates: Requirements 7.2**
    - _Requirements: 3.5, 7.2_

  - [ ]* 4.4 Write unit tests for normalization and freshness weighting
    - Test tenure-in-months computation, not-present marking for absent fields, monotonic non-increasing freshness weight, and empty-profile rejection
    - _Requirements: 3.1, 3.4, 3.6_

- [x] 5. Implement Embedding Generator with section-aligned chunking (Phase 1)
  - [x] 5.1 Implement embedding with chunking and weighted aggregation
    - Implement `EmbeddingGenerator.embed(profile)` and `embedRequirement(reqs)` using the embedding provider: single L2-normalized vector when within the token limit; otherwise split into role/section-aligned chunks (one block per chunk, no fixed-size splits) and return a recency/relevance-weighted-mean vector, L2-normalized to 1.0 ±0.001; candidate and requirement vectors share model and dimensionality for cosine comparability
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 5.2 Write property test for embedding norm and dimensionality
    - **Property: L2 norm = 1.0 ±0.001 and dimensionality matches configured model for both short and chunked-aggregated profiles** (supports cosine comparability)
    - **Validates: Requirements 6.1, 6.3, 6.4**
    - _Requirements: 6.1, 6.3, 6.4_

  - [ ]* 5.3 Write unit tests for section-aligned chunking
    - Test that chunks align to role/section boundaries (not fixed-size) and that more recent/relevant chunks receive weights ≥ older/less-relevant ones
    - _Requirements: 6.2, 6.3_

- [x] 6. Implement persistence layer (Phase 1)
  - [x] 6.1 Implement PostgreSQL + pgvector repositories
    - Create schema and async repositories for `JobDescription`/`RequirementVector`, candidate profiles, and embedded vectors (pgvector as system-of-record); store and retrieve embeddings with dimensionality matching the configured model
    - _Requirements: 2.2_

  - [x] 6.2 Wire ingestion persistence into a Phase 1 ingestion path
    - Add a function that, for a JD and a candidate pool, runs decompose → normalize → enrich → embed and persists the RequirementVector and one embedding per candidate (Phase 1 end-to-end ingestion)
    - _Requirements: 2.1, 2.2_

  - [ ]* 6.3 Write integration test for the ingestion path
    - Test that a JD + small pool yields a persisted RequirementVector and one stored embedding per candidate
    - _Requirements: 2.1, 2.2_

- [x] 7. Checkpoint - Phase 1 ingestion and embedding foundation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement hard-filter gate (Phase 2)
  - [x] 8.1 Implement deterministic hard-filter gate
    - Implement `HybridScoringEngine.hardFilter(reqs, cand)`: exclude a candidate only when it positively matches at least one DISQUALIFYING criterion; never exclude on absence of data or on signal_availability = 0; retain all candidates matching no disqualifier
    - _Requirements: 4.4, 7.3_

  - [ ]* 8.2 Write property test for hard-filter soundness
    - **Property 4: Hard-filter soundness** (a candidate positively matching any DISQUALIFYING criterion never appears in survivors; missing data never excludes)
    - **Validates: Requirements 1.4, 4.4**
    - _Requirements: 4.4, 7.3_

- [x] 9. Implement weight profiles (Phase 2)
  - [x] 9.1 Implement weight-profile registry, selection, and validation
    - Implement default and per-job-type `WeightProfile`s as data; select by job type with fallback to default for unknown types; validate that w1+w2+w3+w4 = 1.0 within ±0.001 and reject + fall back to default with a recorded indication when invalid
    - _Requirements: 4.2, 4.3, 4.8_

  - [ ]* 9.2 Write property test for weight normalization
    - **Property 6: Weight normalization** (for every selected/accepted weight profile, w1+w2+w3+w4 = 1.0 within ±0.001)
    - **Validates: Requirements 4.2, 4.3**
    - _Requirements: 4.2, 4.3, 4.8_

- [x] 10. Implement sub-score computation and dense/sparse retrieval (Phase 2)
  - [x] 10.1 Implement dense and sparse similarity and semantic fit blend
    - Implement dense cosine similarity over vectors and sparse BM25/TF-IDF (rank-bm25) with normalization of each to [0,1], blended into the semantic fit sub-score in [0,1]; wire Qdrant/pgvector ANN search for dense candidate retrieval
    - _Requirements: 4.6_

  - [x] 10.2 Implement trajectory, behavioral, hard-filter-pass, and penalty sub-scores
    - Implement CareerTrajectoryScore, BehavioralSignalScore (neutral prior 0.5 when behavioral availability = 0), HardFilterPassScore (must-have satisfaction ratio), and DisqualifyingFlagPenalty (soft flags); each sub-score normalized to [0,1]; record an indication and exclude a candidate when a required sub-score (semantic/trajectory/behavioral) cannot be computed
    - _Requirements: 4.5, 4.7, 7.2_

  - [ ]* 10.3 Write unit tests for sub-score normalization and neutral prior
    - Test each sub-score stays in [0,1], behavioral neutral prior applied at availability 0, and exclusion + indication when a sub-score is uncomputable
    - _Requirements: 4.5, 4.7, 7.2_

- [x] 11. Implement composite score fusion (Phase 2)
  - [x] 11.1 Implement weighted composite fusion with clamping
    - Implement `composite(signals, weights)`: normalize each sub-score to [0,1] before weighting, apply the selected WeightProfile, subtract the independent penalty term, and clamp the result to [0,1]
    - _Requirements: 4.1, 4.5_

  - [ ]* 11.2 Write property test for score bounds
    - **Property 1: Score bounds** (every composite/final score ∈ [0,1])
    - **Validates: Requirements 4.1**
    - _Requirements: 4.1, 4.5_

  - [ ]* 11.3 Write property test for determinism of deterministic stages
    - **Property 7: Determinism of deterministic stages** (identical inputs + fixed model versions yield identical hard-filter, dense, sparse, and composite outputs)
    - **Validates: Requirements 2.3**
    - _Requirements: 2.3_

- [x] 12. Checkpoint - Phase 2 hybrid scoring and retrieval
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement LLM contextual reranker (Phase 3)
  - [x] 13.1 Implement top-K reranking with score blending
    - Implement `rerank(topK, reqs)`: select the K highest composite-scored candidates (K configurable in 1..50; rerank all when survivors ≤ K) using Claude Sonnet 4; build a prompt that includes each candidate's signal breakdown and excludes all Protected_Proxy attributes; compute Final_Score as a fixed nonzero-weight blend of the [0,1]-normalized LLM score and composite score (weights sum to 1.0), clamped to [0,1]
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 13.2 Write property test for score bounds and proxy exclusion in rerank
    - **Property 1: Score bounds** applied to reranked Final_Score, plus assertion that Protected_Proxy attributes never appear in the rerank prompt
    - **Validates: Requirements 4.1, 8.2, 8.4**
    - _Requirements: 8.1, 8.2, 8.4, 8.5_

- [x] 14. Implement explanation generator (Phase 3)
  - [x] 14.1 Implement recruiter-readable explanation generation
    - Implement `explain(cand, reqs)` using Gemini 1.5 Pro to produce a non-empty summary (≤ 1000 chars), at least one driving signal, the candidate's gaps (empty only when no unmet requirements), and unmet_must_haves containing only unsatisfied MUST_HAVE requirements (excluding NICE_TO_HAVE, DISQUALIFYING, and satisfied must-haves); exclude all Protected_Proxy attributes from explanations
    - _Requirements: 5.2, 5.4, 7.1_

  - [ ]* 14.2 Write property test for explanation consistency
    - **Property 8: Explanation consistency** (every listed unmet_must_have corresponds to an unsatisfied MUST_HAVE requirement; no NICE_TO_HAVE/DISQUALIFYING/satisfied must-haves)
    - **Validates: Requirements 5.4**
    - _Requirements: 5.2, 5.4_

- [x] 15. Implement confidence computation and fairness controls (Phase 3)
  - [x] 15.1 Implement confidence computation
    - Implement `computeConfidence(cand, ranked)` from signal coverage and score margin to neighbors, clamped to [0,1], such that higher coverage and larger margin yield confidence ≥ that of an otherwise-identical lower-coverage/smaller-margin candidate
    - _Requirements: 5.5_

  - [x] 15.2 Implement fairness and missing-signal guarantees
    - Enforce Protected_Proxy exclusion from scoring inputs; for any tier with availability 0 assign the neutral prior (0.5) and reduce confidence without lowering rank on the missing tier; for a candidate with availability 0 in every tier, score with neutral priors and include at minimum confidence rather than excluding
    - _Requirements: 7.1, 7.2, 7.4, 7.5_

  - [ ]* 15.3 Write property test for confidence coherence
    - **Property 9: Confidence coherence** (strictly higher coverage with equal margin ⟹ confidence ≥)
    - **Validates: Requirements 5.5, 7.2**
    - _Requirements: 5.5, 7.2_

  - [ ]* 15.4 Write property test for counterfactual fairness
    - **Property 5: Missing-data fairness** + counterfactual check: perturbing Protected_Proxy attributes while holding job-relevant signals constant yields a rank delta of 0
    - **Validates: Requirements 7.2, 7.4**
    - _Requirements: 7.1, 7.2, 7.4_

- [x] 16. Implement ranking orchestrator and rank assignment (Phase 3)
  - [x] 16.1 Wire the full pipeline in the orchestrator
    - Implement `rankCandidates(rawJD, pool, jobType)`: run stages in order (decompose → enrich → embed → hard filter → composite score → rerank → explain), emit exactly one result per surviving candidate, and assign unique, contiguous ranks 1..N with higher final_score ranked ahead and a deterministic tie-break for equal scores
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 5.3, 5.6, 2.6_

  - [x] 16.2 Implement error handling and resilience
    - Behavioral fetch timeout (10s)/unavailable ⟹ set tier availability 0 and use neutral prior; embedding errors ⟹ retry up to 3 times with backoff and reuse cached embeddings, excluding only un-embeddable candidates while others proceed; reranker failure ⟹ fall back to composite ordering flagged un-reranked; explanation failure ⟹ mark explanation unavailable without fabricating content
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 16.3 Write property tests for rank integrity and monotonic ordering
    - **Property 2: Rank integrity** (ranks unique and contiguous 1..N) and **Property 3: Monotonic ordering** (higher final_score ⟹ lower rank number)
    - **Validates: Requirements 2.5, 5.3**
    - _Requirements: 2.4, 2.5, 5.3, 5.6_

  - [ ]* 16.4 Write integration tests for error-handling scenarios
    - Failure-injection tests for behavioral-fetch timeout, embedding retry/exclusion, reranker fallback, and explanation-unavailable paths
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 17. Expose the pipeline via FastAPI (Phase 3/4 wiring)
  - [x] 17.1 Implement async FastAPI endpoints
    - Add endpoints to submit a JD + candidate pool + job type and return the ranked shortlist (final score, breakdown, summary, confidence per result), reusing the orchestrator and running independent candidate enrichment/embedding concurrently
    - _Requirements: 2.1, 5.1_

  - [ ]* 17.2 Write integration test for the ranking endpoint
    - Test that the endpoint returns one result per survivor with required fields populated for a small labeled pool
    - _Requirements: 2.1, 5.1_

- [x] 18. Checkpoint - Phase 3 reranking, explanations, and orchestration
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. Implement ranked shortlist output and Streamlit UI (Phase 4)
  - [x] 19.1 Implement shortlist output assembly
    - Assemble the ranked shortlist output object: per result a final score in [0,1], a per-signal breakdown (semantic-fit, trajectory, behavioral, hard-filter each in [0,1]), a non-empty summary ≤ 1000 chars, and a confidence in [0,1]; surface the un-reranked and explanation-unavailable flags honestly
    - _Requirements: 5.1, 9.4, 9.5_

  - [x] 19.2 Build the recruiter-facing Streamlit dashboard
    - Build the Streamlit PoC UI: upload/paste a JD, upload a candidate pool, choose job type, trigger ranking via the FastAPI endpoint, and display the ranked shortlist with expandable per-candidate explanations, signal breakdowns, and confidence (presenting uncertainty without false precision)
    - _Requirements: 5.1, 5.2_

  - [ ]* 19.3 Write integration test for output assembly
    - Test that the assembled shortlist conforms to field/range/length constraints and correctly reflects un-reranked and explanation-unavailable flags
    - _Requirements: 5.1, 9.4, 9.5_

- [x] 20. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each task references specific requirement sub-clauses for traceability.
- Property-based tests use Hypothesis and validate the nine Correctness Properties from the design; they are placed close to the code they validate so errors surface early.
- Checkpoints align with the design's four-phase PoC roadmap (Phase 1 ingestion/embedding, Phase 2 hybrid scoring, Phase 3 reranking/explanations/orchestration, Phase 4 output/UI).
- Property/requirement coverage: P1→4.1; P2→2.5,5.3; P3→5.3; P4→1.4,4.4; P5→7.2; P6→4.2,4.3; P7→2.3; P8→5.4; P9→5.5,7.2.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "2.2", "2.3"] },
    { "id": 2, "tasks": ["2.4", "3.1", "4.1", "9.1"] },
    { "id": 3, "tasks": ["3.2", "4.2", "9.2"] },
    { "id": 4, "tasks": ["4.3", "4.4", "5.1", "8.1"] },
    { "id": 5, "tasks": ["5.2", "5.3", "6.1", "8.2", "10.1"] },
    { "id": 6, "tasks": ["6.2", "10.2"] },
    { "id": 7, "tasks": ["6.3", "10.3", "11.1"] },
    { "id": 8, "tasks": ["11.2", "11.3", "13.1", "14.1"] },
    { "id": 9, "tasks": ["13.2", "14.2", "15.1"] },
    { "id": 10, "tasks": ["15.2"] },
    { "id": 11, "tasks": ["15.3", "15.4", "16.1"] },
    { "id": 12, "tasks": ["16.2"] },
    { "id": 13, "tasks": ["16.3", "16.4", "17.1"] },
    { "id": 14, "tasks": ["17.2", "19.1"] },
    { "id": 15, "tasks": ["19.2", "19.3"] }
  ]
}
```

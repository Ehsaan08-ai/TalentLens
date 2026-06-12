# Requirements Document

## Introduction

The Intelligent Candidate Ranking System (ICRS) ranks job candidates the way a world-class human recruiter would: through semantic understanding, behavioral inference, and multi-signal scoring rather than brittle keyword matching. ICRS decomposes a job description into a structured, weighted requirement schema, enriches each candidate profile with inferred semantic and behavioral signals, and fuses four complementary scoring methods (rule-based hard filters, dense vector similarity, sparse keyword matching, and LLM contextual reranking) into a single explainable composite score. Every ranking is accompanied by a recruiter-readable rationale that names the driving signals and surfaces gaps, so the system augments rather than replaces human judgment.

This document specifies the requirements for a Proof of Concept (PoC), derived from the approved design blueprint. The requirements cover JD understanding, three-tier signal interpretation, candidate ingestion and enrichment, embedding generation, the hybrid scoring engine, LLM contextual reranking, explainable ranking and output, fairness and missing-signal handling, and error handling.

## Glossary

- **ICRS**: The Intelligent Candidate Ranking System as a whole.
- **JD_Decomposer**: The component that converts a raw job description into a structured, weighted RequirementVector via a single LLM extraction call.
- **RequirementVector**: A structured representation of a job description containing role intent, seniority band, classified and weighted requirements, implicit expectations, and culture signals.
- **Candidate_Enricher**: The component that normalizes a raw candidate profile into a canonical schema and attaches inferred Tier 2 (semantic) and Tier 3 (behavioral) signals.
- **Embedding_Generator**: The component that produces a unit-normalized vector embedding from an enriched candidate profile or a requirement vector.
- **Scoring_Engine**: The hybrid scoring engine that applies hard filters, computes dense and sparse sub-scores, fuses them into a composite score, and reranks the top-K.
- **Reranker**: The LLM contextual reranking stage that refines the ordering of the top-K composite-scored candidates.
- **Explanation_Generator**: The component that produces a recruiter-readable rationale, driving signals, gaps, and unmet must-haves per shortlisted candidate.
- **Orchestrator**: The component that coordinates the full pipeline and assembles the ranked output.
- **Structural_Signal**: A Tier 1 signal extracted directly from structured profile fields (e.g., title progression, tenure, education, explicit skills).
- **Semantic_Signal**: A Tier 2 signal inferred from free text via embeddings and LLM extraction (e.g., inferred responsibilities, implicit skills, trajectory arc, depth/breadth signature).
- **Behavioral_Signal**: A Tier 3 signal derived from external platform activity (e.g., GitHub commits, publications), freshness-weighted.
- **Signal_Availability**: A per-tier coverage value in [0,1] recording how much data is present for a candidate in each signal tier, computed as the fraction of that tier's expected fields that are populated.
- **Neutral_Prior**: The default sub-score value (0.5) assigned when a signal tier has no data, used so absence is treated as "unknown" rather than zero.
- **WeightProfile**: A configurable set of weights (w1..w5) applied during composite score fusion, selectable per job type.
- **Composite_Score**: The fused, normalized score in [0,1] produced from the weighted combination of sub-scores.
- **Final_Score**: The score in [0,1] assigned to a candidate after reranking, used to determine rank.
- **Confidence**: A value in [0,1] reflecting signal coverage and score margin, reduced when data is sparse or the score margin to neighbors is small.
- **Hard_Filter**: The deterministic gate that removes candidates positively matching a DISQUALIFYING criterion.
- **Disqualifying_Criterion**: A requirement that, when positively matched by a candidate, acts as an absolute gate excluding that candidate from results.
- **Must_Have**: A requirement classified as mandatory for the role.
- **Protected_Proxy**: A demographic or proxy attribute (name, gender markers, age proxies, photos, location-as-proxy) excluded from scoring inputs and explanations.

## Requirements

### Requirement 1: Job Description Understanding & Decomposition

**User Story:** As a recruiter, I want the system to understand the true intent of a job description, so that candidates are matched against what the role actually requires rather than against literal keywords.

#### Acceptance Criteria

1. WHEN a recruiter submits a raw job description containing at least one non-whitespace character, THE JD_Decomposer SHALL produce a RequirementVector in which the role intent is derived from the description body and stored in a field distinct from the job title.
2. WHEN the JD_Decomposer produces a RequirementVector, THE JD_Decomposer SHALL classify each extracted requirement as exactly one of MUST_HAVE, NICE_TO_HAVE, or DISQUALIFYING.
3. WHEN the JD_Decomposer produces a RequirementVector, THE JD_Decomposer SHALL infer a seniority band selected from a predefined enumerated set of bands, and SHALL populate implicit expectations and culture signals as distinct fields that each default to an empty collection when none are present.
4. WHEN the JD_Decomposer classifies requirements, THE JD_Decomposer SHALL record each DISQUALIFYING criterion as an absolute gate that is excluded from weighted scoring contribution.
5. WHEN the JD_Decomposer produces a RequirementVector, THE JD_Decomposer SHALL include at least one MUST_HAVE requirement.
6. IF the JD_Decomposer returns a RequirementVector that fails schema validation, THEN THE JD_Decomposer SHALL retry exactly once with a stricter prompt, and IF the second attempt also fails schema validation, THEN THE JD_Decomposer SHALL return a parse error and SHALL NOT produce a ranking.
7. IF the submitted raw job description is empty or contains only whitespace, THEN THE JD_Decomposer SHALL return a validation error and SHALL NOT produce a RequirementVector.
8. IF the JD_Decomposer extracts zero MUST_HAVE requirements from the job description, THEN THE JD_Decomposer SHALL return a decomposition error and SHALL NOT produce a ranking.

### Requirement 2: Ranking Pipeline Orchestration & Determinism

**User Story:** As a recruiter, I want a single pipeline that takes a job description and a candidate pool and returns a ranked shortlist, so that I can review candidates end-to-end without manual coordination.

#### Acceptance Criteria

1. WHEN a recruiter submits a non-empty raw job description, a candidate pool containing at least one candidate, and a job type, THE Orchestrator SHALL execute the pipeline stages in the order JD decomposition, candidate enrichment, hard filtering, composite scoring, reranking, and explanation, and SHALL return a list of ranking results.
2. WHILE processing the candidate pool, THE Orchestrator SHALL pass each candidate through normalization, then enrichment, then embedding, and SHALL complete all three stages for a candidate before that candidate enters composite scoring.
3. WHEN identical inputs and fixed model versions are supplied, THE Scoring_Engine SHALL produce identical hard-filter, dense-similarity, sparse-similarity, and composite-fusion outputs.
4. WHEN the pipeline completes, THE Orchestrator SHALL emit exactly one ranking result per candidate surviving the hard-filter gate, each carrying a final score in [0,1], a unique integer rank, a signal breakdown of its per-tier sub-scores, an explanation, and a confidence value in [0,1].
5. WHEN the Orchestrator assigns ranks for a job, THE Orchestrator SHALL assign ranks that are unique and contiguous starting from 1 to N, where N is the number of candidates surviving the hard-filter gate.
6. IF the raw job description is empty or the candidate pool contains zero candidates, THEN THE Orchestrator SHALL reject the request with an error indicating the invalid input and SHALL NOT produce a ranking.

### Requirement 3: Candidate Ingestion, Normalization & Three-Tier Signal Enrichment

**User Story:** As a recruiter, I want heterogeneous candidate profiles normalized and enriched with inferred signals across structural, semantic, and behavioral tiers, so that strong candidates are recognized even when their resumes do not echo the job description's vocabulary.

#### Acceptance Criteria

1. WHEN a non-empty raw candidate profile is submitted, THE Candidate_Enricher SHALL normalize it into a canonical profile containing roles, education, certifications, explicit skills, and total tenure expressed in whole months.
2. WHEN a normalized profile is enriched, THE Candidate_Enricher SHALL derive Structural_Signal values from structured fields — including title progression, tenure, industry, education, certifications, and explicit skills — and SHALL mark any absent structured field as not-present rather than substituting a default value.
3. WHEN a normalized profile is enriched, THE Candidate_Enricher SHALL infer Semantic_Signal values from free text, including inferred responsibilities, implicit skills, career trajectory arc, and depth/breadth signature.
4. WHERE external platform handles are available, THE Candidate_Enricher SHALL derive Behavioral_Signal values from external platform activity, each weighted by a freshness weight in [0,1] that is monotonically non-increasing with the age of the activity.
5. WHEN a profile is enriched, THE Candidate_Enricher SHALL record Signal_Availability per tier as the fraction of that tier's expected fields that are populated, in [0,1], so that missing data is recorded as unknown rather than as zero.
6. IF a submitted raw candidate profile is empty or fails schema validation, THEN THE Candidate_Enricher SHALL reject it with an error and SHALL NOT produce a canonical profile.

### Requirement 4: Hybrid Scoring Engine

**User Story:** As a recruiter, I want candidates scored by fusing semantic, lexical, structural, and behavioral signals under role-appropriate weights, so that the ranking reflects holistic fit rather than a single matching method.

#### Acceptance Criteria

1. WHEN the Scoring_Engine computes a score for a candidate, THE Scoring_Engine SHALL produce a single numeric final score within the inclusive range [0.0, 1.0].
2. WHERE a job type is supplied, THE Scoring_Engine SHALL select the WeightProfile configured for that job type, and IF no WeightProfile is defined for the supplied job type, THEN THE Scoring_Engine SHALL select the default WeightProfile.
3. WHEN the Scoring_Engine applies a WeightProfile, THE Scoring_Engine SHALL use four weight components (w1 semantic, w2 trajectory, w3 behavioral, w4 hard-filter), each within the inclusive range [0.0, 1.0], whose sum equals 1.0 within a tolerance of ±0.001.
4. WHEN the Scoring_Engine applies the Hard_Filter gate, THE Scoring_Engine SHALL exclude from results any candidate whose attributes satisfy the match condition of at least one Disqualifying_Criterion, and SHALL retain in results any candidate that satisfies no Disqualifying_Criterion.
5. WHEN the Scoring_Engine fuses sub-scores, THE Scoring_Engine SHALL normalize each sub-score to the inclusive range [0.0, 1.0] before applying its weight, and SHALL clamp the weighted composite score to the inclusive range [0.0, 1.0].
6. WHEN the Scoring_Engine computes the semantic fit sub-score, THE Scoring_Engine SHALL combine dense vector similarity and sparse BM25/TF-IDF similarity, each normalized to the inclusive range [0.0, 1.0], into a single semantic fit sub-score in the inclusive range [0.0, 1.0].
7. IF a required sub-score (semantic, trajectory, or behavioral) cannot be computed for a candidate, THEN THE Scoring_Engine SHALL exclude that candidate from the scored results and SHALL record an indication identifying the unavailable sub-score.
8. IF the selected WeightProfile's components (w1, w2, w3, w4) do not sum to 1.0 within a tolerance of ±0.001, THEN THE Scoring_Engine SHALL reject that WeightProfile, SHALL fall back to the default WeightProfile, and SHALL record an indication that the WeightProfile was rejected.

### Requirement 5: Explainable Ranking & Output

**User Story:** As a recruiter, I want a ranked shortlist with plain-language rationale, confidence scores, and signal breakdowns, so that I can trust and act on the ranking with human judgment.

#### Acceptance Criteria

1. WHEN the pipeline completes, THE Orchestrator SHALL emit a ranked shortlist in which each result includes a final score in [0,1], a per-signal breakdown reporting the semantic-fit, career-trajectory, behavioral, and hard-filter sub-scores each in [0,1], a non-empty recruiter-facing summary of at most 1000 characters, and a confidence value in [0,1].
2. WHEN the Explanation_Generator produces an explanation for a shortlisted candidate, THE Explanation_Generator SHALL name at least one driving signal that contributed to the candidate's final score and SHALL list the candidate's gaps, listing an empty set of gaps only when the candidate has no unmet requirements.
3. WHEN two candidates have different final scores, THE Orchestrator SHALL assign the candidate with the strictly higher final score a rank ahead of (a numerically lower rank number than) the candidate with the lower final score.
4. WHEN the Explanation_Generator lists the unmet must-haves for a candidate, THE Explanation_Generator SHALL include only requirements classified as MUST_HAVE that the candidate did not satisfy, and SHALL exclude NICE_TO_HAVE requirements, DISQUALIFYING requirements, and any satisfied MUST_HAVE requirements.
5. WHEN the Orchestrator assigns a confidence value to a candidate, THE Orchestrator SHALL compute a confidence value in [0,1] from signal coverage and score margin to adjacent-ranked candidates, such that a candidate with higher signal coverage and a larger score margin receives a confidence value greater than or equal to that of an otherwise-identical candidate with lower coverage or a smaller margin.
6. WHEN two candidates have equal final scores, THE Orchestrator SHALL apply a deterministic tie-break that yields identical ordering for identical inputs, so that the assigned ranks remain unique and contiguous.

### Requirement 6: Embedding Generation & Section-Aligned Chunking

**User Story:** As a recruiter, I want long candidate profiles embedded so that recent and role-relevant experience carries appropriate weight, so that current expertise is not diluted by old or irrelevant history.

#### Acceptance Criteria

1. WHEN the Embedding_Generator embeds an enriched profile whose serialized text is within the configured embedding model's maximum input token limit, THE Embedding_Generator SHALL return a single vector whose dimensionality equals the configured model's output dimensionality and whose L2 norm equals 1.0 within a tolerance of ±0.001.
2. WHEN the Embedding_Generator embeds an enriched profile whose serialized text exceeds the configured embedding model's maximum input token limit, THE Embedding_Generator SHALL split the text into chunks aligned to role or section boundaries such that each chunk corresponds to exactly one role or section block, and SHALL NOT produce chunks defined by fixed-size character or token splits.
3. WHEN the Embedding_Generator aggregates chunk embeddings, THE Embedding_Generator SHALL compute a weighted mean of the chunk vectors using non-negative weights in which a more recent or more role-relevant chunk receives a weight greater than or equal to that of a less recent or less role-relevant chunk, and SHALL return a single vector whose L2 norm equals 1.0 within a tolerance of ±0.001.
4. THE Embedding_Generator SHALL produce candidate profile vectors and requirement vectors using the same configured embedding model and the same output dimensionality, such that any candidate vector and any requirement vector are directly comparable by cosine similarity.

### Requirement 7: Fairness & Missing-Signal Handling

**User Story:** As a recruiter, I want candidates with sparse profiles or missing platform data treated fairly, so that lacking a public footprint or a single signal does not unfairly penalize an otherwise strong candidate.

#### Acceptance Criteria

1. THE Scoring_Engine SHALL exclude all Protected_Proxy attributes from scoring inputs, and THE Explanation_Generator SHALL exclude all Protected_Proxy attributes from generated explanations.
2. IF a candidate's Signal_Availability for a tier equals 0, THEN THE Scoring_Engine SHALL assign the Neutral_Prior (0.5) for that tier's sub-score rather than zero, and THE Orchestrator SHALL reduce that candidate's confidence relative to an otherwise-identical candidate whose coverage for that tier is greater than 0, without lowering that candidate's rank on the basis of the missing tier.
3. WHEN a candidate's Signal_Availability for a tier equals 0, THE Hard_Filter SHALL NOT exclude the candidate, and SHALL exclude a candidate only when the candidate positively matches a Disqualifying_Criterion.
4. WHEN Protected_Proxy attributes are perturbed while all job-relevant signals are held constant, THE Orchestrator SHALL assign the affected candidates ranks identical to those assigned before the perturbation (a rank delta of 0).
5. WHEN a candidate has Signal_Availability equal to 0 in every tier, THE Scoring_Engine SHALL score the candidate using the Neutral_Prior for every sub-score, and THE Orchestrator SHALL include the candidate in the output at the minimum Confidence rather than excluding the candidate.

### Requirement 8: LLM Contextual Reranking

**User Story:** As a recruiter, I want the top candidates reranked by nuanced LLM judgment, so that close calls are decided with reasoning beyond numeric scores while keeping cost bounded.

#### Acceptance Criteria

1. WHEN composite scoring completes and the number of surviving candidates exceeds K, THE Reranker SHALL apply LLM contextual reranking to only the K highest Composite_Score candidates, where K is a configured bound in the range 1 to 50.
2. WHEN the Reranker refines the ordering, THE Reranker SHALL compute each candidate's Final_Score as a fixed-weight combination of the LLM-derived score and the Composite_Score in which both scores receive a nonzero weight, the weights sum to 1.0, and the LLM-derived score is normalized to [0,1], and SHALL clamp the Final_Score to [0,1].
3. WHEN the Reranker builds its prompt, THE Reranker SHALL include each candidate's signal breakdown in the prompt.
4. WHEN the Reranker builds its prompt, THE Reranker SHALL exclude all Protected_Proxy attributes from the prompt.
5. WHEN composite scoring completes and the number of surviving candidates is less than or equal to K, THE Reranker SHALL apply LLM contextual reranking to all surviving candidates.

### Requirement 9: Error Handling & Resilience

**User Story:** As a recruiter, I want the system to degrade gracefully when individual stages fail, so that a single failure does not block the entire ranking or fabricate results.

#### Acceptance Criteria

1. IF an external behavioral signal fetch for a candidate does not complete within 10 seconds or is otherwise unavailable, THEN THE Candidate_Enricher SHALL set that candidate's Signal_Availability for the affected tier to 0 and SHALL proceed using the Neutral_Prior for that tier.
2. IF the embedding service returns an error, THEN THE Embedding_Generator SHALL retry the request up to a maximum of 3 times with increasing backoff between attempts, and SHALL reuse a cached embedding when one is available instead of retrying.
3. IF a candidate's embedding cannot be produced after the maximum retries and no cached embedding is available, THEN THE Orchestrator SHALL exclude that candidate from composite scoring and SHALL allow all successfully embedded candidates to proceed.
4. IF the Reranker fails, THEN THE Orchestrator SHALL fall back to composite-score ordering and SHALL return the shortlist flagged as un-reranked.
5. IF the Explanation_Generator fails for a candidate, THEN THE Orchestrator SHALL mark that candidate's explanation as unavailable and SHALL NOT substitute placeholder, inferred, or fabricated explanation content.

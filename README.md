# Intelligent Candidate Ranking System (ICRS)

ICRS ranks job candidates the way a world-class human recruiter would: through
semantic understanding, behavioral inference, and multi-signal scoring rather
than brittle keyword matching. It decomposes a job description into a structured,
weighted requirement schema, enriches each candidate profile with inferred
semantic and behavioral signals, and fuses four complementary scoring methods
(rule-based hard filters, dense vector similarity, sparse keyword matching, and
LLM contextual reranking) into a single explainable composite score.

This repository is a Proof of Concept (PoC).

## Project layout

```
icrs/
  config.py        # typed settings / config loader (env + .env)
  models/          # typed data models (RequirementVector, profiles, results)
  providers/       # abstract provider interfaces + task-keyed LLM registry
  pipeline/        # JD decomposition, enrichment, embedding, orchestration
  scoring/         # hybrid scoring engine (hard filter, dense/sparse, fusion)
  persistence/     # PostgreSQL + pgvector / Qdrant repositories
  ui/              # Streamlit PoC dashboard
tests/             # pytest + Hypothesis suite (deterministic profile)
```

## Providers (swappable, free-tier by default)

The pipeline depends only on the abstract interfaces in `icrs/providers/base.py`,
so concrete models are config-level swaps. The PoC defaults avoid paid APIs:

| Role | Default backend | Why |
|------|-----------------|-----|
| Embeddings | `BAAI/bge-large-en-v1.5` via `sentence-transformers` (local, CPU, 1024-dim) | Free, no paid API, strong open-source quality |
| LLM — JD decomposition + reranking | Groq-hosted `llama-3.3-70b-versatile` via the `groq` SDK (OpenAI-compatible) | Free tier, fast structured output |
| LLM — explanation generation | `gemini-2.5-flash` via `google-generativeai` | Free tier, long context, fluent prose |

`LLMProviderRegistry` routes each task (`DECOMPOSE`, `RERANK`, `EXPLAIN`) to a
concrete provider, so different tasks can use different backends. All model ids
and the embedding dimensionality are configurable.

## Setup

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux

# 2. install runtime + dev dependencies
pip install -r requirements-dev.txt
# or, editable install with extras:
# pip install -e ".[dev]"

# 3. configure environment
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
# then edit .env and fill in your keys
```

## Required environment variables

Secrets are read from the environment only and are never hardcoded. Copy
`.env.example` to `.env` and fill in:

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GROQ_API_KEY` | yes (for decompose/rerank) | — | Groq API key |
| `GOOGLE_API_KEY` | yes (for explanations) | — | Google Gemini API key |
| `ICRS_GROQ_MODEL` | no | `llama-3.3-70b-versatile` | Groq model id |
| `ICRS_GEMINI_MODEL` | no | `gemini-2.5-flash` | Gemini model id |
| `ICRS_EMBEDDING_MODEL` | no | `BAAI/bge-large-en-v1.5` | Embedding model id |
| `ICRS_EMBEDDING_DIM` | no | `1024` | Embedding dimensionality |
| `ICRS_EMBEDDING_DEVICE` | no | `cpu` | Embedding device (`cpu`/`cuda`) |
| `ICRS_MAX_INPUT_TOKENS` | no | `512` | Token limit before chunking |
| `ICRS_RERANK_K` | no | `10` | Rerank bound K (1..50) |
| `ICRS_DATABASE_URL` | no | local Postgres | PostgreSQL + pgvector DSN |
| `ICRS_QDRANT_URL` | no | `http://localhost:6333` | Qdrant URL |
| `ICRS_RANDOM_SEED` | no | `1729` | Fixed seed for deterministic stages |

## Running tests

```bash
pytest
```

Hypothesis uses a deterministic profile (`icrs-deterministic`, fixed seed) by
default so the deterministic pipeline stages are reproducible. Override with
`HYPOTHESIS_PROFILE=icrs-fast` for quicker local runs.

## Running the PoC dashboard

The recruiter-facing UI is a Streamlit app (`icrs/ui/dashboard.py`) that talks
to the FastAPI ranking endpoint. Start the backend first, then the dashboard:

```bash
# 1. start the ranking API (FastAPI) — defaults to http://localhost:8000
uvicorn icrs.api.app:create_app --factory --reload

# 2. in a second terminal, launch the Streamlit dashboard
streamlit run icrs/ui/dashboard.py
```

In the dashboard you paste or upload a job description, choose the job type,
upload or paste a candidate pool (a JSON array of objects with optional
`structured_fields`, `free_text`, and `external_handles`), and click **Rank
candidates**. The ranked shortlist shows each candidate's relative score and
confidence, with an expandable section per candidate for the recruiter summary,
driving signals, gaps, unmet must-haves, and the per-signal breakdown. The
backend URL is configurable in the sidebar (default `http://localhost:8000`).

Scores are presented as *relative* ranking scores in `[0,1]`, rounded to two
decimals — not absolute hiring probabilities — and run-level degradations (an
ordering that was not LLM-reranked, unavailable explanations, or candidates
excluded before ranking) are surfaced honestly as banners.

> Note: the `/rank` endpoint is an **unauthenticated** PoC endpoint. Do not
> expose it publicly without adding authentication, authorization, and rate
> limiting.

## Status

Task 1 (project scaffolding, abstract provider interfaces, typed config loader,
and the deterministic test harness) is complete. Concrete data models, pipeline
components, scoring, persistence, and the UI are implemented in subsequent tasks.

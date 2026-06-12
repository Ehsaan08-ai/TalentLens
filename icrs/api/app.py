"""FastAPI application for the asynchronous ICRS ranking API (Task 17.1).

This module exposes :func:`create_app`, a FastAPI app factory that serves the
end-to-end ranking pipeline over HTTP. A client submits a job description, a
job type, and a candidate pool to ``POST /rank`` and receives the ranked
shortlist — each entry carrying the final score, per-signal breakdown,
recruiter explanation, and confidence (Requirement 5.1) — together with the
run-level resilience flags from the orchestrator (Requirement 9). ``GET /health``
is a lightweight liveness probe.

Async + concurrency (Requirements 2.1, 5.1):
    The endpoint is ``async`` and awaits
    :meth:`~icrs.pipeline.orchestrator.RankingOrchestrator.rank_candidates_run`,
    so the server can service requests concurrently. Per-candidate enrichment /
    embedding concurrency is owned by the orchestrator's pipeline stages; the API
    layer awaits that single coroutine and does not duplicate or reorder its
    work, preserving the orchestrator's deterministic ordering guarantees
    (Requirements 2.5 / 5.3 / 5.6). The orchestrator is reused across requests.

Dependency injection / testability:
    ``create_app(orchestrator=...)`` accepts a fully-built orchestrator so tests
    (and alternate deployments) can inject a stub-backed pipeline that never
    touches the network. When no orchestrator is injected, the default
    production orchestrator is built **lazily** on first use from config-driven
    providers (see :mod:`icrs.api.providers`); nothing calls a real LLM /
    embedding API at import or app-creation time.

SECURITY — UNAUTHENTICATED ENDPOINT (PoC ONLY):
    This service exposes the ranking pipeline over the network with **NO
    authentication or authorization**. That is acceptable only for the PoC. A
    production deployment MUST add authentication (e.g. API keys / OAuth) and
    authorization in front of ``/rank`` before exposing it, and SHOULD add rate
    limiting and request-size limits. See the TODO in :func:`create_app`.
"""

from __future__ import annotations

from typing import Callable

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Request, status

from icrs.api.schemas import DecomposeJDRequest, DecomposeJDResponse, RankRequest, RankResponse
from icrs.config import Settings, get_settings
from icrs.models.candidate import RawCandidate
from icrs.pipeline.orchestrator import InvalidRankingInputError, RankingOrchestrator


def create_app(
    orchestrator: RankingOrchestrator | None = None,
    *,
    orchestrator_factory: Callable[[], RankingOrchestrator] | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build the ICRS ranking FastAPI app.

    Args:
        orchestrator: a fully-constructed orchestrator to serve every request
            with. Inject a stub-backed orchestrator in tests so no real LLM /
            embedding API is ever called.
        orchestrator_factory: optional zero-arg callable that lazily builds the
            orchestrator on first request (used when ``orchestrator`` is not
            supplied). Defaults to the config-driven
            :func:`icrs.api.providers.build_default_orchestrator`.
        settings: optional settings passed to the default factory.

    Returns:
        A configured :class:`fastapi.FastAPI` application.

    Note:
        # TODO(security): This PoC endpoint is UNAUTHENTICATED. Before any
        # production / shared deployment, add authentication + authorization
        # (and ideally rate limiting and request-size caps) in front of /rank.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        current_settings = app.state.settings or get_settings()

        # Setup Redis Client
        app.state.redis_client = None
        if current_settings.redis_url:
            import redis.asyncio as aioredis
            try:
                client = aioredis.from_url(current_settings.redis_url, decode_responses=True)
                await client.ping()
                app.state.redis_client = client
                print(f"Connected to Redis cache at {current_settings.redis_url}")
            except Exception as e:
                print(f"WARNING: Redis connection failed: {e}")

        # Setup PostgreSQL Store
        app.state.postgres_store = None
        from icrs.persistence.postgres import POSTGRES_AVAILABLE, PostgresRankingStore
        if POSTGRES_AVAILABLE and current_settings.database_url:
            try:
                store = PostgresRankingStore(settings=current_settings)
                await store.create_schema()
                app.state.postgres_store = store
                print("Connected to PostgreSQL database and verified schema")
            except Exception as e:
                print(f"WARNING: PostgreSQL initialization failed: {e}")

        yield

        # Shutdown
        if app.state.redis_client:
            await app.state.redis_client.close()
        if app.state.postgres_store:
            try:
                await app.state.postgres_store.dispose()
            except Exception:
                pass

    app = FastAPI(
        title="ICRS Ranking API",
        version="0.1.0",
        description=(
            "Intelligent Candidate Ranking System — PoC ranking API.\n\n"
            "WARNING: This is an UNAUTHENTICATED proof-of-concept endpoint. Do "
            "not expose it publicly without adding authentication, "
            "authorization, and rate limiting."
        ),
        lifespan=lifespan,
    )

    # The injected orchestrator (if any) is cached on app state; otherwise the
    # factory builds it lazily on the first request so no provider is constructed
    # or called at import / app-creation time.
    app.state.orchestrator = orchestrator
    app.state.orchestrator_factory = orchestrator_factory
    app.state.settings = settings

    def get_orchestrator() -> RankingOrchestrator:
        """Return the shared orchestrator, building it lazily on first use."""

        if app.state.orchestrator is not None:
            return app.state.orchestrator
        factory = app.state.orchestrator_factory
        if factory is not None:
            app.state.orchestrator = factory()
        else:
            # Lazy import so the (heavier) default-provider wiring is only loaded
            # when actually needed — never at import time.
            from icrs.api.providers import build_default_orchestrator

            app.state.orchestrator = build_default_orchestrator(app.state.settings)
        return app.state.orchestrator

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe. Returns ``{"status": "ok"}``."""

        return {"status": "ok"}

    @app.post("/rank", response_model=RankResponse)
    async def rank(
        payload: RankRequest,
        request: Request,
        orchestrator: RankingOrchestrator = Depends(get_orchestrator),
    ) -> RankResponse:
        """Rank ``payload.candidates`` against ``payload.raw_jd``.

        Check Redis cache first. If cache miss, check Postgres database.
        If both miss, run orchestrator, and cache response in both stores.
        """
        import hashlib
        import json
        import uuid

        # 1. Deterministically hash request payload (stable representation)
        payload_dict = {
            "raw_jd": payload.raw_jd,
            "job_type": payload.job_type.value if payload.job_type else None,
            "title": payload.title,
            "candidates": [
                {
                    "structured_fields": c.structured_fields,
                    "free_text": c.free_text,
                    "external_handles": c.external_handles,
                }
                for c in payload.candidates
            ]
        }
        payload_bytes = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        redis_key = f"icrs:rank:{payload_hash}"

        # 2. Check Redis Cache
        redis_client = getattr(request.app.state, "redis_client", None)
        if redis_client:
            try:
                cached_response = await redis_client.get(redis_key)
                if cached_response:
                    print(f"Redis Cache Hit for key {redis_key}")
                    return RankResponse.model_validate_json(cached_response)
            except Exception as e:
                print(f"WARNING: Redis get error: {e}")

        # 3. Check Postgres Database Cache
        postgres_store = getattr(request.app.state, "postgres_store", None)
        if postgres_store:
            try:
                db_response = await postgres_store.get_ranking_run(payload_hash)
                if db_response:
                    print(f"Database Cache Hit for key {redis_key}")
                    # Cache back to Redis
                    if redis_client:
                        try:
                            current_settings = request.app.state.settings or get_settings()
                            await redis_client.setex(
                                redis_key,
                                current_settings.redis_ttl,
                                json.dumps(db_response),
                            )
                        except Exception as e:
                            print(f"WARNING: Redis set error: {e}")
                    return RankResponse.model_validate(db_response)
            except Exception as e:
                print(f"WARNING: Database lookup error: {e}")

        # 4. Cache Miss: Run Orchestrator Pipeline
        print(f"Cache Miss for key {redis_key} - running ranking orchestrator")
        pool = []
        for c in payload.candidates:
            # Bug 4 fix: pop client_assigned_id from structured_fields so it
            # doesn't flow into normalization/embedding as noise text. The UUID
            # is only needed for response mapping, not for scoring.
            structured = dict(c.structured_fields)
            client_id = structured.pop("client_assigned_id", None)
            candidate_uuid = None
            if client_id:
                try:
                    candidate_uuid = uuid.UUID(str(client_id))
                except ValueError:
                    pass
            pool.append(
                RawCandidate(
                    id=candidate_uuid if candidate_uuid is not None else uuid.uuid4(),
                    structured_fields=structured,
                    free_text=c.free_text,
                    external_handles=c.external_handles,
                )
            )

        try:
            run = await orchestrator.rank_candidates_run(
                payload.raw_jd, pool, payload.job_type
            )
        except InvalidRankingInputError as exc:
            # Requirement 2.6: empty/whitespace JD or empty pool -> 400.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

        response = RankResponse.from_run(run)
        response_dict = response.model_dump(mode="json")

        # 5. Save to Postgres Database
        if postgres_store:
            try:
                await postgres_store.save_ranking_run(payload_hash, response_dict)
            except Exception as e:
                print(f"WARNING: Database save error: {e}")

        # 6. Save to Redis Cache
        if redis_client:
            try:
                current_settings = request.app.state.settings or get_settings()
                await redis_client.setex(
                    redis_key,
                    current_settings.redis_ttl,
                    json.dumps(response_dict),
                )
            except Exception as e:
                print(f"WARNING: Redis set error: {e}")

        return response

    @app.post("/decompose-jd", response_model=DecomposeJDResponse)
    async def decompose_jd(
        payload: DecomposeJDRequest,
        orchestrator: RankingOrchestrator = Depends(get_orchestrator),
    ) -> DecomposeJDResponse:
        """Decompose a raw JD into intent, must-haves, nice-to-haves, and behavioral signals."""
        if not payload.raw_jd or not payload.raw_jd.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="raw_jd must contain at least one non-whitespace character",
            )
        try:
            from fastapi.concurrency import run_in_threadpool

            # Use the public decompose_jd method instead of reaching into
            # the private _decomposer attribute — preserves layering contract.
            vector = await run_in_threadpool(orchestrator.decompose_jd, payload.raw_jd)

            must_have = []
            nice_to_have = []
            behavioral_signals = []

            for req in vector.requirements:
                if req.category.value == "MUST_HAVE":
                    must_have.append(req.text)
                elif req.category.value == "NICE_TO_HAVE":
                    nice_to_have.append(req.text)

                if req.tier.value == "BEHAVIORAL":
                    behavioral_signals.append(req.text)

            # Add implicit expectations and culture signals to behavioral signals
            for imp in vector.implicit_expectations:
                if imp not in behavioral_signals:
                    behavioral_signals.append(imp)
            for cult in vector.culture_signals:
                if cult not in behavioral_signals:
                    behavioral_signals.append(cult)

            return DecomposeJDResponse(
                role_intent=vector.role_intent,
                must_have=must_have,
                nice_to_have=nice_to_have,
                behavioral_signals=behavioral_signals,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    return app


__all__ = ["create_app"]

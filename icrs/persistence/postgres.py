"""PostgreSQL + pgvector implementation of :class:`RankingStore` (Task 6.1).

This is the real Phase-1 *system of record* (see the design's data-layer
section): relational rows for jobs and candidate profiles, and a ``pgvector``
column for per-candidate embeddings so candidate and requirement vectors share a
dimensionality and are directly comparable by cosine similarity (Requirement
2.2, 6.4).

Heavy / optional dependencies (``asyncpg`` and ``pgvector``) are imported
defensively: this module always imports cleanly even when those packages — or a
running PostgreSQL server — are unavailable. When they are missing, the ORM
models are not defined and constructing :class:`PostgresRankingStore` raises a
clear, actionable error. This lets the rest of ICRS (and the test suite, which
uses :class:`~icrs.persistence.memory.InMemoryRankingStore`) import and run
without a database.

The schema is intentionally simple for the PoC: the full pydantic models are
persisted as JSON documents, while the embedding is stored in a native
``vector`` column to keep ANN search (Task 10) on the real system-of-record.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from icrs.config import Settings, get_settings
from icrs.models.candidate import EnrichedProfile
from icrs.models.job import JobDescription
from icrs.persistence.base import RankingStore, validate_dimensionality
from icrs.providers.base import Vector

# --- Defensive import of the heavy/optional persistence stack. ---------------
# SQLAlchemy is a declared core dependency, but asyncpg (the async driver) and
# pgvector (the column type) may be absent in environments without a database.
# We degrade gracefully so importing this module never fails.
try:  # pragma: no cover - exercised only when the stack is installed
    from pgvector.sqlalchemy import Vector as PgVector
    from sqlalchemy import JSON, String, select
    from sqlalchemy.dialects.postgresql import UUID as PgUUID
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

    import asyncpg  # noqa: F401  (presence check: the async driver must be installed)

    _IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - import-guard branch
    PgVector = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

#: Whether the PostgreSQL + pgvector stack is importable in this environment.
POSTGRES_AVAILABLE: bool = _IMPORT_ERROR is None


if POSTGRES_AVAILABLE:  # pragma: no cover - requires the optional stack installed

    class _Base(DeclarativeBase):
        """Declarative base for ICRS persistence tables."""

    class JobRow(_Base):
        """Relational row for a :class:`JobDescription` and its parsed vector."""

        __tablename__ = "icrs_jobs"

        id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
        # Full JobDescription (including the parsed RequirementVector) as JSON.
        document: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    class CandidateRow(_Base):
        """Relational row for an :class:`EnrichedProfile`, optionally job-scoped."""

        __tablename__ = "icrs_candidates"

        id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
        job_id: Mapped[UUID | None] = mapped_column(
            PgUUID(as_uuid=True), nullable=True, index=True
        )
        document: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    class EmbeddingRow(_Base):
        """Per-candidate embedding stored in a native pgvector column.

        The column dimensionality is fixed at table-creation time to the
        configured embedding model dimensionality, enforcing Requirement 2.2 at
        the schema level in addition to the application-level check.
        """

        __tablename__ = "icrs_embeddings"

        candidate_id: Mapped[UUID] = mapped_column(
            PgUUID(as_uuid=True), primary_key=True
        )
        # The vector dimensionality is bound when the store builds the table DDL.
        model_id: Mapped[str] = mapped_column(String, nullable=False)

    class RankingRunRow(_Base):
        """Relational row for a full RankingRun response keyed by payload hash."""

        __tablename__ = "icrs_ranking_runs"

        id: Mapped[str] = mapped_column(String, primary_key=True)
        # Full RankResponse as JSON.
        document: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class PostgresRankingStore(RankingStore):
    """PostgreSQL + pgvector system-of-record implementation.

    Construction requires the optional ``asyncpg`` + ``pgvector`` stack. When it
    is unavailable a :class:`RuntimeError` is raised immediately with guidance,
    rather than failing later at query time.

    Note: exercising this store requires a live PostgreSQL server with the
    ``vector`` extension enabled. It is structured so that unit tests run against
    :class:`~icrs.persistence.memory.InMemoryRankingStore` instead.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedding_dim: int | None = None,
    ) -> None:
        if not POSTGRES_AVAILABLE:
            raise RuntimeError(
                "PostgresRankingStore requires the 'asyncpg' and 'pgvector' "
                "packages. Install them (see requirements.txt) or use "
                "InMemoryRankingStore for tests / DB-less runs."
            ) from _IMPORT_ERROR

        self._settings = settings or get_settings()
        self.embedding_dim = (
            embedding_dim if embedding_dim is not None else self._settings.embedding_dim
        )
        self._model_id = self._settings.embedding_model

        # The embedding table's vector column is dimensioned to the configured
        # model so the database itself rejects mismatched vectors.
        self._embedding_table = self._build_embedding_table(self.embedding_dim)

        self._engine = create_async_engine(self._settings.database_url, future=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)

    @staticmethod
    def _build_embedding_table(dim: int):  # pragma: no cover - requires stack
        """Attach a correctly-dimensioned pgvector column to the embedding row.

        The dimensionality is only known at runtime (from settings), so the
        ``vector(dim)`` column is added dynamically rather than declared
        statically on :class:`EmbeddingRow`.
        """

        if not hasattr(EmbeddingRow, "vector"):
            EmbeddingRow.vector = mapped_column(  # type: ignore[attr-defined]
                PgVector(dim), nullable=False
            )
        return EmbeddingRow

    async def create_schema(self) -> None:  # pragma: no cover - requires live DB
        """Create the pgvector extension and all tables if they do not exist."""

        from sqlalchemy import text

        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(_Base.metadata.create_all)

    # ----- Jobs + parsed RequirementVector -----

    async def save_job(self, job: JobDescription) -> None:  # pragma: no cover
        document = job.model_dump(mode="json")
        async with self._sessionmaker() as session:
            await session.merge(JobRow(id=job.id, document=document))
            await session.commit()

    async def get_job(self, job_id: UUID) -> JobDescription | None:  # pragma: no cover
        async with self._sessionmaker() as session:
            row = await session.get(JobRow, job_id)
            if row is None:
                return None
            return JobDescription.model_validate(row.document)

    # ----- Candidate profiles -----

    async def save_candidate(  # pragma: no cover - requires live DB
        self, profile: EnrichedProfile, *, job_id: UUID | None = None
    ) -> None:
        if profile.embedding is not None:
            validate_dimensionality(profile.embedding, self.embedding_dim)

        document = profile.model_dump(mode="json")
        async with self._sessionmaker() as session:
            await session.merge(
                CandidateRow(id=profile.id, job_id=job_id, document=document)
            )
            if profile.embedding is not None:
                await session.merge(
                    self._embedding_table(
                        candidate_id=profile.id,
                        model_id=self._model_id,
                        vector=list(profile.embedding),
                    )
                )
            await session.commit()

    async def get_candidate(  # pragma: no cover - requires live DB
        self, candidate_id: UUID
    ) -> EnrichedProfile | None:
        async with self._sessionmaker() as session:
            row = await session.get(CandidateRow, candidate_id)
            if row is None:
                return None
            return EnrichedProfile.model_validate(row.document)

    async def list_candidates_for_job(  # pragma: no cover - requires live DB
        self, job_id: UUID
    ) -> list[EnrichedProfile]:
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(CandidateRow).where(CandidateRow.job_id == job_id)
            )
            return [
                EnrichedProfile.model_validate(row.document)
                for row in result.scalars().all()
            ]

    # ----- Per-candidate embeddings -----

    async def upsert_embedding(  # pragma: no cover - requires live DB
        self, candidate_id: UUID, vector: Vector
    ) -> None:
        validate_dimensionality(vector, self.embedding_dim)
        async with self._sessionmaker() as session:
            await session.merge(
                self._embedding_table(
                    candidate_id=candidate_id,
                    model_id=self._model_id,
                    vector=list(vector),
                )
            )
            await session.commit()

    async def get_embedding(  # pragma: no cover - requires live DB
        self, candidate_id: UUID
    ) -> Vector | None:
        async with self._sessionmaker() as session:
            row = await session.get(self._embedding_table, candidate_id)
            if row is None:
                return None
            return list(row.vector)

    async def save_ranking_run(  # pragma: no cover - requires live DB
        self, run_id: str, document: dict[str, Any]
    ) -> None:
        async with self._sessionmaker() as session:
            await session.merge(RankingRunRow(id=run_id, document=document))
            await session.commit()

    async def get_ranking_run(  # pragma: no cover - requires live DB
        self, run_id: str
    ) -> dict[str, Any] | None:
        async with self._sessionmaker() as session:
            row = await session.get(RankingRunRow, run_id)
            if row is None:
                return None
            return row.document

    async def dispose(self) -> None:  # pragma: no cover - requires live DB
        """Dispose of the underlying engine / connection pool."""

        await self._engine.dispose()


__all__ = ["POSTGRES_AVAILABLE", "PostgresRankingStore"]

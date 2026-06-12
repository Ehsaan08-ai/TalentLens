"""Persistence layer for ICRS (Task 6.1).

Defines an abstract async :class:`RankingStore` contract plus two concrete
implementations:

    - :class:`PostgresRankingStore` — the PostgreSQL + pgvector system-of-record
      (requires the optional ``asyncpg`` + ``pgvector`` stack and a live DB).
    - :class:`InMemoryRankingStore` — a dependency-free async store used by tests
      and by the Phase 1 ingestion path without a database.

Both enforce that stored embedding dimensionality matches the configured
embedding model (Requirement 2.2), raising :class:`DimensionalityError` on a
mismatch.

The PostgreSQL implementation is imported lazily-guarded so importing this
package never fails when the optional database stack is absent;
``POSTGRES_AVAILABLE`` reports whether it can be constructed.
"""

from icrs.persistence.base import (
    DimensionalityError,
    PersistenceError,
    RankingStore,
    validate_dimensionality,
)
from icrs.persistence.memory import InMemoryRankingStore
from icrs.persistence.postgres import POSTGRES_AVAILABLE, PostgresRankingStore

__all__ = [
    "RankingStore",
    "PersistenceError",
    "DimensionalityError",
    "validate_dimensionality",
    "InMemoryRankingStore",
    "PostgresRankingStore",
    "POSTGRES_AVAILABLE",
]

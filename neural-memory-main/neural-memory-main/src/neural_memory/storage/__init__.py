"""Storage backends for NeuralMemory."""

from neural_memory.storage.base import CoreStorage, ExtendedStorage, NeuralStorage
from neural_memory.storage.factory import HybridStorage, create_storage
from neural_memory.storage.memory_store import InMemoryStorage
from neural_memory.storage.shared_store import SharedStorage
from neural_memory.storage.shared_store_collections import SharedStorageError

# Unified SQL adapter (new — supersedes per-engine implementations)
from neural_memory.storage.sql import Dialect, SQLStorage
from neural_memory.storage.sql.postgres_dialect import PostgresDialect
from neural_memory.storage.sql.sqlite_dialect import SQLiteDialect
from neural_memory.storage.sqlite_store import SQLiteStorage

__all__ = [
    "CoreStorage",
    "Dialect",
    "ExtendedStorage",
    "HybridStorage",
    "InMemoryStorage",
    "NeuralStorage",
    "PostgresDialect",
    "SQLiteDialect",
    "SQLiteStorage",
    "SQLStorage",
    "SharedStorage",
    "SharedStorageError",
    "create_storage",
]


# Lazy imports for optional backends (avoid requiring packages for SQLite users)
def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "PostgreSQLStorage":
        from neural_memory.storage.postgres.postgres_store import PostgreSQLStorage

        return PostgreSQLStorage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

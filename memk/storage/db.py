"""
memk.storage.db
===============
SQLite storage adapter — the *only* layer that speaks SQL.

Schema
------
memories   : raw immutable chat / event log
facts      : reconciled Subject-Predicate-Object knowledge graph
decisions  : agent decision audit trail

v0.3 additions (metadata / scoring support)
--------------------------------------------
Both memories and facts now carry:
  importance       REAL [0,1]   — domain priority set at insert time
  confidence       REAL [0,1]   — epistemic certainty
  access_count     INTEGER      — incremented on each retrieval
  last_accessed_at TEXT         — ISO UTC timestamp of last retrieval

Migration is fully backward-compatible: the init_db() method applies
ALTER TABLE statements guarded by OperationalError so the same code
runs on any database version without wiping data.
"""

import sqlite3
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatabaseError(Exception):
    """Raised for any unrecoverable storage-layer error."""


# ---------------------------------------------------------------------------
# MemoryDB
# ---------------------------------------------------------------------------

class MemoryDB:
    def __init__(self, db_path: Optional[str] = None):
        import os
        self.db_path = db_path or os.getenv("MEMK_DB_PATH", "mem.db")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return a configured SQLite connection (row_factory = Row)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"DB connect failed [{self.db_path}]: {e}")
            raise DatabaseError(f"Connection failed: {e}") from e

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """
        Create tables and apply all pending migrations.
        Safe to call multiple times — fully idempotent.
        """
        create_memories = """
            CREATE TABLE IF NOT EXISTS memories (
                id               TEXT    PRIMARY KEY,
                content          TEXT    NOT NULL,
                embedding        BLOB,
                importance       REAL    DEFAULT 0.5,
                confidence       REAL    DEFAULT 1.0,
                access_count     INTEGER DEFAULT 0,
                last_accessed_at TEXT,
                decay_score      REAL    DEFAULT 1.0,
                created_at       TEXT    NOT NULL
            )
        """
        create_facts = """
            CREATE TABLE IF NOT EXISTS facts (
                id               TEXT    PRIMARY KEY,
                subject          TEXT    NOT NULL,
                predicate        TEXT    NOT NULL,
                object           TEXT    NOT NULL,
                confidence       REAL    DEFAULT 1.0,
                importance       REAL    DEFAULT 0.5,
                embedding        BLOB,
                access_count     INTEGER DEFAULT 0,
                last_accessed_at TEXT,
                decay_score      REAL    DEFAULT 1.0,
                created_at       TEXT    NOT NULL,
                is_active        INTEGER DEFAULT 1
            )
        """
        create_decisions = """
            CREATE TABLE IF NOT EXISTS decisions (
                id           TEXT PRIMARY KEY,
                action       TEXT NOT NULL,
                reason       TEXT NOT NULL,
                used_fact_ids TEXT,
                created_at   TEXT NOT NULL
            )
        """

        # Migration map: (table, column, definition)
        migrations = [
            # v0.1 → v0.2 embeddings
            ("facts",    "is_active",        "INTEGER DEFAULT 1"),
            ("memories", "embedding",         "BLOB"),
            ("facts",    "embedding",         "BLOB"),
            # v0.2 → v0.3 scoring metadata
            ("memories", "importance",        "REAL DEFAULT 0.5"),
            ("memories", "confidence",        "REAL DEFAULT 1.0"),
            ("memories", "access_count",      "INTEGER DEFAULT 0"),
            ("memories", "last_accessed_at",  "TEXT"),
            ("facts",    "access_count",      "INTEGER DEFAULT 0"),
            ("facts",    "last_accessed_at",  "TEXT"),
            ("memories", "decay_score",       "REAL DEFAULT 1.0"),
            ("facts",    "decay_score",       "REAL DEFAULT 1.0"),
            # facts.importance was INTEGER in v0.1 — it stays as-is; new rows use REAL
        ]

        try:
            with self._get_connection() as conn:
                conn.execute(create_memories)
                conn.execute(create_facts)
                conn.execute(create_decisions)
                for table, col, defn in migrations:
                    try:
                        conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {col} {defn}"
                        )
                    except sqlite3.OperationalError:
                        pass  # Column already exists — expected on re-init
        except sqlite3.Error as e:
            logger.error(f"Schema init failed: {e}")
            raise DatabaseError(f"Initialization failed: {e}") from e

    # ------------------------------------------------------------------
    # Memories — CRUD
    # ------------------------------------------------------------------

    def insert_memory(
        self,
        content: str,
        *,
        embedding: Optional[np.ndarray] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
    ) -> str:
        """
        Insert a raw memory. Returns its UUID.

        Parameters
        ----------
        content    : Raw text to store (must be non-empty).
        embedding  : Optional pre-computed float32 vector.
        importance : Domain priority [0, 1]. Default 0.5 = neutral.
        confidence : How certain this memory is [0, 1].
        """
        if not content or not content.strip():
            raise ValueError("Memory content cannot be empty.")

        mem_id = str(uuid.uuid4())
        created_at = _utcnow()
        blob = _encode_blob(embedding) if embedding is not None else None

        sql = """
            INSERT INTO memories
                (id, content, embedding, importance, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (
                    mem_id, content.strip(), blob,
                    float(importance), float(confidence), created_at,
                ))
            return mem_id
        except sqlite3.Error as e:
            logger.error(f"insert_memory failed: {e}")
            raise DatabaseError(f"Insertion failed: {e}") from e

    def update_memory_embedding(self, mem_id: str, embedding: np.ndarray) -> None:
        """Backfill / refresh the stored embedding for a memory row."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (_encode_blob(embedding), mem_id),
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Embedding update failed: {e}") from e

    def touch_memory(self, mem_id: str) -> None:
        """
        Increment access_count and refresh last_accessed_at.
        Called by retrievers after surfacing a memory to the user/agent.
        """
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE memories
                    SET access_count     = access_count + 1,
                        last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (_utcnow(), mem_id),
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"touch_memory failed: {e}") from e

    def search_memory(self, keyword: str) -> List[Dict[str, Any]]:
        """Case-insensitive LIKE search over memory content."""
        if not keyword:
            return []
        sql = """
            SELECT * FROM memories
            WHERE content LIKE ?
            ORDER BY created_at DESC
        """
        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(sql, (f"%{keyword}%",)).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"search_memory failed: {e}") from e

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Fetch every memory row for full-corpus vector scan."""
        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC"
                ).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_all_memories failed: {e}") from e

    def get_memories_without_embedding(self) -> List[Dict[str, Any]]:
        """Return rows that lack an embedding (for async backfill jobs)."""
        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(
                    "SELECT id, content FROM memories WHERE embedding IS NULL"
                ).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_memories_without_embedding failed: {e}") from e

    def get_top_memories_by_metadata(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return memories ranked by (importance DESC, confidence DESC, created_at DESC).
        Used by `memk doctor` for a cold-start ranking without a query.
        """
        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(
                    """
                    SELECT * FROM memories
                    ORDER BY importance DESC, confidence DESC, created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_top_memories_by_metadata failed: {e}") from e

    # ------------------------------------------------------------------
    # Facts — CRUD
    # ------------------------------------------------------------------

    def insert_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        importance: float = 0.5,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """
        Insert a structured fact (SPO triplet). Returns its UUID.

        Auto-reconciles: any previous active fact with the same
        (subject, predicate) is set is_active=0 before insertion.

        Parameters
        ----------
        importance : Domain priority [0, 1]. LLM-extracted facts default to 0.5;
                     manually annotated critical facts should be set higher.
        """
        if not subject or not predicate or not obj:
            raise ValueError("subject, predicate, and object cannot be empty.")

        fact_id = str(uuid.uuid4())
        created_at = _utcnow()
        blob = _encode_blob(embedding) if embedding is not None else None

        reconcile = """
            UPDATE facts SET is_active = 0
            WHERE subject = ? AND predicate = ? AND is_active = 1
        """
        insert = """
            INSERT INTO facts
                (id, subject, predicate, object, confidence, importance,
                 embedding, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(reconcile, (subject.strip(), predicate.strip()))
                conn.execute(insert, (
                    fact_id,
                    subject.strip(), predicate.strip(), obj.strip(),
                    float(confidence), float(importance),
                    blob, created_at,
                ))
            return fact_id
        except sqlite3.Error as e:
            logger.error(f"insert_fact failed: {e}")
            raise DatabaseError(f"Fact insertion failed: {e}") from e

    def update_fact_embedding(self, fact_id: str, embedding: np.ndarray) -> None:
        """Backfill / refresh the stored embedding for a fact row."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE facts SET embedding = ? WHERE id = ?",
                    (_encode_blob(embedding), fact_id),
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Fact embedding update failed: {e}") from e

    def touch_fact(self, fact_id: str) -> None:
        """Increment access_count and refresh last_accessed_at for a fact."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE facts
                    SET access_count     = access_count + 1,
                        last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (_utcnow(), fact_id),
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"touch_fact failed: {e}") from e

    def search_facts(
        self,
        subject: Optional[str] = None,
        keyword: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return active facts matching subject and/or keyword.
        Returns only is_active=1 rows (non-reconciled current truths).
        """
        sql = "SELECT * FROM facts WHERE is_active = 1"
        params: List[Any] = []

        if subject:
            sql += " AND subject = ?"
            params.append(subject)

        if keyword:
            sql += " AND (subject LIKE ? OR predicate LIKE ? OR object LIKE ?)"
            like = f"%{keyword}%"
            params.extend([like, like, like])

        sql += " ORDER BY created_at DESC"

        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"search_facts failed: {e}") from e

    def get_all_active_facts(self) -> List[Dict[str, Any]]:
        """Fetch all active fact rows for full-corpus vector scan."""
        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(
                    "SELECT * FROM facts WHERE is_active = 1 ORDER BY created_at DESC"
                ).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_all_active_facts failed: {e}") from e

    def get_facts_without_embedding(self) -> List[Dict[str, Any]]:
        """Return active facts that lack an embedding."""
        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(
                    """
                    SELECT id, subject, predicate, object
                    FROM facts
                    WHERE is_active = 1 AND embedding IS NULL
                    """
                ).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_facts_without_embedding failed: {e}") from e

    def get_top_facts_by_metadata(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Active facts ranked by priority metadata (for cold doctor report)."""
        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(
                    """
                    SELECT * FROM facts
                    WHERE is_active = 1
                    ORDER BY importance DESC, confidence DESC, created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_top_facts_by_metadata failed: {e}") from e

    def get_fact_conflicts(self, active_fact_ids: List[str]) -> List[Dict[str, Any]]:
        """
        For a given set of active fact IDs, find their historical (inactive) versions.
        Useful for highlighting where knowledge has evolved or conflicted.
        """
        if not active_fact_ids:
            return []
            
        placeholders = ",".join("?" * len(active_fact_ids))
        sql = f"""
            WITH active_keys AS (
                SELECT subject, predicate FROM facts WHERE id IN ({placeholders})
            )
            SELECT f.* FROM facts f
            JOIN active_keys ak ON f.subject = ak.subject AND f.predicate = ak.predicate
            WHERE f.is_active = 0
            ORDER BY f.created_at DESC
        """
        try:
            with self._get_connection() as conn:
                return [_to_dict(r) for r in conn.execute(sql, active_fact_ids).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_fact_conflicts failed: {e}") from e

    def get_all_subjects(self) -> List[str]:
        """Return a unique list of subjects from all active facts."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT subject FROM facts WHERE is_active = 1"
                ).fetchall()
                return [r[0] for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_all_subjects failed: {e}") from e

    def update_decay_scores(self, scorer_fn) -> int:
        """
        Iterate through all memories and facts, apply decay scoring logic,
        and update the stored decay_score.
        """
        updated = 0
        try:
            with self._get_connection() as conn:
                # 1. Update Memories
                rows = conn.execute("SELECT id, importance, access_count, created_at FROM memories").fetchall()
                for r in rows:
                    new_score = scorer_fn(r['importance'], r['access_count'], r['created_at'])
                    conn.execute("UPDATE memories SET decay_score = ? WHERE id = ?", (new_score, r['id']))
                    updated += 1
                
                # 2. Update Facts
                rows = conn.execute("SELECT id, importance, access_count, created_at FROM facts").fetchall()
                for r in rows:
                    new_score = scorer_fn(r['importance'], r['access_count'], r['created_at'])
                    conn.execute("UPDATE facts SET decay_score = ? WHERE id = ?", (new_score, r['id']))
                    updated += 1
                
                conn.commit()
            return updated
        except sqlite3.Error as e:
            raise DatabaseError(f"update_decay_scores failed: {e}") from e

    def prune_cold_memories(self, threshold: float) -> int:
        """Delete memories and inactive facts scoring below the threshold."""
        try:
            with self._get_connection() as conn:
                # We typically don't delete active facts unless they are really irrelevant,
                # but raw memories can be pruned safely.
                c1 = conn.execute("DELETE FROM memories WHERE decay_score < ?", (threshold,)).rowcount
                # For facts, maybe we just deactivate them or delete if inactive
                c2 = conn.execute("DELETE FROM facts WHERE decay_score < ? AND is_active = 0", (threshold,)).rowcount
                conn.commit()
                return c1 + c2
        except sqlite3.Error as e:
            raise DatabaseError(f"prune_cold_memories failed: {e}") from e

    def get_state_counts(self, cold_th: float, warm_th: float) -> Dict[str, int]:
        """Group all items into hot/warm/cold counts."""
        try:
            with self._get_connection() as conn:
                sql = """
                    SELECT 
                        SUM(CASE WHEN decay_score >= ? THEN 1 ELSE 0 END) as hot,
                        SUM(CASE WHEN decay_score >= ? AND decay_score < ? THEN 1 ELSE 0 END) as warm,
                        SUM(CASE WHEN decay_score < ? THEN 1 ELSE 0 END) as cold
                    FROM (SELECT decay_score FROM memories UNION ALL SELECT decay_score FROM facts)
                """
                row = conn.execute(sql, (warm_th, cold_th, warm_th, cold_th)).fetchone()
                return {"hot": row[0] or 0, "warm": row[1] or 0, "cold": row[2] or 0}
        except sqlite3.Error as e:
            raise DatabaseError(f"get_state_counts failed: {e}") from e

    # ------------------------------------------------------------------
    # Decisions / Audit Log
    # ------------------------------------------------------------------

    def log_decision(
        self,
        action: str,
        reason: str,
        used_fact_ids: Optional[List[str]] = None,
    ) -> str:
        """Append a decision event for agent observability."""
        import json
        decision_id = str(uuid.uuid4())
        sql = """
            INSERT INTO decisions (id, action, reason, used_fact_ids, created_at)
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (
                    decision_id, action, reason,
                    json.dumps(used_fact_ids or []),
                    _utcnow(),
                ))
            return decision_id
        except sqlite3.Error as e:
            raise DatabaseError(f"log_decision failed: {e}") from e

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics for `memk doctor`."""
        try:
            with self._get_connection() as conn:
                def scalar(q: str) -> int:
                    return conn.execute(q).fetchone()[0]

                return {
                    "total_memories":       scalar("SELECT COUNT(*) FROM memories"),
                    "total_active_facts":   scalar("SELECT COUNT(*) FROM facts WHERE is_active = 1"),
                    "embedded_memories":    scalar("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"),
                    "embedded_facts":       scalar(
                        "SELECT COUNT(*) FROM facts WHERE is_active = 1 AND embedding IS NOT NULL"
                    ),
                    "total_decisions":      scalar("SELECT COUNT(*) FROM decisions"),
                    "most_accessed_memory": _most_accessed(conn, "memories"),
                    "most_accessed_fact":   _most_accessed(conn, "facts"),
                }
        except sqlite3.Error as e:
            raise DatabaseError(f"get_stats failed: {e}") from e


# ---------------------------------------------------------------------------
# Module-level helpers (also imported by retriever / tests)
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    """Current UTC time as ISO 8601 string with microseconds."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")


def _encode_blob(vec: np.ndarray) -> bytes:
    """Serialize float32 numpy vector → raw bytes (4 bytes/element)."""
    import struct
    arr = vec.astype(np.float32)
    return struct.pack(f"{len(arr)}f", *arr)


def _decode_blob(blob: bytes) -> np.ndarray:
    """Deserialize raw bytes → float32 numpy vector."""
    import struct
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def _to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert sqlite3.Row to plain dict (BLOB columns kept as raw bytes)."""
    return dict(row)


def _most_accessed(conn: sqlite3.Connection, table: str) -> Optional[str]:
    """Return the content/subject of the most-accessed row, or None."""
    try:
        col = "content" if table == "memories" else "subject"
        row = conn.execute(
            f"SELECT {col} FROM {table} ORDER BY access_count DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None

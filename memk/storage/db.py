"""
memk.storage.db
===============
SQLite storage adapter — the *only* layer that speaks SQL.

Schema
------
memories   : raw immutable chat / event log
facts      : reconciled Subject-Predicate-Object knowledge graph
decisions  : agent decision audit trail

v0.4 production hardening
--------------------------
- Schema versioning with migration framework
- WAL mode for concurrent access
- Performance indexes
- Background job tracking
- Database metadata
"""

import sqlite3
import uuid
import logging
import json
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from memk.storage.migrations import auto_migrate, check_schema_version
from memk.storage.config import configure_connection, get_database_info
from memk.core.hlc import GLOBAL_HLC

logger = logging.getLogger("memk.storage")


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
        """Return a configured SQLite connection with WAL mode and optimizations."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Apply production configuration (WAL mode, pragmas)
            configure_connection(conn)
            
            return conn
        except sqlite3.Error as e:
            logger.error(f"DB connect failed [{self.db_path}]: {e}")
            raise DatabaseError(f"Connection failed: {e}") from e

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """
        Yield a configured connection and always close it on exit.

        sqlite3.Connection's native context manager manages transactions only;
        it does not close the file handle. Keeping this lifecycle explicit avoids
        locked database files on Windows, especially in tempfile-backed tests.
        """
        conn = self._get_connection()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """
        Create tables and apply all pending migrations.
        Safe to call multiple times — fully idempotent.
        
        Uses migration framework for schema versioning.
        """
        # Run auto-migration first (handles schema versioning)
        try:
            migrated = auto_migrate(self.db_path)
            if migrated:
                logger.info(f"Database migrations applied: {self.db_path}")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise DatabaseError(f"Migration failed: {e}") from e
        
        # Create core tables (idempotent - IF NOT EXISTS)
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
                created_at       TEXT    NOT NULL,
                centroid_id      TEXT,
                heat_tier        INTEGER DEFAULT 0,
                archived         INTEGER DEFAULT 0,
                version_hlc      INTEGER DEFAULT 0,
                version_node     TEXT,
                version_seq      INTEGER DEFAULT 0
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
                is_active        INTEGER DEFAULT 1,
                version_hlc      INTEGER DEFAULT 0,
                version_node     TEXT,
                version_seq      INTEGER DEFAULT 0
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

        try:
            with self.connection() as conn:
                conn.execute(create_memories)
                conn.execute(create_facts)
                conn.execute(create_decisions)
                
                # Log database info
                info = get_database_info(conn)
                logger.info(f"Database initialized: {info.get('journal_mode')} mode, "
                           f"{info.get('size_mb', 0)}MB")
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
        
        hlc, node, seq = GLOBAL_HLC.next_version()

        sql = """
            INSERT INTO memories
                (id, content, embedding, importance, confidence, created_at, version_hlc, version_node, version_seq)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self.connection() as conn:
                conn.execute(sql, (
                    mem_id, content.strip(), blob,
                    float(importance), float(confidence), created_at,
                    hlc, node, seq
                ))
                
                # Automatically log to sync tables
                row_data = {
                    "id": mem_id, "content": content.strip(), 
                    "importance": importance, "confidence": confidence,
                    "created_at": created_at,
                    "version_hlc": hlc, "version_node": node, "version_seq": seq
                }
                self._log_sync_operation(conn, "memories", mem_id, "INSERT", row_data, (hlc, node, seq))
                
            return mem_id
        except sqlite3.Error as e:
            logger.error(f"insert_memory failed: {e}")
            raise DatabaseError(f"Insertion failed: {e}") from e

    def update_memory_content(self, mem_id: str, content: str) -> None:
        """Fully update memory content and bump sync version."""
        hlc, node, seq = GLOBAL_HLC.next_version()
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE memories SET content = ?, version_hlc = ?, version_node = ?, version_seq = ? WHERE id = ?",
                    (content.strip(), hlc, node, seq, mem_id)
                )
                self._log_sync_operation(conn, "memories", mem_id, "UPDATE", {}, (hlc, node, seq))
        except sqlite3.Error as e:
            raise DatabaseError(f"update_memory_content failed: {e}") from e

    def update_memory_embedding(self, mem_id: str, embedding: np.ndarray) -> None:
        """Backfill / refresh the stored embedding for a memory row."""
        try:
            with self.connection() as conn:
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
            with self.connection() as conn:
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

    def update_memory_heat(self, mem_id: str, heat_tier: int) -> None:
        """Update heat tier (for sharding/cache purging routines)."""
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE memories SET heat_tier = ? WHERE id = ?",
                    (heat_tier, mem_id)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"heat update failed: {e}") from e

    def update_memory_centroid(self, mem_id: str, centroid_id: str) -> None:
        """Assign specific centroid to memory for vector partitioned retrieval."""
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE memories SET centroid_id = ? WHERE id = ?",
                    (centroid_id, mem_id)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"centroid update failed: {e}") from e

    def archive_memory(self, mem_id: str) -> None:
        """Mark memory as archived (soft-delete / consolidated)."""
        hlc, node, seq = GLOBAL_HLC.next_version()
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE memories SET archived = 1, version_hlc = ?, version_node = ?, version_seq = ? WHERE id = ?", 
                    (hlc, node, seq, mem_id)
                )
                self._log_sync_operation(conn, "memories", mem_id, "UPDATE", {}, (hlc, node, seq))
        except sqlite3.Error as e:
            raise DatabaseError(f"archive_memory failed: {e}") from e

    def unarchive_memory(self, mem_id: str) -> None:
        """Mark an archived memory as active again (rejuvenation)."""
        hlc, node, seq = GLOBAL_HLC.next_version()
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE memories SET archived = 0, version_hlc = ?, version_node = ?, version_seq = ? WHERE id = ?", 
                    (hlc, node, seq, mem_id)
                )
                self._log_sync_operation(conn, "memories", mem_id, "UPDATE", {}, (hlc, node, seq))
        except sqlite3.Error as e:
            raise DatabaseError(f"unarchive_memory failed: {e}") from e

    def _log_sync_operation(self, conn: sqlite3.Connection, table: str, row_id: str, op: str, row_data: dict, hlc_tuple: tuple) -> None:
        """Centralized write log for Delta Sync using Oplog and Row Hashes."""
        import hashlib
        import json
        
        h, n, s = hlc_tuple
        conn.execute(
            "INSERT INTO oplog (version_hlc, version_node, version_seq, table_name, row_id, operation) VALUES (?, ?, ?, ?, ?, ?)",
            (h, n, s, table, row_id, op)
        )
        # Fetch the canonical row to ensure hash matches schema identically across devices
        full_row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        if full_row:
            row_dict = dict(full_row)
            
            # Sanitize bytes for JSON serialization (e.g., embeddings)
            for k, v in row_dict.items():
                if isinstance(v, bytes):
                    row_dict[k] = v.hex()
                    
            dict_str = json.dumps(row_dict, sort_keys=True)
            hash_val = hashlib.sha256(dict_str.encode("utf-8")).hexdigest()
            
            conn.execute(
                "INSERT OR REPLACE INTO row_hash (table_name, row_id, hash_val, version_hlc) VALUES (?, ?, ?, ?)",
                (table, row_id, hash_val, h)
            )
        elif op == "DELETE":
            # Physically deleted items drop their bucket hash
            conn.execute("DELETE FROM row_hash WHERE table_name = ? AND row_id = ?", (table, row_id))

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
            with self.connection() as conn:
                return [_to_dict(r) for r in conn.execute(sql, (f"%{keyword}%",)).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"search_memory failed: {e}") from e

    def stream_all_memories(self):
        """Yield memory rows one by one to avoid large memory overhead."""
        try:
            with self.connection() as conn:
                cursor = conn.execute("SELECT * FROM memories ORDER BY created_at DESC")
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                    for r in rows:
                        yield _to_dict(r)
        except sqlite3.Error as e:
            raise DatabaseError(f"stream_all_memories failed: {e}") from e

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Fetch every memory row (Legacy, use stream_all_memories for large DBs)."""
        return list(self.stream_all_memories())

    def get_memory_by_id(self, mem_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific memory by ID."""
        try:
            with self.connection() as conn:
                row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
                return _to_dict(row) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"get_memory_by_id failed: {e}") from e

    def get_memories_without_embedding(self) -> List[Dict[str, Any]]:
        """Return rows that lack an embedding (for async backfill jobs)."""
        try:
            with self.connection() as conn:
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
            with self.connection() as conn:
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

        find_previous = """
            SELECT id FROM facts
            WHERE subject = ? AND predicate = ? AND is_active = 1
        """
        reconcile = """
            UPDATE facts
            SET is_active = 0,
                version_hlc = ?,
                version_node = ?,
                version_seq = ?
            WHERE id = ?
        """
        insert = """
            INSERT INTO facts
                (id, subject, predicate, object, confidence, importance,
                 embedding, created_at, is_active,
                 version_hlc, version_node, version_seq)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
        """
        try:
            with self.connection() as conn:
                previous = conn.execute(
                    find_previous,
                    (subject.strip(), predicate.strip()),
                ).fetchall()
                for row in previous:
                    hlc, node, seq = GLOBAL_HLC.next_version()
                    conn.execute(reconcile, (hlc, node, seq, row["id"]))
                    self._log_sync_operation(
                        conn, "facts", row["id"], "UPDATE", {}, (hlc, node, seq)
                    )

                hlc, node, seq = GLOBAL_HLC.next_version()
                conn.execute(insert, (
                    fact_id,
                    subject.strip(), predicate.strip(), obj.strip(),
                    float(confidence), float(importance),
                    blob, created_at,
                    hlc, node, seq,
                ))
                self._log_sync_operation(
                    conn, "facts", fact_id, "INSERT", {}, (hlc, node, seq)
                )
            return fact_id
        except sqlite3.Error as e:
            logger.error(f"insert_fact failed: {e}")
            raise DatabaseError(f"Fact insertion failed: {e}") from e

    def update_fact_embedding(self, fact_id: str, embedding: np.ndarray) -> None:
        """Backfill / refresh the stored embedding for a fact row."""
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE facts SET embedding = ? WHERE id = ?",
                    (_encode_blob(embedding), fact_id),
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"Fact embedding update failed: {e}") from e

    def touch_fact(self, fact_id: str) -> None:
        """Increment access_count and refresh last_accessed_at for a fact."""
        try:
            with self.connection() as conn:
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
            with self.connection() as conn:
                return [_to_dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"search_facts failed: {e}") from e

    def stream_all_active_facts(self):
        """Yield active facts for large-scale hydration."""
        sql = "SELECT * FROM facts WHERE is_active = 1"
        try:
            with self.connection() as conn:
                cursor = conn.execute(sql)
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                    for r in rows:
                        yield _to_dict(r)
        except sqlite3.Error as e:
            raise DatabaseError(f"stream_all_active_facts failed: {e}") from e

    def get_all_active_facts(self) -> List[Dict[str, Any]]:
        """Fetch all active facts (Legacy, use stream_all_active_facts)."""
        return list(self.stream_all_active_facts())

    def get_facts_without_embedding(self) -> List[Dict[str, Any]]:
        """Return active facts that lack an embedding."""
        try:
            with self.connection() as conn:
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
            with self.connection() as conn:
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
            with self.connection() as conn:
                return [_to_dict(r) for r in conn.execute(sql, active_fact_ids).fetchall()]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_fact_conflicts failed: {e}") from e

    def get_all_subjects(self) -> List[str]:
        """Return a unique list of subjects from all active facts."""
        try:
            with self.connection() as conn:
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
            with self.connection() as conn:
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
            with self.connection() as conn:
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
            with self.connection() as conn:
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
            with self.connection() as conn:
                conn.execute(sql, (
                    decision_id, action, reason,
                    json.dumps(used_fact_ids or []),
                    _utcnow(),
                ))
            return decision_id
        except sqlite3.Error as e:
            raise DatabaseError(f"log_decision failed: {e}") from e

    # ------------------------------------------------------------------
    # Background Job Persistence (Storage side)
    # ------------------------------------------------------------------

    def insert_background_job(self, job_type: str, status: str) -> str:
        """Create a persistent record of a background job."""
        job_id = str(uuid.uuid4())[:8]
        now = _utcnow()
        try:
            with self.connection() as conn:
                conn.execute(
                    "INSERT INTO background_jobs (id, job_type, status, created_at) VALUES (?, ?, ?, ?)",
                    (job_id, job_type, status, now)
                )
            return job_id
        except sqlite3.Error as e:
            raise DatabaseError(f"insert_background_job failed: {e}") from e

    def complete_background_job(self, job_id: str, result: dict) -> None:
        """Mark a job as completed and store its result JSON."""
        now = _utcnow()
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE background_jobs SET status = 'completed', completed_at = ?, result = ? WHERE id = ?",
                    (now, json.dumps(result), job_id)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"complete_background_job failed: {e}") from e

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Replica Checkpoint (Delta Sync)
    # ------------------------------------------------------------------

    def prune_oplog_entries(self, boundary_hlc: int, batch_size: int = 1000, dry_run: bool = False) -> int:
        """
        Delete oplog entries older than boundary_hlc (strictly less than).
        """
        try:
            with self.connection() as conn:
                if dry_run:
                    cnt = conn.execute(
                        "SELECT COUNT(*) as c FROM oplog WHERE version_hlc < ?",
                        (boundary_hlc,)
                    ).fetchone()["c"]
                    return cnt
                else:
                    if batch_size > 0:
                        cur = conn.execute(
                            """
                            DELETE FROM oplog WHERE rowid IN (
                                SELECT rowid FROM oplog WHERE version_hlc < ? LIMIT ?
                            )
                            """,
                            (boundary_hlc, batch_size)
                        )
                    else:
                        cur = conn.execute("DELETE FROM oplog WHERE version_hlc < ?", (boundary_hlc,))
                    return cur.rowcount
        except sqlite3.Error as e:
            raise DatabaseError(f"prune_oplog_entries failed: {e}") from e

    def get_oplog_range(self) -> Dict[str, Optional[int]]:
        """Return the min and max version_hlc currently available in the oplog."""
        with self.connection() as conn:
            row = conn.execute("SELECT MIN(version_hlc) as min_hlc, MAX(version_hlc) as max_hlc FROM oplog").fetchone()
            return {"min": row["min_hlc"], "max": row["max_hlc"]}

    def get_latest_version_hlc(self) -> int:
        """Get the absolute latest HLC version seen by this node across all tables."""
        with self.connection() as conn:
            # Check semantic tables and sync metadata for the latest version.
            m_max = conn.execute("SELECT MAX(version_hlc) FROM memories").fetchone()[0] or 0
            fact_max = conn.execute("SELECT MAX(version_hlc) FROM facts").fetchone()[0] or 0
            kg_max = conn.execute("SELECT MAX(version_hlc) FROM kg_fact").fetchone()[0] or 0
            hash_max = conn.execute("SELECT MAX(version_hlc) FROM row_hash").fetchone()[0] or 0
            return max(m_max, fact_max, kg_max, hash_max)

    def upsert_replica_checkpoint(
        self, replica_id: str, hlc: int, node: str, seq: int, note: Optional[str] = None
    ) -> None:
        """
        Record or blindly update a synchronized replica's watermark.
        """
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO replica_checkpoint (replica_id, last_applied_hlc, last_applied_node, last_applied_seq, updated_ts, note)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(replica_id) DO UPDATE SET
                        last_applied_hlc=excluded.last_applied_hlc,
                        last_applied_node=excluded.last_applied_node,
                        last_applied_seq=excluded.last_applied_seq,
                        updated_ts=excluded.updated_ts,
                        note=excluded.note
                    WHERE excluded.last_applied_hlc >= last_applied_hlc
                    """,
                    (replica_id, hlc, node, seq, _utcnow(), note)
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"upsert_replica_checkpoint failed: {e}") from e

    def get_replica_checkpoint(self, replica_id: str) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM replica_checkpoint WHERE replica_id = ?", (replica_id,)).fetchone()
            return dict(row) if row else None

    def list_replica_checkpoints(self) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM replica_checkpoint ORDER BY updated_ts DESC").fetchall()
            return [dict(r) for r in rows]

    def get_min_acknowledged_hlc(self) -> Optional[int]:
        """
        Calculate the lowest synced cursor threshold across all recorded replica branches locally.
        Used primarily by Garbage Collection daemons to identify safe prune boundaries for oplogs.
        """
        with self.connection() as conn:
            row = conn.execute("SELECT MIN(last_applied_hlc) as min_hlc FROM replica_checkpoint").fetchone()
            if row and row["min_hlc"] is not None:
                return int(row["min_hlc"])
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate statistics for `memk doctor` with production metrics."""
        try:
            with self.connection() as conn:
                def scalar(q: str) -> int:
                    return conn.execute(q).fetchone()[0]

                # Get schema version
                schema_version = 0
                try:
                    schema_version = conn.execute(
                        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                    ).fetchone()[0]
                except:
                    pass
                
                # Get database info
                db_info = get_database_info(conn)

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
                    "schema_version":       schema_version,
                    "database_size_mb":     db_info.get("size_mb", 0),
                    "journal_mode":         db_info.get("journal_mode", "unknown"),
                    "wal_size_mb":          db_info.get("wal_size_mb", 0) if db_info.get("journal_mode") == "wal" else 0,
                }
        except sqlite3.Error as e:
            raise DatabaseError(f"get_stats failed: {e}") from e
    def get_delta_since(self, since_hlc: int) -> List[Dict[str, Any]]:
        """
        Fetch all changes recorded in the oplog after since_hlc.
        Returns a list of dicts: {table, row_id, payload}.
        """
        try:
            with self.connection() as conn:
                # Get unique modified rows from oplog since HLC
                rows = conn.execute(
                    """
                    SELECT DISTINCT table_name, row_id 
                    FROM oplog 
                    WHERE version_hlc > ? 
                    ORDER BY version_hlc ASC
                    """,
                    (since_hlc,)
                ).fetchall()
                
                results = []
                for r in rows:
                    tbl = r["table_name"]
                    row_id = r["row_id"]
                    
                    # Fetch current state of the row
                    try:
                        data = conn.execute(f"SELECT * FROM {tbl} WHERE id = ?", (row_id,)).fetchone()
                        if data:
                            results.append({
                                "table": tbl,
                                "row_id": row_id,
                                "payload": _to_dict(data)
                            })
                    except Exception as e:
                        logger.warning(f"get_delta_since: skipped {tbl}:{row_id}: {e}")
                        continue
                        
                return results
        except sqlite3.Error as e:
            raise DatabaseError(f"get_delta_since failed: {e}") from e


# ---------------------------------------------------------------------------
# Module-level helpers (also imported by retriever / tests)
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    """Current UTC time as ISO 8601 string with microseconds."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


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

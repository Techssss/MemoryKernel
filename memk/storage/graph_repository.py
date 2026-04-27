"""
memk.storage.graph_repository
==============================
Repository / DAO layer for the knowledge graph sidecar tables.

Follows the same patterns as MemoryDB in db.py:
  - Uses ``_get_connection()`` context manager for each operation.
  - Wraps sqlite3 errors in ``DatabaseError``.
  - Returns plain dicts or dataclass instances.
  - Logging via standard ``logging`` module.

This module does NOT contain extraction logic or retrieval logic.
It is purely CRUD + query for entity, mention, edge, kg_fact tables.
"""

import sqlite3
import uuid
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from memk.storage.graph_models import (
    EntityRecord,
    MentionRecord,
    EdgeRecord,
    KGFactRecord,
    normalize_entity_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions (reuse from db.py)
# ---------------------------------------------------------------------------

class DatabaseError(Exception):
    """Raised for any unrecoverable storage-layer error."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    """Current UTC time as ISO 8601 string with microseconds."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _row_to_entity(row: sqlite3.Row) -> EntityRecord:
    return EntityRecord(
        id=row["id"],
        workspace_id=row["workspace_id"],
        canonical_text=row["canonical_text"],
        normalized_text=row["normalized_text"],
        entity_type=row["entity_type"],
        first_seen_ts=row["first_seen_ts"],
        last_seen_ts=row["last_seen_ts"],
        confidence=float(row["confidence"]),
    )


def _row_to_mention(row: sqlite3.Row) -> MentionRecord:
    return MentionRecord(
        memory_id=row["memory_id"],
        entity_id=row["entity_id"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        role_hint=row["role_hint"],
        weight=float(row["weight"]),
    )


def _row_to_edge(row: sqlite3.Row) -> EdgeRecord:
    return EdgeRecord(
        id=row["id"],
        workspace_id=row["workspace_id"],
        src_entity_id=row["src_entity_id"],
        rel_type=row["rel_type"],
        dst_entity_id=row["dst_entity_id"],
        weight=float(row["weight"]),
        confidence=float(row["confidence"]),
        provenance_memory_id=row["provenance_memory_id"],
        archived=int(row["archived"]),
        created_at=row["created_at"],
    )


def _row_to_kg_fact(row: sqlite3.Row) -> KGFactRecord:
    return KGFactRecord(
        id=row["id"],
        workspace_id=row["workspace_id"],
        canonical_text=row["canonical_text"],
        summary_json=row["summary_json"],
        confidence=float(row["confidence"]),
        created_ts=row["created_ts"],
    )


# ---------------------------------------------------------------------------
# GraphRepository
# ---------------------------------------------------------------------------

class GraphRepository:
    """
    Data access layer for the knowledge graph sidecar tables.

    Usage
    -----
    >>> from memk.storage.db import MemoryDB
    >>> db = MemoryDB("my.db")
    >>> db.init_db()
    >>> repo = GraphRepository(db.db_path)
    >>> eid = repo.upsert_entity("ws1", "FastAPI", entity_type="TECH")
    >>> repo.add_mention("mem-abc", eid, role_hint="subject")
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Return a configured connection (matches MemoryDB pattern)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Graph DB connect failed [{self.db_path}]: {e}")
            raise DatabaseError(f"Connection failed: {e}") from e

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """
        Yield a configured connection and close it after transaction handling.

        The sqlite3 native context manager does not close connections, so this
        wrapper prevents lingering file handles on Windows.
        """
        conn = self._get_connection()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        workspace_id: str,
        canonical_text: str,
        *,
        entity_type: Optional[str] = None,
        confidence: float = 0.5,
    ) -> int:
        """
        Insert or update an entity. Returns the entity id (INTEGER PK).

        Dedup logic: if (workspace_id, normalized_text, entity_type) already
        exists, update last_seen_ts and bump confidence to max(old, new).
        Otherwise insert a new row.

        Parameters
        ----------
        workspace_id   : Workspace scope.
        canonical_text : Human-readable form (e.g. "Google LLC").
        entity_type    : Optional tag ("ORG", "PERSON", "TECH", ...).
        confidence     : Extraction confidence [0, 1].

        Returns
        -------
        int : The entity row id (existing or newly created).
        """
        if not canonical_text or not canonical_text.strip():
            raise ValueError("Entity canonical_text cannot be empty.")

        normalized = normalize_entity_text(canonical_text)
        now = _utcnow()

        try:
            with self.connection() as conn:
                # Check for existing entity
                row = conn.execute(
                    """
                    SELECT id, confidence FROM entity
                    WHERE workspace_id = ? AND normalized_text = ? AND entity_type IS ?
                    """,
                    (workspace_id, normalized, entity_type),
                ).fetchone()

                if row is not None:
                    # Update: refresh timestamp, keep max confidence
                    new_conf = max(float(row["confidence"]), float(confidence))
                    conn.execute(
                        """
                        UPDATE entity
                        SET last_seen_ts = ?, confidence = ?
                        WHERE id = ?
                        """,
                        (now, new_conf, row["id"]),
                    )
                    logger.debug(
                        f"Entity updated: id={row['id']} '{normalized}' "
                        f"conf={new_conf:.2f}"
                    )
                    return int(row["id"])
                else:
                    # Insert new
                    conn.execute(
                        """
                        INSERT INTO entity
                            (workspace_id, canonical_text, normalized_text,
                             entity_type, first_seen_ts, last_seen_ts, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (workspace_id, canonical_text.strip(), normalized,
                         entity_type, now, now, float(confidence)),
                    )
                    entity_id = conn.execute(
                        "SELECT last_insert_rowid()"
                    ).fetchone()[0]
                    logger.debug(
                        f"Entity created: id={entity_id} '{normalized}' "
                        f"type={entity_type}"
                    )
                    return int(entity_id)

        except sqlite3.Error as e:
            logger.error(f"upsert_entity failed: {e}")
            raise DatabaseError(f"upsert_entity failed: {e}") from e

    def get_entity(self, entity_id: int) -> Optional[EntityRecord]:
        """Get a single entity by id."""
        try:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM entity WHERE id = ?", (entity_id,)
                ).fetchone()
                return _row_to_entity(row) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"get_entity failed: {e}") from e

    def find_entity(
        self,
        workspace_id: str,
        text: str,
        entity_type: Optional[str] = None,
    ) -> Optional[EntityRecord]:
        """Find entity by normalized text lookup."""
        normalized = normalize_entity_text(text)
        try:
            with self.connection() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM entity
                    WHERE workspace_id = ? AND normalized_text = ?
                      AND entity_type IS ?
                    """,
                    (workspace_id, normalized, entity_type),
                ).fetchone()
                return _row_to_entity(row) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"find_entity failed: {e}") from e

    def get_all_entities(self, workspace_id: str) -> List[EntityRecord]:
        """Get all entities for a workspace."""
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM entity WHERE workspace_id = ? ORDER BY id",
                    (workspace_id,),
                ).fetchall()
                return [_row_to_entity(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_all_entities failed: {e}") from e

    # ------------------------------------------------------------------
    # Mention CRUD
    # ------------------------------------------------------------------

    def add_mention(
        self,
        memory_id: str,
        entity_id: int,
        *,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        role_hint: Optional[str] = None,
        weight: float = 1.0,
    ) -> None:
        """
        Record that a memory mentions an entity.

        Idempotent: INSERT OR IGNORE on the composite PK
        (memory_id, entity_id, role_hint).

        Parameters
        ----------
        memory_id  : FK to memories.id.
        entity_id  : FK to entity.id.
        start_char : Optional start offset of entity span in text.
        end_char   : Optional end offset.
        role_hint  : "subject", "object", "context", or None.
        weight     : Mention importance weight [0, 1].
        """
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO mention
                        (memory_id, entity_id, start_char, end_char,
                         role_hint, weight)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, entity_id, start_char, end_char,
                     role_hint, float(weight)),
                )
            logger.debug(
                f"Mention added: mem={memory_id[:8]}.. ent={entity_id} "
                f"role={role_hint}"
            )
        except sqlite3.Error as e:
            logger.error(f"add_mention failed: {e}")
            raise DatabaseError(f"add_mention failed: {e}") from e

    def get_entities_for_memory(self, memory_id: str) -> List[EntityRecord]:
        """
        Get all entities mentioned in a specific memory.

        Returns entity records joined through the mention table.
        """
        sql = """
            SELECT e.* FROM entity e
            INNER JOIN mention m ON m.entity_id = e.id
            WHERE m.memory_id = ?
            ORDER BY m.weight DESC, e.confidence DESC
        """
        try:
            with self.connection() as conn:
                rows = conn.execute(sql, (memory_id,)).fetchall()
                return [_row_to_entity(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_entities_for_memory failed: {e}") from e

    def get_mentions_for_memory(self, memory_id: str) -> List[MentionRecord]:
        """Get all mention records for a specific memory."""
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM mention WHERE memory_id = ?", (memory_id,)
                ).fetchall()
                return [_row_to_mention(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_mentions_for_memory failed: {e}") from e

    def get_memories_for_entity(self, entity_id: int) -> List[str]:
        """Get all memory IDs that mention a specific entity."""
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    "SELECT memory_id FROM mention WHERE entity_id = ?",
                    (entity_id,),
                ).fetchall()
                return [row["memory_id"] for row in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_memories_for_entity failed: {e}") from e

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def add_edge(
        self,
        workspace_id: str,
        src_entity_id: int,
        rel_type: str,
        dst_entity_id: int,
        *,
        weight: float = 1.0,
        confidence: float = 0.5,
        provenance_memory_id: str,
    ) -> int:
        """
        Insert a directional edge between two entities.

        Parameters
        ----------
        workspace_id         : Workspace scope.
        src_entity_id        : Source entity (FK to entity.id).
        rel_type             : Relationship label ("manages", "uses", ...).
        dst_entity_id        : Destination entity (FK to entity.id).
        weight               : Edge strength [0, 1].
        confidence           : Extraction confidence [0, 1].
        provenance_memory_id : The memory this edge was extracted from.

        Returns
        -------
        int : The edge row id.
        """
        if not rel_type or not rel_type.strip():
            raise ValueError("Edge rel_type cannot be empty.")
        if not provenance_memory_id:
            raise ValueError("Edge must have a provenance_memory_id.")

        now = _utcnow()
        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO edge
                        (workspace_id, src_entity_id, rel_type, dst_entity_id,
                         weight, confidence, provenance_memory_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (workspace_id, src_entity_id, rel_type.strip(),
                     dst_entity_id, float(weight), float(confidence),
                     provenance_memory_id, now),
                )
                edge_id = conn.execute(
                    "SELECT last_insert_rowid()"
                ).fetchone()[0]

            logger.debug(
                f"Edge created: {src_entity_id} --[{rel_type}]--> "
                f"{dst_entity_id} (prov={provenance_memory_id[:8]}..)"
            )
            return int(edge_id)

        except sqlite3.Error as e:
            logger.error(f"add_edge failed: {e}")
            raise DatabaseError(f"add_edge failed: {e}") from e

    def get_edges_from_entity(
        self,
        workspace_id: str,
        entity_id: int,
        *,
        include_archived: bool = False,
    ) -> List[EdgeRecord]:
        """Get outbound edges from a specific entity."""
        sql = """
            SELECT * FROM edge
            WHERE workspace_id = ? AND src_entity_id = ?
        """
        params: list = [workspace_id, entity_id]
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY weight DESC"

        try:
            with self.connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [_row_to_edge(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_edges_from_entity failed: {e}") from e

    def get_edges_to_entity(
        self,
        workspace_id: str,
        entity_id: int,
        *,
        include_archived: bool = False,
    ) -> List[EdgeRecord]:
        """Get inbound edges to a specific entity."""
        sql = """
            SELECT * FROM edge
            WHERE workspace_id = ? AND dst_entity_id = ?
        """
        params: list = [workspace_id, entity_id]
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY weight DESC"

        try:
            with self.connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [_row_to_edge(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_edges_to_entity failed: {e}") from e

    def get_edges_for_workspace(
        self,
        workspace_id: str,
        *,
        include_archived: bool = False,
    ) -> List[EdgeRecord]:
        """Get all edges for a workspace (for hydrating the RAM graph)."""
        sql = "SELECT * FROM edge WHERE workspace_id = ?"
        params: list = [workspace_id]
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY id"

        try:
            with self.connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [_row_to_edge(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_edges_for_workspace failed: {e}") from e

    def archive_edge(self, edge_id: int) -> None:
        """Soft-delete an edge by setting archived=1."""
        try:
            with self.connection() as conn:
                conn.execute(
                    "UPDATE edge SET archived = 1 WHERE id = ?", (edge_id,)
                )
            logger.debug(f"Edge archived: id={edge_id}")
        except sqlite3.Error as e:
            raise DatabaseError(f"archive_edge failed: {e}") from e

    def get_edges_for_memory(self, memory_id: str) -> List[EdgeRecord]:
        """Get all edges that were extracted from a specific memory."""
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM edge
                    WHERE provenance_memory_id = ?
                    ORDER BY id
                    """,
                    (memory_id,),
                ).fetchall()
                return [_row_to_edge(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_edges_for_memory failed: {e}") from e

    # ------------------------------------------------------------------
    # KG Fact CRUD
    # ------------------------------------------------------------------

    def create_fact(
        self,
        workspace_id: str,
        canonical_text: str,
        *,
        summary_json: Optional[str] = None,
        confidence: float = 0.5,
    ) -> str:
        """
        Insert a consolidated knowledge fact. Returns its UUID.

        Parameters
        ----------
        workspace_id   : Workspace scope.
        canonical_text : Human-readable summary text.
        summary_json   : Optional JSON blob with structured details.
        confidence     : Aggregated confidence [0, 1].

        Returns
        -------
        str : The kg_fact row UUID.
        """
        if not canonical_text or not canonical_text.strip():
            raise ValueError("KG fact canonical_text cannot be empty.")

        fact_id = str(uuid.uuid4())
        now = _utcnow()

        try:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO kg_fact
                        (id, workspace_id, canonical_text, summary_json,
                         confidence, created_ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (fact_id, workspace_id, canonical_text.strip(),
                     summary_json, float(confidence), now),
                )
            logger.debug(f"KG fact created: id={fact_id[:8]}.. '{canonical_text[:40]}'")
            return fact_id

        except sqlite3.Error as e:
            logger.error(f"create_fact failed: {e}")
            raise DatabaseError(f"create_fact failed: {e}") from e

    def get_facts_for_workspace(
        self,
        workspace_id: str,
        limit: int = 100,
    ) -> List[KGFactRecord]:
        """Get consolidated facts for a workspace, newest first."""
        try:
            with self.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM kg_fact
                    WHERE workspace_id = ?
                    ORDER BY created_ts DESC
                    LIMIT ?
                    """,
                    (workspace_id, limit),
                ).fetchall()
                return [_row_to_kg_fact(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"get_facts_for_workspace failed: {e}") from e

    # ------------------------------------------------------------------
    # Graph stats (diagnostics)
    # ------------------------------------------------------------------

    def get_graph_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get summary counts for the graph sidecar tables."""
        try:
            with self.connection() as conn:
                def scalar(sql):
                    return conn.execute(sql, (workspace_id,)).fetchone()[0]

                return {
                    "entity_count": scalar(
                        "SELECT COUNT(*) FROM entity WHERE workspace_id = ?"
                    ),
                    "mention_count": scalar(
                        "SELECT COUNT(*) FROM mention m "
                        "INNER JOIN entity e ON m.entity_id = e.id "
                        "WHERE e.workspace_id = ?"
                    ),
                    "edge_count": scalar(
                        "SELECT COUNT(*) FROM edge "
                        "WHERE workspace_id = ? AND archived = 0"
                    ),
                    "archived_edge_count": scalar(
                        "SELECT COUNT(*) FROM edge "
                        "WHERE workspace_id = ? AND archived = 1"
                    ),
                    "kg_fact_count": scalar(
                        "SELECT COUNT(*) FROM kg_fact WHERE workspace_id = ?"
                    ),
                }
        except sqlite3.Error as e:
            raise DatabaseError(f"get_graph_stats failed: {e}") from e

"""
memk.storage.migrations
=======================
Schema versioning and migration framework for SQLite.

Provides safe, forward-only migrations with version tracking.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Callable, Tuple
from dataclasses import dataclass

logger = logging.getLogger("memk.migrations")

# Current schema version
CURRENT_SCHEMA_VERSION = 13

@dataclass
class Migration:
    """A single database migration."""
    version: int
    description: str
    up: Callable[[sqlite3.Connection], None]


# ---------------------------------------------------------------------------
# Migration Definitions
# ---------------------------------------------------------------------------

def migrate_v0_to_v1(conn: sqlite3.Connection):
    """Initial schema with version tracking."""
    # Create schema_version table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT NOT NULL
        )
    """)
    
    # Note: Version 1 record will be inserted by apply_migration()
    # No need to insert here


def migrate_v1_to_v2(conn: sqlite3.Connection):
    """Add core tables and performance indexes."""
    # Create core tables first (if they don't exist)
    conn.execute("""
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
    """)
    
    conn.execute("""
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
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id           TEXT PRIMARY KEY,
            action       TEXT NOT NULL,
            reason       TEXT NOT NULL,
            used_fact_ids TEXT,
            created_at   TEXT NOT NULL
        )
    """)
    
    # Index for importance-based queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_importance 
        ON memories(importance DESC, created_at DESC)
    """)
    
    # Index for active facts
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_active 
        ON facts(is_active, importance DESC)
    """)
    
    # Index for decay score queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_decay 
        ON memories(decay_score DESC)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_decay 
        ON facts(decay_score DESC)
    """)


def migrate_v2_to_v3(conn: sqlite3.Connection):
    """Add background job tracking."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS background_jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            progress REAL DEFAULT 0.0,
            result TEXT,
            error TEXT
        )
    """)
    
    # Index for job queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status 
        ON background_jobs(status, created_at DESC)
    """)


def migrate_v3_to_v4(conn: sqlite3.Connection):
    """Add database metadata table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    # Store database identity
    import uuid
    db_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    conn.execute("""
        INSERT OR IGNORE INTO db_metadata (key, value, updated_at)
        VALUES ('db_id', ?, ?)
    """, (db_id, now))
    
    conn.execute("""
        INSERT OR IGNORE INTO db_metadata (key, value, updated_at)
        VALUES ('created_at', ?, ?)
    """, (now, now))

def migrate_v4_to_v5(conn: sqlite3.Connection):
    """Add knowledge graph sidecar tables (entity, mention, edge, kg_fact).

    These tables form a lightweight graph layer alongside the existing
    flat memories/facts storage. They enable multi-hop reasoning without
    replacing the current retrieval pipeline.

    Tables
    ------
    entity   : Canonical entity store. One row per unique real-world entity
               within a workspace. `normalized_text` is lowercase/stripped
               for fast dedup lookups.
    mention  : Links a memory row to the entities it references, with
               optional character-span and role metadata. Composite PK,
               WITHOUT ROWID for compact storage.
    edge     : Directional relationship between two entities, extracted
               from a specific memory (provenance_memory_id). Supports
               archiving without deletion.
    kg_fact  : Consolidated knowledge — summaries produced by the future
               consolidation pipeline. Separate from the existing `facts`
               table to avoid coupling.
    """
    # -- entity --------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id    TEXT    NOT NULL,
            canonical_text  TEXT    NOT NULL,
            normalized_text TEXT    NOT NULL,
            entity_type     TEXT,
            first_seen_ts   TEXT    NOT NULL,
            last_seen_ts    TEXT    NOT NULL,
            confidence      REAL    NOT NULL DEFAULT 0.5
        )
    """)
    # Fast dedup: (workspace, normalized_text, type) must be unique
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_norm
        ON entity(workspace_id, normalized_text, entity_type)
    """)

    # -- mention -------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mention (
            memory_id       TEXT    NOT NULL,
            entity_id       INTEGER NOT NULL,
            start_char      INTEGER,
            end_char         INTEGER,
            role_hint       TEXT,
            weight          REAL    NOT NULL DEFAULT 1.0,
            PRIMARY KEY (memory_id, entity_id, role_hint)
        ) WITHOUT ROWID
    """)

    # -- edge ----------------------------------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id         TEXT    NOT NULL,
            src_entity_id        INTEGER NOT NULL,
            rel_type             TEXT    NOT NULL,
            dst_entity_id        INTEGER NOT NULL,
            weight               REAL    NOT NULL DEFAULT 1.0,
            confidence           REAL    NOT NULL DEFAULT 0.5,
            provenance_memory_id TEXT    NOT NULL,
            archived             INTEGER NOT NULL DEFAULT 0,
            created_at           TEXT    NOT NULL
        )
    """)
    # Traversal from a source entity
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_edge_src
        ON edge(workspace_id, src_entity_id)
    """)
    # Reverse traversal (inbound edges)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_edge_dst
        ON edge(workspace_id, dst_entity_id)
    """)
    # Lookup edges originating from a specific memory
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_edge_provenance
        ON edge(provenance_memory_id)
    """)

    # -- kg_fact (consolidated knowledge) ------------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kg_fact (
            id              TEXT    PRIMARY KEY,
            workspace_id    TEXT    NOT NULL,
            canonical_text  TEXT    NOT NULL,
            summary_json    TEXT,
            confidence      REAL    NOT NULL DEFAULT 0.5,
            created_ts      TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_kg_fact_ws
        ON kg_fact(workspace_id, created_ts DESC)
    """)


def migrate_v5_to_v6(conn: sqlite3.Connection):
    """Add partitioning and sharding fields to memories."""
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN centroid_id TEXT")
    except sqlite3.OperationalError:
        pass
        
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN heat_tier INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_heat ON memories(heat_tier DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_centroid ON memories(centroid_id)")

def migrate_v6_to_v7(conn: sqlite3.Connection):
    """Add archived field to memories for consolidation."""
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN archived INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived)")

def migrate_v7_to_v8(conn: sqlite3.Connection):
    """Add multi-device sync version fields using HLC logic."""
    for table in ["memories", "kg_fact"]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN version_hlc INTEGER DEFAULT 0")
            conn.execute(f"ALTER TABLE {table} ADD COLUMN version_node TEXT")
            conn.execute(f"ALTER TABLE {table} ADD COLUMN version_seq INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
            
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_hlc ON memories(version_hlc DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kg_fact_hlc ON kg_fact(version_hlc DESC)")

def migrate_v8_to_v9(conn: sqlite3.Connection):
    """Add delta sync infrastructure (oplog, row_hash, merkle_bucket)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS oplog (
            version_hlc INTEGER NOT NULL,
            version_node TEXT NOT NULL,
            version_seq INTEGER NOT NULL,
            table_name TEXT NOT NULL,
            row_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            PRIMARY KEY (version_hlc, version_node, version_seq)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS row_hash (
            table_name TEXT NOT NULL,
            row_id TEXT NOT NULL,
            hash_val TEXT NOT NULL,
            version_hlc INTEGER,
            PRIMARY KEY (table_name, row_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS merkle_bucket (
            bucket_id INTEGER PRIMARY KEY,
            hash_val TEXT NOT NULL,
            updated_hlc INTEGER NOT NULL
        )
    """)

def migrate_v9_to_v10(conn: sqlite3.Connection):
    """Add replica checkpointing for tracking replica state."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS replica_checkpoint (
            replica_id TEXT PRIMARY KEY,
            last_applied_hlc INTEGER NOT NULL,
            last_applied_node TEXT NOT NULL,
            last_applied_seq INTEGER NOT NULL,
            updated_ts TEXT NOT NULL,
            note TEXT
        )
    """)

def migrate_v10_to_v11(conn: sqlite3.Connection):
    """Add conflict_record table for semantic conflict tracking.

    LWW (Last Writer Wins) silently overwrites data based on version_hlc.
    This table captures cases where the overwritten data carried distinct
    semantic meaning that a user or agent may want to review later.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conflict_record (
            conflict_id     TEXT    PRIMARY KEY,
            table_name      TEXT    NOT NULL,
            row_id          TEXT    NOT NULL,
            local_hlc       INTEGER NOT NULL,
            remote_hlc      INTEGER NOT NULL,
            local_snapshot  TEXT    NOT NULL,
            remote_snapshot TEXT    NOT NULL,
            detected_ts     TEXT    NOT NULL,
            status          TEXT    NOT NULL DEFAULT 'open',
            resolution      TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_conflict_status
        ON conflict_record(status, detected_ts DESC)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_conflict_row
        ON conflict_record(table_name, row_id)
    """)

def migrate_v11_to_v12(conn: sqlite3.Connection):
    """Add resolved_ts to conflict_record for resolution timestamp tracking."""
    try:
        conn.execute("ALTER TABLE conflict_record ADD COLUMN resolved_ts TEXT")
    except sqlite3.OperationalError:
        pass


def migrate_v12_to_v13(conn: sqlite3.Connection):
    """Add missing HLC sync fields to facts.

    New databases already get these columns from MemoryDB.init_db(), but
    databases created through the migration path before this version did not.
    Keeping this migration additive preserves upgrade safety.
    """
    for ddl in (
        "ALTER TABLE facts ADD COLUMN version_hlc INTEGER DEFAULT 0",
        "ALTER TABLE facts ADD COLUMN version_node TEXT",
        "ALTER TABLE facts ADD COLUMN version_seq INTEGER DEFAULT 0",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass

    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_hlc ON facts(version_hlc DESC)")


# ---------------------------------------------------------------------------
# Migration Registry
# ---------------------------------------------------------------------------

MIGRATIONS: List[Migration] = [
    Migration(1, "Initial schema with versioning", migrate_v0_to_v1),
    Migration(2, "Add core tables and performance indexes", migrate_v1_to_v2),
    Migration(3, "Add background job tracking", migrate_v2_to_v3),
    Migration(4, "Add database metadata", migrate_v3_to_v4),
    Migration(5, "Add knowledge graph sidecar tables", migrate_v4_to_v5),
    Migration(6, "Add centroid and heat tier to memories", migrate_v5_to_v6),
    Migration(7, "Add archived field to memories", migrate_v6_to_v7),
    Migration(8, "Add HLC sync fields to semantic tables", migrate_v7_to_v8),
    Migration(9, "Add oplog and merkle tables for sync", migrate_v8_to_v9),
    Migration(10, "Add replica checkpoint tracking", migrate_v9_to_v10),
    Migration(11, "Add conflict record table", migrate_v10_to_v11),
    Migration(12, "Add resolved_ts to conflict_record", migrate_v11_to_v12),
    Migration(13, "Add HLC sync fields to facts", migrate_v12_to_v13),
]


# ---------------------------------------------------------------------------
# Migration Engine
# ---------------------------------------------------------------------------

class MigrationEngine:
    """Manages schema migrations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_current_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version from database."""
        try:
            cursor = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            # schema_version table doesn't exist yet
            return 0
    
    def apply_migration(self, conn: sqlite3.Connection, migration: Migration):
        """Apply a single migration in a transaction."""
        logger.info(f"Applying migration v{migration.version}: {migration.description}")
        
        try:
            # Run migration
            migration.up(conn)
            
            # Record migration
            conn.execute("""
                INSERT INTO schema_version (version, applied_at, description)
                VALUES (?, ?, ?)
            """, (migration.version, datetime.now(timezone.utc).isoformat(), migration.description))
            
            conn.commit()
            logger.info(f"✓ Migration v{migration.version} applied successfully")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"✗ Migration v{migration.version} failed: {e}")
            raise
    
    def migrate(self) -> Tuple[int, int]:
        """
        Run all pending migrations.
        
        Returns:
            (old_version, new_version)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            current_version = self.get_current_version(conn)
            old_version = current_version
            
            if current_version >= CURRENT_SCHEMA_VERSION:
                logger.info(f"Database already at version {current_version}")
                return (old_version, current_version)
            
            logger.info(f"Migrating database from v{current_version} to v{CURRENT_SCHEMA_VERSION}")
            
            # Apply pending migrations
            for migration in MIGRATIONS:
                if migration.version > current_version:
                    self.apply_migration(conn, migration)
                    current_version = migration.version
            
            logger.info(f"✓ Database migrated successfully: v{old_version} → v{current_version}")
            return (old_version, current_version)
            
        finally:
            conn.close()
    
    def get_migration_history(self) -> List[dict]:
        """Get list of applied migrations."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            current_version = self.get_current_version(conn)
            if current_version == 0:
                return []
            
            cursor = conn.execute("""
                SELECT version, applied_at, description
                FROM schema_version
                ORDER BY version ASC
            """)
            
            return [dict(row) for row in cursor.fetchall()]
            
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def check_schema_version(db_path: str) -> dict:
    """
    Check database schema version and migration status.
    
    Returns:
        dict with current_version, target_version, needs_migration
    """
    engine = MigrationEngine(db_path)
    conn = sqlite3.connect(db_path)
    
    try:
        current_version = engine.get_current_version(conn)
        needs_migration = current_version < CURRENT_SCHEMA_VERSION
        
        return {
            "current_version": current_version,
            "target_version": CURRENT_SCHEMA_VERSION,
            "needs_migration": needs_migration,
            "migrations_pending": CURRENT_SCHEMA_VERSION - current_version
        }
    finally:
        conn.close()


def auto_migrate(db_path: str) -> bool:
    """
    Automatically run pending migrations if needed.
    
    Returns:
        True if migrations were applied, False if already up-to-date
    """
    status = check_schema_version(db_path)
    
    if not status["needs_migration"]:
        return False
    
    engine = MigrationEngine(db_path)
    old_ver, new_ver = engine.migrate()
    
    return old_ver != new_ver


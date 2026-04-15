"""
memk.storage.migrations
=======================
Schema versioning and migration framework for SQLite.

Provides safe, forward-only migrations with version tracking.
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Callable, Tuple
from dataclasses import dataclass

logger = logging.getLogger("memk.migrations")

# Current schema version
CURRENT_SCHEMA_VERSION = 4

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
    now = datetime.utcnow().isoformat()
    
    conn.execute("""
        INSERT OR IGNORE INTO db_metadata (key, value, updated_at)
        VALUES ('db_id', ?, ?)
    """, (db_id, now))
    
    conn.execute("""
        INSERT OR IGNORE INTO db_metadata (key, value, updated_at)
        VALUES ('created_at', ?, ?)
    """, (now, now))


# ---------------------------------------------------------------------------
# Migration Registry
# ---------------------------------------------------------------------------

MIGRATIONS: List[Migration] = [
    Migration(1, "Initial schema with versioning", migrate_v0_to_v1),
    Migration(2, "Add core tables and performance indexes", migrate_v1_to_v2),
    Migration(3, "Add background job tracking", migrate_v2_to_v3),
    Migration(4, "Add database metadata", migrate_v3_to_v4),
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
            """, (migration.version, datetime.utcnow().isoformat(), migration.description))
            
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


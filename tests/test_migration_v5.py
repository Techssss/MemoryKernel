"""
Test: V5 Knowledge Graph Schema Migration
==========================================
Verifies that the V5 migration (entity, mention, edge, kg_fact tables)
applies cleanly on both fresh databases and existing V4 databases,
and that basic CRUD operations work correctly.

Run: python -m pytest tests/test_migration_v5.py -v
  or: python tests/test_migration_v5.py
"""

import os
import sys
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone

# Ensure memk package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memk.storage.migrations import (
    auto_migrate,
    check_schema_version,
    MigrationEngine,
    CURRENT_SCHEMA_VERSION,
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _make_temp_db() -> str:
    """Create a temp DB file path in the project directory (not /tmp)."""
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_test_v5")
    os.makedirs(tmp_dir, exist_ok=True)
    return os.path.join(tmp_dir, f"test_{uuid.uuid4().hex[:8]}.db")


class TestMigrationV5:
    """Test suite for V5 schema migration."""

    def setup_method(self):
        self.db_path = _make_temp_db()

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        wal = self.db_path + "-wal"
        shm = self.db_path + "-shm"
        if os.path.exists(wal):
            os.remove(wal)
        if os.path.exists(shm):
            os.remove(shm)

    def _migrate(self):
        engine = MigrationEngine(self.db_path)
        return engine.migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------------------------------------------------------
    # 1. Migration applies cleanly on fresh DB
    # ---------------------------------------------------------------

    def test_fresh_db_migrates_to_v5(self):
        old_ver, new_ver = self._migrate()
        assert old_ver == 0
        assert new_ver == CURRENT_SCHEMA_VERSION
        assert new_ver >= 5

    def test_schema_version_reports_v5(self):
        self._migrate()
        status = check_schema_version(self.db_path)
        assert status["current_version"] >= 5
        assert status["needs_migration"] is False

    # ---------------------------------------------------------------
    # 2. Auto-migrate idempotency
    # ---------------------------------------------------------------

    def test_auto_migrate_idempotent(self):
        # First run
        result1 = auto_migrate(self.db_path)
        assert result1 is True

        # Second run — already at V5
        result2 = auto_migrate(self.db_path)
        assert result2 is False

    # ---------------------------------------------------------------
    # 3. Tables exist with correct columns
    # ---------------------------------------------------------------

    def test_entity_table_exists(self):
        self._migrate()
        conn = self._connect()
        cursor = conn.execute("PRAGMA table_info(entity)")
        cols = {row["name"] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "id", "workspace_id", "canonical_text", "normalized_text",
            "entity_type", "first_seen_ts", "last_seen_ts", "confidence",
        }
        assert expected == cols, f"Missing columns: {expected - cols}"

    def test_mention_table_exists(self):
        self._migrate()
        conn = self._connect()
        cursor = conn.execute("PRAGMA table_info(mention)")
        cols = {row["name"] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "memory_id", "entity_id", "start_char", "end_char",
            "role_hint", "weight",
        }
        assert expected == cols, f"Missing columns: {expected - cols}"

    def test_edge_table_exists(self):
        self._migrate()
        conn = self._connect()
        cursor = conn.execute("PRAGMA table_info(edge)")
        cols = {row["name"] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "id", "workspace_id", "src_entity_id", "rel_type",
            "dst_entity_id", "weight", "confidence",
            "provenance_memory_id", "archived", "created_at",
        }
        assert expected == cols, f"Missing columns: {expected - cols}"

    def test_kg_fact_table_exists(self):
        self._migrate()
        conn = self._connect()
        cursor = conn.execute("PRAGMA table_info(kg_fact)")
        cols = {row["name"] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "id", "workspace_id", "canonical_text", "summary_json",
            "confidence", "created_ts",
        }
        assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    # ---------------------------------------------------------------
    # 4. Indexes exist
    # ---------------------------------------------------------------

    def test_indexes_created(self):
        self._migrate()
        conn = self._connect()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = {row["name"] for row in cursor.fetchall()}
        conn.close()

        # V5 indexes
        v5_indexes = {
            "idx_entity_norm",
            "idx_edge_src",
            "idx_edge_dst",
            "idx_edge_provenance",
            "idx_kg_fact_ws",
        }
        missing = v5_indexes - indexes
        assert not missing, f"Missing indexes: {missing}"

    # ---------------------------------------------------------------
    # 5. CRUD smoke tests
    # ---------------------------------------------------------------

    def test_entity_insert_and_query(self):
        self._migrate()
        conn = self._connect()
        now = _utcnow()

        conn.execute("""
            INSERT INTO entity (workspace_id, canonical_text, normalized_text,
                                entity_type, first_seen_ts, last_seen_ts, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("ws1", "Google LLC", "google llc", "ORG", now, now, 0.9))
        conn.commit()

        row = conn.execute(
            "SELECT * FROM entity WHERE normalized_text = ?", ("google llc",)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["canonical_text"] == "Google LLC"
        assert row["entity_type"] == "ORG"
        assert row["confidence"] == 0.9

    def test_entity_unique_constraint(self):
        """Same (workspace, normalized_text, type) should conflict."""
        self._migrate()
        conn = self._connect()
        now = _utcnow()

        conn.execute("""
            INSERT INTO entity (workspace_id, canonical_text, normalized_text,
                                entity_type, first_seen_ts, last_seen_ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("ws1", "Python", "python", "TECH", now, now))
        conn.commit()

        try:
            conn.execute("""
                INSERT INTO entity (workspace_id, canonical_text, normalized_text,
                                    entity_type, first_seen_ts, last_seen_ts)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("ws1", "python", "python", "TECH", now, now))
            conn.commit()
            assert False, "Should have raised IntegrityError"
        except sqlite3.IntegrityError:
            pass  # Expected
        finally:
            conn.close()

    def test_mention_insert(self):
        self._migrate()
        conn = self._connect()
        now = _utcnow()

        # Insert entity first
        conn.execute("""
            INSERT INTO entity (workspace_id, canonical_text, normalized_text,
                                entity_type, first_seen_ts, last_seen_ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("ws1", "FastAPI", "fastapi", "TECH", now, now))
        entity_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        mem_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO mention (memory_id, entity_id, start_char, end_char,
                                 role_hint, weight)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (mem_id, entity_id, 10, 16, "subject", 1.0))
        conn.commit()

        row = conn.execute(
            "SELECT * FROM mention WHERE memory_id = ?", (mem_id,)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["entity_id"] == entity_id
        assert row["role_hint"] == "subject"

    def test_edge_insert_and_traversal(self):
        self._migrate()
        conn = self._connect()
        now = _utcnow()

        # Insert two entities
        conn.execute("""
            INSERT INTO entity (workspace_id, canonical_text, normalized_text,
                                entity_type, first_seen_ts, last_seen_ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("ws1", "Team Alpha", "team alpha", "TEAM", now, now))
        src_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute("""
            INSERT INTO entity (workspace_id, canonical_text, normalized_text,
                                entity_type, first_seen_ts, last_seen_ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("ws1", "Project X", "project x", "PROJECT", now, now))
        dst_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        mem_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO edge (workspace_id, src_entity_id, rel_type, dst_entity_id,
                              weight, confidence, provenance_memory_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, ("ws1", src_id, "manages", dst_id, 1.0, 0.85, mem_id, now))
        conn.commit()

        # Forward traversal: edges from Team Alpha
        rows = conn.execute(
            "SELECT * FROM edge WHERE workspace_id = ? AND src_entity_id = ?",
            ("ws1", src_id)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["rel_type"] == "manages"
        assert rows[0]["dst_entity_id"] == dst_id

        # Reverse traversal: edges into Project X
        rows = conn.execute(
            "SELECT * FROM edge WHERE workspace_id = ? AND dst_entity_id = ?",
            ("ws1", dst_id)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["src_entity_id"] == src_id

        conn.close()

    def test_kg_fact_insert(self):
        self._migrate()
        conn = self._connect()
        now = _utcnow()

        fact_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO kg_fact (id, workspace_id, canonical_text, summary_json,
                                 confidence, created_ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (fact_id, "ws1", "Team Alpha manages Project X",
              '{"subject":"Team Alpha","rel":"manages","object":"Project X"}',
              0.85, now))
        conn.commit()

        row = conn.execute(
            "SELECT * FROM kg_fact WHERE id = ?", (fact_id,)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["canonical_text"] == "Team Alpha manages Project X"
        assert row["confidence"] == 0.85

    # ---------------------------------------------------------------
    # 6. Existing tables untouched
    # ---------------------------------------------------------------

    def test_existing_tables_preserved(self):
        """V5 migration must NOT alter memories, facts, or decisions."""
        self._migrate()
        conn = self._connect()

        # Verify old tables still exist with original columns
        for table in ("memories", "facts", "decisions"):
            cursor = conn.execute(f"PRAGMA table_info({table})")
            cols = cursor.fetchall()
            assert len(cols) > 0, f"Table {table} missing after V5 migration"

        # Insert into old table — must still work
        mem_id = str(uuid.uuid4())
        now = _utcnow()
        conn.execute("""
            INSERT INTO memories (id, content, importance, confidence, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (mem_id, "test memory", 0.5, 1.0, now))
        conn.commit()

        row = conn.execute(
            "SELECT content FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        assert row["content"] == "test memory"

        conn.close()

    # ---------------------------------------------------------------
    # 7. Migration on pre-existing V4 database
    # ---------------------------------------------------------------

    def test_upgrade_from_v4(self):
        """Simulate a V4 DB and verify V5 applies on top."""
        # Create DB and migrate only to V4
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        engine = MigrationEngine(self.db_path)

        from memk.storage.migrations import MIGRATIONS
        for m in MIGRATIONS:
            if m.version <= 4:
                engine.apply_migration(conn, m)
        conn.close()

        # Verify at V4
        status = check_schema_version(self.db_path)
        assert status["current_version"] == 4

        # Now auto-migrate should bring to V5
        result = auto_migrate(self.db_path)
        assert result is True

        status = check_schema_version(self.db_path)
        assert status["current_version"] >= 5

        # Verify new tables
        conn = sqlite3.connect(self.db_path)
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert "entity" in tables
        assert "mention" in tables
        assert "edge" in tables
        assert "kg_fact" in tables


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_standalone():
    """Run all tests without pytest (for quick verification)."""
    test = TestMigrationV5()
    methods = [m for m in dir(test) if m.startswith("test_")]
    passed = 0
    failed = 0

    for method_name in sorted(methods):
        test.setup_method()
        try:
            getattr(test, method_name)()
            print(f"  PASS  {method_name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {method_name}: {e}")
            failed += 1
        finally:
            test.teardown_method()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    print("V5 Knowledge Graph Schema Migration Tests")
    print("=" * 50)
    run_standalone()

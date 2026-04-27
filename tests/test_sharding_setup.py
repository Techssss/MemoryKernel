import os
import uuid
import numpy as np
import pytest

from memk.storage.db import MemoryDB
from memk.storage.sharding_service import ShardingService

def test_migration_and_sharding_fields():
    test_db_path = f"test_shard_{uuid.uuid4().hex[:8]}.db"

    try:
        # 1. Init DB and apply migrations
        db = MemoryDB(test_db_path)
        db.init_db()

        # 2. Extract schema
        with db.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = [row["name"] for row in cursor.fetchall()]

            assert "centroid_id" in columns, "centroid_id field missing from memories after v6 migration"
            assert "heat_tier" in columns, "heat_tier field missing from memories after v6 migration"

            # 3. Test insert & default values
            mem_id = db.insert_memory("Test content", importance=0.5, confidence=1.0)

            row = conn.execute("SELECT centroid_id, heat_tier FROM memories WHERE id = ?", (mem_id,)).fetchone()
            assert row["centroid_id"] is None
            assert row["heat_tier"] == 0

            # 4. Test ShardingService logic
            sharding = ShardingService(db)

            # Verify heat update
            sharding.compute_and_update_heat(mem_id, access_count=10, importance=0.9)
            row = conn.execute("SELECT heat_tier FROM memories WHERE id = ?", (mem_id,)).fetchone()
            assert row["heat_tier"] == 2  # Hot memory!

            # Verify centroid update (with simulated cluster centroids)
            c0 = np.array([1.0, 0.0], dtype=np.float32)
            c1 = np.array([0.0, 1.0], dtype=np.float32)

            emb = np.array([0.1, 0.9], dtype=np.float32) # closer to c1
            assigned = sharding.assign_centroid(mem_id, emb, centroids=[c0, c1])

            assert assigned == "c_1"
            row = conn.execute("SELECT centroid_id FROM memories WHERE id = ?", (mem_id,)).fetchone()
            assert row["centroid_id"] == "c_1"

    finally:
        # Cleanup
        if os.path.exists(test_db_path):
            try:
                os.remove(test_db_path)
                os.remove(test_db_path + "-wal")
                os.remove(test_db_path + "-shm")
            except:
                pass

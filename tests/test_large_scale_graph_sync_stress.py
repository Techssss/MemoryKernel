import hashlib
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from memk.core.graph_index import GraphIndex
from memk.storage.db import MemoryDB
from memk.storage.graph_repository import GraphRepository
from memk.sync.merkle import MerkleService
from memk.sync.protocol import SyncProtocolNode


WORKSPACE_ID = "large_stress_ws"
MEMORY_COUNT = 512
FACT_COUNT = 160
ENTITY_COUNT = 128
MENTIONS_PER_MEMORY = 3
EDGES_PER_MEMORY = 2
UPDATE_COUNT = 64
ARCHIVE_COUNT = 48
UNARCHIVE_COUNT = 12
FACT_REPLACEMENTS = 40
EMBEDDING_DIM = 64

pytestmark = [
    pytest.mark.skipif(
        os.getenv("MEMK_RUN_LARGE_STRESS") != "1",
        reason="Set MEMK_RUN_LARGE_STRESS=1 to run the large graph/sync stress test.",
    ),
]


class Replica:
    def __init__(self, node_id: str, db_path: str):
        self.node_id = node_id
        self.db = MemoryDB(db_path)
        self.db.init_db()
        self.runtime = SimpleNamespace(db=self.db, workspace_id=WORKSPACE_ID)
        self.merkle = MerkleService(self.runtime)
        self.protocol = SyncProtocolNode(self.merkle)

    def refresh_merkle(self) -> None:
        self.merkle.cleanup_stale_row_hashes()
        self.merkle.rebuild_or_refresh_merkle_buckets(self.db.get_latest_version_hlc())

    def sync_from(self, remote: "Replica") -> int:
        remote.refresh_merkle()
        self.refresh_merkle()

        mismatched = self.protocol.diff_buckets(remote.protocol.get_bucket_hashes())
        deltas = remote.protocol.fetch_delta_for_buckets(mismatched)
        self.protocol.apply_remote_delta(deltas, remote_replica_id=remote.node_id)
        self.refresh_merkle()
        return len(deltas)


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(0.0, 1.0, size=EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _load_large_graph_dataset(replica: Replica) -> tuple[list[str], list[int]]:
    repo = GraphRepository(replica.db.db_path)

    entity_ids = [
        repo.upsert_entity(WORKSPACE_ID, f"Entity {i:03d}", entity_type="SERVICE")
        for i in range(ENTITY_COUNT)
    ]

    memory_ids = []
    with patch("memk.storage.db.GLOBAL_HLC.node_id", replica.node_id):
        for i in range(MEMORY_COUNT):
            primary = i % ENTITY_COUNT
            secondary = (i * 7 + 13) % ENTITY_COUNT
            tertiary = (i * 17 + 31) % ENTITY_COUNT

            mem_id = replica.db.insert_memory(
                (
                    f"project-{i % 32:02d} event-{i:04d} "
                    f"entity-{primary:03d} depends on entity-{secondary:03d} "
                    f"and writes shard-{i % 64:02d}"
                ),
                embedding=_vector(i),
                importance=0.3 + (i % 7) * 0.08,
                confidence=0.7 + (i % 5) * 0.05,
            )
            memory_ids.append(mem_id)

            repo.add_mention(mem_id, entity_ids[primary], role_hint="subject", weight=0.9)
            repo.add_mention(mem_id, entity_ids[secondary], role_hint="object", weight=0.7)
            repo.add_mention(mem_id, entity_ids[tertiary], role_hint="context", weight=0.5)
            repo.add_edge(
                WORKSPACE_ID,
                entity_ids[primary],
                "depends_on",
                entity_ids[secondary],
                weight=0.8,
                confidence=0.9,
                provenance_memory_id=mem_id,
            )
            repo.add_edge(
                WORKSPACE_ID,
                entity_ids[secondary],
                "writes_to",
                entity_ids[tertiary],
                weight=0.6,
                confidence=0.8,
                provenance_memory_id=mem_id,
            )

        for i in range(FACT_COUNT):
            replica.db.insert_fact(
                f"project-{i:03d}",
                "status",
                f"status-green-{i % 11}",
                embedding=_vector(10_000 + i),
                importance=0.4 + (i % 4) * 0.1,
                confidence=0.8,
            )

    replica.refresh_merkle()
    return memory_ids, entity_ids


def _memory_state(db: MemoryDB) -> list[tuple]:
    state = []
    for row in db.get_all_memories():
        embedding_hash = hashlib.sha256(row["embedding"] or b"").hexdigest()
        state.append(
            (
                row["id"],
                row["content"],
                int(row["archived"]),
                int(row["version_hlc"]),
                embedding_hash,
            )
        )
    return sorted(state)


def _fact_state(db: MemoryDB) -> list[tuple]:
    with db.connection() as conn:
        rows = conn.execute(
            """
            SELECT id, subject, predicate, object, is_active, version_hlc
            FROM facts
            ORDER BY id
            """
        ).fetchall()
    return [
        (
            row["id"],
            row["subject"],
            row["predicate"],
            row["object"],
            int(row["is_active"]),
            int(row["version_hlc"]),
        )
        for row in rows
    ]


def _row_hash_count(db: MemoryDB) -> int:
    with db.connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM row_hash").fetchone()[0]


def test_large_scale_graph_sync_and_reconciliation_stress():
    """
    Exercise storage, graph indexing, Merkle delta sync, BLOB propagation,
    fact reconciliation, checkpointing, and archive state with a large dataset.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        replica_a = Replica("node_alpha", os.path.join(tmpdir, "replica_a.db"))
        replica_b = Replica("node_beta", os.path.join(tmpdir, "replica_b.db"))

        memory_ids, entity_ids = _load_large_graph_dataset(replica_a)

        graph_index = GraphIndex(replica_a.db.db_path)
        graph_index.build_from_db(WORKSPACE_ID)
        graph_stats = graph_index.get_stats()

        assert graph_stats["num_entities"] == ENTITY_COUNT
        assert graph_stats["num_memories"] == MEMORY_COUNT
        assert graph_stats["m2e_mentions"] == MEMORY_COUNT * MENTIONS_PER_MEMORY
        assert graph_stats["e2m_mentions"] == MEMORY_COUNT * MENTIONS_PER_MEMORY
        assert graph_stats["e2e_edges"] == MEMORY_COUNT * EDGES_PER_MEMORY

        hub_entity_idx = graph_index.entity_id_map[entity_ids[0]]
        hub_mem_start = graph_index.e2m_indptr[hub_entity_idx]
        hub_mem_end = graph_index.e2m_indptr[hub_entity_idx + 1]
        assert hub_mem_end - hub_mem_start >= MEMORY_COUNT // ENTITY_COUNT

        initial_delta_count = replica_b.sync_from(replica_a)
        assert initial_delta_count == MEMORY_COUNT + FACT_COUNT
        assert _row_hash_count(replica_b.db) == MEMORY_COUNT + FACT_COUNT
        assert _memory_state(replica_a.db) == _memory_state(replica_b.db)
        assert _fact_state(replica_a.db) == _fact_state(replica_b.db)

        assert len(replica_b.db.search_memory("project-07")) >= MEMORY_COUNT // 32
        assert len(replica_b.db.search_facts(keyword="status-green")) == FACT_COUNT

        with patch("memk.storage.db.GLOBAL_HLC.node_id", replica_a.node_id):
            for i, mem_id in enumerate(memory_ids[:UPDATE_COUNT]):
                replica_a.db.update_memory_content(
                    mem_id,
                    f"revision-{i:03d} moved project-{i % 32:02d} to shard-hot-{i % 16:02d}",
                )

            for mem_id in memory_ids[UPDATE_COUNT:UPDATE_COUNT + ARCHIVE_COUNT]:
                replica_a.db.archive_memory(mem_id)

            archive_start = UPDATE_COUNT
            for mem_id in memory_ids[archive_start:archive_start + UNARCHIVE_COUNT]:
                replica_a.db.unarchive_memory(mem_id)

            for i in range(FACT_REPLACEMENTS):
                replica_a.db.insert_fact(
                    f"project-{i:03d}",
                    "status",
                    f"status-red-reconciled-{i:03d}",
                    embedding=_vector(20_000 + i),
                    importance=0.95,
                    confidence=0.99,
                )

        second_delta_count = replica_b.sync_from(replica_a)
        assert second_delta_count >= UPDATE_COUNT + ARCHIVE_COUNT + FACT_REPLACEMENTS
        assert _row_hash_count(replica_b.db) == MEMORY_COUNT + FACT_COUNT + FACT_REPLACEMENTS
        assert _memory_state(replica_a.db) == _memory_state(replica_b.db)
        assert _fact_state(replica_a.db) == _fact_state(replica_b.db)

        archived_b = {
            row["id"]: int(row["archived"])
            for row in replica_b.db.get_all_memories()
            if row["id"] in memory_ids[UPDATE_COUNT:UPDATE_COUNT + ARCHIVE_COUNT]
        }
        assert sum(archived_b.values()) == ARCHIVE_COUNT - UNARCHIVE_COUNT

        active_facts_b = replica_b.db.get_all_active_facts()
        assert len(active_facts_b) == FACT_COUNT
        assert len(replica_b.db.search_facts(keyword="status-red-reconciled")) == FACT_REPLACEMENTS

        checkpoint = replica_b.db.get_replica_checkpoint(replica_a.node_id)
        assert checkpoint is not None
        assert checkpoint["last_applied_hlc"] == replica_a.db.get_latest_version_hlc()

import time
import hashlib
import re
from typing import List

import numpy as np

from memk.storage.db import MemoryDB
from memk.retrieval.retriever import ScoredRetriever
from bench.dataset import SyntheticDataset
from bench.metrics import MetricsCollector


class TokenHashEmbedder:
    """Deterministic offline embedder for benchmark runs."""

    def __init__(self, dim: int = 128):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, 16, 4):
                raw = int.from_bytes(digest[offset:offset + 4], "little")
                idx = raw % self._dim
                vec[idx] += 1.0 if raw & 1 else -1.0

        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.embed(text) for text in texts]

class RetrievalStress:
    def __init__(self, db_path: str, collector: MetricsCollector):
        self.db = MemoryDB(db_path)
        self.db.init_db()
        self.collector = collector
        self.dataset = SyntheticDataset()
        self.retriever = ScoredRetriever(
            self.db,
            embedder=TokenHashEmbedder(),
            track_access=False,
        )

    def run_semantic_bench(self, query_count: int = 50):
        snap = self.collector.start_test(f"Retrieval_Semantic_{query_count}")
        queries = [f"What is {topic}?" for topic in self.dataset.topics]
        snap.extras["backfilled_embeddings"] = self.retriever.backfill_all_embeddings()
        
        start_time = time.perf_counter()
        for i in range(query_count):
            query = queries[i % len(queries)]
            q_start = time.perf_counter()
            try:
                results = self.retriever.retrieve(query, limit=10)
                snap.latency_ms.append((time.perf_counter() - q_start) * 1000)
            except Exception:
                snap.errors += 1
        
        self.collector.record_batch(snap, start_time, query_count)
        return snap.summarize()

    def run_multi_hop_bench(self):
        """
        Tests multi-hop reasoning via graph-enhanced retrieval.
        Alice -> works at GroupX -> Tokyo
        Query: "Where is Alice located?"
        """
        snap = self.collector.start_test("Retrieval_MultiHop")
        
        # Setup specific multi-hop scenario
        self.db.insert_fact(subject="Alice", predicate="works at", obj="TechCorp")
        self.db.insert_fact(subject="TechCorp", predicate="is located in", obj="Tokyo")
        
        # Need to backfill embeddings for facts to be searchable
        self.retriever.backfill_all_embeddings()
        
        # Test query
        start_time = time.perf_counter()
        # We query for Alice and expect Tokyo related content if graph propagation or reasoning works
        results = self.retriever.retrieve("Where is Alice located?", limit=5)
        snap.latency_ms.append((time.perf_counter() - start_time) * 1000)
        
        # Check recall
        found = any("Tokyo" in r.content for r in results)
        snap.extras["multi_hop_recall"] = 1.0 if found else 0.0
        
        self.collector.record_batch(snap, start_time, 1)
        return snap.summarize()

import os
import shutil
from datetime import datetime

from bench.graph_bench import GraphStress
from bench.ingestion import IngestionStress
from bench.metrics import MetricsCollector
from bench.retrieval import RetrievalStress
from bench.sync_bench import SyncStress


class BenchmarkRunner:
    def __init__(self, dataset_size: int = 1000):
        self.dataset_size = dataset_size
        self.collector = MetricsCollector()
        self.tmp_dir = "bench_tmp_final"
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)

        self.db_path = os.path.join(self.tmp_dir, "main.db")
        from memk.storage.db import MemoryDB

        MemoryDB(self.db_path).init_db()

    def run_all(self):
        print(f"Starting Deep Stress Test (Size: {self.dataset_size})...")

        ingest = IngestionStress(self.db_path, self.collector)
        print("  - Testing ingestion scale...")
        ingest.run(self.dataset_size)

        retrieval = RetrievalStress(self.db_path, self.collector)
        print("  - Testing semantic and multi-hop retrieval...")
        retrieval.run_semantic_bench(query_count=20)
        retrieval.run_multi_hop_bench()

        graph = GraphStress(self.db_path, self.collector)
        print("  - Testing graph propagation stress...")
        graph.run_propagation_stress(
            num_entities=min(self.dataset_size, 5000),
            density=0.01,
        )

        sync = SyncStress(self.tmp_dir, self.collector)
        print("  - Testing hybrid sync and conflict detection...")
        sync.run_hybrid_recovery_scenario(item_count=max(200, self.dataset_size // 10))
        sync.run_conflict_scenario()

        print("Benchmarks completed. Generating report...")
        self.generate_report()

    def generate_report(self):
        summaries = self.collector.get_summary()
        report_path = "docs/deep_stress_test_report.md"

        content = "# Deep Stress Test & Capability Evaluation Report\n\n"
        content += f"**Date**: {datetime.now().isoformat()}\n"
        content += f"**Dataset Size**: {self.dataset_size} items\n\n"

        content += "## System Strengths\n"
        strengths = 0

        ingest_thru = next(
            (s.get("throughput", 0) for s in summaries if "Ingestion" in s["name"]),
            0,
        )
        if ingest_thru > 100:
            content += "- **High Throughput Ingestion**: Bulk writes stayed efficient in this run.\n"
            strengths += 1

        hop_recall = next(
            (s.get("multi_hop_recall", 0) for s in summaries if "MultiHop" in s["name"]),
            0,
        )
        if hop_recall > 0.5:
            content += "- **Semantic Connectivity**: Multi-hop fact retrieval surfaced the linked target.\n"
            strengths += 1

        sync_conv = next(
            (s.get("converged", False) for s in summaries if "Sync_Hybrid" in s["name"]),
            False,
        )
        if sync_conv:
            content += "- **Robust Hybrid Sync**: Merkle recovery converged after oplog loss.\n"
            strengths += 1

        conflict_count = next(
            (s.get("conflicts_detected", 0) for s in summaries if "Conflict" in s["name"]),
            0,
        )
        if conflict_count > 0:
            content += "- **Conflict Visibility**: Divergent concurrent writes were recorded.\n"
            strengths += 1

        if strengths == 0:
            content += "- No automatic strength threshold was crossed in this run.\n"

        content += "\n## Current Bottlenecks & Weaknesses\n"
        weaknesses = 0
        for summary in summaries:
            if summary.get("p99", 0) > 500:
                content += (
                    f"- **Tail Latency in {summary['name']}**: "
                    f"P99 reached {summary['p99']}ms.\n"
                )
                weaknesses += 1
            if summary.get("errors", 0) > 0:
                content += (
                    f"- **Reliability Issues in {summary['name']}**: "
                    f"Encountered {summary['errors']} errors.\n"
                )
                weaknesses += 1

        if weaknesses == 0:
            content += "- No automatic bottleneck threshold was crossed in this run.\n"

        content += "\n## Performance Metrics Table\n\n"
        content += "| Metric Class | Count | P50 (ms) | P95 (ms) | P99 (ms) | Avg (ms) | Throughput (it/s) | Memory (MB) | Errors |\n"
        content += "|:---|:---|:---|:---|:---|:---|:---|:---|:---|\n"

        for summary in summaries:
            content += (
                f"| {summary['name']} "
                f"| {summary.get('count', '-')} "
                f"| {summary.get('p50', '-')} "
                f"| {summary.get('p95', '-')} "
                f"| {summary.get('p99', '-')} "
                f"| {summary.get('avg', '-')} "
                f"| {summary.get('throughput', '-')} "
                f"| {summary.get('memory_mb', '-')} "
                f"| {summary.get('errors', '-')} |\n"
            )

        content += "\n## Performance Insights\n"
        content += "1. **Sync State Control**: Merkle recovery is heavier than oplog replay, but protects stale replicas when history is gone.\n"
        content += "2. **Graph Scaling**: Propagation latency scales with density; `max_active_entities` remains the main stability lever.\n"
        content += "3. **Retrieval Consistency**: This benchmark uses a deterministic offline embedder, so it validates pipeline behavior rather than production model quality.\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000)
    args = parser.parse_args()

    runner = BenchmarkRunner(dataset_size=args.size)
    runner.run_all()

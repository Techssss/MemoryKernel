"""
benchmarks/run_bench.py
========================
Performance benchmark with tail-latency diagnostics.

Measures:
  - Insertion (embed + persist + extract)
  - Search Cold (first query through the full pipeline)
  - Search Hot (cache layer hit)
  - Context Building (retrieve + reconcile + assemble)

Outputs:
  - benchmarks/results.json  — percentile table
  - benchmarks/tail_report.json — root-cause breakdown of slow requests
"""

import time
import os
import json
import logging
import statistics
import sys
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memk.core.service import MemoryKernelService
from memk.core.tracing import get_collector

# Quiet logs except warnings from tracing
logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")

RESULTS_DIR = os.path.join(os.path.dirname(__file__))


class Benchmarker:
    def __init__(self, db_path="bench_performance_test.db"):
        self.db_path = db_path

        # Clean previous bench DB
        for p in [db_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        os.environ["MEMK_DB_PATH"] = self.db_path

        # Reset singletons for clean benchmark
        from memk.core.runtime import RuntimeManager
        RuntimeManager._instance = None
        import memk.core.tracing as tracing_mod
        tracing_mod._collector = None
        import memk.core.embedder as embedder_mod
        embedder_mod._DEFAULT_EMBEDDER = None
        embedder_mod._DEFAULT_PIPELINE = None

        print("Initializing MemoryKernel Service...")
        self.service = MemoryKernelService()
        self.service.ensure_initialized()
        print("Initialization complete.\n")
        self.results = {}

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    async def run_insertion_bench(self, count=50):
        latencies = []
        print(f"[1/4] Benchmarking {count} Insertions...")
        for i in range(count):
            start = time.perf_counter()
            await self.service.add_memory(
                content=f"Fact number {i}: The system performance is critical for AI agents. Latency must be sub-10ms.",
                importance=0.8,
                confidence=0.9,
            )
            latencies.append((time.perf_counter() - start) * 1000)
            if (i + 1) % 10 == 0:
                avg_so_far = statistics.mean(latencies[-10:])
                print(f"  Completed {i+1}/{count}  (last-10 avg={avg_so_far:.1f}ms)")

        self.results["Insertion"] = latencies

    async def run_search_bench(self, count=50):
        cold_latencies = []
        hot_latencies = []
        queries = ["performance", "latency", "system", "AI agents", "critical",
                    "memory", "facts", "embedding", "sub-10ms", "architecture"] * (count // 10 + 1)
        queries = queries[:count]

        print(f"[2/4] Benchmarking {len(queries)} Cold Searches...")
        for i, q in enumerate(queries):
            start = time.perf_counter()
            await self.service.search(q, limit=5)
            cold_latencies.append((time.perf_counter() - start) * 1000)
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{len(queries)}...")

        print(f"[3/4] Benchmarking {len(queries)} Hot Searches (Cached)...")
        for i, q in enumerate(queries):
            start = time.perf_counter()
            await self.service.search(q, limit=5)
            hot_latencies.append((time.perf_counter() - start) * 1000)
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{len(queries)}...")

        self.results["Search (Cold)"] = cold_latencies
        self.results["Search (Hot)"] = hot_latencies

    async def run_context_bench(self, count=20):
        latencies = []
        context_queries = ["performance metrics", "system architecture",
                           "AI agent latency", "memory management"]
        print(f"[4/4] Benchmarking {count} Context Buildings...")
        for i in range(count):
            q = context_queries[i % len(context_queries)]
            start = time.perf_counter()
            await self.service.build_context(q, max_chars=1000)
            latencies.append((time.perf_counter() - start) * 1000)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{count}...")

        self.results["Context Building"] = latencies

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def display_results(self):
        print("\n" + "=" * 70)
        print("  MEMORYKERNEL PERFORMANCE BENCHMARK")
        print("=" * 70)

        json_results = {}
        for op, data in self.results.items():
            sorted_data = sorted(data)
            n = len(sorted_data)
            avg = statistics.mean(data)
            med = statistics.median(data)
            min_val = min(data)
            max_val = max(data)
            p90 = sorted_data[int(n * 0.90)] if n >= 10 else max_val
            p95 = sorted_data[int(n * 0.95)] if n >= 20 else max_val
            p99 = sorted_data[int(n * 0.99)] if n >= 100 else max_val
            stdev = statistics.stdev(data) if len(data) > 1 else 0

            print(f"\n  {op}")
            print(f"    Avg: {avg:8.2f} ms   Med: {med:8.2f} ms   Stdev: {stdev:6.2f} ms")
            print(f"    Min: {min_val:8.2f} ms   Max: {max_val:8.2f} ms")
            print(f"    P90: {p90:8.2f} ms   P95: {p95:8.2f} ms   P99: {p99:8.2f} ms")

            # Target assessment
            targets = {
                "Insertion": 50,
                "Search (Cold)": 20,
                "Search (Hot)": 1,
                "Context Building": 20,
            }
            target = targets.get(op, 50)
            status = "PASS" if p95 <= target else "FAIL"
            symbol = "[OK]" if status == "PASS" else "[FAIL]"
            print(f"    Target P95 < {target}ms: {symbol} {status}")

            json_results[op] = {
                "avg": round(avg, 3),
                "median": round(med, 3),
                "min": round(min_val, 3),
                "max": round(max_val, 3),
                "p90": round(p90, 3),
                "p95": round(p95, 3),
                "p99": round(p99, 3),
                "stdev": round(stdev, 3),
                "target_p95": target,
                "pass": status == "PASS",
            }

        # Save results
        results_path = os.path.join(RESULTS_DIR, "results.json")
        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=4)

        print(f"\n  Results saved to: {results_path}")

    def display_tail_report(self):
        """Display the tracing-based tail-latency report."""
        report = self.service.get_tail_latency_report()

        print("\n" + "=" * 70)
        print("  TAIL-LATENCY ANALYSIS REPORT")
        print("=" * 70)

        total = report.get("total_requests", 0)
        slow = report.get("slow_request_count", 0)
        rate = report.get("slow_rate_pct", 0)
        pcts = report.get("latency_percentiles", {})

        print(f"\n  Total Requests:  {total}")
        print(f"  Slow Requests:   {slow}  ({rate:.1f}%)")
        print(f"  Percentiles:     P50={pcts.get('p50', 0):.1f}ms  "
              f"P90={pcts.get('p90', 0):.1f}ms  "
              f"P95={pcts.get('p95', 0):.1f}ms  "
              f"P99={pcts.get('p99', 0):.1f}ms")

        causes = report.get("cause_frequency", {})
        if causes:
            print("\n  Root Cause Frequency:")
            for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
                pct_of_total = count / total * 100 if total > 0 else 0
                bar = "#" * int(pct_of_total / 2)
                print(f"    {cause:25s} {count:4d}  ({pct_of_total:5.1f}%) {bar}")

        slow_paths = report.get("top_slow_paths", [])
        if slow_paths:
            print("\n  Top Slow Paths:")
            for sp in slow_paths:
                degrade_info = f" [DEGRADED: {sp['degradation_reason']}]" if sp.get('degraded') else ""
                print(f"    [{sp['operation']}] cause={sp['root_cause']}{degrade_info}  "
                      f"count={sp['count']}  avg={sp['avg_ms']:.1f}ms  max={sp['max_ms']:.1f}ms")

        # Save report
        report_path = os.path.join(RESULTS_DIR, "tail_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"\n  Report saved to: {report_path}")
        print("=" * 70)

    def display_embedding_telemetry(self):
        """Display embedding pipeline telemetry."""
        telem = self.service.get_embedding_telemetry()

        print("\n" + "=" * 70)
        print("  EMBEDDING PIPELINE TELEMETRY")
        print("=" * 70)

        pipeline = telem.get("pipeline", {})
        cache = telem.get("cache", {})

        print(f"\n  Total Calls:      {pipeline.get('total_calls', 0)}")
        print(f"  Cache Hits:       {pipeline.get('cache_hits', 0)}  (exact)")
        print(f"  Semantic Hits:    {pipeline.get('semantic_hits', 0)}  (neighborhood)")
        print(f"  Model Calls:      {pipeline.get('model_calls', 0)}  (cold)")
        print(f"  Cache Hit Rate:   {pipeline.get('cache_hit_rate_pct', 0):.1f}%")
        print(f"  Avg Latency:      {pipeline.get('avg_latency_ms', 0):.3f}ms")
        print(f"  Max Latency:      {pipeline.get('max_latency_ms', 0):.3f}ms")

        dist = pipeline.get("latency_distribution", {})
        if dist:
            print("\n  Latency Distribution:")
            for bucket, count in dist.items():
                bar = "#" * min(count, 40)
                print(f"    {bucket:8s} {count:4d}  {bar}")

        print(f"\n  Cache Size:       {cache.get('size', 0)} / {cache.get('max_size', 0)}")
        print(f"  Cache Hit Rate:   {cache.get('hit_rate', '0%')}")

        concurrency = telem.get("concurrency", {})
        if concurrency:
            print(f"\n  Concurrency:")
            print(f"    Pending Tasks:   {concurrency.get('pending_tasks', 0)}")
            print(f"    Max Queue Size:  {concurrency.get('max_queue_size', 0)}")
            print(f"    Is Busy (Guard): {concurrency.get('is_busy', False)}")

        # Save
        telem_path = os.path.join(RESULTS_DIR, "embedding_telemetry.json")
        with open(telem_path, "w") as f:
            json.dump(telem, f, indent=4)

        print(f"\n  Telemetry saved to: {telem_path}")
        print("=" * 70)


async def main():
    bench = Benchmarker()
    print("Starting Performance Test (Async)...\n")

    await bench.run_insertion_bench(count=50)
    await bench.run_search_bench(count=50)
    await bench.run_context_bench(count=20)

    bench.display_results()
    bench.display_tail_report()
    bench.display_embedding_telemetry()


if __name__ == "__main__":
    asyncio.run(main())

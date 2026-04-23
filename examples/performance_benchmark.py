"""
Performance Benchmark - MemoryKernel
====================================
Comprehensive performance testing with large datasets.

Metrics:
- RAM consumption
- Query latency (P50, P95, P99)
- Throughput (ops/sec)
- Accuracy (precision, recall)
- Index build time
- Cache hit rate
"""

import asyncio
import sys
import time
import psutil
import os
from pathlib import Path
from typing import List, Dict, Tuple
import random
import string

sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.core.runtime_v2 import get_runtime_v2
from memk.workspace.manager import WorkspaceManager


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self, workspace_id: str = "perf-test"):
        self.workspace_id = workspace_id
        self.runtime_manager = None
        self.workspace = None
        self.process = psutil.Process(os.getpid())
        
        # Test data
        self.test_memories = []
        self.test_queries = []
        
        # Results
        self.results = {
            "memory": {},
            "latency": {},
            "throughput": {},
            "accuracy": {},
            "cache": {},
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        mem_info = self.process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        }
    
    def generate_test_data(self, num_memories: int = 1000):
        """Generate synthetic test data."""
        print(f"[*] Generating {num_memories} test memories...")
        
        # Categories for realistic data
        categories = [
            ("Python", ["function", "class", "module", "import", "async", "decorator"]),
            ("JavaScript", ["const", "function", "async", "promise", "react", "component"]),
            ("Database", ["SQL", "query", "index", "table", "join", "transaction"]),
            ("API", ["REST", "endpoint", "request", "response", "authentication", "authorization"]),
            ("Architecture", ["microservices", "monolith", "pattern", "design", "SOLID", "DRY"]),
        ]
        
        self.test_memories = []
        for i in range(num_memories):
            category, keywords = random.choice(categories)
            
            # Generate realistic content
            num_keywords = random.randint(3, 6)
            selected_keywords = random.sample(keywords, min(num_keywords, len(keywords)))
            
            content = f"{category}: " + " ".join([
                f"{kw} is important for {random.choice(['development', 'production', 'testing', 'deployment'])}"
                for kw in selected_keywords
            ])
            
            importance = random.uniform(0.3, 0.9)
            
            self.test_memories.append({
                "content": content,
                "importance": importance,
                "category": category,
            })
        
        # Generate test queries
        self.test_queries = [
            "How to use Python functions?",
            "What is REST API?",
            "Database query optimization",
            "JavaScript async programming",
            "Microservices architecture patterns",
            "SOLID principles in design",
            "SQL join operations",
            "React component lifecycle",
            "Authentication best practices",
            "Python decorators usage",
        ]
        
        print(f"✓ Generated {len(self.test_memories)} memories and {len(self.test_queries)} queries")
        print()
    
    async def initialize(self):
        """Initialize workspace and measure baseline memory."""
        print("[*] Initializing Performance Benchmark")
        print("=" * 70)
        print()
        
        # Baseline memory
        baseline_mem = self.get_memory_usage()
        print(f"Baseline memory: {baseline_mem['rss_mb']:.2f} MB")
        print()
        
        # Initialize workspace
        print("Initializing workspace...")
        self.runtime_manager = get_runtime_v2()
        self.runtime_manager.initialize_global()
        
        try:
            ws_manager = WorkspaceManager()
            if not ws_manager.is_initialized():
                ws_manager.initialize()
        except:
            pass
        
        self.workspace = self.runtime_manager.get_workspace_runtime(self.workspace_id)
        
        init_mem = self.get_memory_usage()
        print(f"After init memory: {init_mem['rss_mb']:.2f} MB")
        print(f"Init overhead: {init_mem['rss_mb'] - baseline_mem['rss_mb']:.2f} MB")
        print()
        
        self.results["memory"]["baseline_mb"] = baseline_mem["rss_mb"]
        self.results["memory"]["after_init_mb"] = init_mem["rss_mb"]
        self.results["memory"]["init_overhead_mb"] = init_mem["rss_mb"] - baseline_mem["rss_mb"]
    
    async def benchmark_insertion(self, batch_size: int = 100):
        """Benchmark memory insertion performance."""
        print("[+] Benchmark: Memory Insertion")
        print("-" * 70)
        
        embedder = self.runtime_manager.container.get_embedder()
        
        start_mem = self.get_memory_usage()
        start_time = time.time()
        
        inserted_ids = []
        batch_times = []
        
        for i in range(0, len(self.test_memories), batch_size):
            batch_start = time.time()
            batch = self.test_memories[i:i+batch_size]
            
            for mem_data in batch:
                # Embed
                embedding = embedder.embed(mem_data["content"])
                
                # Insert
                mem_id = self.workspace.db.insert_memory(
                    mem_data["content"],
                    embedding=embedding,
                    importance=mem_data["importance"]
                )
                
                # Add to index
                from memk.retrieval.index import IndexEntry
                import datetime
                
                entry = IndexEntry(
                    id=mem_id,
                    item_type="memory",
                    content=mem_data["content"],
                    importance=mem_data["importance"],
                    confidence=1.0,
                    created_at=datetime.datetime.now().isoformat(),
                    decay_score=1.0,
                    access_count=0,
                )
                self.workspace.index.add_entry(entry, embedding)
                
                inserted_ids.append(mem_id)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if (i + batch_size) % 500 == 0:
                print(f"  Inserted {i + batch_size}/{len(self.test_memories)} memories...")
        
        total_time = time.time() - start_time
        end_mem = self.get_memory_usage()
        
        # Bump generation
        new_gen = self.workspace.bump_generation()
        
        # Calculate metrics
        throughput = len(self.test_memories) / total_time
        avg_latency = (total_time / len(self.test_memories)) * 1000  # ms
        memory_per_item = (end_mem["rss_mb"] - start_mem["rss_mb"]) / len(self.test_memories)
        
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} ops/sec")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  Memory used: {end_mem['rss_mb'] - start_mem['rss_mb']:.2f} MB")
        print(f"  Memory per item: {memory_per_item * 1024:.2f} KB")
        print(f"  Generation: {new_gen}")
        print()
        
        self.results["throughput"]["insertion_ops_per_sec"] = throughput
        self.results["latency"]["insertion_avg_ms"] = avg_latency
        self.results["memory"]["after_insertion_mb"] = end_mem["rss_mb"]
        self.results["memory"]["insertion_overhead_mb"] = end_mem["rss_mb"] - start_mem["rss_mb"]
        self.results["memory"]["memory_per_item_kb"] = memory_per_item * 1024
    
    async def benchmark_search(self, num_iterations: int = 100):
        """Benchmark search performance and accuracy."""
        print("[?] Benchmark: Search Performance")
        print("-" * 70)
        
        latencies = []
        cache_hits = 0
        
        # Warm up cache
        for query in self.test_queries[:3]:
            _ = self.workspace.retriever.retrieve(query, limit=10)
        
        # Benchmark
        for i in range(num_iterations):
            query = random.choice(self.test_queries)
            
            start = time.perf_counter()
            results = self.workspace.retriever.retrieve(query, limit=10)
            latency = (time.perf_counter() - start) * 1000  # ms
            
            latencies.append(latency)
            
            if i % 20 == 0:
                print(f"  Completed {i}/{num_iterations} searches...")
        
        # Calculate percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = sum(latencies) / len(latencies)
        
        # Get cache stats
        cache_stats = self.workspace.cache.get_stats()
        
        print(f"\n✓ Search complete:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Avg latency: {avg:.2f}ms")
        print(f"  P50 latency: {p50:.2f}ms")
        print(f"  P95 latency: {p95:.2f}ms")
        print(f"  P99 latency: {p99:.2f}ms")
        print(f"  Cache hit rate: {cache_stats.get('query', {}).get('hit_rate', 0):.2%}")
        print()
        
        self.results["latency"]["search_avg_ms"] = avg
        self.results["latency"]["search_p50_ms"] = p50
        self.results["latency"]["search_p95_ms"] = p95
        self.results["latency"]["search_p99_ms"] = p99
        self.results["cache"]["hit_rate"] = cache_stats.get('query', {}).get('hit_rate', 0)
    
    async def benchmark_accuracy(self):
        """Benchmark search accuracy with known ground truth."""
        print("[!] Benchmark: Search Accuracy")
        print("-" * 70)
        
        # Create test cases with known relevant results
        test_cases = [
            {
                "query": "Python function development",
                "expected_category": "Python",
                "min_results": 5,
            },
            {
                "query": "JavaScript async programming",
                "expected_category": "JavaScript",
                "min_results": 5,
            },
            {
                "query": "Database SQL query",
                "expected_category": "Database",
                "min_results": 5,
            },
            {
                "query": "REST API endpoint",
                "expected_category": "API",
                "min_results": 5,
            },
            {
                "query": "Microservices architecture",
                "expected_category": "Architecture",
                "min_results": 5,
            },
        ]
        
        total_precision = 0
        total_recall = 0
        
        for test_case in test_cases:
            query = test_case["query"]
            expected_cat = test_case["expected_category"]
            
            # Search
            results = self.workspace.retriever.retrieve(query, limit=10)
            
            # Calculate precision: how many results are relevant?
            relevant_count = 0
            for result in results:
                # Check if result content contains expected category
                if expected_cat.lower() in result.content.lower():
                    relevant_count += 1
            
            precision = relevant_count / len(results) if results else 0
            
            # Calculate recall: how many relevant items were found?
            total_relevant = sum(1 for m in self.test_memories if m["category"] == expected_cat)
            recall = relevant_count / total_relevant if total_relevant > 0 else 0
            
            total_precision += precision
            total_recall += recall
            
            print(f"  Query: {query[:40]}...")
            print(f"    Precision: {precision:.2%}, Recall: {recall:.2%}")
        
        avg_precision = total_precision / len(test_cases)
        avg_recall = total_recall / len(test_cases)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print(f"\n✓ Accuracy metrics:")
        print(f"  Avg Precision: {avg_precision:.2%}")
        print(f"  Avg Recall: {avg_recall:.2%}")
        print(f"  F1 Score: {f1_score:.2%}")
        print()
        
        self.results["accuracy"]["precision"] = avg_precision
        self.results["accuracy"]["recall"] = avg_recall
        self.results["accuracy"]["f1_score"] = f1_score
    
    async def benchmark_concurrent_access(self, num_threads: int = 10):
        """Benchmark concurrent read performance."""
        print("[~] Benchmark: Concurrent Access")
        print("-" * 70)
        
        import concurrent.futures
        
        def search_task(query_id):
            query = self.test_queries[query_id % len(self.test_queries)]
            start = time.perf_counter()
            results = self.workspace.retriever.retrieve(query, limit=10)
            latency = (time.perf_counter() - start) * 1000
            return latency
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(search_task, i) for i in range(100)]
            latencies = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        throughput = len(latencies) / total_time
        avg_latency = sum(latencies) / len(latencies)
        
        print(f"✓ Concurrent access complete:")
        print(f"  Threads: {num_threads}")
        print(f"  Total queries: {len(latencies)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} ops/sec")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print()
        
        self.results["throughput"]["concurrent_ops_per_sec"] = throughput
        self.results["latency"]["concurrent_avg_ms"] = avg_latency
    
    def print_summary(self):
        """Print comprehensive benchmark summary."""
        print("\n" + "=" * 70)
        print("[=] BENCHMARK SUMMARY")
        print("=" * 70)
        print()
        
        # Memory
        print("[MEM] Memory Usage:")
        print(f"  Baseline: {self.results['memory']['baseline_mb']:.2f} MB")
        print(f"  After init: {self.results['memory']['after_init_mb']:.2f} MB")
        print(f"  After insertion: {self.results['memory']['after_insertion_mb']:.2f} MB")
        print(f"  Total overhead: {self.results['memory']['after_insertion_mb'] - self.results['memory']['baseline_mb']:.2f} MB")
        print(f"  Per item: {self.results['memory']['memory_per_item_kb']:.2f} KB")
        print()
        
        # Latency
        print("[LAT] Latency:")
        print(f"  Insertion avg: {self.results['latency']['insertion_avg_ms']:.2f}ms")
        print(f"  Search avg: {self.results['latency']['search_avg_ms']:.2f}ms")
        print(f"  Search P50: {self.results['latency']['search_p50_ms']:.2f}ms")
        print(f"  Search P95: {self.results['latency']['search_p95_ms']:.2f}ms")
        print(f"  Search P99: {self.results['latency']['search_p99_ms']:.2f}ms")
        print()
        
        # Throughput
        print("[THR] Throughput:")
        print(f"  Insertion: {self.results['throughput']['insertion_ops_per_sec']:.2f} ops/sec")
        print(f"  Concurrent: {self.results['throughput']['concurrent_ops_per_sec']:.2f} ops/sec")
        print()
        
        # Accuracy
        print("[ACC] Accuracy:")
        print(f"  Precision: {self.results['accuracy']['precision']:.2%}")
        print(f"  Recall: {self.results['accuracy']['recall']:.2%}")
        print(f"  F1 Score: {self.results['accuracy']['f1_score']:.2%}")
        print()
        
        # Cache
        print("[CHE] Cache:")
        print(f"  Hit rate: {self.results['cache']['hit_rate']:.2%}")
        print()
        
        # Pass/Fail
        print("[CHK] Performance Targets:")
        targets = [
            ("Insertion throughput > 10 ops/sec", self.results['throughput']['insertion_ops_per_sec'] > 10),
            ("Search P50 < 50ms", self.results['latency']['search_p50_ms'] < 50),
            ("Search P95 < 100ms", self.results['latency']['search_p95_ms'] < 100),
            ("Precision > 60%", self.results['accuracy']['precision'] > 0.6),
            ("Memory per item < 100KB", self.results['memory']['memory_per_item_kb'] < 100),
        ]
        
        for target, passed in targets:
            status = "✓" if passed else "✗"
            print(f"  {status} {target}")
        
        print()
        print("=" * 70)


async def main():
    """Run comprehensive benchmark."""
    
    # Configuration
    NUM_MEMORIES = 1000  # Adjust for larger tests
    NUM_SEARCH_ITERATIONS = 100
    
    benchmark = PerformanceBenchmark()
    
    # Generate test data
    benchmark.generate_test_data(num_memories=NUM_MEMORIES)
    
    # Initialize
    await benchmark.initialize()
    
    # Run benchmarks
    await benchmark.benchmark_insertion()
    await benchmark.benchmark_search(num_iterations=NUM_SEARCH_ITERATIONS)
    await benchmark.benchmark_accuracy()
    await benchmark.benchmark_concurrent_access()
    
    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    asyncio.run(main())

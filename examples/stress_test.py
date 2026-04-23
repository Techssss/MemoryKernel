"""
MemoryKernel Stress Test & Performance Benchmark
=================================================
Comprehensive testing with large datasets to measure:
- Memory consumption (RAM usage)
- Speed (latency, throughput)
- Accuracy (search relevance)
- Scalability (performance under load)

Test Scenarios:
1. Large-scale memory insertion (1K, 10K, 100K records)
2. Concurrent search operations
3. Memory usage tracking
4. Search accuracy validation
5. Cache performance
6. Index performance
"""

import asyncio
import sys
import time
import psutil
import os
from pathlib import Path
from typing import List, Dict, Any
import random
import string

sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.core.runtime_v2 import get_runtime_v2


class PerformanceMonitor:
    """Monitor system resources during tests."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_memory = self.get_memory_mb()
        self.start_time = time.time()
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_delta(self) -> float:
        """Get memory increase since start."""
        return self.get_memory_mb() - self.start_memory
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance stats."""
        return {
            "memory_mb": self.get_memory_mb(),
            "memory_delta_mb": self.get_memory_delta(),
            "elapsed_sec": self.get_elapsed_time(),
            "cpu_percent": self.process.cpu_percent(),
        }


class DataGenerator:
    """Generate realistic test data."""
    
    TOPICS = [
        "Python programming", "Machine learning", "Web development",
        "Database design", "API architecture", "Cloud computing",
        "DevOps practices", "Security best practices", "Testing strategies",
        "Code review", "Performance optimization", "Microservices",
        "Docker containers", "Kubernetes", "CI/CD pipelines",
        "Git workflows", "Agile methodology", "System design",
        "Data structures", "Algorithms", "Design patterns",
    ]
    
    TEMPLATES = [
        "Always use {topic} for better {benefit}",
        "Best practice: {topic} improves {benefit}",
        "Remember: {topic} is essential for {benefit}",
        "Tip: Implement {topic} to achieve {benefit}",
        "Important: {topic} helps with {benefit}",
        "Note: {topic} enables {benefit}",
        "Consider {topic} when working on {benefit}",
        "Use {topic} to optimize {benefit}",
    ]
    
    BENEFITS = [
        "code quality", "performance", "maintainability", "scalability",
        "reliability", "security", "testability", "readability",
        "efficiency", "productivity", "collaboration", "deployment",
    ]
    
    @classmethod
    def generate_memory(cls, index: int = 0) -> str:
        """Generate a realistic memory content."""
        template = random.choice(cls.TEMPLATES)
        topic = random.choice(cls.TOPICS)
        benefit = random.choice(cls.BENEFITS)
        
        content = template.format(topic=topic, benefit=benefit)
        
        # Add some variation
        if random.random() > 0.7:
            content += f" (Reference: doc-{index})"
        
        return content
    
    @classmethod
    def generate_query(cls) -> str:
        """Generate a realistic search query."""
        queries = [
            f"How to improve {random.choice(cls.BENEFITS)}?",
            f"Best practices for {random.choice(cls.TOPICS)}",
            f"What is {random.choice(cls.TOPICS)}?",
            f"Tips for {random.choice(cls.BENEFITS)}",
            f"How to use {random.choice(cls.TOPICS)}?",
        ]
        return random.choice(queries)


class StressTest:
    """Main stress test orchestrator."""
    
    def __init__(self, workspace_id: str = "stress-test"):
        self.workspace_id = workspace_id
        self.monitor = PerformanceMonitor()
        self.runtime_manager = None
        self.workspace = None
        self.results = {}
        
    async def initialize(self):
        """Initialize test environment."""
        print("🔧 Initializing test environment...")
        
        self.runtime_manager = get_runtime_v2()
        self.runtime_manager.initialize_global()
        self.workspace = self.runtime_manager.get_workspace_runtime(self.workspace_id)
        
        stats = self.monitor.get_stats()
        print(f"✓ Initialized (Memory: {stats['memory_mb']:.1f} MB)")
        print()
    
    async def test_insertion_performance(self, count: int):
        """Test memory insertion performance."""
        print(f"📝 Test 1: Inserting {count:,} memories...")
        print("-" * 70)
        
        embedder = self.runtime_manager.container.get_embedder()
        
        start_time = time.time()
        start_memory = self.monitor.get_memory_mb()
        
        # Batch embedding for better performance
        batch_size = 100
        for i in range(0, count, batch_size):
            batch_end = min(i + batch_size, count)
            
            # Generate batch content
            batch_contents = [DataGenerator.generate_memory(j) for j in range(i, batch_end)]
            batch_importances = [random.uniform(0.3, 0.9) for _ in range(len(batch_contents))]
            
            # Batch embed
            batch_embeddings = embedder.embed_batch(batch_contents)
            
            # Insert batch
            for content, embedding, importance in zip(batch_contents, batch_embeddings, batch_importances):
                mem_id = self.workspace.db.insert_memory(
                    content,
                    embedding=embedding,
                    importance=importance
                )
                
                # Add to index for faster search
                from memk.retrieval.index import IndexEntry
                import datetime
                
                entry = IndexEntry(
                    id=mem_id,
                    item_type="memory",
                    content=content,
                    importance=importance,
                    confidence=1.0,
                    created_at=datetime.datetime.now().isoformat(),
                    decay_score=1.0,
                    access_count=0,
                )
                self.workspace.index.add_entry(entry, embedding)
            
            # Progress update
            if (i + batch_size) % 1000 == 0 or (i + batch_size) >= count:
                elapsed = time.time() - start_time
                rate = (i + batch_size) / elapsed
                memory = self.monitor.get_memory_mb()
                print(f"  Progress: {i + batch_size:,}/{count:,} "
                      f"({rate:.0f} ops/sec, {memory:.1f} MB)")
        
        # Bump generation
        self.workspace.bump_generation()
        
        elapsed = time.time() - start_time
        memory_used = self.monitor.get_memory_mb() - start_memory
        ops_per_sec = count / elapsed
        
        self.results['insertion'] = {
            "count": count,
            "elapsed_sec": elapsed,
            "ops_per_sec": ops_per_sec,
            "memory_mb": memory_used,
            "mb_per_1k": (memory_used / count) * 1000,
        }
        
        print()
        print(f"✓ Insertion complete:")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - Speed: {ops_per_sec:.0f} ops/sec")
        print(f"  - Memory: {memory_used:.1f} MB ({self.results['insertion']['mb_per_1k']:.2f} MB/1K records)")
        print(f"  - Index size: {len(self.workspace.index):,} entries")
        print()
    
    async def test_search_performance(self, num_queries: int = 100):
        """Test search performance."""
        print(f"🔍 Test 2: Running {num_queries} search queries...")
        print("-" * 70)
        
        queries = [DataGenerator.generate_query() for _ in range(num_queries)]
        
        start_time = time.time()
        latencies = []
        
        for i, query in enumerate(queries):
            query_start = time.time()
            results = self.workspace.retriever.retrieve(query, limit=10)
            query_time = (time.time() - query_start) * 1000  # ms
            
            latencies.append(query_time)
            
            if (i + 1) % 20 == 0:
                avg_latency = sum(latencies) / len(latencies)
                print(f"  Progress: {i + 1}/{num_queries} "
                      f"(avg: {avg_latency:.1f}ms)")
        
        elapsed = time.time() - start_time
        
        # Calculate percentiles
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        avg = sum(latencies) / len(latencies)
        
        self.results['search'] = {
            "num_queries": num_queries,
            "elapsed_sec": elapsed,
            "qps": num_queries / elapsed,
            "latency_avg_ms": avg,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
        }
        
        print()
        print(f"✓ Search complete:")
        print(f"  - Queries: {num_queries}")
        print(f"  - QPS: {self.results['search']['qps']:.1f}")
        print(f"  - Latency (avg): {avg:.1f}ms")
        print(f"  - Latency (P50): {p50:.1f}ms")
        print(f"  - Latency (P95): {p95:.1f}ms")
        print(f"  - Latency (P99): {p99:.1f}ms")
        print()
    
    async def test_search_accuracy(self, num_tests: int = 50):
        """Test search accuracy with known queries."""
        print(f"🎯 Test 3: Testing search accuracy ({num_tests} tests)...")
        print("-" * 70)
        
        # Create test cases with known content
        test_cases = [
            ("Python programming best practices", "Python programming"),
            ("How to improve code quality?", "code quality"),
            ("Machine learning tips", "Machine learning"),
            ("Database design patterns", "Database design"),
            ("API architecture guidelines", "API architecture"),
        ]
        
        correct = 0
        total = 0
        
        for query, expected_keyword in test_cases * (num_tests // len(test_cases)):
            results = self.workspace.retriever.retrieve(query, limit=5)
            
            # Check if any result contains the expected keyword
            found = any(expected_keyword.lower() in r.content.lower() for r in results)
            
            if found:
                correct += 1
            total += 1
        
        accuracy = (correct / total) * 100
        
        self.results['accuracy'] = {
            "total_tests": total,
            "correct": correct,
            "accuracy_percent": accuracy,
        }
        
        print(f"✓ Accuracy test complete:")
        print(f"  - Tests: {total}")
        print(f"  - Correct: {correct}")
        print(f"  - Accuracy: {accuracy:.1f}%")
        print()
    
    async def test_concurrent_operations(self, num_concurrent: int = 10):
        """Test concurrent search operations."""
        print(f"⚡ Test 4: Testing {num_concurrent} concurrent searches...")
        print("-" * 70)
        
        queries = [DataGenerator.generate_query() for _ in range(num_concurrent)]
        
        async def search_task(query: str):
            start = time.time()
            results = self.workspace.retriever.retrieve(query, limit=10)
            return time.time() - start
        
        start_time = time.time()
        
        # Run concurrent searches
        tasks = [search_task(q) for q in queries]
        latencies = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        avg_latency = (sum(latencies) / len(latencies)) * 1000
        
        self.results['concurrent'] = {
            "num_concurrent": num_concurrent,
            "total_time_sec": elapsed,
            "avg_latency_ms": avg_latency,
            "throughput": num_concurrent / elapsed,
        }
        
        print(f"✓ Concurrent test complete:")
        print(f"  - Concurrent ops: {num_concurrent}")
        print(f"  - Total time: {elapsed:.2f}s")
        print(f"  - Avg latency: {avg_latency:.1f}ms")
        print(f"  - Throughput: {self.results['concurrent']['throughput']:.1f} ops/sec")
        print()
    
    async def test_cache_performance(self, num_queries: int = 100):
        """Test cache hit rate and performance."""
        print(f"💾 Test 5: Testing cache performance ({num_queries} queries)...")
        print("-" * 70)
        
        # Generate queries (with duplicates to test cache)
        unique_queries = [DataGenerator.generate_query() for _ in range(10)]
        queries = unique_queries * (num_queries // 10)
        random.shuffle(queries)
        
        cache_hits = 0
        cache_misses = 0
        
        for query in queries:
            # Check cache before search
            cache_key = (self.workspace_id, query, 10)
            cached = self.workspace.cache.search_results.get(cache_key)
            
            if cached is not None:
                cache_hits += 1
            else:
                cache_misses += 1
                # Perform search (will populate cache)
                self.workspace.retriever.retrieve(query, limit=10)
        
        hit_rate = (cache_hits / num_queries) * 100
        
        self.results['cache'] = {
            "total_queries": num_queries,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate_percent": hit_rate,
        }
        
        print(f"✓ Cache test complete:")
        print(f"  - Total queries: {num_queries}")
        print(f"  - Cache hits: {cache_hits}")
        print(f"  - Cache misses: {cache_misses}")
        print(f"  - Hit rate: {hit_rate:.1f}%")
        print()
    
    async def test_memory_scaling(self):
        """Test memory usage at different scales."""
        print(f"📊 Test 6: Testing memory scaling...")
        print("-" * 70)
        
        scales = [100, 500, 1000, 5000]
        scaling_results = []
        
        for scale in scales:
            # Clear workspace
            self.workspace = self.runtime_manager.get_workspace_runtime(
                f"{self.workspace_id}-scale-{scale}"
            )
            
            start_memory = self.monitor.get_memory_mb()
            
            # Insert memories
            embedder = self.runtime_manager.container.get_embedder()
            for i in range(scale):
                content = DataGenerator.generate_memory(i)
                embedding = embedder.embed(content)
                self.workspace.db.insert_memory(content, embedding=embedding)
            
            end_memory = self.monitor.get_memory_mb()
            memory_used = end_memory - start_memory
            
            scaling_results.append({
                "scale": scale,
                "memory_mb": memory_used,
                "mb_per_record": memory_used / scale,
            })
            
            print(f"  Scale {scale:,}: {memory_used:.1f} MB "
                  f"({memory_used / scale * 1000:.2f} MB/1K)")
        
        self.results['scaling'] = scaling_results
        print()
    
    def print_summary(self):
        """Print comprehensive test summary."""
        print("=" * 70)
        print("📊 STRESS TEST SUMMARY")
        print("=" * 70)
        print()
        
        # System info
        stats = self.monitor.get_stats()
        print("💻 System Resources:")
        print(f"  - Total memory: {stats['memory_mb']:.1f} MB")
        print(f"  - Memory delta: {stats['memory_delta_mb']:.1f} MB")
        print(f"  - CPU usage: {stats['cpu_percent']:.1f}%")
        print(f"  - Total time: {stats['elapsed_sec']:.1f}s")
        print()
        
        # Insertion performance
        if 'insertion' in self.results:
            r = self.results['insertion']
            print("📝 Insertion Performance:")
            print(f"  - Records: {r['count']:,}")
            print(f"  - Speed: {r['ops_per_sec']:.0f} ops/sec")
            print(f"  - Memory: {r['mb_per_1k']:.2f} MB/1K records")
            print()
        
        # Search performance
        if 'search' in self.results:
            r = self.results['search']
            print("🔍 Search Performance:")
            print(f"  - QPS: {r['qps']:.1f}")
            print(f"  - Latency (P50): {r['latency_p50_ms']:.1f}ms")
            print(f"  - Latency (P95): {r['latency_p95_ms']:.1f}ms")
            print(f"  - Latency (P99): {r['latency_p99_ms']:.1f}ms")
            print()
        
        # Accuracy
        if 'accuracy' in self.results:
            r = self.results['accuracy']
            print("🎯 Search Accuracy:")
            print(f"  - Accuracy: {r['accuracy_percent']:.1f}%")
            print(f"  - Correct: {r['correct']}/{r['total_tests']}")
            print()
        
        # Cache performance
        if 'cache' in self.results:
            r = self.results['cache']
            print("💾 Cache Performance:")
            print(f"  - Hit rate: {r['hit_rate_percent']:.1f}%")
            print(f"  - Hits: {r['cache_hits']}/{r['total_queries']}")
            print()
        
        # Overall assessment
        print("✅ Overall Assessment:")
        
        # Check targets
        checks = []
        
        if 'search' in self.results:
            p50 = self.results['search']['latency_p50_ms']
            p95 = self.results['search']['latency_p95_ms']
            checks.append(("P50 < 15ms", p50 < 15, f"{p50:.1f}ms"))
            checks.append(("P95 < 50ms", p95 < 50, f"{p95:.1f}ms"))
        
        if 'insertion' in self.results:
            ops = self.results['insertion']['ops_per_sec']
            checks.append(("Insertion > 20 ops/sec", ops > 20, f"{ops:.0f} ops/sec"))
        
        if 'accuracy' in self.results:
            acc = self.results['accuracy']['accuracy_percent']
            checks.append(("Accuracy > 80%", acc > 80, f"{acc:.1f}%"))
        
        if 'cache' in self.results:
            hit_rate = self.results['cache']['hit_rate_percent']
            checks.append(("Cache hit > 60%", hit_rate > 60, f"{hit_rate:.1f}%"))
        
        for check_name, passed, value in checks:
            status = "✓" if passed else "✗"
            print(f"  {status} {check_name}: {value}")
        
        print()
        print("=" * 70)


async def main():
    """Run comprehensive stress test."""
    print("🚀 MemoryKernel Stress Test & Performance Benchmark")
    print("=" * 70)
    print()
    
    # Configuration
    SMALL_SCALE = 1000    # 1K records
    MEDIUM_SCALE = 5000   # 5K records
    LARGE_SCALE = 10000   # 10K records
    
    # Choose scale based on argument
    if len(sys.argv) > 1:
        scale_arg = sys.argv[1].lower()
        if scale_arg == "small":
            scale = SMALL_SCALE
        elif scale_arg == "medium":
            scale = MEDIUM_SCALE
        elif scale_arg == "large":
            scale = LARGE_SCALE
        else:
            scale = int(sys.argv[1])
    else:
        scale = SMALL_SCALE
    
    print(f"📏 Test scale: {scale:,} records")
    print()
    
    # Run tests
    test = StressTest()
    
    try:
        await test.initialize()
        await test.test_insertion_performance(scale)
        await test.test_search_performance(100)
        await test.test_search_accuracy(50)
        await test.test_concurrent_operations(10)
        await test.test_cache_performance(100)
        await test.test_memory_scaling()
        
        test.print_summary()
        
    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Usage: python stress_test.py [small|medium|large|<number>]")
    print("  small:  1,000 records")
    print("  medium: 5,000 records")
    print("  large:  10,000 records")
    print()
    
    asyncio.run(main())

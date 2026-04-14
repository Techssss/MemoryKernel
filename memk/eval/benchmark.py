import time
import statistics
import requests
import json
import logging
from typing import List, Dict, Any, Callable
from rich.console import Console
from rich.table import Table
from memk.server.manager import URL, is_running
from memk.core.service import MemoryKernelService

console = Console()
logger = logging.getLogger("memk.bench")

class LatencyMetric:
    def __init__(self, name: str):
        self.name = name
        self.values: List[float] = []

    def record(self, val: float):
        self.values.append(val)

    def stats(self) -> Dict[str, float]:
        if not self.values: return {}
        vals = sorted(self.values)
        count = len(vals)
        return {
            "p50": vals[int(count * 0.50)],
            "p95": vals[int(count * 0.95)] if count > 20 else vals[-1],
            "p99": vals[int(count * 0.99)] if count > 100 else vals[-1],
            "avg": sum(vals) / count,
            "min": vals[0],
            "max": vals[-1]
        }

class BenchmarkSuite:
    def __init__(self, iterations: int = 50):
        self.iterations = iterations
        self.results: Dict[str, LatencyMetric] = {}

    def run_op(self, name: str, op: Callable):
        metric = self.results.get(name, LatencyMetric(name))
        for _ in range(self.iterations):
            start = time.perf_counter()
            op()
            metric.record((time.perf_counter() - start) * 1000)
        self.results[name] = metric

    def report(self):
        table = Table(title=f"MemoryKernel Latency Benchmark ({self.iterations} iterations)")
        table.add_column("Operation", style="cyan")
        table.add_column("Avg (ms)", justify="right")
        table.add_column("p50 (ms)", justify="right")
        table.add_column("p95 (ms)", justify="right")
        table.add_column("p99 (ms)", justify="right")

        for name, metric in self.results.items():
            s = metric.stats()
            table.add_row(
                name, 
                f"{s['avg']:.2f}", f"{s['p50']:.2f}", 
                f"{s['p95']:.2f}", f"{s['p99']:.2f}"
            )
        console.print(table)

def run_benchmarks(service_mode: bool = True):
    suite = BenchmarkSuite(iterations=20 if not service_mode else 50)
    
    if service_mode:
        if not is_running():
            console.print("[red]Daemon must be running for 'daemon' benchmarks.[/red]")
            return

        # 1. Search (Daemon + RAM Index)
        suite.run_op("Daemon: Search (Fresh)", lambda: requests.post(f"{URL}/search", json={"query": f"bench_{time.time()}", "limit": 5}))
        
        # 2. Search (Daemon + Cache)
        query = "repeated_query"
        # Prime the cache
        requests.post(f"{URL}/search", json={"query": query})
        suite.run_op("Daemon: Search (Cached)", lambda: requests.post(f"{URL}/search", json={"query": query}))

        # 3. Context (Daemon + RAM Index)
        suite.run_op("Daemon: Context (Fresh)", lambda: requests.post(f"{URL}/context", json={"query": f"bench_{time.time()}"}))

        # 4. Add (Daemon - Async extraction/index)
        suite.run_op("Daemon: Add Memory", lambda: requests.post(f"{URL}/add", json={"content": f"Memory at {time.time()}"}))

    else:
        # Standalone Service Mode (Direct call overhead)
        service = MemoryKernelService()
        service.ensure_initialized()
        
        suite.run_op("Service: Search (Direct)", lambda: service.search(f"bench_{time.time()}"))
        suite.run_op("Service: Add (Direct)", lambda: service.add_memory(f"Direct memory {time.time()}"))

    suite.report()

def profile_breakdown():
    """Deep dive into a single request lifecycle."""
    service = MemoryKernelService()
    service.ensure_initialized()
    query = "performance profile query"
    
    console.print("\n[bold]🔍 Hot-Path Latency Breakdown[/bold]")
    
    # 1. Embed time
    start = time.perf_counter()
    service.runtime.embedder.embed(query)
    embed_ms = (time.perf_counter() - start) * 1000
    
    # 2. Vector search time (RAM)
    q_vec = service.runtime.embedder.embed(query)
    start = time.perf_counter()
    results = service.runtime.index.search(q_vec, top_k=10)
    search_ms = (time.perf_counter() - start) * 1000
    
    # 3. Context building
    items = service.runtime.retriever.retrieve(query)
    start = time.perf_counter()
    service.runtime.builder.build_context(items)
    assembly_ms = (time.perf_counter() - start) * 1000
    
    table = Table()
    table.add_column("Component", style="magenta")
    table.add_column("Latency (ms)", justify="right")
    table.add_row("Query Embedding", f"{embed_ms:.2f}")
    table.add_row("Vector Search (RAM)", f"{search_ms:.2f}")
    table.add_row("Context Assembly", f"{assembly_ms:.2f}")
    console.print(table)

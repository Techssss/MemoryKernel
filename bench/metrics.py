import time
import statistics
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - depends on local benchmark environment
    psutil = None

@dataclass
class MetricSnapshot:
    name: str
    latency_ms: List[float] = field(default_factory=list)
    memory_mb: Optional[float] = None
    throughput: float = 0.0
    errors: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)

    def summarize(self) -> Dict[str, Any]:
        if not self.latency_ms:
            return {"name": self.name, "error": "No data"}
        
        lat = sorted(self.latency_ms)
        return {
            "name": self.name,
            "count": len(self.latency_ms),
            "p50": statistics.median(lat),
            "p95": lat[int(len(lat) * 0.95)] if len(lat) >= 20 else lat[-1],
            "p99": lat[int(len(lat) * 0.99)] if len(lat) >= 100 else lat[-1],
            "avg": statistics.mean(lat),
            "memory_mb": round(self.memory_mb, 2) if self.memory_mb is not None else "-",
            "throughput": round(self.throughput, 2),
            "errors": self.errors,
            **self.extras
        }

class MetricsCollector:
    def __init__(self):
        self.snapshots: Dict[str, MetricSnapshot] = {}
        self.process = psutil.Process(os.getpid()) if psutil is not None else None

    def start_test(self, name: str) -> MetricSnapshot:
        self.snapshots[name] = MetricSnapshot(name=name)
        return self.snapshots[name]

    def measure_memory(self) -> Optional[float]:
        if self.process is None:
            return None
        return self.process.memory_info().rss / (1024 * 1024)

    def record_batch(self, snapshot: MetricSnapshot, start_time: float, count: int, errors: int = 0):
        duration = time.perf_counter() - start_time
        snapshot.throughput = count / duration if duration > 0 else 0
        snapshot.errors += errors
        snapshot.memory_mb = self.measure_memory()

    def get_summary(self) -> List[Dict[str, Any]]:
        return [s.summarize() for s in self.snapshots.values()]

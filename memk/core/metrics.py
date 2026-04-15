"""
memk.core.metrics
=================
Production metrics collection and aggregation.

Tracks request metrics, database metrics, and runtime health.
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

logger = logging.getLogger("memk.metrics")


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    operation: str
    latency_ms: float
    cache_hit: bool
    degraded: bool
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Thread-safe metrics collector for production observability.
    
    Tracks:
    - Request latency percentiles
    - Cache hit rates
    - Degraded request rates
    - Database statistics
    - Runtime health
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._requests: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
        # Counters
        self.total_requests = 0
        self.cache_hits = 0
        self.degraded_requests = 0
        
        # Start time
        self.start_time = time.time()
    
    def record_request(self, metrics: RequestMetrics):
        """Record a completed request."""
        with self._lock:
            self._requests.append(metrics)
            self.total_requests += 1
            if metrics.cache_hit:
                self.cache_hits += 1
            if metrics.degraded:
                self.degraded_requests += 1
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles from recent requests."""
        with self._lock:
            if not self._requests:
                return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
            
            latencies = sorted([r.latency_ms for r in self._requests])
            n = len(latencies)
            
            def pct(p: float) -> float:
                idx = min(int(n * p / 100), n - 1)
                return round(latencies[idx], 2)
            
            return {
                "p50": pct(50),
                "p90": pct(90),
                "p95": pct(95),
                "p99": pct(99),
            }
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return round(self.cache_hits / self.total_requests, 3)
    
    def get_degraded_rate(self) -> float:
        """Calculate degraded request rate."""
        if self.total_requests == 0:
            return 0.0
        return round(self.degraded_requests / self.total_requests, 3)
    
    def get_request_rate(self) -> float:
        """Calculate requests per second."""
        uptime = time.time() - self.start_time
        if uptime == 0:
            return 0.0
        return round(self.total_requests / uptime, 2)
    
    def get_operation_breakdown(self) -> Dict[str, int]:
        """Get count of requests by operation type."""
        with self._lock:
            breakdown = {}
            for req in self._requests:
                breakdown[req.operation] = breakdown.get(req.operation, 0) + 1
            return breakdown
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        return {
            "requests": {
                "total": self.total_requests,
                "rate_per_sec": self.get_request_rate(),
                "window_size": len(self._requests),
            },
            "latency": self.get_latency_percentiles(),
            "cache": {
                "hit_rate": self.get_cache_hit_rate(),
                "total_hits": self.cache_hits,
            },
            "degraded": {
                "rate": self.get_degraded_rate(),
                "total": self.degraded_requests,
            },
            "operations": self.get_operation_breakdown(),
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }
    
    def reset(self):
        """Reset all metrics (for testing)."""
        with self._lock:
            self._requests.clear()
            self.total_requests = 0
            self.cache_hits = 0
            self.degraded_requests = 0
            self.start_time = time.time()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


def record_request(operation: str, latency_ms: float, cache_hit: bool = False, degraded: bool = False):
    """Convenience function to record a request."""
    collector = get_metrics_collector()
    metrics = RequestMetrics(
        operation=operation,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        degraded=degraded,
    )
    collector.record_request(metrics)

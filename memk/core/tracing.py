"""
memk.core.tracing
=================
Per-request tracing and slow-path diagnostics.

Every service call creates a RequestTrace that records microsecond-precision
spans for each phase of the hot path:

    embed → retrieve → rank → assemble → total

Traces exceeding a configurable threshold are automatically logged with
full breakdown, enabling root-cause analysis of tail-latency spikes.

The TracingCollector accumulates traces over the process lifetime and
can produce a structured tail-latency report on demand.
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger("memk.tracing")


# ---------------------------------------------------------------------------
# Span — a single timed phase within a request
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """One timed phase of a request (e.g. 'embed', 'retrieve')."""
    name: str
    start_ns: int = 0
    end_ns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000


# ---------------------------------------------------------------------------
# RequestTrace — full breakdown for one service call
# ---------------------------------------------------------------------------

@dataclass
class RequestTrace:
    """Complete trace for a single service request."""
    operation: str                              # "search" | "build_context" | "add_memory"
    request_id: int = 0
    start_ns: int = 0
    end_ns: int = 0
    spans: List[Span] = field(default_factory=list)
    cache_hit: bool = False
    degraded: bool = False                      # True if latency guard triggered
    degradation_reason: str = ""                # Why it was degraded
    item_count: int = 0                         # items processed
    root_cause: str = ""                        # auto-detected cause of slowness

    @property
    def total_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000

    def span_ms(self, name: str) -> float:
        """Return duration of the named span, or 0 if not found."""
        for s in self.spans:
            if s.name == name:
                return s.duration_ms
        return 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "total_ms": round(self.total_ms, 3),
            "cache_hit": self.cache_hit,
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
            "item_count": self.item_count,
            "root_cause": self.root_cause,
            "spans": {s.name: round(s.duration_ms, 3) for s in self.spans},
        }

    def breakdown_str(self) -> str:
        parts = [f"[{self.operation}] total={self.total_ms:.2f}ms"]
        for s in self.spans:
            parts.append(f"  {s.name}={s.duration_ms:.2f}ms")
            if s.metadata:
                for k, v in s.metadata.items():
                    parts.append(f"    {k}={v}")
        if self.degraded:
            parts.append(f"  ⚠ DEGRADED ({self.degradation_reason})")
        if self.root_cause:
            parts.append(f"  cause={self.root_cause}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# TraceContext — builder pattern for recording spans in a request
# ---------------------------------------------------------------------------

class TraceContext:
    """
    Context manager that accumulates spans for a single request.

    Usage:
        with TraceContext("search") as tc:
            with tc.span("embed"):
                vec = embed(query)
            with tc.span("retrieve"):
                results = index.search(vec)
        trace = tc.trace   # RequestTrace with all spans
    """

    _counter = 0
    _counter_lock = threading.Lock()

    def __init__(self, operation: str):
        with TraceContext._counter_lock:
            TraceContext._counter += 1
            self._id = TraceContext._counter

        self.trace = RequestTrace(
            operation=operation,
            request_id=self._id,
        )
        self._current_span: Optional[Span] = None

    def __enter__(self) -> "TraceContext":
        self.trace.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.trace.end_ns = time.perf_counter_ns()

    @contextmanager
    def span(self, name: str, **metadata):
        """Time a named phase within the request."""
        s = Span(name=name, start_ns=time.perf_counter_ns(), metadata=metadata)
        try:
            yield s
        finally:
            s.end_ns = time.perf_counter_ns()
            self.trace.spans.append(s)

    def elapsed_ms(self) -> float:
        """How many ms have elapsed since the trace started (for deadline checks)."""
        return (time.perf_counter_ns() - self.trace.start_ns) / 1_000_000

    def mark_cache_hit(self):
        self.trace.cache_hit = True

    def mark_degraded(self, reason: str = "latency guard triggered"):
        self.trace.degraded = True
        self.trace.degradation_reason = reason

    def set_item_count(self, n: int):
        self.trace.item_count = n


# ---------------------------------------------------------------------------
# Root-cause classifier
# ---------------------------------------------------------------------------

def classify_root_cause(trace: RequestTrace, cold_embed_threshold_ms: float = 15.0) -> str:
    """
    Heuristic classifier for slow traces.
    Returns a short tag describing the most likely cause.
    """
    embed_ms = trace.span_ms("embed")
    retrieve_ms = trace.span_ms("retrieve")
    rank_ms = trace.span_ms("rank")
    assemble_ms = trace.span_ms("assemble")
    db_ms = trace.span_ms("db_persist")

    if trace.cache_hit:
        return "cache_hit"

    # Order by most impactful cause
    if embed_ms > cold_embed_threshold_ms:
        return "cold_embedding"
    if retrieve_ms > 10.0:
        return "slow_retrieval"
    if db_ms > 10.0:
        return "slow_db_write"
    if rank_ms > 5.0:
        return "large_candidate_set"
    if assemble_ms > 5.0:
        return "complex_assembly"
    if trace.item_count > 100:
        return "large_result_set"
    if trace.degraded:
        return "deadline_exceeded"

    return "normal"


# ---------------------------------------------------------------------------
# TracingCollector — process-level trace aggregation
# ---------------------------------------------------------------------------

class TracingCollector:
    """
    Accumulates RequestTraces and produces tail-latency reports.
    Thread-safe. Keeps a rolling window of the last N traces.
    """

    def __init__(
        self,
        max_traces: int = 1000,
        slow_threshold_ms: float = 20.0,
    ):
        self.max_traces = max_traces
        self.slow_threshold_ms = slow_threshold_ms
        self._traces: List[RequestTrace] = []
        self._slow_traces: List[RequestTrace] = []
        self._lock = threading.Lock()

        # Aggregate counters by root cause
        self._cause_counts: Dict[str, int] = defaultdict(int)
        self._cause_total_ms: Dict[str, float] = defaultdict(float)

    def record(self, trace: RequestTrace):
        """Record a completed trace. Classifies and logs slow ones."""
        trace.root_cause = classify_root_cause(trace)
        with self._lock:
            # Rolling window
            self._traces.append(trace)
            if len(self._traces) > self.max_traces:
                self._traces.pop(0)

            # Aggregate
            self._cause_counts[trace.root_cause] += 1
            self._cause_total_ms[trace.root_cause] += trace.total_ms

            # Slow trace detection
            if trace.total_ms > self.slow_threshold_ms:
                self._slow_traces.append(trace)
                if len(self._slow_traces) > 200:
                    self._slow_traces.pop(0)
                logger.warning(
                    f"SLOW REQUEST detected ({trace.total_ms:.1f}ms > {self.slow_threshold_ms}ms):\n"
                    f"{trace.breakdown_str()}"
                )

    def get_report(self) -> Dict[str, Any]:
        """
        Generate a structured tail-latency report.

        Returns
        -------
        dict with:
            total_requests     : int
            slow_request_count : int
            slow_rate_pct      : float
            top_slow_paths     : list of dicts (operation, avg_ms, count, root_cause)
            cause_frequency    : dict of cause -> count
            latency_percentiles: dict of p50, p90, p95, p99
        """
        with self._lock:
            traces = list(self._traces)
            slow = list(self._slow_traces)
            causes = dict(self._cause_counts)
            cause_ms = dict(self._cause_total_ms)

        if not traces:
            return {"total_requests": 0, "message": "No traces recorded yet."}

        # Compute percentiles
        all_ms = sorted([t.total_ms for t in traces])
        n = len(all_ms)

        def pct(p: float) -> float:
            idx = min(int(n * p / 100), n - 1)
            return round(all_ms[idx], 3)

        # Top slow paths: group slow traces by (operation, root_cause, degradation_reason)
        slow_groups: Dict[tuple, List[RequestTrace]] = defaultdict(list)
        for t in slow:
            key = (t.operation, t.root_cause, t.degradation_reason, t.degraded)
            slow_groups[key].append(t)

        top_slow = []
        for (op, cause, reason, is_degraded), traces in sorted(slow_groups.items(), key=lambda x: -len(x[1])):
            ms_list = [t.total_ms for t in traces]
            top_slow.append({
                "operation": op,
                "root_cause": cause,
                "degradation_reason": reason,
                "degraded": is_degraded,
                "count": len(ms_list),
                "avg_ms": round(sum(ms_list) / len(ms_list), 2),
                "max_ms": round(max(ms_list), 2),
            })

        return {
            "total_requests": n,
            "slow_request_count": len(slow),
            "slow_rate_pct": round(len(slow) / n * 100, 2) if n > 0 else 0,
            "latency_percentiles": {
                "p50": pct(50),
                "p90": pct(90),
                "p95": pct(95),
                "p99": pct(99),
            },
            "top_slow_paths": top_slow[:10],
            "cause_frequency": causes,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_collector: Optional[TracingCollector] = None


def get_collector(slow_threshold_ms: float = 20.0) -> TracingCollector:
    global _collector
    if _collector is None:
        _collector = TracingCollector(slow_threshold_ms=slow_threshold_ms)
    return _collector

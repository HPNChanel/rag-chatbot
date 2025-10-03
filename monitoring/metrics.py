"""In-memory metrics store for the RAG services."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class MetricsSnapshot:
    uptime_seconds: float
    queries_total: int
    avg_latency_ms: float
    throughput_per_minute: float
    retrieval_hits: float
    coverage_score: float


@dataclass
class MetricsStore:
    """Track latency and retrieval metrics in memory."""

    start_time: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _total_latency_ms: float = 0.0
    _query_count: int = 0
    _retrieval_hits_sum: float = 0.0
    _coverage_sum: float = 0.0

    def record(
        self,
        latency_ms: float,
        retrieval_hit_ratio: float,
        coverage_score: float,
    ) -> None:
        with self._lock:
            self._total_latency_ms += max(0.0, latency_ms)
            self._retrieval_hits_sum += max(0.0, min(1.0, retrieval_hit_ratio))
            self._coverage_sum += max(0.0, min(1.0, coverage_score))
            self._query_count += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            count = self._query_count
            avg_latency = self._total_latency_ms / count if count else 0.0
            avg_hits = self._retrieval_hits_sum / count if count else 0.0
            avg_coverage = self._coverage_sum / count if count else 0.0
        uptime = time.perf_counter() - self.start_time
        throughput = (count / (uptime / 60.0)) if uptime > 0 else 0.0
        return MetricsSnapshot(
            uptime_seconds=uptime,
            queries_total=count,
            avg_latency_ms=avg_latency,
            throughput_per_minute=throughput,
            retrieval_hits=avg_hits,
            coverage_score=avg_coverage,
        )

    def as_dict(self) -> Dict[str, float | int]:
        snapshot = self.snapshot()
        return {
            "uptime_seconds": round(snapshot.uptime_seconds, 3),
            "queries_total": snapshot.queries_total,
            "avg_latency_ms": round(snapshot.avg_latency_ms, 3),
            "throughput_per_minute": round(snapshot.throughput_per_minute, 3),
            "retrieval_hits": round(snapshot.retrieval_hits, 3),
            "coverage_score": round(snapshot.coverage_score, 3),
        }


__all__ = ["MetricsStore", "MetricsSnapshot"]


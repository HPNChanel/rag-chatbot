"""Lightweight load test for the RAG API."""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import time
from typing import List

import httpx


def percentile(latencies: List[float], pct: float) -> float:
    if not latencies:
        return 0.0
    index = max(0, min(len(latencies) - 1, math.ceil(pct / 100.0 * len(latencies)) - 1))
    return latencies[index]


async def _worker(
    client: httpx.AsyncClient,
    query: str,
    top_k: int,
    request_count: int,
    delay: float,
    latencies: List[float],
) -> None:
    for _ in range(request_count):
        start = time.perf_counter()
        response = await client.post(
            "/chat",
            json={"query": query, "top_k": top_k},
        )
        response.raise_for_status()
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        if delay:
            await asyncio.sleep(delay)


async def run_load_test(
    base_url: str,
    total_requests: int,
    concurrency: int,
    query: str,
    top_k: int,
    rps: float | None = None,
) -> dict[str, float]:
    per_worker = max(1, total_requests // concurrency)
    delay = 0.0
    if rps and rps > 0:
        delay = max(0.0, concurrency / rps - 1e-6)
    latencies: List[float] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        tasks = [
            asyncio.create_task(
                _worker(client, query, top_k, per_worker, delay, latencies)
            )
            for _ in range(concurrency)
        ]
        await asyncio.gather(*tasks)
    latencies.sort()
    return {
        "requests": len(latencies),
        "avg_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_ms": percentile(latencies, 50),
        "p90_ms": percentile(latencies, 90),
        "p99_ms": percentile(latencies, 99),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark the RAG API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--query", default="What is the project mission?")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--rps", type=float, help="Target requests per second")
    args = parser.parse_args(argv)

    results = asyncio.run(
        run_load_test(
            base_url=args.base_url,
            total_requests=args.requests,
            concurrency=args.concurrency,
            query=args.query,
            top_k=args.top_k,
            rps=args.rps,
        )
    )
    print("Load test results:")
    for key, value in results.items():
        if key.endswith("ms"):
            print(f"  {key}: {value:.2f} ms")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


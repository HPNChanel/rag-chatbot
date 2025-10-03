from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from configs.loader import DeploymentConfig
from monitoring.metrics import MetricsStore
from ui.server import create_app


def test_metrics_store_records_basic_stats() -> None:
    store = MetricsStore()
    store.record(latency_ms=100, retrieval_hit_ratio=0.5, coverage_score=0.7)
    store.record(latency_ms=200, retrieval_hit_ratio=0.9, coverage_score=0.3)
    snapshot = store.as_dict()
    assert snapshot["queries_total"] == 2
    assert snapshot["avg_latency_ms"] > 0
    assert 0 <= snapshot["retrieval_hits"] <= 1


def test_metrics_endpoint_returns_json_payload() -> None:
    config = DeploymentConfig(
        mode="local",
        dataset=Path("data/sample_docs"),
        index_type="bm25",
        api_port=8100,
        ui_port=8600,
        reranker="coverage",
    )
    store = MetricsStore()
    app = create_app(config=config, metrics_store=store)
    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "What is Python?", "top_k": 2})
        assert response.status_code == 200
        metrics = client.get("/metrics").json()
        assert "uptime_seconds" in metrics
        assert metrics["queries_total"] >= 1


from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ui.server import create_app


def _prepare_dataset(tmp_path: Path) -> Path:
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    (data_dir / "python.txt").write_text(
        "Python emphasises readability and has a vast standard library.",
        encoding="utf-8",
    )
    (data_dir / "streamlit.txt").write_text(
        "Streamlit builds interactive dashboards with minimal code.",
        encoding="utf-8",
    )
    return data_dir


def test_query_endpoint_returns_documents_with_scores(tmp_path: Path) -> None:
    data_dir = _prepare_dataset(tmp_path)
    app = create_app(data_dir)

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "Python", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "Python"
    documents = payload["documents"]
    assert len(documents) == 2
    for doc in documents:
        assert "scores" in doc
        assert "ranks" in doc
        assert doc["scores"]["reranked"] >= 0
    metrics = payload["metrics"]
    assert metrics["candidate_count"] >= 2
    assert metrics["total_time"] >= 0


def test_chat_endpoint_returns_answer_and_citations(tmp_path: Path) -> None:
    data_dir = _prepare_dataset(tmp_path)
    app = create_app(data_dir)

    with TestClient(app) as client:
        response = client.post("/chat", json={"query": "What is Streamlit?", "top_k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert "Streamlit" in payload["answer"]
    assert payload["citations"]
    assert payload["documents"]
    assert "References:" in payload["answer"]
    assert payload["metrics"]["end_to_end_time"] >= payload["metrics"]["retrieval_time"]

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import ui.streamlit_app as streamlit_app


def test_highlight_matches_marks_terms() -> None:
    text = "Python makes building apps delightful"
    highlighted = streamlit_app.highlight_matches(text, "Python apps")
    assert highlighted.count("<mark>") >= 2


def test_metrics_panel_renders(tmp_path: Path) -> None:
    payload = {
        "query": "Tell me about Python",
        "answer": "Answer body\n\nReferences:\n- Doc 1",
        "documents": [
            {
                "doc_id": "doc-1",
                "content": "Python is a popular language for rapid development.",
                "metadata": {"title": "Doc 1"},
                "scores": {"candidate": 0.5, "reranked": 0.8},
                "ranks": {"candidate": 1, "reranked": 1},
            },
            {
                "doc_id": "doc-2",
                "content": "Streamlit focuses on interactive data apps.",
                "metadata": {"title": "Doc 2"},
                "scores": {"candidate": 0.4, "reranked": 0.7},
                "ranks": {"candidate": 2, "reranked": 2},
            },
        ],
        "metrics": {
            "retrieval_time": 0.05,
            "rerank_time": 0.01,
            "total_time": 0.06,
            "reranker_impact": 0.5,
            "generation_time": 0.02,
            "end_to_end_time": 0.08,
        },
        "citations": [
            {
                "doc_id": "doc-1",
                "title": "Doc 1",
                "source": str(tmp_path / "doc1.txt"),
                "excerpt": "Python is a popular language for rapid development.",
                "rank": "1",
            }
        ],
    }

    app_path = Path(__file__).resolve().parents[1] / "ui" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path))
    app.session_state["messages"] = [
        {"role": "user", "content": "Tell me about Python"},
        {"role": "assistant", "content": payload["answer"]},
    ]
    app.session_state["last_result"] = payload
    app.run()

    metric_values = {metric.label: metric.value for metric in app.metric}
    assert "Latency" in metric_values
    assert metric_values["Retrieved"] == str(len(payload["documents"]))

    # Retrieved document content should be rendered with highlights.
    assert any("<mark>Python</mark>" in markdown.value for markdown in app.markdown)

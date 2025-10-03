from __future__ import annotations

from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from configs.loader import load_config
from ui.server import create_app


def test_local_config_overrides(tmp_path) -> None:
    config = load_config(
        Path("deploy/configs/local.yaml"),
        {
            "api_port": 8200,
            "index_type": "bm25",
            "index_path": str(tmp_path / "index"),
        },
    )
    assert config.api_port == 8200
    assert config.index_type == "bm25"
    assert config.index_path == (tmp_path / "index").resolve()


def test_local_app_health_endpoint(tmp_path) -> None:
    config = load_config(
        Path("deploy/configs/local.yaml"),
        {
            "index_type": "bm25",
            "api_port": 8300,
            "ui_port": 8700,
            "index_path": str(tmp_path / "index"),
        },
    )
    app = create_app(config=config)
    with TestClient(app) as client:
        health = client.get("/healthz").json()
        assert health["status"] in {"ok", "starting"}
        response = client.post("/query", json={"query": "test", "top_k": 2})
        assert response.status_code == 200


def test_org_config_validation(tmp_path) -> None:
    config_path = tmp_path / "org.yaml"
    config_path.write_text(
        """
mode: org
dataset: data/sample_docs
index_type: bm25
api_port: 9000
ui_port: 9500
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(config_path)


from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from configs.loader import DeploymentConfig
from ui.server import create_app


def test_org_mode_enforces_api_key_and_redaction(tmp_path) -> None:
    config = DeploymentConfig(
        mode="org",
        dataset=Path("data/organization_corpus"),
        index_type="bm25",
        api_port=9000,
        ui_port=9500,
        reranker="coverage",
        index_path=tmp_path / "index",
        workers=1,
        enable_redaction=True,
        logging={"directory": tmp_path / "logs"},
        api_keys=["secret"],
        rate_limit={"requests_per_minute": 1, "burst": 1},
    )
    app = create_app(config=config)
    with TestClient(app) as client:
        unauth = client.post("/chat", json={"query": "hello", "top_k": 2})
        assert unauth.status_code == 401
        headers = {"X-API-Key": "secret"}
        first = client.post(
            "/chat",
            json={"query": "Contact me at example@org.com", "top_k": 2},
            headers=headers,
        )
        assert first.status_code == 200
        assert "[REDACTED_EMAIL]" in first.json()["answer"]
        second = client.post(
            "/chat",
            json={"query": "Contact me at example@org.com", "top_k": 2},
            headers=headers,
        )
        assert second.status_code == 429


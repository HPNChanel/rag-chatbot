.PHONY: run server ui test tiny-bench report lint fmt analyze run-local run-org deploy-check load-test

run:
        python scripts/run_demo.py

run-local:
        python deploy/local_run.py --config deploy/configs/local.yaml --index-type bm25

run-org:
        python deploy/org_run.py --config deploy/configs/org.yaml --index-type bm25

server:
	uvicorn ui.server:app --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py --server.port 8501

test:
	pytest -q

tiny-bench:
	ragx-run --config experiments/configs/baseline.yaml --repeat 1 --override num_queries=3

report:
	ragx-report --run-dir $$(ls -td runs/* | head -1) --format md --title "RAGX Benchmark"

analyze:
        ragx-analyze --run-dir $$(ls -td runs/* | head -1) --redact

lint:
        ruff check .
        mypy .
        flake8 || true

fmt:
        black .
        isort .

deploy-check:
        python - <<'PY'
import subprocess
import sys
import tempfile
import time
import requests
from pathlib import Path

project_root = Path(__file__).resolve().parent

def wait_for(url: str, timeout: float = 20.0) -> None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    local = subprocess.Popen(
        [
            sys.executable,
            "deploy/local_run.py",
            "--config",
            "deploy/configs/local.yaml",
            "--index-type",
            "bm25",
            "--port-api",
            "8123",
            "--port-ui",
            "8623",
            "--run-duration",
            "8",
            "--no-ui",
            "--index-path",
            str(tmp_path / "local-index"),
        ],
        cwd=project_root,
    )
    wait_for("http://localhost:8123/healthz")
    resp = requests.post("http://localhost:8123/chat", json={"query": "hi", "top_k": 1}, timeout=10)
    resp.raise_for_status()
    local.wait()

    org = subprocess.Popen(
        [
            sys.executable,
            "deploy/org_run.py",
            "--config",
            "deploy/configs/org.yaml",
            "--index-type",
            "bm25",
            "--port-api",
            "9123",
            "--port-ui",
            "9623",
            "--run-duration",
            "8",
            "--no-ui",
            "--index-path",
            str(tmp_path / "org-index"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--api-key",
            "ci-key",
        ],
        cwd=project_root,
    )
    wait_for("http://localhost:9123/healthz")
    headers = {"X-API-Key": "ci-key"}
    resp = requests.post("http://localhost:9123/chat", json={"query": "hello", "top_k": 1}, headers=headers, timeout=10)
    resp.raise_for_status()
    metrics = requests.get("http://localhost:9123/metrics", headers=headers, timeout=10)
    metrics.raise_for_status()
    org.wait()
PY

BASE_URL ?= http://localhost:8000
REQUESTS ?= 20
CONCURRENCY ?= 4
TOP_K ?= 4

load-test:
        python deploy/scripts/benchmark_load.py --base-url $(BASE_URL) --requests $(REQUESTS) --concurrency $(CONCURRENCY) --top-k $(TOP_K)

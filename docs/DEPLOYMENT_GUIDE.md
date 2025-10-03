# Deployment Guide

This guide explains how to launch the Retrieval-Augmented Generation (RAG) stack
locally and in a small-organisation setting. The tooling is entirely
Python-driven – no containers required – and is optimised for reproducibility.

## 1. Local Mode

Local mode is intended for a single developer experimenting on a laptop with
toy datasets.

### Prerequisites

1. Install dependencies: `pip install -r requirements.txt`
2. Ensure the sample documents exist (`data/sample_docs`).

### Quickstart

```bash
make run-local
```

The shortcut expands to:

```bash
python deploy/local_run.py --config deploy/configs/local.yaml --index-type bm25
```

Key behaviours:

- Builds the index at `~/.ragx/index/` (can be overridden via CLI).
- Starts the FastAPI service on `http://localhost:8000`.
- Launches the Streamlit UI on `http://localhost:8501`.
- Uses the lightweight Python-only indexer by default to minimise optional
  dependencies.

Once the services are running visit the UI, enter a question, and observe the
retrieval diagnostics and answer references.

### Local Architecture Overview

```
+---------+        HTTP        +-----------+        In-memory         +------+
| Browser |  <---------------- | Streamlit |  <--------------------> | RAG  |
+---------+                    +-----------+                         | Core |
                                                                   +------+ 
```

## 2. Organisation Mode

Organisation mode is optimised for a small lab or research team sharing an
index on a network-attached storage (NAS) drive.

### Shared Setup

1. Mount the shared drive on each machine (e.g. `/shared/ragx/`).
2. Copy or mount the corpus under `data/organization_corpus/`.
3. Create log and index directories: `/shared/ragx/index/` and `/shared/ragx/logs/`.

### Start the Services

```bash
make run-org
```

This resolves to:

```bash
python deploy/org_run.py --config deploy/configs/org.yaml --index-type bm25
```

Organisation-specific behaviour:

- Shared index stored at `/shared/ragx/index/`.
- Structured query logs emitted to `/shared/ragx/logs/` with anonymised IDs.
- Multiple API workers (configurable) served through Uvicorn.
- Optional API keys via `X-API-Key` header, plus token-bucket rate limiting.
- Automatic redaction enabled to guard against accidental PII exposure.
- Health and metrics endpoints (`/healthz`, `/metrics`) for lightweight monitoring.

### Organisation Architecture Overview

```
+---------+     +--------------+     +-------------------+     +-----------+
| Users   | --> | Load Balancer| --> | FastAPI Workers   | --> | Shared    |
| (Teams) |     |   (optional) |     | (uvicorn --workers)|     | Index NAS |
+---------+     +--------------+     +-------------------+     +-----------+
                                            ^
                                            |
                                    +---------------+
                                    | Streamlit UI  |
                                    +---------------+
```

## 3. Scaling Strategies

Even small teams benefit from planned scaling pathways:

### Vertical Scaling

- Upgrade to machines with additional RAM to cache larger indexes.
- Prefer NVMe SSDs on the shared NAS for faster FAISS persistence.
- Enable FAISS (`--index-type faiss`) when the dependency is installed.

### Horizontal Scaling

- Deploy the FastAPI server on 2–3 machines behind a simple round-robin load
  balancer while sharing the same NAS index path.
- Run `python deploy/org_run.py --workers 4` on each node to increase throughput.
- Use the `/metrics` endpoint to track query latency and throughput per node.

### Caching Layer (Optional)

- Add Redis or SQLite as a query cache sitting in front of the API to store
  recent results. The current design exposes hooks in the deployment scripts
  (`ServiceRuntime`) for injecting additional middleware if desired.

## 4. Monitoring & Health Checks

- `/healthz` returns `{"status": "ok"}` once the pipeline is ready.
- `/metrics` returns aggregated latency, throughput, and coverage stats.
- Query logs are emitted as JSON lines inside the configured log directory.
- JSON structured logs (`ragx.log`) rotate automatically.

## 5. Troubleshooting

| Issue | Resolution |
|-------|------------|
| Port already in use | Override ports via `--port-api` / `--port-ui`.
| Shared drive missing | Update `index_path` and `logging.directory` to local paths or ensure the NAS is mounted.
| Index too large for memory | Switch to FAISS with an ANN index, or increase RAM / use sharded indexes.
| 401 responses in org mode | Ensure `X-API-Key` matches one of the configured API keys.
| 429 responses | Increase `rate_limit.requests_per_minute` or add more workers.
| UI cannot reach API | Confirm `RAG_API_URL` environment variable is set and accessible from Streamlit.

## 6. Example Configurations

- Local: `deploy/configs/local.yaml`
- Organisation: `deploy/configs/org.yaml`

Override settings on the CLI to avoid editing the files directly. Example:

```bash
python deploy/org_run.py --config deploy/configs/org.yaml \
    --index-path /mnt/rag/index \
    --log-dir /mnt/rag/logs \
    --api-key team-alpha --api-key team-beta \
    --rate-limit 200 --rate-burst 250
```

This launches the API with custom index/log directories and two API keys, while
raising rate limits for power users.

## 7. Load Testing

Use the bundled benchmark script to run a quick smoke test:

```bash
make load-test BASE_URL=http://localhost:9000 REQUESTS=40 CONCURRENCY=8
```

Inspect the percentile latencies and compare across scaling experiments. The
script uses HTTPX and respects the `/chat` endpoint contract.

---

For deeper automation and integration with CI/CD, see the updated workflow in
`.github/workflows/ci.yml`, which exercises both local and organisation modes.

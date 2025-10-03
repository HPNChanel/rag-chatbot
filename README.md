# RAG Chatbot Scaffold

A production-ready scaffold for building retrieval-augmented generation (RAG) chatbots. The project provides modular components for ingestion, indexing, retrieval, reranking, generation, evaluation, and interactive demos.

## Features

- **Data ingestion** for PDF, Markdown, and text documents.
- **Hybrid retrieval** combining BM25 and FAISS-based dense search with pure Python fallbacks when FAISS isn't available.
- **Coverage-aware reranking** with hooks for custom scoring logic.
- **LLM generation** via an OpenAI API wrapper with dummy generator for offline testing.
- **Dependency-free hashing embedder** that keeps tests and demos working without heavy ML packages.
- **Evaluation suite** covering retrieval (precision@k, recall@k, coverage) and generation (BLEU, ROUGE, human eval placeholder).
- **FastAPI server** exposing `/query` and `/chat` endpoints.
- **Streamlit UI** for quick experimentation.
- **Extensive tests** for core modules and end-to-end flow.

## Getting Started

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) or `pip`

### Installation (Poetry)

```bash
poetry install
```

### Installation (pip)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional Dependencies

- Install `faiss-cpu` and `numpy` to enable the FAISS-backed vector store.
- Install `sentence-transformers` and `scikit-learn` for higher-quality embeddings.
- Install `rank-bm25` for the production-ready lexical retriever (falls back to a simple scorer otherwise).
- Install `openai` (already included) when deploying with the OpenAI API.

### Environment Variables

Set the `OPENAI_API_KEY` environment variable to enable the OpenAI generator.

```bash
export OPENAI_API_KEY=your-key
```

## Usage

### Indexing and Chatting

The `RAGPipeline` orchestrates the full workflow.

```python
from pipelines.chatbot import RAGPipeline
from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder

pipeline = RAGPipeline(
    data_path="data/sample_docs",
    embedder=get_default_embedder(prefer_lightweight=True),
    generator=EchoGenerator(),
    use_faiss=False,
)

print(pipeline.chat("What is Python?"))
```

### FastAPI Server

```bash
uvicorn ui.server:app --reload
```

Endpoints:

- `POST /query` – Retrieve top documents along with candidate/reranked scores and latency metrics.
- `POST /chat` – Generate an answer that includes inline references plus a structured citation payload.

Each request is logged via middleware that records payloads and response times for easy observability.

### Streamlit Demo

```bash
streamlit run ui/streamlit_app.py
```

The UI provides a chat-style interface, a retrieval panel with highlighted matches, and a metrics sidebar summarising latency, reranker impact, and citation details. The app communicates with the FastAPI server, so ensure it is running locally or point the sidebar configuration to a deployed instance.

### One-command Demo

Use the bundled runner to start both services together:

```bash
make run
```

Both interfaces default to lightweight embeddings and the pure-Python vector store so they run without optional dependencies. Set `use_faiss=True` when constructing `RAGPipeline` to leverage FAISS if it is installed.

## Configuration

Configuration files live in `configs/`. Update `configs/default.yaml` to point to different datasets or tweak retrieval settings. The indexing section accepts a `use_faiss` flag so deployments can opt into FAISS when it is installed.

## Benchmarking & Reporting

The `experiments` package adds a reproducible benchmarking workflow:

- Run a baseline experiment on the bundled tiny dataset:

  ```bash
  ragx-run --config experiments/configs/baseline.yaml --repeat 3
  ```

- Launch a PRF ablation:

  ```bash
  ragx-ablations --suite experiments/configs/prf_grid.yaml --repeat 2
  ```

- Generate a markdown report for the latest run:

  ```bash
  make report
  ```

Example config snippet for a PRF sweep:

```yaml
retrieval:
  type: bm25
  top_k: 10
  prf:
    enabled: true
    feedback_k: 5
    strategy: sum
```

Example reranker weight sweep:

```yaml
reranking:
  enabled: true
  lambda_similarity: 0.6
  lambda_coverage: 0.3
```

To switch between the mock and OpenAI generators update the `generation.provider` field in your config (`mock` for offline tests, `openai` for production). The mock generator is deterministic and ideal for CI.

All experiments write to `runs/<timestamp>_<hash>/` with resolved configs, raw results, metrics, and plots. See `docs/EXPERIMENT_GUIDE.md` for best practices.

## Testing

```bash
make test
```

All tests run against the lightweight hashing embedder and the pure Python similarity search path, so they succeed even when FAISS/numpy are unavailable.

## Extensibility

- Implement new embedders by subclassing `BaseEmbedder` and plug them into `FaissVectorStore`. The provided `HashingEmbedder` offers a dependency-free starting point.
- Add rerankers by following the `CoverageReranker` interface.
- Extend evaluation metrics in `evaluation/` modules.
- Integrate custom UIs by reusing the `RAGPipeline` orchestration layer.

## Roadmap

- Add streaming responses from the LLM.
- Support structured document loaders (HTML, CSV).
- Integrate observability for latency and retrieval quality.

## License

MIT

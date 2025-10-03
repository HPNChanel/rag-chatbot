"""FastAPI server exposing chat and query endpoints with monitoring hooks."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from analysis.redaction import RedactionSettings, TextRedactor
from configs.loader import DeploymentConfig
from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder
from monitoring import MetricsStore
from monitoring.logging import configure_logging
from pipelines.chatbot import (
    ChatResponse,
    RAGPipeline,
    RetrievalResponse,
    RetrievalResult,
)


class TokenBucket:
    """Simple token bucket implementation for rate limiting."""

    def __init__(self, rate_per_minute: int, capacity: int | None = None) -> None:
        self.capacity = capacity or rate_per_minute
        self.tokens = float(self.capacity)
        self.rate_per_second = rate_per_minute / 60.0
        self.updated_at = time.perf_counter()

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.perf_counter()
        elapsed = now - self.updated_at
        self.updated_at = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class LoggingMiddleware(BaseHTTPMiddleware):
    """Record structured telemetry for each request."""

    def __init__(self, app: FastAPI, logger: logging.Logger) -> None:
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start = time.perf_counter()
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8") if body_bytes else ""
        except Exception:  # pragma: no cover - defensive guard
            body_text = "<unavailable>"
        self.logger.info(
            "incoming_request",
            extra={
                "extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "payload": body_text,
                }
            },
        )
        response = await call_next(request)
        duration = time.perf_counter() - start
        self.logger.info(
            "request_complete",
            extra={
                "extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": getattr(response, "status_code", "unknown"),
                    "duration_ms": round(duration * 1000, 3),
                }
            },
        )
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Guard endpoints behind API key authentication when configured."""

    def __init__(self, app: FastAPI, valid_keys: Iterable[str]) -> None:
        super().__init__(app)
        self.valid_keys = {key.strip() for key in valid_keys if key.strip()}

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.url.path in {"/healthz", "/metrics"}:
            return await call_next(request)
        key = request.headers.get("X-API-Key")
        if not key or key not in self.valid_keys:
            return Response(status_code=401, content="Unauthorized")
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic token bucket rate limiting."""

    def __init__(self, app: FastAPI, rate_per_minute: int, burst: int | None = None) -> None:
        super().__init__(app)
        self.rate = rate_per_minute
        self.burst = burst or rate_per_minute
        self.buckets: dict[str, TokenBucket] = {}

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        client_host = getattr(request.client, "host", None)
        identifier = request.headers.get("X-API-Key") or client_host or "anonymous"
        bucket = self.buckets.get(identifier)
        if bucket is None:
            bucket = self.buckets[identifier] = TokenBucket(self.rate, self.burst)
        if not bucket.consume():
            return Response(status_code=429, content="Rate limit exceeded")
        return await call_next(request)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class AppState:
    pipeline: RAGPipeline | None = None
    metrics: MetricsStore
    logger: logging.Logger
    redactor: TextRedactor | None
    config: DeploymentConfig | None


def _build_pipeline(dataset: Path, index_type: str) -> RAGPipeline:
    use_faiss = index_type.lower() == "faiss"
    return RAGPipeline(
        data_path=dataset,
        embedder=get_default_embedder(prefer_lightweight=not use_faiss),
        generator=EchoGenerator(),
        use_faiss=use_faiss,
    )


def create_app(
    data_path: str | Path = "data/sample_docs",
    *,
    config: DeploymentConfig | None = None,
    metrics_store: MetricsStore | None = None,
    logger: logging.Logger | None = None,
    redactor: TextRedactor | None = None,
) -> FastAPI:
    """Create the FastAPI application with monitoring and security hooks."""

    resolved_config = config or DeploymentConfig(
        mode="local",
        dataset=Path(os.environ.get("RAG_DATA_PATH", data_path)),
        index_type="faiss",
        api_port=8000,
        ui_port=8501,
        reranker="coverage",
    )

    logger = logger or configure_logging(
        resolved_config.logging.directory if resolved_config.logging else None
    )
    metrics = metrics_store or MetricsStore()
    app = FastAPI(title="RAG Chatbot")
    state = AppState()
    state.metrics = metrics
    state.logger = logger
    if redactor is None and resolved_config.enable_redaction:
        redactor = TextRedactor(RedactionSettings(enable_redaction=True))
    state.redactor = redactor
    state.config = resolved_config

    app.add_middleware(LoggingMiddleware, logger=logger)
    if resolved_config.api_keys:
        app.add_middleware(APIKeyMiddleware, valid_keys=resolved_config.api_keys)
    if resolved_config.rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            rate_per_minute=resolved_config.rate_limit.requests_per_minute,
            burst=resolved_config.rate_limit.burst,
        )

    @app.on_event("startup")
    async def startup_event() -> None:  # pragma: no cover - exercised in integration tests
        dataset = resolved_config.dataset
        pipeline_builder = lambda: _build_pipeline(dataset, resolved_config.index_type)
        state.pipeline = await asyncio.get_event_loop().run_in_executor(None, pipeline_builder)
        logger.info(
            "pipeline_ready",
            extra={
                "extra_data": {
                    "dataset": str(dataset),
                    "index_type": resolved_config.index_type,
                }
            },
        )

    @app.post("/query")
    async def query(request: QueryRequest) -> Dict[str, Any]:
        if state.pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        retrieval: RetrievalResponse = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: state.pipeline.query_with_details(
                request.query, top_k=request.top_k
            ),
        )
        state.metrics.record(
            latency_ms=retrieval.metrics.get("total_time", 0.0) * 1000,
            retrieval_hit_ratio=
            (
                len(retrieval.results)
                / max(1.0, retrieval.metrics.get("candidate_count", len(retrieval.results)))
            ),
            coverage_score=retrieval.metrics.get("reranker_impact", 0.0),
        )
        _record_query_log(state, request.query)
        return {
            "query": retrieval.query,
            "documents": [
                _serialise_result(result) for result in retrieval.results
            ],
            "metrics": retrieval.metrics,
        }

    @app.post("/chat")
    async def chat(request: ChatRequest) -> Dict[str, Any]:
        if state.pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        chat_payload: ChatResponse = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: state.pipeline.chat_with_details(
                request.query, top_k=request.top_k
            ),
        )
        answer = _append_references(chat_payload.answer, chat_payload.citations)
        if state.redactor and state.config and state.config.enable_redaction:
            answer = state.redactor.redact(answer)
            chat_payload.citations = [
                {
                    **citation,
                    "excerpt": state.redactor.redact(citation.get("excerpt", "")),
                }
                for citation in chat_payload.citations
            ]
        state.metrics.record(
            latency_ms=chat_payload.metrics.get("end_to_end_time", 0.0) * 1000,
            retrieval_hit_ratio=
            (
                len(chat_payload.retrieval.results)
                / max(
                    1.0,
                    chat_payload.retrieval.metrics.get(
                        "candidate_count", len(chat_payload.retrieval.results)
                    ),
                )
            ),
            coverage_score=chat_payload.retrieval.metrics.get("reranker_impact", 0.0),
        )
        _record_query_log(state, request.query)
        return {
            "query": chat_payload.query,
            "answer": answer,
            "citations": chat_payload.citations,
            "documents": [
                _serialise_result(result) for result in chat_payload.retrieval.results
            ],
            "metrics": chat_payload.metrics,
        }

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:  # pragma: no cover - trivial
        status = "ok" if state.pipeline is not None else "starting"
        return {"status": status}

    @app.get("/metrics")
    async def metrics_endpoint():
        return state.metrics.as_dict()

    return app


app = create_app()


def _serialise_result(result: RetrievalResult) -> Dict[str, Any]:
    document = result.document
    return {
        "doc_id": document.doc_id,
        "content": document.content,
        "metadata": document.metadata,
        "scores": {
            "candidate": result.candidate_score,
            "reranked": result.reranked_score,
        },
        "ranks": {
            "candidate": result.candidate_rank,
            "reranked": result.reranked_rank,
        },
    }


def _append_references(answer: str, citations: list[dict[str, str]]) -> str:
    if not citations:
        return answer
    references = []
    for citation in citations:
        title = citation.get("title") or citation.get("doc_id", "unknown")
        source = citation.get("source")
        if source:
            references.append(f"- {title} ({source})")
        else:
            references.append(f"- {title}")
    references_text = "\n".join(references)
    if "References:" in answer:
        return answer
    return f"{answer}\n\nReferences:\n{references_text}"


def _record_query_log(state: AppState, query: str) -> None:
    config = state.config
    if not config or not config.logging:
        return
    log_dir = config.logging.directory
    log_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    payload = {
        "timestamp": time.time(),
        "query_id": digest,
        "length": len(query),
    }
    log_path = log_dir / "queries.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

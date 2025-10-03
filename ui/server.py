"""FastAPI server exposing chat and query endpoints."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder
from pipelines.chatbot import ChatResponse, RAGPipeline, RetrievalResponse, RetrievalResult


logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Record basic telemetry for each request."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        start = time.perf_counter()
        try:
            body_bytes = await request.body()
            body_text = body_bytes.decode("utf-8") if body_bytes else ""
        except Exception:  # pragma: no cover - defensive guard
            body_text = "<unavailable>"
        logger.info("%s %s payload=%s", request.method, request.url.path, body_text)
        response = await call_next(request)
        duration = time.perf_counter() - start
        logger.info(
            "%s %s completed status=%s duration=%.3fs",
            request.method,
            request.url.path,
            getattr(response, "status_code", "unknown"),
            duration,
        )
        return response


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class AppState:
    pipeline: RAGPipeline | None = None


def create_app(data_path: str | Path = "data/sample_docs") -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    base_path = Path(os.environ.get("RAG_DATA_PATH", data_path))
    app = FastAPI(title="RAG Chatbot")
    state = AppState()
    app.add_middleware(LoggingMiddleware)

    @app.on_event("startup")
    async def startup_event() -> None:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: setattr(
                state,
                "pipeline",
                RAGPipeline(
                    data_path=base_path,
                    embedder=get_default_embedder(prefer_lightweight=True),
                    generator=EchoGenerator(),
                    use_faiss=False,
                ),
            ),
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
        return {
            "query": chat_payload.query,
            "answer": answer,
            "citations": chat_payload.citations,
            "documents": [
                _serialise_result(result) for result in chat_payload.retrieval.results
            ],
            "metrics": chat_payload.metrics,
        }

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

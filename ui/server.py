"""FastAPI server exposing chat and query endpoints."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder
from pipelines.chatbot import RAGPipeline


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5


class AppState:
    pipeline: RAGPipeline | None = None


def create_app(data_path: str | Path = "data/sample_docs") -> FastAPI:
    app = FastAPI(title="RAG Chatbot")
    state = AppState()

    @app.on_event("startup")
    async def startup_event() -> None:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: setattr(
                state,
                "pipeline",
                RAGPipeline(
                    data_path=data_path,
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
        documents = await asyncio.get_event_loop().run_in_executor(
            None, lambda: state.pipeline.query(request.query, top_k=request.top_k)
        )
        return {"documents": [{"id": doc.doc_id, "content": doc.content} for doc in documents]}

    @app.post("/chat")
    async def chat(request: ChatRequest) -> Dict[str, Any]:
        if state.pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not ready")
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: state.pipeline.chat(request.query, top_k=request.top_k)
        )
        return {"response": response}

    return app


app = create_app()

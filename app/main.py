from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.llm_client import LLMClient
from app.rag import ChatMessage, RAGPipeline
from app.security import is_authorized_api_request
from app.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)

app = FastAPI(title="Knowledge Base Assistant", version="0.1.0")

INDEX_HTML_PATH = Path(__file__).with_name("static") / "index.html"
INDEX_HTML_FALLBACK = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Knowledge Base Assistant</title>
</head>
<body>
  <main>
    <h1>Knowledge Base Assistant</h1>
    <p>The web UI file is missing. Check app/static/index.html.</p>
  </main>
</body>
</html>
"""


class ChatMessageRequest(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(
        ...,
        min_length=1,
        max_length=settings.api_message_max_chars,
    )


class RAGRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=settings.api_question_max_chars,
    )
    top_k: int | None = Field(default=None, ge=1, le=settings.api_top_k_max)
    history: list[ChatMessageRequest] = Field(
        default_factory=list,
        max_length=settings.api_history_max_messages,
    )
    conversation_summary: str | None = Field(
        default=None,
        max_length=settings.api_summary_max_chars,
    )


class ContextResponse(BaseModel):
    text: str
    source: str
    chunk_id: int
    score: float
    content_type: str
    h1: str
    h2: str
    h3: str
    headings: list[str]
    start_line: int
    end_line: int

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "ContextResponse":
        return cls(
            text=result.text,
            source=result.source,
            chunk_id=result.chunk_id,
            score=result.score,
            content_type=result.content_type,
            h1=result.h1,
            h2=result.h2,
            h3=result.h3,
            headings=list(result.headings),
            start_line=result.start_line,
            end_line=result.end_line,
        )


class RAGResponse(BaseModel):
    answer: str
    contexts: list[ContextResponse]
    conversation_summary: str
    compacted_history_messages: int
    used_rag: bool
    route: str
    route_reason: str


vector_store = VectorStore(settings)
llm_client = LLMClient(settings)
rag_pipeline = RAGPipeline(
    settings,
    vector_store=vector_store,
    llm_client=llm_client,
)


def require_api_auth(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if is_authorized_api_request(settings, authorization):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if INDEX_HTML_PATH.exists():
        return INDEX_HTML_PATH.read_text(encoding="utf-8")
    return INDEX_HTML_FALLBACK


@app.on_event("shutdown")
def shutdown_resources() -> None:
    llm_client.close()


@app.get("/health", dependencies=[Depends(require_api_auth)])
async def health() -> dict[str, object]:
    qdrant_ok, llm_ok = await asyncio.gather(
        asyncio.to_thread(_qdrant_health),
        asyncio.to_thread(llm_client.health),
    )

    return {
        "status": "ok" if qdrant_ok and llm_ok else "degraded",
        "qdrant": qdrant_ok,
        "llm": llm_ok,
        "collection": settings.collection_name,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "llm_max_tokens": settings.llm_max_tokens,
        "retrieve_top_k": settings.retrieve_top_k,
        "history_recent_turns": settings.history_recent_turns,
        "api_top_k_max": settings.api_top_k_max,
        "intent_router": settings.intent_router_enabled,
    }


@app.post(
    "/rag",
    response_model=RAGResponse,
    dependencies=[Depends(require_api_auth)],
)
async def rag(request: RAGRequest) -> RAGResponse:
    history = [
        ChatMessage(role=item.role, content=item.content)
        for item in request.history
    ]
    try:
        result = await asyncio.to_thread(
            rag_pipeline.answer,
            request.question,
            top_k=request.top_k,
            history=history,
            conversation_summary=request.conversation_summary,
        )
    except UnexpectedResponse as exc:
        logger.exception("Vector store request failed during RAG.")
        raise HTTPException(
            status_code=502,
            detail=_error_detail("Upstream vector store error.", exc),
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("LLM provider request failed during RAG.")
        raise HTTPException(
            status_code=502,
            detail=_error_detail("Upstream LLM provider error.", exc),
        ) from exc
    except Exception as exc:
        logger.exception("RAG request failed.")
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Internal server error.", exc),
        ) from exc

    return RAGResponse(
        answer=result.answer,
        contexts=[
            ContextResponse.from_search_result(item) for item in result.contexts
        ],
        conversation_summary=result.conversation_summary,
        compacted_history_messages=result.compacted_history_messages,
        used_rag=result.used_rag,
        route=result.route,
        route_reason=result.route_reason,
    )


def _qdrant_health() -> bool:
    try:
        vector_store.client.get_collections()
    except Exception:
        logger.exception("Qdrant health check failed.")
        return False
    return True


def _error_detail(message: str, exc: Exception) -> str:
    if settings.debug:
        return str(exc)
    return message

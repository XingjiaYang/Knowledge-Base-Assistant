from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import csv
from io import StringIO
import logging
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.llm_client import LLMClient
from app.rag import RAGPipeline
from app.reranker import Reranker
from app.security import bearer_token
from app.session_store import (
    ChatSessionRecord,
    CurrentUser,
    SessionStore,
    StoredChatMessage,
    UserRecord,
)
from app.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.vector_store = VectorStore(settings)
    _app.state.llm_client = LLMClient(settings)
    _app.state.session_store = SessionStore(settings)
    _app.state.reranker = Reranker(settings)
    if settings.reranker_enabled and settings.reranker_preload:
        await asyncio.to_thread(_app.state.reranker.warmup)
    await asyncio.to_thread(_app.state.session_store.init_db)
    _app.state.rag_pipeline = RAGPipeline(
        settings,
        vector_store=_app.state.vector_store,
        llm_client=_app.state.llm_client,
        reranker=_app.state.reranker,
    )
    try:
        yield
    finally:
        _app.state.llm_client.close()
        _app.state.vector_store.close()
        if _app.state.session_store is not None:
            _app.state.session_store.close()


app = FastAPI(
    title="Knowledge Base Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

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
INDEX_HTML = (
    INDEX_HTML_PATH.read_text(encoding="utf-8")
    if INDEX_HTML_PATH.exists()
    else INDEX_HTML_FALLBACK
)

STATIC_DIR = INDEX_HTML_PATH.parent
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
    session_id: UUID | None = None
    top_k: int | None = Field(default=None, ge=1, le=settings.api_top_k_max)
    recall_top_k: int | None = Field(
        default=None,
        ge=1,
        le=settings.api_recall_top_k_max,
    )
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
    rerank_score: float | None = None
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
            rerank_score=result.rerank_score,
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
    session_id: str | None = None
    contexts: list[ContextResponse]
    conversation_summary: str
    compacted_history_messages: int
    used_rag: bool
    route: str
    route_reason: str


class UserResponse(BaseModel):
    id: str
    username: str
    is_admin: bool
    must_change_password: bool

    @classmethod
    def from_user(cls, user: CurrentUser) -> "UserResponse":
        return cls(
            id=str(user.id),
            username=user.username,
            is_admin=user.is_admin,
            must_change_password=user.must_change_password,
        )


class AuthConfigResponse(BaseModel):
    auth_enabled: bool


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1, max_length=256)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=1, max_length=256)


class LoginResponse(BaseModel):
    token: str
    user: UserResponse


class AdminUserResponse(BaseModel):
    id: str
    username: str
    is_active: bool
    is_admin: bool
    must_change_password: bool
    created_at: str
    updated_at: str
    last_login_at: str | None
    session_count: int
    message_count: int

    @classmethod
    def from_user_record(cls, user: UserRecord) -> "AdminUserResponse":
        return cls(
            id=str(user.id),
            username=user.username,
            is_active=user.is_active,
            is_admin=user.is_admin,
            must_change_password=user.must_change_password,
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat(),
            last_login_at=user.last_login_at.isoformat()
            if user.last_login_at
            else None,
            session_count=user.session_count,
            message_count=user.message_count,
        )


class AdminUserCreateRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1, max_length=256)
    is_admin: bool = False


class AdminPasswordUpdateRequest(BaseModel):
    password: str = Field(..., min_length=1, max_length=256)


class AdminRoleUpdateRequest(BaseModel):
    is_admin: bool


class AdminUsersCsvImportRequest(BaseModel):
    csv_text: str = Field(..., min_length=1, max_length=1_000_000)


class AdminUsersCsvImportResponse(BaseModel):
    created: int
    users: list[AdminUserResponse]


class SessionRenameRequest(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=settings.session_title_max_chars,
    )


class SessionSummaryResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

    @classmethod
    def from_session(cls, session: ChatSessionRecord) -> "SessionSummaryResponse":
        return cls(
            id=str(session.id),
            title=session.title,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
        )


class StoredMessageResponse(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    contexts: list[ContextResponse]
    used_rag: bool | None = None
    route: str = ""
    route_reason: str = ""
    created_at: str

    @classmethod
    def from_message(cls, message: StoredChatMessage) -> "StoredMessageResponse":
        return cls(
            id=message.id,
            role=message.role,
            content=message.content,
            contexts=[ContextResponse(**context) for context in message.contexts],
            used_rag=message.used_rag,
            route=message.route,
            route_reason=message.route_reason,
            created_at=message.created_at.isoformat(),
        )


class SessionDetailResponse(SessionSummaryResponse):
    conversation_summary: str
    compacted_message_count: int
    messages: list[StoredMessageResponse]

    @classmethod
    def from_session_detail(
        cls,
        session: ChatSessionRecord,
        messages: list[StoredChatMessage],
    ) -> "SessionDetailResponse":
        return cls(
            id=str(session.id),
            title=session.title,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            conversation_summary=session.conversation_summary,
            compacted_message_count=session.compacted_message_count,
            messages=[
                StoredMessageResponse.from_message(message)
                for message in messages
            ],
        )


def require_login_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> CurrentUser:
    token = bearer_token(authorization)
    if token is None:
        raise_unauthorized()
    user = _session_store(request).get_user_by_token(token)
    if user is None:
        raise_unauthorized()
    return user


def require_password_ready_user(
    user: Annotated[CurrentUser, Depends(require_login_auth)],
) -> CurrentUser:
    if user.must_change_password:
        raise HTTPException(status_code=403, detail="Password change required.")
    return user


def require_admin_auth(
    user: Annotated[CurrentUser, Depends(require_password_ready_user)],
) -> CurrentUser:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


def raise_unauthorized() -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok"}


@app.get("/auth/config", response_model=AuthConfigResponse)
def auth_config() -> AuthConfigResponse:
    return AuthConfigResponse(auth_enabled=True)


@app.post("/auth/login", response_model=LoginResponse)
async def login(request: Request, login_request: LoginRequest) -> LoginResponse:
    session_store = _session_store(request)
    try:
        user = await asyncio.to_thread(
            session_store.authenticate_user,
            login_request.username,
            login_request.password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = await asyncio.to_thread(
        session_store.create_auth_session,
        user.id,
        request.headers.get("user-agent", ""),
        request.client.host if request.client else "",
    )
    return LoginResponse(token=token, user=UserResponse.from_user(user))


@app.get("/auth/me", response_model=UserResponse)
def current_user(
    user: Annotated[CurrentUser, Depends(require_login_auth)],
) -> UserResponse:
    return UserResponse.from_user(user)


@app.post("/auth/logout")
async def logout(
    request: Request,
    _user: Annotated[CurrentUser, Depends(require_login_auth)],
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, object]:
    token = bearer_token(authorization)
    if token:
        await asyncio.to_thread(_session_store(request).delete_auth_session, token)
    return {"status": "ok"}


@app.post("/auth/password", response_model=UserResponse)
async def change_password(
    request: Request,
    password_request: PasswordChangeRequest,
    user: Annotated[CurrentUser, Depends(require_login_auth)],
) -> UserResponse:
    try:
        updated = await asyncio.to_thread(
            _session_store(request).change_user_password,
            user.id,
            password_request.current_password,
            password_request.new_password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserResponse.from_user(
        CurrentUser(
            id=user.id,
            username=user.username,
            is_admin=user.is_admin,
            must_change_password=False,
        )
    )


@app.get("/admin/users", response_model=list[AdminUserResponse])
async def admin_list_users(
    request: Request,
    _admin: Annotated[CurrentUser, Depends(require_admin_auth)],
) -> list[AdminUserResponse]:
    users = await asyncio.to_thread(_session_store(request).list_users)
    return [AdminUserResponse.from_user_record(user) for user in users]


@app.post("/admin/users", response_model=AdminUserResponse)
async def admin_create_user(
    request: Request,
    create_request: AdminUserCreateRequest,
    _admin: Annotated[CurrentUser, Depends(require_admin_auth)],
) -> AdminUserResponse:
    try:
        user = await asyncio.to_thread(
            _session_store(request).create_user,
            create_request.username,
            create_request.password,
            create_request.is_admin,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AdminUserResponse.from_user_record(user)


@app.post("/admin/users/import-csv", response_model=AdminUsersCsvImportResponse)
async def admin_import_users_csv(
    request: Request,
    import_request: AdminUsersCsvImportRequest,
    _admin: Annotated[CurrentUser, Depends(require_admin_auth)],
) -> AdminUsersCsvImportResponse:
    try:
        users = parse_user_csv(import_request.csv_text)
        created_users = await asyncio.to_thread(
            _session_store(request).create_users,
            users,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AdminUsersCsvImportResponse(
        created=len(created_users),
        users=[
            AdminUserResponse.from_user_record(user)
            for user in created_users
        ],
    )


@app.post("/admin/users/{user_id}/password")
async def admin_set_user_password(
    user_id: UUID,
    request: Request,
    password_request: AdminPasswordUpdateRequest,
    _admin: Annotated[CurrentUser, Depends(require_admin_auth)],
) -> dict[str, object]:
    try:
        updated = await asyncio.to_thread(
            _session_store(request).set_user_password,
            user_id,
            password_request.password,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"status": "ok"}


@app.patch("/admin/users/{user_id}/role")
async def admin_set_user_role(
    user_id: UUID,
    request: Request,
    role_request: AdminRoleUpdateRequest,
    _admin: Annotated[CurrentUser, Depends(require_admin_auth)],
) -> dict[str, object]:
    try:
        updated = await asyncio.to_thread(
            _session_store(request).set_user_admin,
            user_id,
            role_request.is_admin,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"status": "ok"}


@app.delete("/admin/users/{user_id}/sessions")
async def admin_clear_user_sessions(
    user_id: UUID,
    request: Request,
    _admin: Annotated[CurrentUser, Depends(require_admin_auth)],
) -> dict[str, object]:
    deleted = await asyncio.to_thread(
        _session_store(request).clear_user_sessions,
        user_id,
    )
    return {"status": "ok", "deleted_sessions": deleted}


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(
    user_id: UUID,
    request: Request,
    admin: Annotated[CurrentUser, Depends(require_admin_auth)],
) -> dict[str, object]:
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account.")

    try:
        deleted = await asyncio.to_thread(_session_store(request).delete_user, user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"status": "ok"}


@app.get("/sessions", response_model=list[SessionSummaryResponse])
async def list_sessions(
    request: Request,
    user: Annotated[CurrentUser, Depends(require_password_ready_user)],
) -> list[SessionSummaryResponse]:
    sessions = await asyncio.to_thread(
        _session_store(request).list_chat_sessions,
        user.id,
    )
    return [SessionSummaryResponse.from_session(session) for session in sessions]


@app.post("/sessions", response_model=SessionDetailResponse)
async def create_session(
    request: Request,
    user: Annotated[CurrentUser, Depends(require_password_ready_user)],
) -> SessionDetailResponse:
    session = await asyncio.to_thread(
        _session_store(request).create_chat_session,
        user.id,
    )
    return SessionDetailResponse.from_session_detail(session, [])


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: UUID,
    request: Request,
    user: Annotated[CurrentUser, Depends(require_password_ready_user)],
) -> SessionDetailResponse:
    session_store = _session_store(request)
    session = await asyncio.to_thread(
        session_store.get_chat_session,
        user.id,
        session_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = await asyncio.to_thread(session_store.list_messages, session.id)
    return SessionDetailResponse.from_session_detail(session, messages)


@app.patch("/sessions/{session_id}", response_model=SessionSummaryResponse)
async def rename_session(
    session_id: UUID,
    request: Request,
    rename_request: SessionRenameRequest,
    user: Annotated[CurrentUser, Depends(require_password_ready_user)],
) -> SessionSummaryResponse:
    session = await asyncio.to_thread(
        _session_store(request).rename_chat_session,
        user.id,
        session_id,
        rename_request.title,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionSummaryResponse.from_session(session)


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    request: Request,
    user: Annotated[CurrentUser, Depends(require_password_ready_user)],
) -> dict[str, object]:
    deleted = await asyncio.to_thread(
        _session_store(request).delete_chat_session,
        user.id,
        session_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "ok"}


@app.get("/health/details", dependencies=[Depends(require_password_ready_user)])
async def health_details(request: Request) -> dict[str, object]:
    qdrant_ok, llm_ok = await asyncio.gather(
        asyncio.to_thread(_qdrant_health, request.app.state.vector_store),
        asyncio.to_thread(request.app.state.llm_client.health),
    )

    return {
        "status": "ok" if qdrant_ok and llm_ok else "degraded",
        "qdrant": qdrant_ok,
        "llm": llm_ok,
        "collection": settings.collection_name,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "llm_max_tokens": settings.llm_max_tokens,
        "cuda_enabled": settings.cuda_enabled,
        "recall_top_k": settings.recall_top_k,
        "retrieve_top_k": settings.retrieve_top_k,
        "retrieve_score_threshold": settings.retrieve_score_threshold,
        "history_recent_turns": settings.history_recent_turns,
        "api_top_k_max": settings.api_top_k_max,
        "api_recall_top_k_max": settings.api_recall_top_k_max,
        "reranker_enabled": settings.reranker_enabled,
        "reranker_preload": settings.reranker_preload,
        "reranker_model": settings.reranker_model,
        "intent_router": settings.intent_router_enabled,
        "auth_enabled": True,
    }


@app.post(
    "/rag",
    response_model=RAGResponse,
)
async def rag(
    http_request: Request,
    rag_request: RAGRequest,
    user: Annotated[CurrentUser, Depends(require_password_ready_user)],
) -> RAGResponse:
    session_store = _session_store(http_request)
    session = await _get_or_create_chat_session(
        session_store,
        user,
        rag_request.session_id,
    )
    history = await asyncio.to_thread(
        session_store.prompt_history,
        session.id,
        session.compacted_message_count,
    )
    conversation_summary = session.conversation_summary

    try:
        result = await asyncio.to_thread(
            http_request.app.state.rag_pipeline.answer,
            rag_request.question,
            top_k=rag_request.top_k,
            recall_top_k=rag_request.recall_top_k,
            history=history,
            conversation_summary=conversation_summary,
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

    await asyncio.to_thread(
        session_store.append_message,
        session.id,
        "user",
        rag_request.question,
    )
    await asyncio.to_thread(
        session_store.append_message,
        session.id,
        "assistant",
        result.answer,
        result.contexts,
        result.used_rag,
        result.route,
        result.route_reason,
    )
    session = await asyncio.to_thread(
        session_store.update_chat_session_after_answer,
        session.id,
        result.conversation_summary,
        result.compacted_history_messages,
        rag_request.question,
    )

    return RAGResponse(
        answer=result.answer,
        session_id=str(session.id),
        contexts=[
            ContextResponse.from_search_result(item) for item in result.contexts
        ],
        conversation_summary=result.conversation_summary,
        compacted_history_messages=result.compacted_history_messages,
        used_rag=result.used_rag,
        route=result.route,
        route_reason=result.route_reason,
    )


def _qdrant_health(vector_store: VectorStore) -> bool:
    try:
        vector_store.client.get_collections()
    except Exception:
        logger.exception("Qdrant health check failed.")
        return False
    return True


async def _get_or_create_chat_session(
    session_store: SessionStore,
    user: CurrentUser,
    session_id: UUID | None,
) -> ChatSessionRecord:
    if session_id is None:
        return await asyncio.to_thread(session_store.create_chat_session, user.id)

    session = await asyncio.to_thread(
        session_store.get_chat_session,
        user.id,
        session_id,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


def _session_store(request: Request) -> SessionStore:
    session_store = getattr(request.app.state, "session_store", None)
    if session_store is None:
        raise HTTPException(status_code=503, detail="Account storage is unavailable.")
    return session_store


def parse_user_csv(csv_text: str) -> list[tuple[str, str]]:
    reader = csv.reader(StringIO(csv_text), skipinitialspace=True)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("CSV must include a header row: email,passwd.") from exc

    if header:
        header[0] = header[0].lstrip("\ufeff")
    if header != ["email", "passwd"]:
        raise ValueError("CSV header must be exactly: email,passwd.")

    users: list[tuple[str, str]] = []
    for line_number, row in enumerate(reader, start=2):
        if len(row) != 2:
            raise ValueError(
                f"CSV row {line_number} must contain exactly two columns."
            )

        email = row[0].strip()
        password = row[1].strip()
        if not email or not password:
            raise ValueError(
                f"CSV row {line_number} must include both email and passwd."
            )
        users.append((email, password))

    if not users:
        raise ValueError("CSV must include at least one user row.")
    return users


def _error_detail(message: str, exc: Exception) -> str:
    if settings.debug:
        return str(exc)
    return message

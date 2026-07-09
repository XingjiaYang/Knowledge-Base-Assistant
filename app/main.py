from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import csv
from io import StringIO
import logging
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import UnexpectedResponse

from app.call_graph import CallGraphStore
from app.config import settings
from app.code_indexer import CodeIndexer, CodeRepository, discover_code_repositories
from app.code_retrieval import (
    CodeIndexUnavailable,
    CodeRetrieval,
    CodeSearchHit,
    CodeFileHit,
)
from app.llm_client import LLMClient
from app.model_actors import warmup_model_actors
from app.rag import RAGPipeline, RAGTimings
from app.reranker import Reranker
from app.security import bearer_token
from app.session_store import (
    ChatSessionRecord,
    CurrentUser,
    SessionStore,
    StoredChatMessage,
    LLMSettingsRecord,
    UserRecord,
)
from app.vector_store import SearchResult, VectorStore


logger = logging.getLogger(__name__)


async def _warmup_model_actors_background() -> None:
    try:
        await asyncio.to_thread(warmup_model_actors, settings)
    except Exception:
        logger.exception(
            "Startup degraded: Ray model actor warmup failed; "
            "falling back to lazy or local model loading when needed."
        )


async def _warmup_code_embedding_background(code_retrieval: CodeRetrieval) -> None:
    for attempt in range(1, settings.code_embedding_preload_retries + 1):
        try:
            vector_size = await asyncio.to_thread(code_retrieval.warmup_embedder)
            logger.info(
                "Code embedding model %s warmed up with vector_size=%s.",
                settings.code_embedding_model,
                vector_size,
            )
            return
        except Exception as exc:
            if attempt >= settings.code_embedding_preload_retries:
                logger.exception(
                    "Startup degraded: code_embedding_degraded=True "
                    "model=%s fallback=text_only_code_search.",
                    settings.code_embedding_model,
                )
                return
            logger.warning(
                "Code embedding warmup failed; retrying attempt %s/%s in %.1fs: %s",
                attempt + 1,
                settings.code_embedding_preload_retries,
                settings.code_embedding_preload_retry_seconds,
                exc,
            )
            await asyncio.sleep(settings.code_embedding_preload_retry_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _app.state.vector_store = VectorStore(settings)
    _app.state.session_store = SessionStore(settings)
    await asyncio.to_thread(_app.state.session_store.init_db)
    _app.state.llm_client = LLMClient(
        settings,
        runtime_settings_provider=_app.state.session_store.get_llm_runtime_settings,
    )
    if settings.ray_enabled:
        _app.state.model_actor_warmup_task = asyncio.create_task(
            _warmup_model_actors_background()
        )
    _app.state.reranker = Reranker(settings)
    if (
        settings.reranker_enabled
        and settings.reranker_preload
        and not settings.ray_enabled
    ):
        try:
            await asyncio.to_thread(_app.state.reranker.warmup)
        except Exception:
            logger.exception(
                "Startup degraded: retrieval_degraded=True "
                "qdrant_degraded=False reranker_degraded=True "
                "fallback=rrf_or_bm25_top_k."
            )
    _app.state.rag_pipeline = RAGPipeline(
        settings,
        vector_store=_app.state.vector_store,
        llm_client=_app.state.llm_client,
        reranker=_app.state.reranker,
    )
    _app.state.code_retrieval = CodeRetrieval(settings)
    if settings.code_embedding_preload and not settings.ray_enabled:
        _app.state.code_embedding_warmup_task = asyncio.create_task(
            _warmup_code_embedding_background(_app.state.code_retrieval)
        )
    _app.state.call_graph = CallGraphStore(settings)
    _app.state.code_index_lock = asyncio.Lock()
    try:
        yield
    finally:
        for task_name in ("model_actor_warmup_task", "code_embedding_warmup_task"):
            task = getattr(_app.state, task_name, None)
            if task is not None and not task.done():
                task.cancel()
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


@app.middleware("http")
async def no_store_static_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


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
    bm25_top_k: int | None = Field(
        default=None,
        ge=1,
        le=settings.api_recall_top_k_max,
    )
    recall_top_k: int | None = Field(
        default=None,
        ge=1,
        le=settings.api_recall_top_k_max,
    )
    rrf_top_k: int | None = Field(
        default=None,
        ge=1,
        le=settings.api_recall_top_k_max,
    )
    rag_only: bool = False
    history: list[ChatMessageRequest] = Field(
        default_factory=list,
        max_length=settings.api_history_max_messages,
    )
    conversation_summary: str | None = Field(
        default=None,
        max_length=settings.api_summary_max_chars,
    )


class CodeSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=settings.api_question_max_chars,
    )
    session_id: UUID | None = None
    repository_ids: list[str] = Field(default_factory=list)
    file_top_k: int | None = Field(
        default=None,
        ge=1,
        le=settings.api_recall_top_k_max,
    )
    function_top_k: int | None = Field(
        default=None,
        ge=1,
        le=settings.api_recall_top_k_max,
    )
    final_top_k: int | None = Field(default=None, ge=1, le=settings.api_top_k_max)


class CodeFileHitResponse(BaseModel):
    file_id: str
    repository_id: str
    repository_name: str
    source_root: str
    path: str
    language: str
    score: float

    @classmethod
    def from_hit(cls, hit: CodeFileHit) -> "CodeFileHitResponse":
        return cls(
            file_id=str(hit.file_id),
            repository_id=hit.repository_id,
            repository_name=hit.repository_name,
            source_root=hit.source_root,
            path=hit.path,
            language=hit.language,
            score=hit.score,
        )


class CodeFunctionHitResponse(BaseModel):
    function_id: str
    file_id: str
    repository_id: str
    repository_name: str
    source_root: str
    path: str
    language: str
    name: str
    qualified_name: str
    kind: str
    signature: str
    docstring: str
    snippet: str
    start_line: int
    end_line: int
    score: float
    vector_score: float
    file_score: float | None = None
    rerank_score: float | None = None

    @classmethod
    def from_hit(cls, hit: CodeSearchHit) -> "CodeFunctionHitResponse":
        return cls(
            function_id=str(hit.function_id),
            file_id=str(hit.file_id),
            repository_id=hit.repository_id,
            repository_name=hit.repository_name,
            source_root=hit.source_root,
            path=hit.path,
            language=hit.language,
            name=hit.name,
            qualified_name=hit.qualified_name,
            kind=hit.kind,
            signature=hit.signature,
            docstring=hit.docstring,
            snippet=hit.snippet,
            start_line=hit.start_line,
            end_line=hit.end_line,
            score=hit.score,
            vector_score=hit.vector_score,
            file_score=hit.file_score,
            rerank_score=hit.rerank_score,
        )


class CodeSearchResponse(BaseModel):
    query: str
    session_id: str | None = None
    answer: str = ""
    retrieval_mode: str = "codebert_vector"
    code_embedding_degraded: bool = False
    degradation_reason: str = ""
    files: list[CodeFileHitResponse]
    functions: list[CodeFunctionHitResponse]
    contexts: list["ContextResponse"] = Field(default_factory=list)
    graph: "CodeCallGraphResponse | None" = None


class CodeRepositoryResponse(BaseModel):
    id: str
    name: str
    source_root: str
    indexed_files: int


class CodeRepositoriesResponse(BaseModel):
    repositories: list[CodeRepositoryResponse]
    default_repository_ids: list[str]


class CodeIndexRequest(BaseModel):
    repository_ids: list[str] = Field(default_factory=list)
    rebuild: bool = True


class CodeIndexResponse(BaseModel):
    repositories: list[CodeRepositoryResponse]
    indexed_repository_ids: list[str]
    files: int
    functions: int
    call_edges: int


class CodeGraphNodeResponse(BaseModel):
    id: str
    label: str
    name: str
    kind: str
    type: Literal["file", "function", "class", "component"] = "function"
    filePath: str = ""
    codeSnippet: str = ""
    description: str = ""
    path: str
    repository_id: str = ""
    repository_name: str = ""
    start_line: int
    end_line: int
    indexed: bool


class CodeGraphEdgeResponse(BaseModel):
    source: str
    target: str
    path: str
    lines: list[int]


class CodeGraphElementResponse(BaseModel):
    group: Literal["nodes", "edges"]
    data: dict[str, Any]


class CodeGraphSourceResponse(BaseModel):
    path: str
    url: str = ""
    content: str = ""


class CodeGraphTodoResponse(BaseModel):
    id: str
    title: str
    description: str = ""
    status: Literal["pending", "in-progress", "completed", "error"]
    result: str = ""


class CodeCallGraphResponse(BaseModel):
    function_name: str
    depth: int
    callers: list[CodeGraphNodeResponse]
    nodes: list[CodeGraphNodeResponse]
    edges: list[CodeGraphEdgeResponse]
    elements: list[CodeGraphElementResponse] = Field(default_factory=list)
    loading: bool = False
    analysing: bool = False
    queries: list[str] = Field(default_factory=list)
    sources: list[CodeGraphSourceResponse] = Field(default_factory=list)
    todos: list[CodeGraphTodoResponse] = Field(default_factory=list)


class ContextResponse(BaseModel):
    text: str
    source: str
    chunk_id: int
    score: float
    rerank_score: float | None = None
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    retrieval_source: str = "vector"
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
            vector_score=result.vector_score,
            bm25_score=result.bm25_score,
            rrf_score=result.rrf_score,
            retrieval_source=result.retrieval_source,
            content_type=result.content_type,
            h1=result.h1,
            h2=result.h2,
            h3=result.h3,
            headings=list(result.headings),
            start_line=result.start_line,
            end_line=result.end_line,
        )


class TimingResponse(BaseModel):
    total_ms: float = 0.0
    history_ms: float = 0.0
    intent_ms: float = 0.0
    retrieval_ms: float = 0.0
    recall_ms: float = 0.0
    bm25_ms: float = 0.0
    vector_ms: float = 0.0
    embedding_ms: float = 0.0
    qdrant_ms: float = 0.0
    rrf_ms: float = 0.0
    reranker_ms: float = 0.0
    llm_ms: float = 0.0
    llm_ttft_ms: float | None = None
    llm_output_chars: int = 0
    llm_estimated_output_tokens: int = 0
    llm_estimated_tps: float = 0.0

    @classmethod
    def from_timings(cls, timings: RAGTimings) -> "TimingResponse":
        return cls(
            total_ms=timings.total_ms,
            history_ms=timings.history_ms,
            intent_ms=timings.intent_ms,
            retrieval_ms=timings.retrieval_ms,
            recall_ms=timings.recall_ms,
            bm25_ms=timings.bm25_ms,
            vector_ms=timings.vector_ms,
            embedding_ms=timings.embedding_ms,
            qdrant_ms=timings.qdrant_ms,
            rrf_ms=timings.rrf_ms,
            reranker_ms=timings.reranker_ms,
            llm_ms=timings.llm_ms,
            llm_ttft_ms=timings.llm_ttft_ms,
            llm_output_chars=timings.llm_output_chars,
            llm_estimated_output_tokens=timings.llm_estimated_output_tokens,
            llm_estimated_tps=timings.llm_estimated_tps,
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
    retrieval_degraded: bool = False
    qdrant_degraded: bool = False
    reranker_degraded: bool = False
    degradation_reason: str = ""
    timings: TimingResponse = Field(default_factory=TimingResponse)


class UserResponse(BaseModel):
    id: str
    username: str
    is_admin: bool
    is_superuser: bool
    must_change_password: bool

    @classmethod
    def from_user(cls, user: CurrentUser) -> "UserResponse":
        return cls(
            id=str(user.id),
            username=user.username,
            is_admin=user.is_admin,
            is_superuser=user.is_superuser,
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
    is_superuser: bool
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
            is_superuser=user.is_superuser,
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


class AdminLLMSettingsResponse(BaseModel):
    provider: str
    base_url: str
    model: str
    context_max_tokens: int
    api_key_configured: bool
    source: str

    @classmethod
    def from_record(cls, record: LLMSettingsRecord) -> "AdminLLMSettingsResponse":
        return cls(
            provider=record.provider,
            base_url=record.base_url,
            model=record.model,
            context_max_tokens=record.context_max_tokens,
            api_key_configured=record.api_key_configured,
            source=record.source,
        )


class AdminLLMSettingsUpdateRequest(BaseModel):
    provider: Literal["openai_compatible", "anthropic"]
    base_url: str = Field(..., min_length=1, max_length=2000)
    model: str = Field(..., min_length=1, max_length=300)
    context_max_tokens: int | None = Field(default=None, ge=4096, le=2_000_000)
    api_key: str | None = Field(default=None, max_length=10000)


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
    retrieval_degraded: bool = False
    qdrant_degraded: bool = False
    reranker_degraded: bool = False
    degradation_reason: str = ""
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
            retrieval_degraded=message.retrieval_degraded,
            qdrant_degraded=message.qdrant_degraded,
            reranker_degraded=message.reranker_degraded,
            degradation_reason=message.degradation_reason,
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


def require_superuser_auth(
    user: Annotated[CurrentUser, Depends(require_admin_auth)],
) -> CurrentUser:
    if not user.is_superuser:
        raise HTTPException(status_code=403, detail="Superuser access required.")
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
            is_superuser=user.is_superuser,
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


@app.get("/admin/llm-settings", response_model=AdminLLMSettingsResponse)
async def admin_get_llm_settings(
    request: Request,
    _superuser: Annotated[CurrentUser, Depends(require_superuser_auth)],
) -> AdminLLMSettingsResponse:
    record = await asyncio.to_thread(
        _session_store(request).get_llm_settings_record,
    )
    return AdminLLMSettingsResponse.from_record(record)


@app.put("/admin/llm-settings", response_model=AdminLLMSettingsResponse)
async def admin_update_llm_settings(
    request: Request,
    update_request: AdminLLMSettingsUpdateRequest,
    _superuser: Annotated[CurrentUser, Depends(require_superuser_auth)],
) -> AdminLLMSettingsResponse:
    try:
        record = await asyncio.to_thread(
            _session_store(request).update_llm_settings,
            update_request.provider,
            update_request.base_url,
            update_request.model,
            update_request.api_key,
            update_request.context_max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request.app.state.llm_client.close()
    return AdminLLMSettingsResponse.from_record(record)


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
    llm_settings = await asyncio.to_thread(
        _session_store(request).get_llm_settings_record,
    )
    qdrant_ok, llm_ok = await asyncio.gather(
        asyncio.to_thread(_qdrant_health, request.app.state.vector_store),
        asyncio.to_thread(request.app.state.llm_client.health),
    )

    return {
        "status": "ok" if qdrant_ok and llm_ok else "degraded",
        "qdrant": qdrant_ok,
        "llm": llm_ok,
        "collection": settings.collection_name,
        "docs_source": settings.docs_source,
        "docs_dir": str(settings.docs_dir),
        "docs_s3_bucket": settings.docs_s3_bucket,
        "docs_s3_prefix": settings.docs_s3_prefix,
        "docs_s3_endpoint_url": settings.docs_s3_endpoint_url,
        "docs_s3_region": settings.docs_s3_region,
        "docs_s3_require_versioning": settings.docs_s3_require_versioning,
        "docs_s3_retain_versions": settings.docs_s3_retain_versions,
        "docs_s3_processing_retain_versions": settings.docs_s3_processing_retain_versions,
        "docs_s3_manifest_prefix": settings.docs_s3_manifest_prefix,
        "qdrant_retain_versions": settings.qdrant_retain_versions,
        "qdrant_processing_retain_versions": settings.qdrant_processing_retain_versions,
        "code_root_dir": str(settings.code_root_dir),
        "code_source_dir": str(settings.code_source_dir),
        "code_files_collection": settings.code_files_collection,
        "code_functions_collection": settings.code_functions_collection,
        "code_embedding_model": settings.code_embedding_model,
        "code_embedding_preload": settings.code_embedding_preload,
        "code_embedding_preload_retries": settings.code_embedding_preload_retries,
        "ray_code_embedding_actor_name": settings.ray_code_embedding_actor_name,
        "ray_code_embedding_actor_num_gpus": settings.ray_code_embedding_actor_num_gpus,
        "ray_code_embedding_actor_resource": settings.ray_code_embedding_actor_resource,
        "code_search_file_top_k": settings.code_search_file_top_k,
        "code_search_function_top_k": settings.code_search_function_top_k,
        "code_search_final_top_k": settings.code_search_final_top_k,
        "llm_provider": llm_settings.provider,
        "llm_base_url": llm_settings.base_url,
        "llm_model": llm_settings.model,
        "llm_context_max_tokens": llm_settings.context_max_tokens,
        "llm_context_safety_margin_tokens": settings.llm_context_safety_margin_tokens,
        "llm_context_prompt_overhead_tokens": settings.llm_context_prompt_overhead_tokens,
        "llm_api_key_configured": llm_settings.api_key_configured,
        "llm_settings_source": llm_settings.source,
        "llm_max_tokens": settings.llm_max_tokens,
        "cuda_enabled": settings.cuda_enabled,
        "embedding_model": settings.embedding_model,
        "embedding_query_task": settings.embedding_query_task,
        "embedding_passage_task": settings.embedding_passage_task,
        "embedding_classification_task": settings.embedding_classification_task,
        "embedding_query_prompt_name": settings.embedding_query_prompt_name,
        "embedding_passage_prompt_name": settings.embedding_passage_prompt_name,
        "embedding_classification_prompt_name": settings.embedding_classification_prompt_name,
        "bm25_top_k": settings.bm25_top_k,
        "recall_top_k": settings.recall_top_k,
        "rrf_top_k": settings.rrf_top_k,
        "retrieve_top_k": settings.retrieve_top_k,
        "retrieve_score_threshold": settings.retrieve_score_threshold,
        "history_recent_turns": settings.history_recent_turns,
        "api_top_k_max": settings.api_top_k_max,
        "api_recall_top_k_max": settings.api_recall_top_k_max,
        "reranker_enabled": settings.reranker_enabled,
        "reranker_preload": settings.reranker_preload,
        "reranker_model": settings.reranker_model,
        "ray_enabled": settings.ray_enabled,
        "ray_address": settings.ray_address,
        "ray_local_fallback": settings.ray_local_fallback,
        "ray_namespace": settings.ray_namespace,
        "ray_embedding_actor_num_gpus": settings.ray_embedding_actor_num_gpus,
        "ray_reranker_actor_num_gpus": settings.ray_reranker_actor_num_gpus,
        "ray_embedding_actor_name": settings.ray_embedding_actor_name,
        "ray_reranker_actor_name": settings.ray_reranker_actor_name,
        "ray_embedding_actor_resource": settings.ray_embedding_actor_resource,
        "ray_reranker_actor_resource": settings.ray_reranker_actor_resource,
        "intent_router": settings.intent_router_enabled,
        "auth_enabled": True,
    }


@app.get(
    "/code/repositories",
    response_model=CodeRepositoriesResponse,
    dependencies=[Depends(require_password_ready_user)],
)
async def code_repositories(http_request: Request) -> CodeRepositoriesResponse:
    try:
        repositories = await asyncio.to_thread(
            _code_retrieval(http_request).repositories
        )
    except Exception as exc:
        logger.exception("Code repository listing failed.")
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Internal server error.", exc),
        ) from exc

    responses = [
        CodeRepositoryResponse(
            id=repository.id,
            name=repository.name,
            source_root=repository.source_root,
            indexed_files=indexed_files,
        )
        for repository, indexed_files in repositories
    ]
    return CodeRepositoriesResponse(
        repositories=responses,
        default_repository_ids=[] if len(responses) != 1 else [responses[0].id],
    )


@app.post(
    "/code/index",
    response_model=CodeIndexResponse,
    dependencies=[Depends(require_password_ready_user)],
)
async def code_index(
    http_request: Request,
    index_request: CodeIndexRequest,
) -> CodeIndexResponse:
    repositories = _select_code_repositories(index_request.repository_ids)
    if not repositories:
        raise HTTPException(
            status_code=404,
            detail="No code repositories found under CODE_ROOT_DIR.",
        )

    lock = getattr(http_request.app.state, "code_index_lock", None)
    if lock is None:
        raise HTTPException(status_code=503, detail="Code indexing is unavailable.")

    async with lock:
        try:
            stats = await asyncio.to_thread(
                CodeIndexer(settings).index_repositories,
                repositories,
                recreate=False,
                clear_existing=index_request.rebuild,
            )
            refreshed = await asyncio.to_thread(
                _code_retrieval(http_request).repositories
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Code indexing failed.")
            raise HTTPException(
                status_code=500,
                detail=_error_detail("Internal server error.", exc),
            ) from exc

    return CodeIndexResponse(
        repositories=[
            CodeRepositoryResponse(
                id=repository.id,
                name=repository.name,
                source_root=repository.source_root,
                indexed_files=indexed_files,
            )
            for repository, indexed_files in refreshed
        ],
        indexed_repository_ids=[repository.id for repository in repositories],
        files=stats.files,
        functions=stats.functions,
        call_edges=stats.call_edges,
    )


@app.post(
    "/code/search",
    response_model=CodeSearchResponse,
    dependencies=[Depends(require_password_ready_user)],
)
async def code_search(
    http_request: Request,
    search_request: CodeSearchRequest,
    user: Annotated[CurrentUser, Depends(require_password_ready_user)],
) -> CodeSearchResponse:
    try:
        outcome = await asyncio.to_thread(
            _code_retrieval(http_request).search,
            search_request.query,
            file_top_k=search_request.file_top_k,
            function_top_k=search_request.function_top_k,
            final_top_k=search_request.final_top_k,
            repository_ids=search_request.repository_ids,
        )
    except CodeIndexUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except UnexpectedResponse as exc:
        logger.exception("Code search vector store request failed.")
        raise HTTPException(
            status_code=502,
            detail=_error_detail("Upstream vector store error.", exc),
        ) from exc
    except Exception as exc:
        logger.exception("Code search request failed.")
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Internal server error.", exc),
        ) from exc

    session_store = _session_store(http_request)
    session = await _get_or_create_chat_session(
        session_store,
        user,
        search_request.session_id,
    )
    contexts = [
        _code_hit_to_search_result(hit, index)
        for index, hit in enumerate(outcome.functions)
    ]
    graph, answer = await asyncio.gather(
        asyncio.to_thread(
            _code_search_graph,
            _call_graph(http_request),
            search_request.query,
            outcome.functions,
            search_request.repository_ids,
        ),
        asyncio.to_thread(
            _code_search_answer,
            _llm_client(http_request),
            search_request.query,
            outcome.functions,
        ),
    )
    await asyncio.to_thread(
        session_store.append_message,
        session.id,
        "user",
        search_request.query,
    )
    await asyncio.to_thread(
        session_store.append_message,
        session.id,
        "assistant",
        answer,
        contexts,
        True,
        "code_search",
        "Code search over selected repositories.",
    )
    session = await asyncio.to_thread(
        session_store.update_chat_session_after_answer,
        session.id,
        session.conversation_summary,
        0,
        search_request.query,
    )

    return CodeSearchResponse(
        query=outcome.query,
        session_id=str(session.id),
        answer=answer,
        retrieval_mode=outcome.retrieval_mode,
        code_embedding_degraded=outcome.code_embedding_degraded,
        degradation_reason=outcome.degradation_reason,
        files=[CodeFileHitResponse.from_hit(hit) for hit in outcome.files],
        functions=[
            CodeFunctionHitResponse.from_hit(hit)
            for hit in outcome.functions
        ],
        contexts=[ContextResponse.from_search_result(item) for item in contexts],
        graph=graph,
    )


@app.get(
    "/code/call-graph",
    response_model=CodeCallGraphResponse,
    dependencies=[Depends(require_password_ready_user)],
)
async def code_call_graph(
    http_request: Request,
    function_name: str = Query(..., min_length=1, max_length=300),
    depth: int = Query(default=settings.code_call_graph_depth, ge=1, le=10),
    repository_ids: list[str] | None = Query(default=None),
) -> CodeCallGraphResponse:
    try:
        callers, chain = await asyncio.gather(
            asyncio.to_thread(
                _call_graph(http_request).get_callers,
                function_name,
                repository_ids,
            ),
            asyncio.to_thread(
                _call_graph(http_request).get_call_chain,
                function_name,
                depth,
                repository_ids,
            ),
        )
    except Exception as exc:
        logger.exception("Code call graph request failed.")
        raise HTTPException(
            status_code=500,
            detail=_error_detail("Internal server error.", exc),
        ) from exc

    graph_nodes = [CodeGraphNodeResponse(**node) for node in chain["nodes"]]
    graph_edges = [CodeGraphEdgeResponse(**edge) for edge in chain["edges"]]
    return CodeCallGraphResponse(
        function_name=function_name,
        depth=depth,
        callers=[CodeGraphNodeResponse(**node) for node in callers],
        nodes=graph_nodes,
        edges=graph_edges,
        elements=_code_graph_elements(graph_nodes, graph_edges),
    )


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
            bm25_top_k=rag_request.bm25_top_k,
            rrf_top_k=rag_request.rrf_top_k,
            history=history,
            conversation_summary=conversation_summary,
            rag_only=rag_request.rag_only,
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
        result.retrieval_degraded,
        result.qdrant_degraded,
        result.reranker_degraded,
        result.degradation_reason,
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
        retrieval_degraded=result.retrieval_degraded,
        qdrant_degraded=result.qdrant_degraded,
        reranker_degraded=result.reranker_degraded,
        degradation_reason=result.degradation_reason,
        timings=TimingResponse.from_timings(result.timings),
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


def _code_retrieval(request: Request) -> CodeRetrieval:
    code_retrieval = getattr(request.app.state, "code_retrieval", None)
    if code_retrieval is None:
        raise HTTPException(status_code=503, detail="Code search is unavailable.")
    return code_retrieval


def _call_graph(request: Request) -> CallGraphStore:
    call_graph = getattr(request.app.state, "call_graph", None)
    if call_graph is None:
        raise HTTPException(status_code=503, detail="Code call graph is unavailable.")
    return call_graph


def _llm_client(request: Request) -> LLMClient:
    llm_client = getattr(request.app.state, "llm_client", None)
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client is unavailable.")
    return llm_client


def _code_graph_elements(
    nodes: list[CodeGraphNodeResponse],
    edges: list[CodeGraphEdgeResponse],
) -> list[CodeGraphElementResponse]:
    elements: list[CodeGraphElementResponse] = []
    for node in nodes:
        elements.append(
            CodeGraphElementResponse(
                group="nodes",
                data={
                    "id": node.id,
                    "label": node.label,
                    "name": node.name,
                    "kind": node.kind,
                    "type": node.type,
                    "filePath": node.filePath,
                    "codeSnippet": node.codeSnippet,
                    "description": node.description,
                    "path": node.path,
                    "repository_id": node.repository_id,
                    "repository_name": node.repository_name,
                    "start_line": node.start_line,
                    "end_line": node.end_line,
                    "indexed": node.indexed,
                },
            )
        )
    for index, edge in enumerate(edges):
        elements.append(
            CodeGraphElementResponse(
                group="edges",
                data={
                    "id": f"{edge.source}->{edge.target}:{index}",
                    "source": edge.source,
                    "target": edge.target,
                    "path": edge.path,
                    "lines": edge.lines,
                    "label": ", ".join(str(line) for line in edge.lines),
                    "type": "calls",
                },
            )
        )
    return elements


def _code_hit_to_search_result(hit: CodeSearchHit, index: int) -> SearchResult:
    text = "\n\n".join(
        part
        for part in (
            hit.signature,
            hit.docstring,
            hit.snippet,
        )
        if part
    )
    return SearchResult(
        text=text,
        source=f"{hit.repository_name or hit.repository_id}/{hit.path}",
        chunk_id=index,
        score=hit.score,
        vector_score=hit.vector_score,
        retrieval_source="code_search",
        content_type="code",
        start_line=hit.start_line,
        end_line=hit.end_line,
    )


def _code_search_graph(
    call_graph: CallGraphStore,
    query: str,
    functions: list[CodeSearchHit],
    repository_ids: list[str],
) -> CodeCallGraphResponse:
    names = [hit.qualified_name for hit in functions[:10]]
    sources = [
        CodeGraphSourceResponse(
            path=hit.path,
            content="\n".join(
                part
                for part in (hit.signature, hit.docstring, hit.snippet)
                if part
            ),
        )
        for hit in functions[:8]
    ]
    try:
        chain = call_graph.get_relevant_graph(
            names,
            repository_ids,
            max_nodes=28,
            neighbors_per_seed=4,
        )
        graph_nodes = [CodeGraphNodeResponse(**node) for node in chain["nodes"]]
        graph_edges = [CodeGraphEdgeResponse(**edge) for edge in chain["edges"]]
        graph_status: Literal["completed", "error"] = "completed"
        graph_result = f"{len(graph_nodes)} nodes, {len(graph_edges)} edges"
    except Exception:
        logger.exception("Code search graph construction failed.")
        graph_nodes = []
        graph_edges = []
        graph_status = "error"
        graph_result = "Graph construction failed"

    todos = [
        CodeGraphTodoResponse(
            id="retrieve",
            title="Retrieve code",
            description="Search indexed files and functions for the query.",
            status="completed",
            result=f"{len(functions)} functions",
        ),
        CodeGraphTodoResponse(
            id="graph",
            title="Build code graph",
            description="Resolve AST call edges around retrieved functions.",
            status=graph_status,
            result=graph_result,
        ),
    ]
    return CodeCallGraphResponse(
        function_name=query,
        depth=2,
        callers=[],
        nodes=graph_nodes,
        edges=graph_edges,
        elements=_code_graph_elements(graph_nodes, graph_edges),
        loading=False,
        analysing=False,
        queries=[query],
        sources=sources,
        todos=todos,
    )


def _code_search_answer(
    llm_client: LLMClient,
    query: str,
    functions: list[CodeSearchHit],
) -> str:
    if not functions:
        return "No matching code functions found."

    messages = _code_answer_messages(query, functions)
    try:
        answer = llm_client.chat(
            messages,
            temperature=0.1,
            max_tokens=1200,
        ).strip()
        if answer:
            return answer
    except Exception:
        logger.exception(
            "Code answer LLM synthesis failed; falling back to retrieved function list."
        )

    lines = ["Found these code entry points:"]
    for index, hit in enumerate(functions[:10], start=1):
        location = f"{hit.path}:{hit.start_line}"
        lines.append(
            f"{index}. `{hit.qualified_name}` in `{location}` "
            f"(score {hit.score:.3f})"
        )
    return "\n".join(lines)


def _code_answer_messages(
    query: str,
    functions: list[CodeSearchHit],
) -> list[dict[str, str]]:
    context_blocks: list[str] = []
    for index, hit in enumerate(functions[:8], start=1):
        snippet = "\n".join(
            part
            for part in (hit.signature, hit.docstring, hit.snippet)
            if part
        )
        if len(snippet) > 2400:
            snippet = f"{snippet[:2400]}\n..."
        context_blocks.append(
            "\n".join(
                [
                    f"[{index}] {hit.qualified_name}",
                    f"Path: {hit.path}:{hit.start_line}-{hit.end_line}",
                    f"Kind: {hit.kind}",
                    "Code:",
                    snippet,
                ]
            )
        )

    system_prompt = (
        "You are a codebase navigation assistant. Answer only from the provided "
        "retrieved source snippets. Explain the likely implementation path and "
        "cite functions inline using `function` and `path:line`. If the snippets "
        "are insufficient, say what is missing instead of guessing. Answer in the "
        "same language as the user query."
    )
    user_prompt = (
        f"User query:\n{query}\n\n"
        "Retrieved code snippets:\n\n"
        + "\n\n---\n\n".join(context_blocks)
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _select_code_repositories(repository_ids: list[str]) -> list[CodeRepository]:
    repositories = discover_code_repositories(settings)
    requested = {
        repository_id.strip()
        for repository_id in repository_ids
        if repository_id.strip()
    }
    if not requested:
        return repositories

    selected = [
        repository
        for repository in repositories
        if repository.id in requested or repository.name in requested
    ]
    matched = {
        value
        for repository in selected
        for value in (repository.id, repository.name)
    }
    missing = requested - matched
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown code repository: {', '.join(sorted(missing))}",
        )
    return selected


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

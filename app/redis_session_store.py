from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from threading import Lock
import time
from typing import Any, Literal
from uuid import UUID, uuid4

from redis import Redis
from redis.exceptions import LockError, RedisError

from app.config import Settings, settings
from app.rag import ChatMessage
from app.session_store import (
    ChatSessionRecord,
    CurrentUser,
    SessionStore,
    StoredChatMessage,
    clean_session_title,
    hash_token,
    next_session_seq,
)
from app.vector_store import SearchResult


logger = logging.getLogger(__name__)


class SessionStoreUnavailableError(RuntimeError):
    """Raised when active Redis session state cannot be read consistently."""


class RedisSessionStore:
    """Redis hot-path facade with idempotent asynchronous PostgreSQL archival."""

    def __init__(
        self,
        config: Settings = settings,
        *,
        pg_store: SessionStore | None = None,
        client: Redis | None = None,
    ) -> None:
        self.config = config
        self.pg_store = pg_store or SessionStore(config)
        self.client = client or Redis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_timeout=config.redis_socket_timeout_seconds,
            socket_connect_timeout=config.redis_socket_timeout_seconds,
            health_check_interval=30,
        )
        self._backlog_lock = Lock()
        self._backlog_checked_at = 0.0
        self._backlog_over_limit = False
        self._availability_lock = Lock()
        self._redis_unavailable = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pg_store, name)

    @property
    def archive_stream_key(self) -> str:
        return self._key(self.config.redis_archive_stream)

    def init_cache(self) -> None:
        self.ping()

    def ping(self) -> bool:
        try:
            ready = bool(self.client.ping())
        except RedisError:
            self._mark_redis_unavailable()
            raise
        if ready:
            self._mark_redis_available()
        return ready

    def health(self) -> dict[str, object]:
        return {
            "ready": self.ping(),
            "archive_backlog": int(self.client.xlen(self.archive_stream_key)),
        }

    def close(self) -> None:
        self.client.close()
        self.pg_store.close()

    def get_user_by_token(self, token: str) -> CurrentUser | None:
        if not token:
            return None
        token_hash = hash_token(token)
        cache_key = self._auth_key(token_hash)
        redis_cache_available = not self._redis_was_unavailable()
        if redis_cache_available:
            try:
                cached = self.client.get(cache_key)
                if cached:
                    return self._user_from_json(cached)
            except (RedisError, ValueError, TypeError) as exc:
                self._mark_redis_unavailable()
                redis_cache_available = False
                logger.warning(
                    "Redis auth cache read failed; using PostgreSQL error=%s: %s",
                    type(exc).__name__,
                    exc,
                )

        user = self.pg_store.get_user_by_token(token)
        if user is not None and redis_cache_available:
            self._cache_auth_user(cache_key, user)
        return user

    def delete_auth_session(self, token: str) -> None:
        self.pg_store.delete_auth_session(token)
        if self._redis_was_unavailable():
            return
        try:
            self.client.delete(self._auth_key(hash_token(token)))
        except RedisError as exc:
            self._mark_redis_unavailable()
            logger.warning(
                "Failed to invalidate Redis auth token cache error=%s: %s",
                type(exc).__name__,
                exc,
            )

    def change_user_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> bool:
        updated = self.pg_store.change_user_password(
            user_id,
            current_password,
            new_password,
        )
        if updated:
            self._invalidate_user_auth(user_id)
        return updated

    def set_user_password(
        self,
        user_id: UUID,
        password: str,
        must_change_password: bool = True,
    ) -> bool:
        updated = self.pg_store.set_user_password(
            user_id,
            password,
            must_change_password,
        )
        if updated:
            self._invalidate_user_auth(user_id)
        return updated

    def set_user_admin(self, user_id: UUID, is_admin: bool) -> bool:
        updated = self.pg_store.set_user_admin(user_id, is_admin)
        if updated:
            self._invalidate_user_auth(user_id)
        return updated

    def list_chat_sessions(self, user_id: UUID) -> list[ChatSessionRecord]:
        self._require_redis_available()
        try:
            session_ids = self.client.zrevrange(
                self._user_sessions_key(user_id),
                0,
                self.config.session_list_limit - 1,
            )
            self.client.expire(
                self._user_sessions_key(user_id),
                self.config.redis_session_ttl_seconds,
            )
        except (RedisError, ValueError, TypeError) as exc:
            self._raise_unavailable("list sessions", exc)

        sessions = {item.id: item for item in self.pg_store.list_chat_sessions(user_id)}
        for raw_session_id in session_ids:
            try:
                meta = self.client.hgetall(self._meta_key(UUID(raw_session_id)))
                if meta and meta.get("user_id") == str(user_id):
                    session = self._session_from_meta(meta)
                    sessions[session.id] = session
            except (RedisError, ValueError, TypeError) as exc:
                self._raise_unavailable("load session list metadata", exc)
        return sorted(
            sessions.values(),
            key=lambda item: item.updated_at,
            reverse=True,
        )[: self.config.session_list_limit]

    def create_chat_session(
        self,
        user_id: UUID,
        title: str | None = None,
    ) -> ChatSessionRecord:
        self._require_redis_available()
        try:
            self.client.ping()
        except RedisError as exc:
            self._raise_unavailable("create session", exc)
        session = self.pg_store.create_chat_session(user_id, title)
        try:
            self._cache_session(session, [])
        except RedisError as exc:
            self._raise_unavailable("cache new session", exc)
        return session

    def get_chat_session(
        self,
        user_id: UUID,
        session_id: UUID,
    ) -> ChatSessionRecord | None:
        self._require_redis_available()
        try:
            meta = self.client.hgetall(self._meta_key(session_id))
            if meta and meta.get("user_id") == str(user_id):
                self._touch_session(session_id, user_id)
                return self._session_from_meta(meta)
        except (RedisError, ValueError, TypeError) as exc:
            self._raise_unavailable("get session", exc)

        session = self.pg_store.get_chat_session(user_id, session_id)
        if session is None:
            return None
        try:
            self._cache_from_pg(session)
        except RedisError as exc:
            self._raise_unavailable("cold-load session", exc)
        return session

    def rename_chat_session(
        self,
        user_id: UUID,
        session_id: UUID,
        title: str,
    ) -> ChatSessionRecord | None:
        self._require_redis_available()
        try:
            cached = bool(self.client.exists(self._meta_key(session_id)))
        except RedisError as exc:
            self._raise_unavailable("rename session", exc)
        session = self.pg_store.rename_chat_session(user_id, session_id, title)
        if session is None:
            return None
        try:
            if cached:
                self.client.hset(
                    self._meta_key(session_id),
                    mapping={
                        "title": session.title,
                        "updated_at": session.updated_at.isoformat(),
                    },
                )
                self._touch_session(session_id, user_id)
        except RedisError as exc:
            self._raise_unavailable("update renamed session", exc)
        return session

    def delete_chat_session(self, user_id: UUID, session_id: UUID) -> bool:
        self._require_redis_available()
        try:
            self.client.ping()
        except RedisError as exc:
            self._raise_unavailable("delete session", exc)
        deleted = self.pg_store.delete_chat_session(user_id, session_id)
        if not deleted:
            return False
        try:
            pipe = self.client.pipeline(transaction=True)
            pipe.delete(
                self._meta_key(session_id),
                self._messages_key(session_id),
                self._pending_key(session_id),
            )
            pipe.zrem(self._user_sessions_key(user_id), str(session_id))
            pipe.execute()
        except RedisError as exc:
            self._raise_unavailable("clear deleted session cache", exc)
        return True

    def clear_user_sessions(self, user_id: UUID) -> int:
        self._require_redis_available()
        try:
            session_ids = self.client.zrange(self._user_sessions_key(user_id), 0, -1)
        except RedisError as exc:
            self._raise_unavailable("clear user sessions", exc)
        deleted = self.pg_store.clear_user_sessions(user_id)
        try:
            pipe = self.client.pipeline(transaction=True)
            for raw_session_id in session_ids:
                session_id = UUID(raw_session_id)
                pipe.delete(
                    self._meta_key(session_id),
                    self._messages_key(session_id),
                    self._pending_key(session_id),
                )
            pipe.delete(self._user_sessions_key(user_id))
            pipe.execute()
        except (RedisError, ValueError) as exc:
            self._raise_unavailable("clear user session cache", exc)
        return deleted

    def delete_user(self, user_id: UUID) -> bool:
        deleted = self.pg_store.delete_user(user_id)
        if deleted:
            self._invalidate_user_auth(user_id)
            try:
                self.client.delete(self._user_sessions_key(user_id))
            except RedisError:
                logger.exception("Failed to clear deleted user's Redis index.")
        return deleted

    def list_messages(self, session_id: UUID) -> list[StoredChatMessage]:
        self._require_redis_available()
        try:
            meta = self.client.hgetall(self._meta_key(session_id))
            if meta and meta.get("messages_cached") == "1":
                payloads = self.client.lrange(self._messages_key(session_id), 0, -1)
                self._touch_session(session_id, UUID(meta["user_id"]))
                return [self._message_from_json(item) for item in payloads]
        except (RedisError, ValueError, TypeError, KeyError) as exc:
            self._raise_unavailable("list session messages", exc)

        messages = self.pg_store.list_messages(session_id)
        session = self.pg_store.get_chat_session_by_id(session_id)
        if session is not None:
            try:
                self._cache_from_pg(session, messages)
            except RedisError as exc:
                self._raise_unavailable("cold-load session messages", exc)
        return messages

    def prompt_history(
        self,
        session_id: UUID,
        compacted_message_count: int,
    ) -> list[ChatMessage]:
        messages = self.list_messages(session_id)[max(0, compacted_message_count) :]
        return [
            ChatMessage(
                role=message.role,
                content=message.content,
                used_rag=message.used_rag,
                route=message.route,
                context_count=len(message.contexts),
            )
            for message in messages
        ]

    def append_exchange(
        self,
        session_id: UUID,
        user_content: str,
        assistant_content: str,
        contexts: list[SearchResult] | None = None,
        used_rag: bool | None = None,
        route: str = "",
        route_reason: str = "",
        retrieval_degraded: bool = False,
        embedding_degraded: bool = False,
        qdrant_degraded: bool = False,
        reranker_degraded: bool = False,
        degradation_reason: str = "",
        conversation_summary: str = "",
        compacted_delta: int = 0,
        title: str | None = None,
    ) -> ChatSessionRecord:
        self._require_redis_available()
        try:
            backlog_exceeded = self._archive_backlog_exceeded()
        except RedisError as exc:
            self._raise_unavailable("check session archive backlog", exc)
        if backlog_exceeded:
            raise SessionStoreUnavailableError(
                "Session archive backlog is full; active session writes are paused."
            )

        try:
            meta = self.client.hgetall(self._meta_key(session_id))
            if not meta or meta.get("messages_cached") != "1":
                session = self.pg_store.get_chat_session_by_id(session_id)
                if session is None:
                    raise ValueError("Chat session not found.")
                self._cache_session(session, self.pg_store.list_messages(session_id))

            lock = self.client.lock(
                self._lock_key(session_id),
                timeout=self.config.redis_session_lock_timeout_seconds,
                blocking_timeout=self.config.redis_socket_timeout_seconds,
            )
            if not lock.acquire():
                raise TimeoutError("Timed out waiting for the Redis session lock.")
            try:
                return self._append_exchange_redis(
                    session_id=session_id,
                    user_content=user_content,
                    assistant_content=assistant_content,
                    contexts=contexts,
                    used_rag=used_rag,
                    route=route,
                    route_reason=route_reason,
                    retrieval_degraded=retrieval_degraded,
                    embedding_degraded=embedding_degraded,
                    qdrant_degraded=qdrant_degraded,
                    reranker_degraded=reranker_degraded,
                    degradation_reason=degradation_reason,
                    conversation_summary=conversation_summary,
                    compacted_delta=compacted_delta,
                    title=title,
                )
            finally:
                try:
                    lock.release()
                except (LockError, RedisError) as exc:
                    self._raise_unavailable("release session write lock", exc)
        except SessionStoreUnavailableError:
            raise
        except (RedisError, TimeoutError) as exc:
            self._raise_unavailable("write session exchange", exc)

    def _append_exchange_redis(
        self,
        *,
        session_id: UUID,
        user_content: str,
        assistant_content: str,
        contexts: list[SearchResult] | None,
        used_rag: bool | None,
        route: str,
        route_reason: str,
        retrieval_degraded: bool,
        embedding_degraded: bool,
        qdrant_degraded: bool,
        reranker_degraded: bool,
        degradation_reason: str,
        conversation_summary: str,
        compacted_delta: int,
        title: str | None,
    ) -> ChatSessionRecord:
        meta = self.client.hgetall(self._meta_key(session_id))
        if not meta:
            raise RedisError("Redis session metadata disappeared before write.")

        last_seq = int(meta.get("last_seq", 0) or 0)
        first_seq = next_session_seq(last_seq)
        updated_at = datetime.now(timezone.utc)
        current_title = meta.get("title", "New chat")
        updated_title = current_title
        if title and current_title == "New chat":
            updated_title = clean_session_title(
                title,
                self.config.session_title_max_chars,
            )
        compacted_count = int(meta.get("compacted_message_count", 0) or 0) + max(
            0,
            compacted_delta,
        )
        user_message = self.pg_store._archive_message_payload(  # noqa: SLF001
            event_id=str(uuid4()),
            session_seq=first_seq,
            role="user",
            content=user_content,
            created_at=updated_at,
        )
        assistant_message = self.pg_store._archive_message_payload(  # noqa: SLF001
            event_id=str(uuid4()),
            session_seq=first_seq + 1,
            role="assistant",
            content=assistant_content,
            contexts=contexts,
            used_rag=used_rag,
            route=route,
            route_reason=route_reason,
            retrieval_degraded=retrieval_degraded,
            embedding_degraded=embedding_degraded,
            qdrant_degraded=qdrant_degraded,
            reranker_degraded=reranker_degraded,
            degradation_reason=degradation_reason,
            created_at=updated_at,
        )
        event_id = str(uuid4())
        event = {
            "event_id": event_id,
            "session_id": str(session_id),
            "last_seq": first_seq + 1,
            "messages": [user_message, assistant_message],
            "session": {
                "conversation_summary": conversation_summary,
                "compacted_message_count": compacted_count,
                "title": updated_title,
                "updated_at": updated_at.isoformat(),
            },
        }
        event_json = self._json(event)
        user_id = UUID(meta["user_id"])
        meta_updates = {
            "title": updated_title,
            "conversation_summary": conversation_summary,
            "compacted_message_count": compacted_count,
            "last_seq": first_seq + 1,
            "updated_at": updated_at.isoformat(),
            "messages_cached": "1",
        }

        pipe = self.client.pipeline(transaction=True)
        pipe.rpush(
            self._messages_key(session_id),
            self._json(user_message),
            self._json(assistant_message),
        )
        pipe.hset(self._meta_key(session_id), mapping=meta_updates)
        pipe.hset(self._pending_key(session_id), event_id, event_json)
        pipe.xadd(
            self.archive_stream_key,
            {
                "event_id": event_id,
                "session_id": str(session_id),
            },
        )
        pipe.zadd(
            self._user_sessions_key(user_id),
            {str(session_id): updated_at.timestamp()},
        )
        pipe.expire(self._meta_key(session_id), self.config.redis_session_ttl_seconds)
        pipe.expire(
            self._messages_key(session_id),
            self.config.redis_session_ttl_seconds,
        )
        pipe.expire(
            self._user_sessions_key(user_id),
            self.config.redis_session_ttl_seconds,
        )
        pipe.execute()

        return ChatSessionRecord(
            id=session_id,
            user_id=user_id,
            title=updated_title,
            conversation_summary=conversation_summary,
            compacted_message_count=compacted_count,
            created_at=self._parse_datetime(meta["created_at"]),
            updated_at=updated_at,
        )

    def _archive_backlog_exceeded(self) -> bool:
        now = time.monotonic()
        with self._backlog_lock:
            if now - self._backlog_checked_at < 1.0:
                return self._backlog_over_limit
            backlog = int(self.client.xlen(self.archive_stream_key))
            self._backlog_over_limit = (
                backlog >= self.config.redis_archive_backlog_max
            )
            self._backlog_checked_at = now
            return self._backlog_over_limit

    def _cache_from_pg(
        self,
        session: ChatSessionRecord,
        messages: list[StoredChatMessage] | None = None,
    ) -> None:
        self._cache_session(
            session,
            messages if messages is not None else self.pg_store.list_messages(session.id),
        )

    def _cache_session(
        self,
        session: ChatSessionRecord,
        messages: list[StoredChatMessage],
    ) -> None:
        pending_events = self._pending_events(session.id)
        merged_messages = self._merge_pending(
            session.id,
            messages,
            pending_events=pending_events,
        )
        last_seq = max((item.session_seq or item.id for item in merged_messages), default=0)
        title = session.title
        conversation_summary = session.conversation_summary
        compacted_message_count = session.compacted_message_count
        updated_at = session.updated_at
        if pending_events:
            latest_event = max(
                pending_events,
                key=lambda item: int(item.get("last_seq", 0) or 0),
            )
            pending_state = latest_event.get("session", {})
            if isinstance(pending_state, dict):
                if title == "New chat":
                    title = str(pending_state.get("title", title))
                conversation_summary = str(
                    pending_state.get("conversation_summary", conversation_summary)
                )
                compacted_message_count = int(
                    pending_state.get(
                        "compacted_message_count",
                        compacted_message_count,
                    )
                    or 0
                )
                updated_at = self._parse_datetime(
                    pending_state.get("updated_at", updated_at)
                )
        meta = {
            "id": str(session.id),
            "user_id": str(session.user_id),
            "title": title,
            "conversation_summary": conversation_summary,
            "compacted_message_count": compacted_message_count,
            "created_at": session.created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
            "last_seq": last_seq,
            "messages_cached": "1",
        }
        pipe = self.client.pipeline(transaction=True)
        pipe.delete(self._messages_key(session.id))
        if merged_messages:
            pipe.rpush(
                self._messages_key(session.id),
                *(self._message_to_json(item) for item in merged_messages),
            )
            pipe.expire(
                self._messages_key(session.id),
                self.config.redis_session_ttl_seconds,
            )
        pipe.hset(self._meta_key(session.id), mapping=meta)
        pipe.expire(self._meta_key(session.id), self.config.redis_session_ttl_seconds)
        pipe.zadd(
            self._user_sessions_key(session.user_id),
            {str(session.id): updated_at.timestamp()},
        )
        pipe.expire(
            self._user_sessions_key(session.user_id),
            self.config.redis_session_ttl_seconds,
        )
        pipe.execute()

    def _merge_pending(
        self,
        session_id: UUID,
        messages: list[StoredChatMessage],
        *,
        pending_events: list[dict[str, Any]] | None = None,
    ) -> list[StoredChatMessage]:
        by_event = {item.event_id: item for item in messages if item.event_id}
        by_seq = {item.session_seq: item for item in messages if item.session_seq}
        events = (
            pending_events
            if pending_events is not None
            else self._pending_events(session_id)
        )
        for event in events:
            try:
                for message_payload in event.get("messages", []):
                    event_id = str(message_payload.get("event_id", ""))
                    session_seq = int(message_payload.get("session_seq", 0) or 0)
                    if event_id in by_event or session_seq in by_seq:
                        continue
                    message = self._message_from_payload(message_payload)
                    if event_id:
                        by_event[event_id] = message
                    if session_seq:
                        by_seq[session_seq] = message
                    messages.append(message)
            except (ValueError, TypeError):
                logger.exception("Ignoring malformed Redis pending session event.")
        return sorted(messages, key=lambda item: (item.session_seq or item.id, item.id))

    def _pending_events(self, session_id: UUID) -> list[dict[str, Any]]:
        pending_payloads = self.client.hvals(self._pending_key(session_id))
        events: list[dict[str, Any]] = []
        for raw_event in pending_payloads:
            try:
                event = json.loads(raw_event)
                if isinstance(event, dict):
                    events.append(event)
            except (TypeError, json.JSONDecodeError):
                logger.exception("Ignoring malformed Redis pending session event.")
        return events

    def _touch_session(self, session_id: UUID, user_id: UUID) -> None:
        pipe = self.client.pipeline(transaction=False)
        pipe.expire(self._meta_key(session_id), self.config.redis_session_ttl_seconds)
        pipe.expire(
            self._messages_key(session_id),
            self.config.redis_session_ttl_seconds,
        )
        pipe.expire(
            self._user_sessions_key(user_id),
            self.config.redis_session_ttl_seconds,
        )
        pipe.execute()

    def _cache_auth_user(self, cache_key: str, user: CurrentUser) -> None:
        if self._redis_was_unavailable():
            return
        try:
            pipe = self.client.pipeline(transaction=True)
            pipe.set(
                cache_key,
                self._json(
                    {
                        "id": str(user.id),
                        "username": user.username,
                        "is_admin": user.is_admin,
                        "is_superuser": user.is_superuser,
                        "must_change_password": user.must_change_password,
                    }
                ),
                ex=self.config.redis_auth_cache_ttl_seconds,
            )
            pipe.sadd(self._user_auth_key(user.id), cache_key)
            pipe.expire(
                self._user_auth_key(user.id),
                self.config.auth_session_ttl_seconds,
            )
            pipe.execute()
        except RedisError as exc:
            self._mark_redis_unavailable()
            logger.warning(
                "Failed to populate Redis auth cache error=%s: %s",
                type(exc).__name__,
                exc,
            )

    def _invalidate_user_auth(self, user_id: UUID) -> None:
        if self._redis_was_unavailable():
            return
        try:
            index_key = self._user_auth_key(user_id)
            auth_keys = self.client.smembers(index_key)
            if auth_keys:
                self.client.delete(*auth_keys)
            self.client.delete(index_key)
        except RedisError as exc:
            self._mark_redis_unavailable()
            logger.warning(
                "Failed to invalidate Redis auth cache error=%s: %s",
                type(exc).__name__,
                exc,
            )

    def _require_redis_available(self) -> None:
        with self._availability_lock:
            if self._redis_unavailable:
                raise SessionStoreUnavailableError(
                    "Active session storage is temporarily unavailable."
                )

    def _mark_redis_unavailable(self) -> None:
        with self._availability_lock:
            self._redis_unavailable = True

    def _mark_redis_available(self) -> None:
        with self._availability_lock:
            self._redis_unavailable = False

    def _redis_was_unavailable(self) -> bool:
        with self._availability_lock:
            return self._redis_unavailable

    def _raise_unavailable(self, operation: str, exc: Exception) -> None:
        self._mark_redis_unavailable()
        logger.warning(
            "Redis session operation failed operation=%s; returning HTTP 503 "
            "error=%s: %s",
            operation,
            type(exc).__name__,
            exc,
        )
        raise SessionStoreUnavailableError(
            "Active session storage is temporarily unavailable."
        ) from exc

    def _session_from_meta(self, meta: dict[str, str]) -> ChatSessionRecord:
        return ChatSessionRecord(
            id=UUID(meta["id"]),
            user_id=UUID(meta["user_id"]),
            title=meta.get("title", "New chat"),
            conversation_summary=meta.get("conversation_summary", ""),
            compacted_message_count=int(meta.get("compacted_message_count", 0) or 0),
            created_at=self._parse_datetime(meta["created_at"]),
            updated_at=self._parse_datetime(meta["updated_at"]),
        )

    def _message_to_json(self, message: StoredChatMessage) -> str:
        return self._json(
            {
                "event_id": message.event_id,
                "session_seq": message.session_seq or message.id,
                "role": message.role,
                "content": message.content,
                "contexts": message.contexts,
                "used_rag": message.used_rag,
                "route": message.route,
                "route_reason": message.route_reason,
                "retrieval_degraded": message.retrieval_degraded,
                "embedding_degraded": message.embedding_degraded,
                "qdrant_degraded": message.qdrant_degraded,
                "reranker_degraded": message.reranker_degraded,
                "degradation_reason": message.degradation_reason,
                "created_at": message.created_at.isoformat(),
            }
        )

    def _message_from_json(self, raw: str) -> StoredChatMessage:
        return self._message_from_payload(json.loads(raw))

    def _message_from_payload(self, payload: dict[str, Any]) -> StoredChatMessage:
        session_seq = int(payload.get("session_seq", 0) or 0)
        return StoredChatMessage(
            id=session_seq,
            role=payload["role"],
            content=str(payload.get("content", "")),
            contexts=list(payload.get("contexts", [])),
            used_rag=payload.get("used_rag"),
            route=str(payload.get("route", "")),
            route_reason=str(payload.get("route_reason", "")),
            retrieval_degraded=bool(payload.get("retrieval_degraded", False)),
            embedding_degraded=bool(payload.get("embedding_degraded", False)),
            qdrant_degraded=bool(payload.get("qdrant_degraded", False)),
            reranker_degraded=bool(payload.get("reranker_degraded", False)),
            degradation_reason=str(payload.get("degradation_reason", "")),
            created_at=self._parse_datetime(payload.get("created_at")),
            event_id=str(payload.get("event_id", "")),
            session_seq=session_seq,
        )

    @staticmethod
    def _user_from_json(raw: str) -> CurrentUser:
        payload = json.loads(raw)
        return CurrentUser(
            id=UUID(payload["id"]),
            username=payload["username"],
            is_admin=bool(payload.get("is_admin", False)),
            is_superuser=bool(payload.get("is_superuser", False)),
            must_change_password=bool(payload.get("must_change_password", False)),
        )

    @staticmethod
    def _parse_datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _key(self, suffix: str) -> str:
        return f"{self.config.redis_key_prefix}:{suffix}"

    def _meta_key(self, session_id: UUID) -> str:
        return self._key(f"session:{session_id}:meta")

    def _messages_key(self, session_id: UUID) -> str:
        return self._key(f"session:{session_id}:messages")

    def _pending_key(self, session_id: UUID) -> str:
        return self._key(f"session:{session_id}:pending")

    def _lock_key(self, session_id: UUID) -> str:
        return self._key(f"session:{session_id}:lock")

    def _user_sessions_key(self, user_id: UUID) -> str:
        return self._key(f"user:{user_id}:active_sessions")

    def _auth_key(self, token_hash: str) -> str:
        return self._key(f"auth:{token_hash}")

    def _user_auth_key(self, user_id: UUID) -> str:
        return self._key(f"user:{user_id}:auth_keys")

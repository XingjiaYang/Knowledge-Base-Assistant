from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import pbkdf2_hmac, sha256
from hmac import compare_digest
import base64
import json
import logging
import re
import secrets
import time
from typing import Any, Literal
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from app.config import LLMRuntimeSettings, Settings, normalize_llm_provider, settings
from app.rag import ChatMessage
from app.vector_store import SearchResult


logger = logging.getLogger(__name__)

_PASSWORD_SCHEME = "pbkdf2_sha256"
_PASSWORD_ITERATIONS = 260_000
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,80}$")


@dataclass(frozen=True)
class CurrentUser:
    id: UUID
    username: str
    is_admin: bool = False
    must_change_password: bool = False
    is_superuser: bool = False


@dataclass(frozen=True)
class UserRecord:
    id: UUID
    username: str
    is_active: bool
    is_admin: bool
    is_superuser: bool
    must_change_password: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime | None
    session_count: int = 0
    message_count: int = 0


@dataclass(frozen=True)
class ChatSessionRecord:
    id: UUID
    user_id: UUID
    title: str
    conversation_summary: str
    compacted_message_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class StoredChatMessage:
    id: int
    role: Literal["user", "assistant"]
    content: str
    contexts: list[dict[str, Any]]
    used_rag: bool | None
    route: str
    route_reason: str
    retrieval_degraded: bool
    embedding_degraded: bool
    qdrant_degraded: bool
    reranker_degraded: bool
    degradation_reason: str
    created_at: datetime
    event_id: str = ""
    session_seq: int = 0


@dataclass(frozen=True)
class LLMSettingsRecord:
    provider: str
    base_url: str
    model: str
    context_max_tokens: int
    api_key_configured: bool
    source: str


class SessionStore:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def init_db(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS app_users (
                        id UUID PRIMARY KEY,
                        username TEXT NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                        is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
                        must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_login_at TIMESTAMPTZ
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE app_users
                    ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE app_users
                    ADD COLUMN IF NOT EXISTS is_superuser BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE app_users
                    ADD COLUMN IF NOT EXISTS must_change_password BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_app_users_single_superuser
                    ON app_users (is_superuser)
                    WHERE is_superuser = TRUE
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_sessions (
                        token_hash TEXT PRIMARY KEY,
                        user_id UUID NOT NULL REFERENCES app_users(id)
                            ON DELETE CASCADE,
                        user_agent TEXT NOT NULL DEFAULT '',
                        client_host TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMPTZ NOT NULL,
                        last_seen_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id
                    ON auth_sessions(user_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at
                    ON auth_sessions(expires_at)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id UUID PRIMARY KEY,
                        user_id UUID NOT NULL REFERENCES app_users(id)
                            ON DELETE CASCADE,
                        title TEXT NOT NULL,
                        conversation_summary TEXT NOT NULL DEFAULT '',
                        compacted_message_count INTEGER NOT NULL DEFAULT 0
                            CHECK (compacted_message_count >= 0),
                        archive_seq BIGINT NOT NULL DEFAULT 0
                            CHECK (archive_seq >= 0),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_sessions
                    ADD COLUMN IF NOT EXISTS archive_seq BIGINT NOT NULL DEFAULT 0
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                    ON chat_sessions(user_id, updated_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id BIGSERIAL PRIMARY KEY,
                        session_id UUID NOT NULL REFERENCES chat_sessions(id)
                            ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        contexts JSONB NOT NULL DEFAULT '[]'::jsonb,
                        used_rag BOOLEAN,
                        route TEXT NOT NULL DEFAULT '',
                        route_reason TEXT NOT NULL DEFAULT '',
                        retrieval_degraded BOOLEAN NOT NULL DEFAULT FALSE,
                        embedding_degraded BOOLEAN NOT NULL DEFAULT FALSE,
                        qdrant_degraded BOOLEAN NOT NULL DEFAULT FALSE,
                        reranker_degraded BOOLEAN NOT NULL DEFAULT FALSE,
                        degradation_reason TEXT NOT NULL DEFAULT '',
                        event_id UUID,
                        session_seq BIGINT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN IF NOT EXISTS retrieval_degraded BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN IF NOT EXISTS embedding_degraded BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN IF NOT EXISTS qdrant_degraded BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN IF NOT EXISTS reranker_degraded BOOLEAN NOT NULL DEFAULT FALSE
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN IF NOT EXISTS degradation_reason TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN IF NOT EXISTS event_id UUID
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN IF NOT EXISTS session_seq BIGINT
                    """
                )
                cur.execute(
                    """
                    WITH existing AS (
                        SELECT session_id, COALESCE(MAX(session_seq), 0) AS max_seq
                        FROM chat_messages
                        WHERE session_seq IS NOT NULL
                        GROUP BY session_id
                    ),
                    ranked AS (
                        SELECT message.id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY message.session_id
                                   ORDER BY message.id
                               ) + COALESCE(existing.max_seq, 0) AS seq
                        FROM chat_messages
                        AS message
                        LEFT JOIN existing ON existing.session_id = message.session_id
                        WHERE message.session_seq IS NULL
                    )
                    UPDATE chat_messages AS message
                    SET session_seq = ranked.seq
                    FROM ranked
                    WHERE message.id = ranked.id
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_messages_event_id
                    ON chat_messages(event_id)
                    WHERE event_id IS NOT NULL
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_messages_session_seq
                    ON chat_messages(session_id, session_seq)
                    WHERE session_seq IS NOT NULL
                    """
                )
                cur.execute(
                    """
                    UPDATE chat_sessions AS session
                    SET archive_seq = archived.max_seq
                    FROM (
                        SELECT session_id, MAX(session_seq) AS max_seq
                        FROM chat_messages
                        WHERE session_seq IS NOT NULL
                        GROUP BY session_id
                    ) AS archived
                    WHERE session.id = archived.session_id
                      AND session.archive_seq < archived.max_seq
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
                    ON chat_messages(session_id, session_seq, id)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS app_settings (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS code_files (
                        id UUID PRIMARY KEY,
                        repository_id TEXT NOT NULL DEFAULT '',
                        repository_name TEXT NOT NULL DEFAULT '',
                        source_root TEXT NOT NULL,
                        path TEXT NOT NULL,
                        language TEXT NOT NULL,
                        full_content TEXT NOT NULL,
                        content_sha256 TEXT NOT NULL,
                        line_count INTEGER NOT NULL DEFAULT 0
                            CHECK (line_count >= 0),
                        indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE code_files
                    ADD COLUMN IF NOT EXISTS repository_id TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE code_files
                    ADD COLUMN IF NOT EXISTS repository_name TEXT NOT NULL DEFAULT ''
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE code_files
                    DROP CONSTRAINT IF EXISTS code_files_path_key
                    """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_code_files_repository_path
                    ON code_files(repository_id, path)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_code_files_repository_id
                    ON code_files(repository_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_code_files_language
                    ON code_files(language)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS code_functions (
                        id UUID PRIMARY KEY,
                        file_id UUID NOT NULL REFERENCES code_files(id)
                            ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        qualified_name TEXT NOT NULL,
                        kind TEXT NOT NULL DEFAULT 'function',
                        signature TEXT NOT NULL,
                        body TEXT NOT NULL,
                        docstring TEXT NOT NULL DEFAULT '',
                        start_line INTEGER NOT NULL CHECK (start_line >= 1),
                        end_line INTEGER NOT NULL CHECK (end_line >= start_line),
                        embedding_text TEXT NOT NULL,
                        indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_code_functions_file_id
                    ON code_functions(file_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_code_functions_name
                    ON code_functions(name)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_code_functions_qualified_name
                    ON code_functions(qualified_name)
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS code_call_edges (
                        id BIGSERIAL PRIMARY KEY,
                        caller_id UUID NOT NULL REFERENCES code_functions(id)
                            ON DELETE CASCADE,
                        caller_name TEXT NOT NULL,
                        callee_name TEXT NOT NULL,
                        file_id UUID NOT NULL REFERENCES code_files(id)
                            ON DELETE CASCADE,
                        call_line INTEGER NOT NULL CHECK (call_line >= 1),
                        UNIQUE (caller_id, callee_name, call_line)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_code_call_edges_callee
                    ON code_call_edges(callee_name)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_code_call_edges_caller
                    ON code_call_edges(caller_name)
                    """
                )
            self._bootstrap_users(conn)
            self.delete_expired_auth_sessions(conn)

    def _bootstrap_users(self, conn: psycopg.Connection[Any]) -> None:
        if self.config.auth_default_admin_enabled:
            self._ensure_default_admin(conn)

        users = parse_bootstrap_users(self.config.auth_bootstrap_users)
        if not users:
            return

        with conn.cursor() as cur:
            for username, password in users:
                cur.execute(
                    """
                    INSERT INTO app_users (
                        id,
                        username,
                        password_hash,
                        is_admin,
                        is_superuser,
                        must_change_password
                    )
                    VALUES (%s, %s, %s, FALSE, FALSE, FALSE)
                    ON CONFLICT (username) DO NOTHING
                    """,
                    (uuid4(), normalize_username(username), hash_password(password)),
                )

    def _ensure_default_admin(self, conn: psycopg.Connection[Any]) -> None:
        username = normalize_username(self.config.auth_default_admin_username)
        password = self.config.auth_default_admin_password
        if not password:
            raise ValueError("AUTH_DEFAULT_ADMIN_PASSWORD must not be empty.")

        conn.execute(
            """
            UPDATE app_users
            SET is_superuser = FALSE,
                updated_at = CURRENT_TIMESTAMP
            WHERE username <> %s AND is_superuser = TRUE
            """,
            (username,),
        )
        conn.execute(
            """
            INSERT INTO app_users (
                id,
                username,
                password_hash,
                is_active,
                is_admin,
                is_superuser,
                must_change_password
            )
            VALUES (%s, %s, %s, TRUE, TRUE, TRUE, FALSE)
            ON CONFLICT (username) DO UPDATE
            SET is_admin = TRUE,
                is_superuser = TRUE,
                is_active = TRUE,
                must_change_password = CASE
                    WHEN app_users.is_admin = FALSE OR app_users.is_active = FALSE
                    THEN FALSE
                    ELSE app_users.must_change_password
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE app_users.is_admin = FALSE
               OR app_users.is_superuser = FALSE
               OR app_users.is_active = FALSE
            """,
            (uuid4(), username, hash_password(password)),
        )

    def authenticate_user(self, username: str, password: str) -> CurrentUser | None:
        normalized = normalize_username(username)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id,
                       username,
                       password_hash,
                       is_admin,
                       is_superuser,
                       must_change_password
                FROM app_users
                WHERE username = %s AND is_active = TRUE
                """,
                (normalized,),
            ).fetchone()
            if row is None or not verify_password(password, row["password_hash"]):
                return None

            conn.execute(
                """
                UPDATE app_users
                SET last_login_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (row["id"],),
            )
            return CurrentUser(
                id=row["id"],
                username=row["username"],
                is_admin=row["is_admin"],
                is_superuser=row["is_superuser"],
                must_change_password=row["must_change_password"],
            )

    def create_auth_session(
        self,
        user_id: UUID,
        user_agent: str = "",
        client_host: str = "",
    ) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.config.auth_session_ttl_seconds,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO auth_sessions (
                    token_hash,
                    user_id,
                    user_agent,
                    client_host,
                    expires_at
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    hash_token(token),
                    user_id,
                    user_agent[:300],
                    client_host[:120],
                    expires_at,
                ),
            )
        return token

    def get_user_by_token(self, token: str) -> CurrentUser | None:
        if not token:
            return None

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id,
                       u.username,
                       u.is_admin,
                       u.is_superuser,
                       u.must_change_password
                FROM auth_sessions s
                JOIN app_users u ON u.id = s.user_id
                WHERE s.token_hash = %s
                  AND s.expires_at > CURRENT_TIMESTAMP
                  AND u.is_active = TRUE
                """,
                (hash_token(token),),
            ).fetchone()
            if row is None:
                return None

            conn.execute(
                """
                UPDATE auth_sessions
                SET last_seen_at = CURRENT_TIMESTAMP
                WHERE token_hash = %s
                """,
                (hash_token(token),),
            )
            return CurrentUser(
                id=row["id"],
                username=row["username"],
                is_admin=row["is_admin"],
                is_superuser=row["is_superuser"],
                must_change_password=row["must_change_password"],
            )

    def list_users(self) -> list[UserRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT u.id,
                       u.username,
                       u.is_active,
                       u.is_admin,
                       u.is_superuser,
                       u.must_change_password,
                       u.created_at,
                       u.updated_at,
                       u.last_login_at,
                       COUNT(DISTINCT cs.id) AS session_count,
                       COUNT(cm.id) AS message_count
                FROM app_users u
                LEFT JOIN chat_sessions cs ON cs.user_id = u.id
                LEFT JOIN chat_messages cm ON cm.session_id = cs.id
                GROUP BY u.id
                ORDER BY u.username
                """
            ).fetchall()
        return [user_record_from_row(row) for row in rows]

    def create_user(
        self,
        username: str,
        password: str,
        is_admin: bool = False,
        must_change_password: bool = True,
    ) -> UserRecord:
        normalized = normalize_username(username)
        if not password:
            raise ValueError("Password must not be empty.")

        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO app_users (
                    id,
                    username,
                    password_hash,
                    is_active,
                    is_admin,
                    is_superuser,
                    must_change_password
                )
                VALUES (%s, %s, %s, TRUE, %s, FALSE, %s)
                ON CONFLICT (username) DO NOTHING
                RETURNING id,
                          username,
                          is_active,
                          is_admin,
                          is_superuser,
                          must_change_password,
                          created_at,
                          updated_at,
                          last_login_at,
                          0 AS session_count,
                          0 AS message_count
                """,
                (
                    uuid4(),
                    normalized,
                    hash_password(password),
                    is_admin,
                    must_change_password,
                ),
            ).fetchone()
            if row is None:
                raise ValueError("User already exists.")
        return user_record_from_row(row)

    def create_users(
        self,
        users: list[tuple[str, str]],
        is_admin: bool = False,
        must_change_password: bool = True,
    ) -> list[UserRecord]:
        normalized_users: list[tuple[str, str]] = []
        seen_usernames: set[str] = set()
        for username, password in users:
            normalized = normalize_username(username)
            if not password:
                raise ValueError("Password must not be empty.")
            if normalized in seen_usernames:
                raise ValueError(f"Duplicate user in CSV: {normalized}.")
            seen_usernames.add(normalized)
            normalized_users.append((normalized, password))

        created: list[UserRecord] = []
        with self._connect() as conn:
            for username, password in normalized_users:
                row = conn.execute(
                    """
                    INSERT INTO app_users (
                        id,
                        username,
                        password_hash,
                        is_active,
                        is_admin,
                        is_superuser,
                        must_change_password
                    )
                    VALUES (%s, %s, %s, TRUE, %s, FALSE, %s)
                    ON CONFLICT (username) DO NOTHING
                    RETURNING id,
                              username,
                              is_active,
                              is_admin,
                              is_superuser,
                              must_change_password,
                              created_at,
                              updated_at,
                              last_login_at,
                              0 AS session_count,
                              0 AS message_count
                    """,
                    (
                        uuid4(),
                        username,
                        hash_password(password),
                        is_admin,
                        must_change_password,
                    ),
                ).fetchone()
                if row is None:
                    raise ValueError(f"User already exists: {username}.")
                created.append(user_record_from_row(row))

        return created

    def set_user_password(
        self,
        user_id: UUID,
        password: str,
        must_change_password: bool = True,
    ) -> bool:
        if not password:
            raise ValueError("Password must not be empty.")

        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE app_users
                SET password_hash = %s,
                    must_change_password = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (hash_password(password), must_change_password, user_id),
            )
        return result.rowcount > 0

    def change_user_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
    ) -> bool:
        if not current_password:
            raise ValueError("Current password must not be empty.")
        if not new_password:
            raise ValueError("New password must not be empty.")

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT password_hash
                FROM app_users
                WHERE id = %s AND is_active = TRUE
                """,
                (user_id,),
            ).fetchone()
            if row is None:
                return False
            if not verify_password(current_password, row["password_hash"]):
                raise ValueError("Current password is incorrect.")

            result = conn.execute(
                """
                UPDATE app_users
                SET password_hash = %s,
                    must_change_password = FALSE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (hash_password(new_password), user_id),
            )
        return result.rowcount > 0

    def set_user_admin(self, user_id: UUID, is_admin: bool) -> bool:
        with self._connect() as conn:
            if not is_admin and self._is_superuser(conn, user_id):
                raise ValueError("Cannot remove administrator role from superuser.")
            if not is_admin and self._is_last_admin(conn, user_id):
                raise ValueError("Cannot remove the last administrator.")

            result = conn.execute(
                """
                UPDATE app_users
                SET is_admin = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (is_admin, user_id),
            )
        return result.rowcount > 0

    def delete_user(self, user_id: UUID) -> bool:
        with self._connect() as conn:
            if self._is_superuser(conn, user_id):
                raise ValueError("Cannot delete the superuser.")
            if self._is_last_admin(conn, user_id):
                raise ValueError("Cannot delete the last administrator.")

            result = conn.execute("DELETE FROM app_users WHERE id = %s", (user_id,))
        return result.rowcount > 0

    def clear_user_sessions(self, user_id: UUID) -> int:
        with self._connect() as conn:
            result = conn.execute(
                "DELETE FROM chat_sessions WHERE user_id = %s",
                (user_id,),
            )
        return result.rowcount

    def get_llm_runtime_settings(self) -> LLMRuntimeSettings:
        values = self._read_settings(
            [
                "llm_provider",
                "llm_base_url",
                "llm_api_key",
                "llm_model",
                "llm_context_max_tokens",
            ]
        )
        return LLMRuntimeSettings.from_settings(
            self.config,
            provider=values.get("llm_provider"),
            base_url=values.get("llm_base_url"),
            api_key=values.get("llm_api_key"),
            model=values.get("llm_model"),
            context_max_tokens=values.get("llm_context_max_tokens"),
        ).validate()

    def get_llm_settings_record(self) -> LLMSettingsRecord:
        values = self._read_settings(
            [
                "llm_provider",
                "llm_base_url",
                "llm_api_key",
                "llm_model",
                "llm_context_max_tokens",
            ]
        )
        runtime = LLMRuntimeSettings.from_settings(
            self.config,
            provider=values.get("llm_provider"),
            base_url=values.get("llm_base_url"),
            api_key=values.get("llm_api_key"),
            model=values.get("llm_model"),
            context_max_tokens=values.get("llm_context_max_tokens"),
        ).validate()
        return LLMSettingsRecord(
            provider=runtime.llm_provider,
            base_url=runtime.llm_base_url,
            model=runtime.llm_model,
            context_max_tokens=runtime.llm_context_max_tokens,
            api_key_configured=bool(runtime.llm_api_key),
            source="database" if values else ".env",
        )

    def update_llm_settings(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        context_max_tokens: int | None = None,
    ) -> LLMSettingsRecord:
        provider = normalize_llm_provider(provider)
        base_url = base_url.strip()
        model = model.strip()
        current = self.get_llm_runtime_settings()
        runtime = LLMRuntimeSettings.from_settings(
            self.config,
            provider=provider,
            base_url=base_url,
            api_key=current.llm_api_key if api_key is None else api_key.strip(),
            model=model,
            context_max_tokens=(
                current.llm_context_max_tokens
                if context_max_tokens is None
                else context_max_tokens
            ),
        ).validate()

        values = {
            "llm_provider": runtime.llm_provider,
            "llm_base_url": runtime.llm_base_url,
            "llm_model": runtime.llm_model,
            "llm_context_max_tokens": str(runtime.llm_context_max_tokens),
        }
        if api_key is not None:
            values["llm_api_key"] = runtime.llm_api_key

        with self._connect() as conn:
            for key, value in values.items():
                conn.execute(
                    """
                    INSERT INTO app_settings (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (key, value),
                )

        return self.get_llm_settings_record()

    def delete_auth_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM auth_sessions WHERE token_hash = %s",
                (hash_token(token),),
            )

    def delete_expired_auth_sessions(
        self,
        conn: psycopg.Connection[Any] | None = None,
    ) -> None:
        if conn is not None:
            conn.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= CURRENT_TIMESTAMP"
            )
            return

        with self._connect() as owned_conn:
            owned_conn.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= CURRENT_TIMESTAMP"
            )

    def list_chat_sessions(self, user_id: UUID) -> list[ChatSessionRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id,
                       user_id,
                       title,
                       conversation_summary,
                       compacted_message_count,
                       created_at,
                       updated_at
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, self.config.session_list_limit),
            ).fetchall()
        return [chat_session_from_row(row) for row in rows]

    def create_chat_session(
        self,
        user_id: UUID,
        title: str | None = None,
    ) -> ChatSessionRecord:
        session_id = uuid4()
        clean_title = clean_session_title(
            title or "New chat",
            self.config.session_title_max_chars,
        )
        with self._connect() as conn:
            row = conn.execute(
                """
                INSERT INTO chat_sessions (id, user_id, title)
                VALUES (%s, %s, %s)
                RETURNING id,
                          user_id,
                          title,
                          conversation_summary,
                          compacted_message_count,
                          created_at,
                          updated_at
                """,
                (session_id, user_id, clean_title),
            ).fetchone()
        return chat_session_from_row(row)

    def get_chat_session(
        self,
        user_id: UUID,
        session_id: UUID,
    ) -> ChatSessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id,
                       user_id,
                       title,
                       conversation_summary,
                       compacted_message_count,
                       created_at,
                       updated_at
                FROM chat_sessions
                WHERE id = %s AND user_id = %s
                """,
                (session_id, user_id),
            ).fetchone()
        return chat_session_from_row(row) if row else None

    def get_chat_session_by_id(self, session_id: UUID) -> ChatSessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id,
                       user_id,
                       title,
                       conversation_summary,
                       compacted_message_count,
                       created_at,
                       updated_at
                FROM chat_sessions
                WHERE id = %s
                """,
                (session_id,),
            ).fetchone()
        return chat_session_from_row(row) if row else None

    def rename_chat_session(
        self,
        user_id: UUID,
        session_id: UUID,
        title: str,
    ) -> ChatSessionRecord | None:
        clean_title = clean_session_title(
            title,
            self.config.session_title_max_chars,
        )
        with self._connect() as conn:
            row = conn.execute(
                """
                UPDATE chat_sessions
                SET title = %s
                WHERE id = %s AND user_id = %s
                RETURNING id,
                          user_id,
                          title,
                          conversation_summary,
                          compacted_message_count,
                          created_at,
                          updated_at
                """,
                (clean_title, session_id, user_id),
            ).fetchone()
        return chat_session_from_row(row) if row else None

    def delete_chat_session(self, user_id: UUID, session_id: UUID) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s",
                (session_id, user_id),
            )
        return result.rowcount > 0

    def list_messages(self, session_id: UUID) -> list[StoredChatMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id,
                       event_id,
                       session_seq,
                       role,
                       content,
                       contexts,
                       used_rag,
                       route,
                       route_reason,
                       retrieval_degraded,
                       embedding_degraded,
                       qdrant_degraded,
                       reranker_degraded,
                       degradation_reason,
                       created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY session_seq NULLS LAST, id
                """,
                (session_id,),
            ).fetchall()
        return [chat_message_from_row(row) for row in rows]

    def prompt_history(
        self,
        session_id: UUID,
        compacted_message_count: int,
    ) -> list[ChatMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role,
                       content,
                       contexts,
                       used_rag,
                       route
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY session_seq NULLS LAST, id
                OFFSET %s
                """,
                (session_id, compacted_message_count),
            ).fetchall()

        messages: list[ChatMessage] = []
        for row in rows:
            if row["role"] not in {"user", "assistant"}:
                continue
            contexts = row["contexts"] or []
            if isinstance(contexts, str):
                contexts = json.loads(contexts)
            messages.append(
                ChatMessage(
                    role=row["role"],
                    content=row["content"],
                    used_rag=row["used_rag"],
                    route=row["route"] or "",
                    context_count=len(contexts),
                )
            )
        return messages

    def append_message(
        self,
        session_id: UUID,
        role: Literal["user", "assistant"],
        content: str,
        contexts: list[SearchResult] | None = None,
        used_rag: bool | None = None,
        route: str = "",
        route_reason: str = "",
        retrieval_degraded: bool = False,
        embedding_degraded: bool = False,
        qdrant_degraded: bool = False,
        reranker_degraded: bool = False,
        degradation_reason: str = "",
    ) -> StoredChatMessage:
        context_payload = [
            search_result_to_dict(context)
            for context in contexts or []
        ]
        with self._connect() as conn:
            session_row = conn.execute(
                "SELECT archive_seq FROM chat_sessions WHERE id = %s FOR UPDATE",
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise ValueError("Chat session not found.")
            session_seq = next_session_seq(
                int(session_row["archive_seq"] or 0),
                floor_offset=500,
            )
            event_id = uuid4()
            row = conn.execute(
                """
                INSERT INTO chat_messages (
                    session_id,
                    event_id,
                    session_seq,
                    role,
                    content,
                    contexts,
                    used_rag,
                    route,
                    route_reason,
                    retrieval_degraded,
                    embedding_degraded,
                    qdrant_degraded,
                    reranker_degraded,
                    degradation_reason
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s::jsonb, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                RETURNING id,
                          event_id,
                          session_seq,
                          role,
                          content,
                          contexts,
                          used_rag,
                          route,
                          route_reason,
                          retrieval_degraded,
                          embedding_degraded,
                          qdrant_degraded,
                          reranker_degraded,
                          degradation_reason,
                          created_at
                """,
                (
                    session_id,
                    event_id,
                    session_seq,
                    role,
                    content,
                    json.dumps(context_payload),
                    used_rag,
                    route,
                    route_reason,
                    retrieval_degraded,
                    embedding_degraded,
                    qdrant_degraded,
                    reranker_degraded,
                    degradation_reason,
                ),
            ).fetchone()
            conn.execute(
                """
                UPDATE chat_sessions
                SET archive_seq = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (session_seq, session_id),
            )
        return chat_message_from_row(row)

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
        """Persist one user/assistant exchange in a single PostgreSQL transaction."""
        with self._connect() as conn:
            session_row = conn.execute(
                """
                SELECT id,
                       user_id,
                       title,
                       conversation_summary,
                       compacted_message_count,
                       archive_seq,
                       created_at,
                       updated_at
                FROM chat_sessions
                WHERE id = %s
                FOR UPDATE
                """,
                (session_id,),
            ).fetchone()
            if session_row is None:
                raise ValueError("Chat session not found.")

            last_seq = int(session_row["archive_seq"] or 0)
            first_seq = next_session_seq(last_seq, floor_offset=500)
            updated_at = datetime.now(timezone.utc)
            clean_title = (
                clean_session_title(title, self.config.session_title_max_chars)
                if title
                else str(session_row["title"])
            )
            event = {
                "event_id": str(uuid4()),
                "session_id": str(session_id),
                "last_seq": first_seq + 1,
                "messages": [
                    self._archive_message_payload(
                        event_id=str(uuid4()),
                        session_seq=first_seq,
                        role="user",
                        content=user_content,
                        created_at=updated_at,
                    ),
                    self._archive_message_payload(
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
                    ),
                ],
                "session": {
                    "conversation_summary": conversation_summary,
                    "compacted_message_count": int(
                        session_row["compacted_message_count"] or 0
                    )
                    + max(0, compacted_delta),
                    "title": clean_title,
                    "updated_at": updated_at.isoformat(),
                },
            }
            self._archive_session_event(conn, event)
            row = conn.execute(
                """
                SELECT id,
                       user_id,
                       title,
                       conversation_summary,
                       compacted_message_count,
                       created_at,
                       updated_at
                FROM chat_sessions
                WHERE id = %s
                """,
                (session_id,),
            ).fetchone()
        return chat_session_from_row(row)

    def archive_session_events(self, events: list[dict[str, Any]]) -> int:
        """Idempotently archive Redis exchange events in one PG transaction."""
        archived = 0
        with self._connect() as conn:
            for event in events:
                if self._archive_session_event(conn, event):
                    archived += 1
        return archived

    def _archive_session_event(
        self,
        conn: psycopg.Connection[Any],
        event: dict[str, Any],
    ) -> bool:
        session_id = UUID(str(event["session_id"]))
        session_exists = conn.execute(
            "SELECT 1 FROM chat_sessions WHERE id = %s",
            (session_id,),
        ).fetchone()
        if session_exists is None:
            return False

        messages = event.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("Session archive event messages must be a list.")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Session archive message must be an object.")
            conn.execute(
                """
                INSERT INTO chat_messages (
                    session_id,
                    event_id,
                    session_seq,
                    role,
                    content,
                    contexts,
                    used_rag,
                    route,
                    route_reason,
                    retrieval_degraded,
                    embedding_degraded,
                    qdrant_degraded,
                    reranker_degraded,
                    degradation_reason,
                    created_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT DO NOTHING
                """,
                (
                    session_id,
                    UUID(str(message["event_id"])),
                    int(message["session_seq"]),
                    message["role"],
                    str(message.get("content", "")),
                    json.dumps(message.get("contexts", [])),
                    message.get("used_rag"),
                    str(message.get("route", "")),
                    str(message.get("route_reason", "")),
                    bool(message.get("retrieval_degraded", False)),
                    bool(message.get("embedding_degraded", False)),
                    bool(message.get("qdrant_degraded", False)),
                    bool(message.get("reranker_degraded", False)),
                    str(message.get("degradation_reason", "")),
                    self._archive_datetime(message.get("created_at")),
                ),
            )

        session_payload = event.get("session", {})
        if not isinstance(session_payload, dict):
            raise ValueError("Session archive state must be an object.")
        last_seq = int(event.get("last_seq", 0) or 0)
        conn.execute(
            """
            UPDATE chat_sessions
            SET conversation_summary = %s,
                compacted_message_count = %s,
                title = CASE
                    WHEN title = 'New chat' THEN %s
                    ELSE title
                END,
                updated_at = %s,
                archive_seq = %s
            WHERE id = %s
              AND archive_seq < %s
            """,
            (
                str(session_payload.get("conversation_summary", "")),
                max(0, int(session_payload.get("compacted_message_count", 0) or 0)),
                clean_session_title(
                    str(session_payload.get("title", "New chat")),
                    self.config.session_title_max_chars,
                ),
                self._archive_datetime(session_payload.get("updated_at")),
                last_seq,
                session_id,
                last_seq,
            ),
        )
        return True

    @staticmethod
    def _archive_datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    @staticmethod
    def _archive_message_payload(
        *,
        event_id: str,
        session_seq: int,
        role: Literal["user", "assistant"],
        content: str,
        contexts: list[SearchResult] | None = None,
        used_rag: bool | None = None,
        route: str = "",
        route_reason: str = "",
        retrieval_degraded: bool = False,
        embedding_degraded: bool = False,
        qdrant_degraded: bool = False,
        reranker_degraded: bool = False,
        degradation_reason: str = "",
        created_at: datetime | None = None,
    ) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "session_seq": session_seq,
            "role": role,
            "content": content,
            "contexts": [search_result_to_dict(item) for item in contexts or []],
            "used_rag": used_rag,
            "route": route,
            "route_reason": route_reason,
            "retrieval_degraded": retrieval_degraded,
            "embedding_degraded": embedding_degraded,
            "qdrant_degraded": qdrant_degraded,
            "reranker_degraded": reranker_degraded,
            "degradation_reason": degradation_reason,
            "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        }

    def update_chat_session_after_answer(
        self,
        session_id: UUID,
        conversation_summary: str,
        compacted_delta: int,
        title: str | None = None,
    ) -> ChatSessionRecord:
        assignments = [
            "conversation_summary = %s",
            "compacted_message_count = compacted_message_count + %s",
            "updated_at = CURRENT_TIMESTAMP",
        ]
        values: list[Any] = [conversation_summary, max(0, compacted_delta)]

        if title:
            assignments.append(
                """
                title = CASE
                    WHEN title = 'New chat' THEN %s
                    ELSE title
                END
                """
            )
            values.append(
                clean_session_title(title, self.config.session_title_max_chars)
            )

        values.append(session_id)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                UPDATE chat_sessions
                SET {", ".join(assignments)}
                WHERE id = %s
                RETURNING id,
                          user_id,
                          title,
                          conversation_summary,
                          compacted_message_count,
                          created_at,
                          updated_at
                """,
                values,
            ).fetchone()
        return chat_session_from_row(row)

    def close(self) -> None:
        return

    def _is_last_admin(self, conn: psycopg.Connection[Any], user_id: UUID) -> bool:
        row = conn.execute(
            "SELECT is_admin FROM app_users WHERE id = %s",
            (user_id,),
        ).fetchone()
        if row is None or not row["is_admin"]:
            return False

        admin_count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM app_users
            WHERE is_admin = TRUE AND is_active = TRUE
            """
        ).fetchone()["count"]
        return int(admin_count) <= 1

    def _is_superuser(self, conn: psycopg.Connection[Any], user_id: UUID) -> bool:
        row = conn.execute(
            "SELECT is_superuser FROM app_users WHERE id = %s",
            (user_id,),
        ).fetchone()
        return bool(row and row["is_superuser"])

    def _read_settings(self, keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT key, value
                FROM app_settings
                WHERE key = ANY(%s)
                """,
                (keys,),
            ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(
            self.config.database_url,
            row_factory=dict_row,
            connect_timeout=self.config.database_connect_timeout_seconds,
        )


def normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if not _USERNAME_RE.fullmatch(normalized):
        raise ValueError(
            "Usernames may contain letters, numbers, underscore, dash, dot, or @."
        )
    return normalized


def parse_bootstrap_users(raw: str) -> list[tuple[str, str]]:
    users: list[tuple[str, str]] = []
    for item in re.split(r"[,;\n]+", raw or ""):
        if not item.strip():
            continue
        username, separator, password = item.partition(":")
        if separator != ":" or not username.strip() or not password:
            raise ValueError(
                "AUTH_BOOTSTRAP_USERS entries must use username:password."
            )
        users.append((normalize_username(username), password.strip()))
    return users


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PASSWORD_ITERATIONS,
    )
    return "$".join(
        [
            _PASSWORD_SCHEME,
            str(_PASSWORD_ITERATIONS),
            _b64encode(salt),
            _b64encode(digest),
        ]
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations_text, salt_text, digest_text = stored_hash.split("$", 3)
        if scheme != _PASSWORD_SCHEME:
            return False
        iterations = int(iterations_text)
        salt = _b64decode(salt_text)
        expected_digest = _b64decode(digest_text)
    except (ValueError, TypeError):
        return False

    candidate_digest = pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return compare_digest(candidate_digest, expected_digest)


def hash_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def clean_session_title(title: str, max_chars: int) -> str:
    collapsed = " ".join(title.strip().split())
    if not collapsed:
        return "New chat"
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[:max_chars].rstrip()


def next_session_seq(last_seq: int, *, floor_offset: int = 0) -> int:
    if not 0 <= floor_offset < 1000:
        raise ValueError("Session sequence floor offset must be between 0 and 999.")
    # Redis starts at 0 and explicit PostgreSQL-only mode starts at 500 per millisecond.
    clock_floor = (time.time_ns() // 1_000_000) * 1000 + floor_offset
    return max(int(last_seq) + 1, clock_floor)


def chat_session_from_row(row: dict[str, Any]) -> ChatSessionRecord:
    return ChatSessionRecord(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        conversation_summary=row["conversation_summary"] or "",
        compacted_message_count=row["compacted_message_count"] or 0,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def user_record_from_row(row: dict[str, Any]) -> UserRecord:
    return UserRecord(
        id=row["id"],
        username=row["username"],
        is_active=row["is_active"],
        is_admin=row["is_admin"],
        is_superuser=row["is_superuser"],
        must_change_password=row["must_change_password"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_login_at=row["last_login_at"],
        session_count=int(row["session_count"] or 0),
        message_count=int(row["message_count"] or 0),
    )


def chat_message_from_row(row: dict[str, Any]) -> StoredChatMessage:
    contexts = row["contexts"] or []
    if isinstance(contexts, str):
        contexts = json.loads(contexts)
    return StoredChatMessage(
        id=int(row.get("session_seq") or row["id"]),
        role=row["role"],
        content=row["content"],
        contexts=contexts,
        used_rag=row["used_rag"],
        route=row["route"] or "",
        route_reason=row["route_reason"] or "",
        retrieval_degraded=bool(row["retrieval_degraded"]),
        embedding_degraded=bool(row.get("embedding_degraded", False)),
        qdrant_degraded=bool(row["qdrant_degraded"]),
        reranker_degraded=bool(row["reranker_degraded"]),
        degradation_reason=row["degradation_reason"] or "",
        created_at=row["created_at"],
        event_id=str(row.get("event_id") or ""),
        session_seq=int(row.get("session_seq") or 0),
    )


def search_result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "source": result.source,
        "chunk_id": result.chunk_id,
        "score": result.score,
        "rerank_score": result.rerank_score,
        "vector_score": result.vector_score,
        "bm25_score": result.bm25_score,
        "rrf_score": result.rrf_score,
        "retrieval_source": result.retrieval_source,
        "content_type": result.content_type,
        "h1": result.h1,
        "h2": result.h2,
        "h3": result.h3,
        "headings": list(result.headings),
        "start_line": result.start_line,
        "end_line": result.end_line,
    }


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)

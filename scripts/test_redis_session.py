from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import sys
import time
from typing import Any

from redis.exceptions import ConnectionError as RedisConnectionError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.redis_session_store import (  # noqa: E402
    RedisSessionStore,
    SessionStoreUnavailableError,
)
from app.session_store import SessionStore  # noqa: E402


class CommitThenDisconnectPipeline:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pipeline, name)

    def execute(self) -> None:
        self.pipeline.execute()
        raise RedisConnectionError("simulated disconnect after EXEC")


class CommitThenDisconnectClient:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.failed_once = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def pipeline(self, *args: object, **kwargs: object) -> Any:
        pipeline = self.client.pipeline(*args, **kwargs)
        if self.failed_once:
            return pipeline
        self.failed_once = True
        return CommitThenDisconnectPipeline(pipeline)


def main() -> None:
    if not settings.redis_session_enabled:
        raise RuntimeError("REDIS_SESSION_ENABLED must be enabled for this smoke test.")

    pg_store = SessionStore(settings)
    pg_store.init_db()
    store = RedisSessionStore(settings, pg_store=pg_store)
    store.init_cache()

    with pg_store._connect() as conn:  # noqa: SLF001 - operational smoke test
        user_row = conn.execute(
            "SELECT id FROM app_users ORDER BY created_at LIMIT 1"
        ).fetchone()
    if user_row is None:
        raise RuntimeError("At least one application user is required.")

    session = store.create_chat_session(user_row["id"], "Redis smoke test")
    try:
        write_started = time.perf_counter()
        session = store.append_exchange(
            session.id,
            "redis smoke user message",
            "redis smoke assistant message",
            used_rag=False,
            route="direct",
            conversation_summary="redis smoke summary",
            title="Redis smoke test",
        )
        redis_write_ms = (time.perf_counter() - write_started) * 1000

        hot_messages = store.list_messages(session.id)
        hot_sequences = [item.session_seq for item in hot_messages]
        if len(hot_sequences) != 2 or hot_sequences[1] != hot_sequences[0] + 1:
            raise AssertionError("Redis hot history should expose consecutive sequences.")
        if session.conversation_summary != "redis smoke summary":
            raise AssertionError("Redis hot metadata should contain the new summary.")

        archive_started = time.perf_counter()
        deadline = archive_started + 10
        archived_count = 0
        while time.perf_counter() < deadline:
            with pg_store._connect() as conn:  # noqa: SLF001
                archived_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM chat_messages
                        WHERE session_id = %s AND event_id IS NOT NULL
                        """,
                        (session.id,),
                    ).fetchone()["count"]
                )
            if archived_count == 2:
                break
            time.sleep(0.1)
        if archived_count != 2:
            raise AssertionError("Archive worker did not persist both messages.")
        archive_ms = (time.perf_counter() - archive_started) * 1000

        pending_count = int(store.client.hlen(store._pending_key(session.id)))  # noqa: SLF001
        if pending_count != 0:
            raise AssertionError("Pending messages should clear after PG commit.")

        store.client.delete(
            store._meta_key(session.id),  # noqa: SLF001
            store._messages_key(session.id),  # noqa: SLF001
        )
        cold_session = store.get_chat_session(session.user_id, session.id)
        cold_messages = store.list_messages(session.id)
        if cold_session is None or len(cold_messages) != 2:
            raise AssertionError("Cold load should restore the full PG session into Redis.")

        real_client = store.client
        store.client = CommitThenDisconnectClient(real_client)
        try:
            store.append_exchange(
                session.id,
                "ambiguous commit user",
                "ambiguous commit assistant",
                used_rag=False,
                route="direct",
                conversation_summary="ambiguous commit recovered",
            )
        except SessionStoreUnavailableError:
            pass
        else:
            raise AssertionError("An uncertain Redis commit must report unavailable.")
        finally:
            store.client = real_client
        store.ping()

        deadline = time.perf_counter() + 10
        while time.perf_counter() < deadline:
            pending_count = int(
                store.client.hlen(store._pending_key(session.id))  # noqa: SLF001
            )
            if pending_count == 0:
                break
            time.sleep(0.1)
        archived_messages = pg_store.list_messages(session.id)
        if len(archived_messages) != 4 or pending_count != 0:
            raise AssertionError(
                "A committed Redis event must still archive once after a 503 response."
            )

        print(
            "Redis session smoke -> ok "
            f"redis_write_ms={redis_write_ms:.2f} archive_ms={archive_ms:.2f}"
        )
    finally:
        store.delete_chat_session(session.user_id, session.id)
        store.close()

    unavailable_session = pg_store.create_chat_session(
        user_row["id"],
        "Redis unavailable test",
    )
    unavailable_config = replace(
        settings,
        redis_url="redis://127.0.0.1:1/0",
        redis_socket_timeout_seconds=0.1,
    )
    unavailable_store = RedisSessionStore(unavailable_config, pg_store=pg_store)
    try:
        try:
            unavailable_store.append_exchange(
                unavailable_session.id,
                "unavailable user",
                "unavailable assistant",
                used_rag=False,
                route="direct",
            )
        except SessionStoreUnavailableError:
            pass
        else:
            raise AssertionError("Redis failure must reject the session write.")
        if pg_store.list_messages(unavailable_session.id):
            raise AssertionError("Redis failure must not fall back to PostgreSQL.")
        fast_fail_started = time.perf_counter()
        try:
            unavailable_store.list_messages(unavailable_session.id)
        except SessionStoreUnavailableError:
            pass
        else:
            raise AssertionError("The Redis circuit breaker must reject session reads.")
        fast_fail_ms = (time.perf_counter() - fast_fail_started) * 1000
        if fast_fail_ms >= 50:
            raise AssertionError("Known Redis outages should fail without a socket wait.")
        print("Redis unavailable strict consistency -> ok")
        print(f"Redis unavailable fast 503 -> ok elapsed_ms={fast_fail_ms:.2f}")
    finally:
        pg_store.delete_chat_session(
            unavailable_session.user_id,
            unavailable_session.id,
        )
        unavailable_store.close()


if __name__ == "__main__":
    main()

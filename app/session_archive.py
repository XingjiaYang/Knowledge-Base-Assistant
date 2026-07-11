from __future__ import annotations

import json
import logging
import os
import signal
import socket
import time
from typing import Any

from redis import Redis
from redis.exceptions import RedisError, ResponseError

from app.config import Settings, settings
from app.session_store import SessionStore


logger = logging.getLogger(__name__)


class SessionArchiveWorker:
    def __init__(
        self,
        config: Settings = settings,
        *,
        redis_client: Redis | None = None,
        pg_store: SessionStore | None = None,
    ) -> None:
        self.config = config
        self.redis = redis_client or Redis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_timeout=config.redis_socket_timeout_seconds,
            socket_connect_timeout=config.redis_socket_timeout_seconds,
            health_check_interval=30,
        )
        self.pg_store = pg_store or SessionStore(config)
        self.stream_key = self._key(config.redis_archive_stream)
        self.group = config.redis_archive_group
        self.consumer = f"{socket.gethostname()}-{os.getpid()}"
        self.heartbeat_key = self._key("session:archiver:heartbeat")
        self.heartbeat_file = "/tmp/session-archiver-heartbeat"
        self._stopping = False

    def run(self) -> None:
        self.pg_store.init_db()
        self._ensure_group()
        self._install_signal_handlers()
        logger.info(
            "Session archive worker started stream=%s group=%s consumer=%s batch=%s.",
            self.stream_key,
            self.group,
            self.consumer,
            self.config.redis_archive_batch_size,
        )
        while not self._stopping:
            try:
                self._heartbeat()
                entries = self._claim_stale_entries()
                if not entries:
                    entries = self._read_new_entries()
                if entries:
                    self._archive_entries(entries)
            except (RedisError, OSError):
                logger.exception("Session archive worker transport failure; retrying.")
                time.sleep(1)
            except Exception:
                logger.exception("Session archive worker batch failed; retrying.")
                time.sleep(1)

        self.redis.close()
        self.pg_store.close()

    def _ensure_group(self) -> None:
        try:
            self.redis.xgroup_create(
                self.stream_key,
                self.group,
                id="0-0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _claim_stale_entries(self) -> list[tuple[str, dict[str, str]]]:
        response = self.redis.xautoclaim(
            self.stream_key,
            self.group,
            self.consumer,
            min_idle_time=self.config.redis_archive_claim_idle_ms,
            start_id="0-0",
            count=self.config.redis_archive_batch_size,
        )
        if not response or len(response) < 2:
            return []
        return [(str(item_id), dict(fields)) for item_id, fields in response[1]]

    def _read_new_entries(self) -> list[tuple[str, dict[str, str]]]:
        response = self.redis.xreadgroup(
            self.group,
            self.consumer,
            {self.stream_key: ">"},
            count=self.config.redis_archive_batch_size,
            block=self.config.redis_archive_block_ms,
        )
        if not response:
            return []
        entries: list[tuple[str, dict[str, str]]] = []
        for _stream_name, stream_entries in response:
            entries.extend(
                (str(item_id), dict(fields))
                for item_id, fields in stream_entries
            )
        return entries

    def _archive_entries(self, entries: list[tuple[str, dict[str, str]]]) -> None:
        valid_entries: list[tuple[str, str, str, dict[str, Any]]] = []
        malformed_ids: list[str] = []
        pending_entries: list[tuple[str, str, str]] = []
        for stream_id, fields in entries:
            try:
                event_id = str(fields["event_id"])
                session_id = str(fields["session_id"])
                if not event_id or not session_id:
                    raise ValueError("Archive event identifiers must not be empty.")
                legacy_payload = fields.get("payload")
                if legacy_payload is None:
                    pending_entries.append((stream_id, event_id, session_id))
                else:
                    valid_entries.append(
                        (
                            stream_id,
                            event_id,
                            session_id,
                            self._decode_payload(legacy_payload),
                        )
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                logger.exception(
                    "Dropping malformed internal session archive event stream_id=%s.",
                    stream_id,
                )
                malformed_ids.append(stream_id)

        if pending_entries:
            pipe = self.redis.pipeline(transaction=False)
            for _stream_id, event_id, session_id in pending_entries:
                pipe.hget(self._key(f"session:{session_id}:pending"), event_id)
            pending_payloads = pipe.execute()
            for entry, raw_payload in zip(pending_entries, pending_payloads, strict=True):
                stream_id, event_id, session_id = entry
                if raw_payload is None:
                    logger.error(
                        "Dropping session archive reference with no Pending payload "
                        "stream_id=%s event_id=%s session_id=%s.",
                        stream_id,
                        event_id,
                        session_id,
                    )
                    malformed_ids.append(stream_id)
                    continue
                try:
                    valid_entries.append(
                        (
                            stream_id,
                            event_id,
                            session_id,
                            self._decode_payload(raw_payload),
                        )
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.exception(
                        "Dropping malformed Pending session archive payload "
                        "stream_id=%s event_id=%s.",
                        stream_id,
                        event_id,
                    )
                    malformed_ids.append(stream_id)

        if malformed_ids:
            pipe = self.redis.pipeline(transaction=True)
            for stream_id in malformed_ids:
                pipe.xack(self.stream_key, self.group, stream_id)
                pipe.xdel(self.stream_key, stream_id)
            pipe.execute()

        if not valid_entries:
            return

        self.pg_store.archive_session_events(
            [payload for _stream_id, _event_id, _session_id, payload in valid_entries]
        )

        pipe = self.redis.pipeline(transaction=True)
        for stream_id, event_id, session_id, _payload in valid_entries:
            pipe.hdel(self._key(f"session:{session_id}:pending"), event_id)
            pipe.xack(self.stream_key, self.group, stream_id)
            pipe.xdel(self.stream_key, stream_id)
        pipe.execute()
        logger.info("Archived %s Redis session events to PostgreSQL.", len(valid_entries))

    @staticmethod
    def _decode_payload(raw_payload: str) -> dict[str, Any]:
        payload = json.loads(raw_payload)
        if not isinstance(payload, dict):
            raise ValueError("Archive payload must be an object.")
        return payload

    def _heartbeat(self) -> None:
        self.redis.set(
            self.heartbeat_key,
            str(time.time()),
            ex=self.config.redis_archiver_heartbeat_ttl_seconds,
        )
        with open(self.heartbeat_file, "a", encoding="ascii"):
            os.utime(self.heartbeat_file)

    def _install_signal_handlers(self) -> None:
        def stop(_signum: int, _frame: object) -> None:
            self._stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)

    def _key(self, suffix: str) -> str:
        return f"{self.config.redis_key_prefix}:{suffix}"


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    SessionArchiveWorker(settings).run()


if __name__ == "__main__":
    main()

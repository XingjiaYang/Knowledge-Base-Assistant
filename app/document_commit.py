from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from threading import Condition
from typing import Iterator, Protocol

import httpx


@dataclass(frozen=True)
class RequestGateSnapshot:
    blocked: bool
    active_requests: int
    queued_requests: int
    generation: int

    def as_dict(self) -> dict[str, object]:
        return {
            "blocked": self.blocked,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "generation": self.generation,
        }


class RetrievalRequestGate:
    """Queues new retrievals while an index-version pointer swap is committed."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._blocked = False
        self._active_requests = 0
        self._queued_requests = 0
        self._generation = 0

    @contextmanager
    def request(self) -> Iterator[None]:
        queued = False
        with self._condition:
            if self._blocked:
                self._queued_requests += 1
                queued = True
            try:
                while self._blocked:
                    self._condition.wait()
            finally:
                if queued:
                    self._queued_requests -= 1
            self._active_requests += 1

        try:
            yield
        finally:
            with self._condition:
                self._active_requests -= 1
                if self._active_requests == 0:
                    self._condition.notify_all()

    @contextmanager
    def exclusive(self, *, drain_timeout_seconds: float) -> Iterator[None]:
        self._close_and_drain(drain_timeout_seconds)
        try:
            yield
        finally:
            self._open()

    def _close_and_drain(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            if self._blocked:
                raise RuntimeError("Document commit request gate is already closed.")
            self._blocked = True
            self._generation += 1
            try:
                while self._active_requests:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Timed out draining active RAG requests before index commit."
                        )
                    self._condition.wait(timeout=remaining)
            except BaseException:
                self._blocked = False
                self._condition.notify_all()
                raise

    def _open(self) -> None:
        with self._condition:
            self._blocked = False
            self._condition.notify_all()

    def snapshot(self) -> RequestGateSnapshot:
        with self._condition:
            return RequestGateSnapshot(
                blocked=self._blocked,
                active_requests=self._active_requests,
                queued_requests=self._queued_requests,
                generation=self._generation,
            )


@dataclass(frozen=True)
class PreparedIndexCommit:
    index_version: str
    candidate_collection: str
    previous_collection: str
    expected_total_chunks: int


class DocumentIndexCommitter(Protocol):
    def prepare(self, index_version: str) -> int: ...

    def commit(self, prepared: PreparedIndexCommit) -> str: ...

    def discard(self, index_version: str) -> None: ...


class HTTPDocumentIndexCommitter:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        timeout_seconds: float,
        attempts: int = 3,
    ) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.attempts = max(1, attempts)

    def prepare(self, index_version: str) -> int:
        body = self._post(
            f"{self.url}/prepare",
            {"index_version": index_version},
        )
        return int(body.get("bm25_chunks", 0) or 0)

    def commit(self, prepared: PreparedIndexCommit) -> str:
        payload = {
            "index_version": prepared.index_version,
            "candidate_collection": prepared.candidate_collection,
            "previous_collection": prepared.previous_collection,
            "expected_total_chunks": prepared.expected_total_chunks,
        }
        body = self._post(f"{self.url}/commit", payload)
        return str(body.get("previous_collection", ""))

    def discard(self, index_version: str) -> None:
        self._post(
            f"{self.url}/discard",
            {"index_version": index_version},
        )

    def _post(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        headers = {"X-KBA-Docs-Commit-Token": self.token}
        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise RuntimeError("Main document commit endpoint returned invalid JSON.")
                return body
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.attempts:
                    time.sleep(min(2.0, 0.5 * attempt))
                    continue
                raise RuntimeError(
                    "Main document commit endpoint remained unavailable."
                ) from exc
        raise RuntimeError("Main document commit endpoint failed.") from last_error

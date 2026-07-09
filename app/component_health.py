from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import logging
from threading import Lock
import time

from app.config import Settings, settings


logger = logging.getLogger(__name__)

Probe = Callable[[], object]


@dataclass(frozen=True)
class ComponentHealthSnapshot:
    name: str
    degraded: bool
    status: str
    consecutive_failures: int
    consecutive_successes: int
    last_error: str
    last_checked_at: float
    last_changed_at: float
    probe_interval_seconds: float


class ComponentHealthState:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int,
        recovery_threshold: int,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_threshold = max(1, recovery_threshold)
        self._lock = Lock()
        self._degraded = False
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_error = ""
        self._last_checked_at = 0.0
        self._last_changed_at = time.time()

    @property
    def degraded(self) -> bool:
        with self._lock:
            return self._degraded

    def record_success(self) -> bool:
        changed = False
        with self._lock:
            self._last_checked_at = time.time()
            self._last_error = ""
            self._consecutive_failures = 0
            self._consecutive_successes += 1
            if (
                self._degraded
                and self._consecutive_successes >= self.recovery_threshold
            ):
                self._degraded = False
                self._last_changed_at = self._last_checked_at
                changed = True
        return changed

    def record_failure(self, error: Exception) -> bool:
        changed = False
        with self._lock:
            self._last_checked_at = time.time()
            self._last_error = str(error)
            self._consecutive_successes = 0
            self._consecutive_failures += 1
            if (
                not self._degraded
                and self._consecutive_failures >= self.failure_threshold
            ):
                self._degraded = True
                self._last_changed_at = self._last_checked_at
                changed = True
        return changed

    def snapshot(self, *, normal_interval: float, degraded_interval: float) -> ComponentHealthSnapshot:
        with self._lock:
            interval = degraded_interval if self._degraded else normal_interval
            if self._degraded:
                status = "degraded"
            elif self._last_checked_at <= 0:
                status = "unknown"
            else:
                status = "healthy"
            return ComponentHealthSnapshot(
                name=self.name,
                degraded=self._degraded,
                status=status,
                consecutive_failures=self._consecutive_failures,
                consecutive_successes=self._consecutive_successes,
                last_error=self._last_error,
                last_checked_at=self._last_checked_at,
                last_changed_at=self._last_changed_at,
                probe_interval_seconds=interval,
            )


class ComponentHealthSupervisor:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._components: dict[str, tuple[ComponentHealthState, Probe]] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._started = False

    def register(self, name: str, probe: Probe) -> None:
        if self._started:
            raise RuntimeError("Cannot register health probes after start().")
        self._components[name] = (
            ComponentHealthState(
                name,
                failure_threshold=self.config.health_probe_failure_threshold,
                recovery_threshold=self.config.health_probe_recovery_threshold,
            ),
            probe,
        )

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for name in sorted(self._components):
            state, probe = self._components[name]
            self._tasks.append(asyncio.create_task(self._run(name, state, probe)))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._started = False

    def is_degraded(self, name: str) -> bool:
        component = self._components.get(name)
        if component is None:
            return False
        state, _probe = component
        return state.degraded

    def snapshots(self) -> dict[str, ComponentHealthSnapshot]:
        return {
            name: state.snapshot(
                normal_interval=self.config.health_probe_interval_seconds,
                degraded_interval=self.config.health_probe_degraded_interval_seconds,
            )
            for name, (state, _probe) in self._components.items()
        }

    async def _run(
        self,
        name: str,
        state: ComponentHealthState,
        probe: Probe,
    ) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(probe),
                    timeout=self.config.health_probe_timeout_seconds,
                )
                if state.record_success():
                    logger.info("Component health recovered: %s.", name)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if state.record_failure(exc):
                    logger.warning(
                        "Component health degraded: %s after %s consecutive failures: %s",
                        name,
                        self.config.health_probe_failure_threshold,
                        exc,
                    )

            snapshot = state.snapshot(
                normal_interval=self.config.health_probe_interval_seconds,
                degraded_interval=self.config.health_probe_degraded_interval_seconds,
            )
            await asyncio.sleep(snapshot.probe_interval_seconds)

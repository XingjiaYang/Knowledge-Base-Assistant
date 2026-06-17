from __future__ import annotations

import logging
from threading import Lock
import time
from typing import Callable, Literal

import httpx

from app.config import LLMRuntimeSettings, Settings, settings


logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


class LLMClient:
    def __init__(
        self,
        config: Settings = settings,
        transport: httpx.BaseTransport | None = None,
        runtime_settings_provider: Callable[[], LLMRuntimeSettings] | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self.runtime_settings_provider = runtime_settings_provider
        self._client: httpx.Client | None = None
        self._client_lock = Lock()

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = self._normalize_messages(messages)
        config = self._active_config()
        if config.llm_provider.lower() == "anthropic":
            return self._anthropic_chat(
                config,
                messages,
                temperature,
                top_p,
                max_tokens,
            )
        return self._openai_compatible_chat(
            config,
            messages,
            temperature,
            top_p,
            max_tokens,
        )

    def _openai_compatible_chat(
        self,
        config: LLMRuntimeSettings,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url = f"{config.llm_base_url.rstrip('/')}/chat/completions"
        headers = self._bearer_headers(config)
        payload = {
            "model": config.llm_model,
            "messages": messages,
            "temperature": config.llm_temperature
            if temperature is None
            else temperature,
            "top_p": config.llm_top_p if top_p is None else top_p,
            "max_tokens": config.llm_max_tokens
            if max_tokens is None
            else max_tokens,
        }

        data = self._post_json_with_retries(config, url, headers, payload)
        return data["choices"][0]["message"]["content"]

    def _anthropic_chat(
        self,
        config: LLMRuntimeSettings,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url = f"{config.llm_base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": config.llm_api_key,
            "anthropic-version": config.llm_anthropic_version,
            "Content-Type": "application/json",
        }
        system_prompt, anthropic_messages = self._to_anthropic_messages(messages)
        payload = {
            "model": config.llm_model,
            "messages": anthropic_messages,
            "temperature": config.llm_temperature
            if temperature is None
            else temperature,
            "top_p": config.llm_top_p if top_p is None else top_p,
            "max_tokens": config.llm_max_tokens
            if max_tokens is None
            else max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt

        data = self._post_json_with_retries(config, url, headers, payload)
        content = data.get("content", [])
        return "".join(
            item.get("text", "")
            for item in content
            if item.get("type") == "text"
        )

    def health(self) -> bool:
        config = self._active_config()
        if not config.llm_health_check_enabled:
            return True

        method, url, payload = self._health_request(config)
        headers = self._health_headers(config)
        try:
            self._request_with_retries(
                config,
                method,
                url,
                headers=headers,
                json_payload=payload,
                timeout=5.0,
                max_attempts=1,
            )
        except httpx.HTTPError:
            logger.exception("LLM health check failed.")
            return False
        return True

    def _active_config(self) -> LLMRuntimeSettings:
        if self.runtime_settings_provider is not None:
            return self.runtime_settings_provider().validate()
        return LLMRuntimeSettings.from_settings(self.config).validate()

    def _bearer_headers(self, config: LLMRuntimeSettings) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if config.llm_api_key:
            headers["Authorization"] = f"Bearer {config.llm_api_key}"
        return headers

    def _health_headers(self, config: LLMRuntimeSettings) -> dict[str, str]:
        if config.llm_provider.lower() == "anthropic":
            return {
                "x-api-key": config.llm_api_key,
                "anthropic-version": config.llm_anthropic_version,
                "Content-Type": "application/json",
            }
        if not config.llm_api_key:
            return {}
        return {"Authorization": f"Bearer {config.llm_api_key}"}

    def _health_request(
        self,
        config: LLMRuntimeSettings | None = None,
    ) -> tuple[str, str, dict[str, object] | None]:
        config = config or self._active_config()
        path = config.llm_health_path.strip()
        if not path:
            path = (
                "/messages/count_tokens"
                if config.llm_provider.lower() == "anthropic"
                else "/models"
            )
        if not path.startswith("/"):
            path = f"/{path}"

        url = f"{config.llm_base_url.rstrip('/')}{path}"
        if (
            config.llm_provider.lower() == "anthropic"
            and path == "/messages/count_tokens"
        ):
            return (
                "POST",
                url,
                {
                    "model": config.llm_model,
                    "messages": [{"role": "user", "content": "health check"}],
                },
            )
        return "GET", url, None

    def _post_json_with_retries(
        self,
        config: LLMRuntimeSettings,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
    ) -> dict[str, object]:
        response = self._request_with_retries(
            config,
            "POST",
            url,
            headers=headers,
            json_payload=payload,
            timeout=config.llm_timeout_seconds,
        )
        return response.json()

    def _request_with_retries(
        self,
        config: LLMRuntimeSettings,
        method: Literal["GET", "POST"],
        url: str,
        headers: dict[str, str],
        json_payload: dict[str, object] | None = None,
        timeout: float | None = None,
        max_attempts: int | None = None,
    ) -> httpx.Response:
        attempts = max(1, max_attempts or config.llm_retry_attempts)
        delay = config.llm_retry_backoff_seconds
        max_delay = config.llm_retry_backoff_max_seconds
        request_timeout = config.llm_timeout_seconds if timeout is None else timeout

        for attempt in range(1, attempts + 1):
            try:
                client = self._http_client()
                if method == "POST":
                    response = client.post(
                        url,
                        headers=headers,
                        json=json_payload,
                        timeout=request_timeout,
                    )
                else:
                    response = client.get(
                        url,
                        headers=headers,
                        timeout=request_timeout,
                    )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if (
                    not self._is_retryable_status(exc.response.status_code)
                    or attempt >= attempts
                ):
                    raise
                self._log_retry(
                    attempt,
                    attempts,
                    exc,
                    f"status={exc.response.status_code}",
                )
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                if attempt >= attempts:
                    raise
                self._log_retry(attempt, attempts, exc, exc.__class__.__name__)

            if delay > 0:
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

        raise RuntimeError("LLM request retry loop exhausted unexpectedly.")

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _http_client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.Client(
                    timeout=self.config.llm_timeout_seconds,
                    transport=self.transport,
                )
            return self._client

    @staticmethod
    def _is_retryable_status(status_code: int) -> bool:
        return status_code in _RETRYABLE_STATUS_CODES

    @staticmethod
    def _log_retry(
        attempt: int,
        attempts: int,
        exc: Exception,
        reason: str,
    ) -> None:
        logger.warning(
            "LLM request failed transiently; retrying attempt %s/%s (%s): %s",
            attempt + 1,
            attempts,
            reason,
            exc.__class__.__name__,
        )

    @staticmethod
    def _normalize_messages(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if not messages:
            raise ValueError("LLM messages must not be empty.")

        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported LLM message role at index {index}.")
            if not content:
                raise ValueError(f"LLM message content is empty at index {index}.")
            normalized.append({"role": role, "content": content})

        if not any(message["role"] != "system" for message in normalized):
            raise ValueError("LLM messages must include a user or assistant message.")
        return normalized

    def _to_anthropic_messages(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        messages = self._normalize_messages(messages)
        system_parts: list[str] = []
        converted: list[dict[str, str]] = []

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_parts.append(content)
                continue

            role = "assistant" if role == "assistant" else "user"
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"] += f"\n\n{content}"
            else:
                converted.append({"role": role, "content": content})

        return "\n\n".join(system_parts), converted

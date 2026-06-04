from __future__ import annotations

import httpx

from app.config import Settings, settings


class LLMClient:
    def __init__(self, config: Settings = settings) -> None:
        self.config = config

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = self._normalize_messages(messages)
        if self.config.llm_provider.lower() == "anthropic":
            return self._anthropic_chat(messages, temperature, top_p, max_tokens)
        return self._openai_compatible_chat(messages, temperature, top_p, max_tokens)

    def _openai_compatible_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url = f"{self.config.llm_base_url.rstrip('/')}/chat/completions"
        headers = self._bearer_headers()
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": self.config.llm_temperature
            if temperature is None
            else temperature,
            "top_p": self.config.llm_top_p if top_p is None else top_p,
            "max_tokens": self.config.llm_max_tokens
            if max_tokens is None
            else max_tokens,
        }

        with httpx.Client(timeout=self.config.llm_timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _anthropic_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        url = f"{self.config.llm_base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": self.config.llm_api_key,
            "anthropic-version": self.config.llm_anthropic_version,
            "Content-Type": "application/json",
        }
        system_prompt, anthropic_messages = self._to_anthropic_messages(messages)
        payload = {
            "model": self.config.llm_model,
            "messages": anthropic_messages,
            "temperature": self.config.llm_temperature
            if temperature is None
            else temperature,
            "top_p": self.config.llm_top_p if top_p is None else top_p,
            "max_tokens": self.config.llm_max_tokens
            if max_tokens is None
            else max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=self.config.llm_timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json()
        content = data.get("content", [])
        return "".join(
            item.get("text", "")
            for item in content
            if item.get("type") == "text"
        )

    def health(self) -> bool:
        if not self.config.llm_health_check_enabled:
            return True

        method, url, payload = self._health_request()
        headers = self._health_headers()
        try:
            with httpx.Client(timeout=5.0) as client:
                if method == "POST":
                    response = client.post(url, headers=headers, json=payload)
                else:
                    response = client.get(url, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError:
            return False
        return True

    def _bearer_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"
        return headers

    def _health_headers(self) -> dict[str, str]:
        if self.config.llm_provider.lower() == "anthropic":
            return {
                "x-api-key": self.config.llm_api_key,
                "anthropic-version": self.config.llm_anthropic_version,
                "Content-Type": "application/json",
            }
        if not self.config.llm_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.llm_api_key}"}

    def _health_request(self) -> tuple[str, str, dict[str, object] | None]:
        path = self.config.llm_health_path.strip()
        if not path:
            path = (
                "/messages/count_tokens"
                if self.config.llm_provider.lower() == "anthropic"
                else "/models"
            )
        if not path.startswith("/"):
            path = f"/{path}"

        url = f"{self.config.llm_base_url.rstrip('/')}{path}"
        if (
            self.config.llm_provider.lower() == "anthropic"
            and path == "/messages/count_tokens"
        ):
            return (
                "POST",
                url,
                {
                    "model": self.config.llm_model,
                    "messages": [{"role": "user", "content": "health check"}],
                },
            )
        return "GET", url, None

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

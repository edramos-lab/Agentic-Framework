"""Unified LLM interface — supports OpenAI-compatible and Anthropic backends."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around chat-completion APIs.

    Parameters
    ----------
    provider : str
        "openai" or "anthropic".
    model : str
        Model identifier, e.g. "gpt-4o" or "claude-sonnet-4-20250514".
    api_key : str
        API key for the chosen provider.
    base_url : str | None
        Override base URL (useful for local vLLM / Ollama endpoints).
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens in the response.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "anthropic":
            from anthropic import Anthropic

            self._anthropic = Anthropic(api_key=api_key)
            self._openai = None
        else:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._openai = OpenAI(**kwargs)
            self._anthropic = None

    # ----- public API --------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant text."""
        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "anthropic":
            return self._chat_anthropic(messages, temp, mt)
        return self._chat_openai(messages, temp, mt, response_format)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Chat and parse the response as JSON."""
        raw = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"} if self.provider != "anthropic" else None,
        )
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        return json.loads(text)

    # ----- private -----------------------------------------------------------

    def _chat_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = self._openai.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _chat_anthropic(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        # Separate system message if present
        system = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                chat_messages.append(m)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system.strip():
            kwargs["system"] = system.strip()

        resp = self._anthropic.messages.create(**kwargs)
        return resp.content[0].text

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

            self._anthropic = Anthropic(api_key=api_key, max_retries=5)
            self._openai = None
        elif self.provider == "openrouter":  # one key, many models, through an OpenAI-compatible API
            from openai import OpenAI

            self._openai = OpenAI(api_key=api_key, base_url=base_url or "https://openrouter.ai/api/v1", max_retries=5,
                                  default_headers={"HTTP-Referer": "https://github.com/edramos-lab/Agentic-Framework", "X-Title": "PaperOrchestra"})
            self._anthropic = None
        else:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 5}  # long generations get dropped now and then
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
        if self.provider == "openrouter":
            return self._chat_openrouter(messages, mt)
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
            response_format={"type": "json_object"} if self.provider not in ("anthropic", "openrouter") else None,
        )
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError:  # a model without a JSON mode may wrap the object in a sentence
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise
            return json.loads(text[start:end + 1])

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

    def _chat_openrouter(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Chat through OpenRouter.  No temperature or JSON mode is sent (not every model accepts them; the prompts ask for
        JSON and chat_json tolerates stray text).  A model that reasons first can spend its token budget before it answers,
        so the budget has a floor, and a reply that stops at the limit is continued instead of returned cut off."""
        msgs = list(messages)
        max_tokens = max(max_tokens, 12000)
        text = ""
        for _ in range(5):
            resp = self._openai.chat.completions.create(model=self.model, messages=msgs, max_tokens=max_tokens)
            choice = resp.choices[0]
            part = choice.message.content or ""
            text += part
            if choice.finish_reason != "length":
                break
            logging.getLogger("paper_orchestra.llm").warning("Reply hit the token limit; continuing")
            if part.strip():
                msgs = [*msgs, {"role": "assistant", "content": text.rstrip()},
                        {"role": "user", "content": "Your reply was cut off by the length limit. Continue exactly where it stopped: "
                         "no repetition, no preamble, no commentary."}]
            else:
                max_tokens = min(max_tokens * 2, 64000)  # all of the budget went on reasoning: ask again with more room
        return text

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

        # The model may think before it answers, and the thinking counts against max_tokens: a reply capped at a few
        # thousand tokens can be cut off mid-sentence.  Give it room, and if it still stops at the limit, make it continue.
        max_tokens = max(max_tokens, 12000)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,  # no temperature: the SDK in use does not accept it
        }
        if system.strip():
            kwargs["system"] = system.strip()

        def text_of(r) -> str:
            # the reply may open with a thinking block; only the text blocks are the answer
            return "".join(b.text for b in r.content if getattr(b, "type", "text") == "text")

        resp = self._anthropic.messages.create(**kwargs)
        text = text_of(resp)
        for _ in range(4):
            if getattr(resp, "stop_reason", "") != "max_tokens":
                break
            if not text.strip():  # all of the budget went on thinking: ask again with more room
                kwargs["max_tokens"] = min(kwargs["max_tokens"] * 2, 20000)
            else:
                kwargs["messages"] = [*kwargs["messages"], {"role": "assistant", "content": text.rstrip()},
                                      {"role": "user", "content": "Your reply was cut off by the length limit. Continue "
                                       "exactly where it stopped: no repetition, no preamble, no commentary."}]
            logging.getLogger("paper_orchestra.llm").warning("Reply hit the token limit; continuing")
            resp = self._anthropic.messages.create(**kwargs)
            text = text + text_of(resp) if text.strip() else text_of(resp)
        return text

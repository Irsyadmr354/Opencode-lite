"""OpenAI-compatible streaming chat client for a local Ollama server."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    reasoning: str | None = None


class LLMError(Exception):
    """Raised on connection or HTTP failures."""


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def chat_stream(self, messages: list[dict], tools_schema: list[dict] | None = None,
                    cancel=None):
        """Yield {"type": "delta"|"reasoning", "text": ...} chunks then
        {"type": "final", "turn": ...}. Reasoning deltas come from the
        ``reasoning``/``reasoning_content`` fields some runtimes use for
        thinking models.

        ``cancel`` is a threading.Event (or anything with ``is_set()``); when set
        mid-stream the response is closed and a final turn with whatever was
        accumulated is still yielded.
        """
        payload: dict = {"model": self.model, "messages": messages, "stream": True}
        if tools_schema:
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        pending: dict[int, dict] = {}  # tool_call index -> {id, name, arguments}
        finish_reason: str | None = None

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                with client.stream("POST", f"{self.base_url}/chat/completions",
                                   json=payload, headers=headers) as resp:
                    if resp.status_code // 100 != 2:
                        body = resp.read().decode("utf-8", "replace")
                        raise LLMError(f"HTTP {resp.status_code}: {body[:300]}")

                    raw_lines: list[str] = []
                    saw_sse = False
                    for line in resp.iter_lines():
                        if cancel is not None and cancel.is_set():
                            break
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", "replace")
                        raw_lines.append(line)
                        if not line.startswith("data:"):
                            continue
                        saw_sse = True
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        thought = (delta.get("reasoning")
                                   or delta.get("reasoning_content"))
                        if thought:
                            reasoning_parts.append(thought)
                            yield {"type": "reasoning", "text": thought}
                        text = delta.get("content")
                        if text:
                            content_parts.append(text)
                            yield {"type": "delta", "text": text}
                        for entry in delta.get("tool_calls") or []:
                            index = entry.get("index") or 0
                            slot = pending.setdefault(
                                index, {"id": "", "name": "", "arguments": ""})
                            if entry.get("id"):
                                slot["id"] = entry["id"]
                            fn = entry.get("function") or {}
                            if fn.get("name"):
                                slot["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                    cancelled = cancel is not None and cancel.is_set()
                    if not saw_sse and not cancelled:
                        # Non-streaming provider: whole body is one completion object.
                        body = "\n".join(raw_lines).strip()
                        try:
                            obj = json.loads(body)
                            choice = (obj.get("choices") or [{}])[0]
                            message = choice.get("message") or {}
                        except (json.JSONDecodeError, AttributeError) as exc:
                            raise LLMError(f"invalid non-streaming response: {exc}") from exc
                        thought = (message.get("reasoning")
                                   or message.get("reasoning_content"))
                        if thought:
                            reasoning_parts.append(thought)
                            yield {"type": "reasoning", "text": thought}
                        text = message.get("content")
                        if text:
                            content_parts.append(text)
                            yield {"type": "delta", "text": text}
                        for entry in message.get("tool_calls") or []:
                            fn = entry.get("function") or {}
                            pending[len(pending)] = {
                                "id": entry.get("id") or "",
                                "name": fn.get("name") or "",
                                "arguments": fn.get("arguments") or "",
                            }
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
        except httpx.HTTPError as exc:
            raise LLMError(f"request failed: {exc}") from exc

        tool_calls: list[ToolCall] = []
        for index in sorted(pending):
            slot = pending[index]
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(ToolCall(id=slot["id"] or f"call_{index}",
                                       name=slot["name"], arguments=args))

        turn = AssistantTurn(content="".join(content_parts) or None,
                             tool_calls=tool_calls, finish_reason=finish_reason,
                             reasoning="".join(reasoning_parts) or None)
        yield {"type": "final", "turn": turn}

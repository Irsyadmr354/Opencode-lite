"""OpenAI-compatible streaming chat client for a local Ollama server."""

from __future__ import annotations

import json
import re
import time
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
    stats: dict | None = None


class LLMError(Exception):
    """Raised on connection or HTTP failures."""


# Inline "thinking" markup some models emit inside ``content`` when the
# runtime does not populate a native reasoning field: closed <think> blocks,
# a trailing unclosed <think> opener, and [thinking: ...] bracket blocks.
_THINK_CLOSED_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>(.*)\Z", re.IGNORECASE | re.DOTALL)
_BRACKET_THINKING_RE = re.compile(r"\[\s*thinking\s*:?\s*([^\]]*)\]", re.IGNORECASE)


def _separate_inline_thinking(text: str) -> tuple[str, str]:
    """Split model ``content`` into ``(visible_text, inline_thoughts)``.

    Removes (a) ``<think>...</think>`` blocks, (b) an unclosed trailing
    ``<think>`` opener plus everything after it (that text IS thinking),
    and (c) ``[thinking: ...]`` blocks up to their closing ``]``. The
    removed text is returned so it can be surfaced as reasoning instead of
    being replayed into conversation history.
    """
    thoughts: list[str] = []

    def _take(match: re.Match) -> str:
        thoughts.append(match.group(1))
        return ""

    clean = _THINK_CLOSED_RE.sub(_take, text)
    clean = _THINK_UNCLOSED_RE.sub(_take, clean)
    clean = _BRACKET_THINKING_RE.sub(_take, clean)
    return clean, "".join(thoughts)


def _strip_fences(s: str) -> str:
    s = s.strip()
    # remove ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        # strip first fence line
        s = re.sub(r"^```[a-z]*\s*\n?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n?```\s*$", "", s)
        s = s.strip()
    return s


_TOOL_LEAK_PATTERNS = (
    re.compile(r">tool_calls?", re.IGNORECASE),
    re.compile(r"<\/?tool_calls?>", re.IGNORECASE),
    re.compile(r"\btool_calls?\s*:\s*\[", re.IGNORECASE),
    re.compile(r"\[\s*[a-zA-Z0-9_]+\s*(?:\(.*?\))?\s*\]"),
)


def _looks_like_tool_leak(s: str) -> bool:
    if not s:
        return False
    if '"name"' in s and ('"arguments"' in s or '"query"' in s or '"path"' in s or '"command"' in s or '"content"' in s):
        return True
    if any(p.search(s) for p in _TOOL_LEAK_PATTERNS):
        return True
    t = _strip_fences(s).strip()
    if t.startswith("{") and ('"name"' in t or '"function"' in t or '"tool"' in t):
        return True
    if "```json" in s and ('"name"' in s or any(p.search(s) for p in _TOOL_LEAK_PATTERNS)):
        return True
    return False


def _looks_like_tool_json(s: str) -> bool:
    return _looks_like_tool_leak(s)


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
        t_start = time.monotonic()
        ollama_stats: dict = {}
        usage: dict | None = None

        def _capture_ollama(obj: dict) -> None:
            nonlocal ollama_stats, usage
            if not isinstance(obj, dict):
                return
            for k in ("total_duration", "load_duration", "prompt_eval_count",
                      "prompt_eval_duration", "eval_count", "eval_duration", "timings"):
                if k in obj:
                    ollama_stats[k] = obj[k]
            if "done" in obj and isinstance(obj["done"], bool):
                ollama_stats["done"] = obj["done"]
            if "usage" in obj and isinstance(obj["usage"], dict):
                usage = obj["usage"]
            # Some providers may emit prompt_tokens etc at top level
            # without wrapping in usage – capture as usage fallback.
            if usage is None and any(k in obj for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
                usage = {k: obj[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in obj}

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

        def _clean_error_body(raw: str) -> str:
            # Ollama often double-encodes Jinja errors: {"error":{"message":"{\"error\":{\"message\":\"...\"}}"}}
            # Extract the innermost human message
            try:
                outer = json.loads(raw)
                # Try outer.error.message
                msg = None
                if isinstance(outer, dict):
                    err = outer.get("error")
                    if isinstance(err, dict):
                        msg = err.get("message")
                    elif isinstance(err, str):
                        msg = err
                    if msg and isinstance(msg, str):
                        # msg may itself be JSON
                        msg_stripped = msg.strip()
                        if msg_stripped.startswith("{"):
                            try:
                                inner = json.loads(msg_stripped)
                                inner_err = inner.get("error") if isinstance(inner, dict) else None
                                if isinstance(inner_err, dict) and inner_err.get("message"):
                                    msg = inner_err["message"]
                                elif isinstance(inner, dict) and inner.get("message"):
                                    msg = inner["message"]
                            except Exception:
                                pass
                        # Clean Jinja trace: keep only first line with raise_exception
                        if "raise_exception" in msg:
                            # Extract the quoted message inside raise_exception('...')
                            m = re.search(r"raise_exception\(['\"]([^'\"]+)['\"]\)", msg)
                            if m:
                                msg = m.group(1)
                        # Truncate long traces
                        msg = msg.split("\n")[0].strip()
                        if msg:
                            return msg
                # Fallback: return raw truncated
                return raw[:300].strip()
            except Exception:
                return raw[:300].strip()

        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                with client.stream("POST", f"{self.base_url}/chat/completions",
                                   json=payload, headers=headers) as resp:
                    if resp.status_code // 100 != 2:
                        raw = resp.read().decode("utf-8", "replace")
                        clean = _clean_error_body(raw)
                        # Hint for common template error
                        hint = ""
                        if "No user query" in clean:
                            hint = " (reset conversation history with /clear or /session new)"
                        raise LLMError(f"HTTP {resp.status_code}: {clean}{hint}")

                    raw_lines: list[str] = []
                    saw_sse = False
                    for line in resp.iter_lines():
                        if cancel is not None and cancel.is_set():
                            break
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", "replace")
                        raw_lines.append(line)
                        # Capture stats from bare NDJSON lines (native Ollama) before SSE check.
                        # Only capture ollama stats here; content handling for bare NDJSON is deferred
                        # to the non-streaming fallback to avoid double-yield when the body is
                        # a single JSON object (e.g., FakeOllama raw_json).
                        _stripped = line.strip()
                        if _stripped.startswith("{"):
                            try:
                                _nd = json.loads(_stripped)
                                _capture_ollama(_nd)
                                # For true streaming NDJSON (not SSE), also capture stats but don't
                                # yield content here – the SSE path handles streaming content.
                                # Native Ollama NDJSON content will be handled if we ever need to,
                                # but stats capture is the primary goal for verbose mode.
                            except json.JSONDecodeError:
                                pass
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
                        _capture_ollama(obj)
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
                            # suppress raw tool JSON from streaming to UI (will be tool_calls) — aggressive
                            accum = "".join(content_parts)
                            if ('"name"' in text and '"arguments"' in text) or ('"name"' in accum and '"arguments"' in accum):
                                pass  # leaked tool JSON, hide
                            elif _looks_like_tool_json(accum) or _looks_like_tool_json(text):
                                pass  # don't yield tool JSON as visible text
                            elif accum.strip().startswith("{") and '"name"' in accum:
                                pass
                            elif accum.strip().startswith("```") and '"name"' in accum:
                                pass
                            elif "```json" in text or "```json" in accum:
                                pass
                            else:
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
                            _capture_ollama(obj)
                            choice = (obj.get("choices") or [{}])[0]
                            message = choice.get("message") or {}
                            # Also capture stats that might be nested inside choice/message
                            if isinstance(choice, dict):
                                _capture_ollama(choice)
                            if isinstance(message, dict):
                                _capture_ollama(message)
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
                            accum = "".join(content_parts)
                            if ('"name"' in text and '"arguments"' in text) or ('"name"' in accum and '"arguments"' in accum):
                                pass
                            elif _looks_like_tool_json(accum) or _looks_like_tool_json(text):
                                pass
                            elif accum.strip().startswith("{") and '"name"' in accum:
                                pass
                            elif accum.strip().startswith("```") and '"name"' in accum:
                                pass
                            elif "```json" in text or "```json" in accum:
                                pass
                            else:
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
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"request timed out after {self.timeout_s}s — "
                f"try increasing timeout_s in config or use a faster model"
            ) from exc
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

        # If native tool_calls were not parsed from SSE, check if model dumped tool calls in content
        raw_content = "".join(content_parts)
        if not tool_calls and raw_content:
            # 1. Check for XML tags <tool_call>...</tool_call>
            for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", raw_content, re.DOTALL):
                try:
                    obj = json.loads(m.group(1).strip())
                    if isinstance(obj, dict) and "name" in obj:
                        args = obj.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        tool_calls.append(ToolCall(id=f"call_{len(tool_calls)}", name=obj["name"], arguments=args if isinstance(args, dict) else {}))
                except Exception:
                    pass

            # 1b. Check for >tool_calls [name] or >tool_calls [name(args)]
            for m in re.finditer(r">tool_calls?\s*\[([a-zA-Z0-9_]+)(?:\((.*?)\))?\]", raw_content):
                fn_name = m.group(1)
                args_str = m.group(2) or ""
                args = {}
                if args_str:
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        pass
                tool_calls.append(ToolCall(id=f"call_{len(tool_calls)}", name=fn_name, arguments=args if isinstance(args, dict) else {}))

            # 2. Check for raw json function call strings (including ```json fences and truncated JSON)
            if not tool_calls and ('"name"' in raw_content):
                for m in re.finditer(r'\{[^{}]*?"name"\s*:\s*"([^"]+)"', raw_content, re.DOTALL):
                    name = m.group(1)
                    start_idx = m.start()
                    sub = raw_content[start_idx:]
                    brace_count = 0
                    end_idx = -1
                    in_str = False
                    escape = False
                    for i, ch in enumerate(sub):
                        if escape:
                            escape = False
                            continue
                        if ch == '\\':
                            escape = True
                            continue
                        if ch == '"':
                            in_str = not in_str
                            continue
                        if not in_str:
                            if ch == '{':
                                brace_count += 1
                            elif ch == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                    if end_idx != -1:
                        candidate = sub[:end_idx]
                        try:
                            obj = json.loads(candidate)
                            fn = obj.get("function") if isinstance(obj, dict) and "function" in obj else obj
                            if isinstance(fn, dict) and "name" in fn:
                                args = fn.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except Exception:
                                        pass
                                elif isinstance(fn, dict):
                                    args = {k: v for k, v in fn.items() if k not in ("name", "function", "id", "type")}
                                tool_calls.append(ToolCall(id=f"call_{len(tool_calls)}", name=fn["name"], arguments=args if isinstance(args, dict) else {}))
                        except Exception:
                            pass
                    else:
                        candidate = sub.rstrip()
                        for suffix in ("}", '"}', '"} }', '"} } }', '"}'):
                            try:
                                obj = json.loads(candidate + suffix)
                                if isinstance(obj, dict) and "name" in obj:
                                    args = obj.get("arguments", {})
                                    if isinstance(args, str):
                                        try:
                                            args = json.loads(args)
                                        except Exception:
                                            pass
                                    elif not isinstance(args, dict):
                                        args = {k: v for k, v in obj.items() if k not in ("name", "function", "id", "type")}
                                    tool_calls.append(ToolCall(id=f"call_{len(tool_calls)}", name=obj["name"], arguments=args if isinstance(args, dict) else {}))
                                    break
                            except Exception:
                                continue
                        if not tool_calls:
                            args = {}
                            for kv_match in re.finditer(r'"([a-zA-Z0-9_]+)"\s*:\s*("(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?|true|false|null)', sub):
                                k = kv_match.group(1)
                                if k in ("name", "function", "id", "type"):
                                    continue
                                v_raw = kv_match.group(2)
                                try:
                                    args[k] = json.loads(v_raw)
                                except Exception:
                                    args[k] = v_raw.strip('"')
                            if name:
                                tool_calls.append(ToolCall(id=f"call_{len(tool_calls)}", name=name, arguments=args))

                if tool_calls:
                    raw_content = ""

        # Store CLEANED content: inline think markup was extracted from the
        # RAW content above (tool-call fallback) and must not be replayed
        # into conversation history. Surface it as reasoning only when the
        # native reasoning field stayed empty; otherwise discard it.
        clean_content, inline_thoughts = _separate_inline_thinking(raw_content)
        if tool_calls:
            clean_content = ""
        else:
            clean_content = re.sub(r">tool_calls?\s*\[[^\]]*\]", "", clean_content, flags=re.IGNORECASE)
            clean_content = re.sub(r"<\/?tool_calls?>", "", clean_content, flags=re.IGNORECASE)
            clean_content = re.sub(r">tool_calls?\s*\{.*?\}", "", clean_content, flags=re.DOTALL | re.IGNORECASE)
            clean_content = clean_content.strip()
        reasoning_text = "".join(reasoning_parts) or None
        if reasoning_text is None and inline_thoughts.strip():
            reasoning_text = inline_thoughts

        wall_duration_s = time.monotonic() - t_start
        stats: dict = {
            "wall_duration_s": wall_duration_s,
            "ollama": ollama_stats or None,
            "usage": usage,
        }
        turn = AssistantTurn(
            content=clean_content if clean_content.strip() else None,
            tool_calls=tool_calls, finish_reason=finish_reason,
            reasoning=reasoning_text, stats=stats)
        yield {"type": "final", "turn": turn}

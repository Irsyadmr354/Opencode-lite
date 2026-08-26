"""Minimal Ollama/OpenAI streaming client + typewriter display."""

from __future__ import annotations

import io
import json
import re
import sys
import threading
import time
from typing import Any, Callable

import httpx

BOLD = "\033[1m"
DIM = "\033[2m"
DARK_GRAY = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

SPINNER_FRAMES = ["|", "/", "-", "\\"]
ANSI_ESCAPE_RE = re.compile(r"(\x1b\[[0-9;]*[a-zA-Z])")


class StreamReasoningParser:
    """Separates <think>...</think> tags from normal content across streamed chunks."""

    def __init__(self):
        self.in_think = False
        self.buffer = ""

    def feed(self, text: str) -> list[tuple[str, bool]]:
        """Feed incoming text chunk, returning list of (segment, is_thinking) tuples."""
        if not text:
            return []

        self.buffer += text
        results: list[tuple[str, bool]] = []

        while self.buffer:
            if not self.in_think:
                pos = self.buffer.find("<think>")
                if pos != -1:
                    if pos > 0:
                        results.append((self.buffer[:pos], False))
                    self.buffer = self.buffer[pos + len("<think>") :]
                    self.in_think = True
                    continue

                # Check if buffer ends with a prefix of "<think>"
                matched_prefix_len = 0
                for l in range(min(len(self.buffer), len("<think>") - 1), 0, -1):
                    if "<think>".startswith(self.buffer[-l:]):
                        matched_prefix_len = l
                        break

                if matched_prefix_len > 0:
                    to_emit = self.buffer[:-matched_prefix_len]
                    if to_emit:
                        results.append((to_emit, False))
                    self.buffer = self.buffer[-matched_prefix_len:]
                    break
                else:
                    results.append((self.buffer, False))
                    self.buffer = ""
                    break
            else:
                pos = self.buffer.find("</think>")
                if pos != -1:
                    if pos > 0:
                        results.append((self.buffer[:pos], True))
                    self.buffer = self.buffer[pos + len("</think>") :]
                    self.in_think = False
                    continue

                # Check if buffer ends with a prefix of "</think>"
                matched_prefix_len = 0
                for l in range(min(len(self.buffer), len("</think>") - 1), 0, -1):
                    if "</think>".startswith(self.buffer[-l:]):
                        matched_prefix_len = l
                        break

                if matched_prefix_len > 0:
                    to_emit = self.buffer[:-matched_prefix_len]
                    if to_emit:
                        results.append((to_emit, True))
                    self.buffer = self.buffer[-matched_prefix_len:]
                    break
                else:
                    results.append((self.buffer, True))
                    self.buffer = ""
                    break

        return results

    def flush(self) -> list[tuple[str, bool]]:
        """Flush remaining buffer at the end of stream."""
        results: list[tuple[str, bool]] = []
        if self.buffer:
            results.append((self.buffer, self.in_think))
            self.buffer = ""
        return results


def _emit_delta(on_delta: Callable | None, token: str, is_thinking: bool = False) -> None:
    """Safely call on_delta supporting both (token, is_thinking) and legacy (token) signatures."""
    if not on_delta or not token:
        return
    try:
        on_delta(token, is_thinking)
    except TypeError:
        on_delta(token)


def _format_http_error(status_code: int, response_bytes: bytes, base_url: str, model: str) -> str:
    """Format structured, human-readable error messages for HTTP response codes."""
    raw_text = response_bytes.decode(errors="replace")[:1000].strip()
    detail = ""
    try:
        err_json = json.loads(raw_text)
        if isinstance(err_json, dict):
            if "error" in err_json:
                err_val = err_json["error"]
                if isinstance(err_val, dict):
                    detail = err_val.get("message") or str(err_val)
                else:
                    detail = str(err_val)
            elif "detail" in err_json:
                detail = str(err_json["detail"])
            elif "message" in err_json:
                detail = str(err_json["message"])
    except Exception:
        pass

    if not detail:
        detail = raw_text or "No error detail returned by server."

    status_descriptions = {
        400: "Bad Request",
        401: "Unauthorized - check API key",
        403: "Forbidden - permission denied",
        404: f"Not Found - check model '{model}' or endpoint at '{base_url}'",
        429: "Rate Limit Exceeded / Server Busy",
        500: "Internal Server Error",
        502: "Bad Gateway - upstream server error",
        503: "Service Unavailable - LLM server is overloaded or down",
        504: "Gateway Timeout - upstream server timed out",
    }
    desc = status_descriptions.get(status_code, "HTTP Error")
    return f"LLM HTTP {status_code} ({desc}): {detail}"


def _is_tool_call_dict(obj: Any) -> bool:
    """Check whether a decoded JSON object is structured as a tool/function call."""
    if not isinstance(obj, dict):
        return False
    if "function" in obj and isinstance(obj["function"], dict) and "name" in obj["function"]:
        return True
    name = obj.get("name") or obj.get("tool") or obj.get("action")
    if isinstance(name, str) and name:
        if any(k in obj for k in ("arguments", "parameters", "action_input", "input")) or obj.get("type") == "function":
            return True
        if set(obj.keys()).issubset({"name", "id", "tool", "action"}):
            return True
    return False


def _normalize_tool_call(obj: dict, idx: int) -> dict:
    """Normalize tool call representation into standard {id, name, arguments} dict."""
    if "function" in obj and isinstance(obj["function"], dict):
        fn = obj["function"]
        name = fn.get("name", "")
        raw_args = fn.get("arguments", {})
        tc_id = obj.get("id") or fn.get("id") or f"call_text_{idx}"
    else:
        name = obj.get("name") or obj.get("tool") or obj.get("action") or ""
        raw_args = obj.get("arguments")
        if raw_args is None:
            raw_args = obj.get("parameters")
        if raw_args is None:
            raw_args = obj.get("action_input")
        if raw_args is None:
            raw_args = obj.get("input")
        if raw_args is None:
            raw_args = {}
        tc_id = obj.get("id") or f"call_text_{idx}"

    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
            if not isinstance(args, dict):
                args = {"raw": raw_args}
        except json.JSONDecodeError:
            args = {"raw": raw_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {"value": raw_args}

    return {
        "id": str(tc_id),
        "name": str(name),
        "arguments": args,
    }


def extract_tool_calls_from_text(text: str) -> tuple[list[dict], str]:
    """Extract tool calls from markdown fences, XML tags, or raw JSON, and return cleaned text."""
    if not text:
        return [], text

    tool_calls: list[dict] = []
    spans_to_remove: list[tuple[int, int]] = []
    decoder = json.JSONDecoder()

    # 1. Look for <tool_call>...</tool_call> tags
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        inner = match.group(1).strip()
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, list):
                valid_items = [item for item in parsed if _is_tool_call_dict(item)]
                if valid_items:
                    for item in valid_items:
                        tool_calls.append(_normalize_tool_call(item, len(tool_calls)))
                    spans_to_remove.append(match.span())
            elif _is_tool_call_dict(parsed):
                tool_calls.append(_normalize_tool_call(parsed, len(tool_calls)))
                spans_to_remove.append(match.span())
        except json.JSONDecodeError:
            pass

    # 2. Look for code fences: ```(?:json)? ... ```
    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL):
        if any(s[0] <= match.start() and match.end() <= s[1] for s in spans_to_remove):
            continue
        inner = match.group(1).strip()
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, list):
                valid_items = [item for item in parsed if _is_tool_call_dict(item)]
                if valid_items:
                    for item in valid_items:
                        tool_calls.append(_normalize_tool_call(item, len(tool_calls)))
                    spans_to_remove.append(match.span())
            elif _is_tool_call_dict(parsed):
                tool_calls.append(_normalize_tool_call(parsed, len(tool_calls)))
                spans_to_remove.append(match.span())
        except json.JSONDecodeError:
            idx = 0
            found_inside = False
            while idx < len(inner):
                if inner[idx] in ("{", "["):
                    try:
                        obj, end_pos = decoder.raw_decode(inner, idx)
                        if isinstance(obj, list):
                            for item in obj:
                                if _is_tool_call_dict(item):
                                    tool_calls.append(_normalize_tool_call(item, len(tool_calls)))
                                    found_inside = True
                        elif _is_tool_call_dict(obj):
                            tool_calls.append(_normalize_tool_call(obj, len(tool_calls)))
                            found_inside = True
                        idx = end_pos
                        continue
                    except json.JSONDecodeError:
                        pass
                idx += 1
            if found_inside:
                spans_to_remove.append(match.span())

    # 3. Look for raw JSON objects or lists outside code fences
    idx = 0
    while idx < len(text):
        in_existing_span = False
        for start, end in spans_to_remove:
            if start <= idx < end:
                idx = end
                in_existing_span = True
                break
        if in_existing_span:
            continue

        if text[idx] in ("{", "["):
            try:
                obj, end_pos = decoder.raw_decode(text, idx)
                if isinstance(obj, list):
                    valid_items = [item for item in obj if _is_tool_call_dict(item)]
                    if valid_items:
                        for item in valid_items:
                            tool_calls.append(_normalize_tool_call(item, len(tool_calls)))
                        spans_to_remove.append((idx, end_pos))
                        idx = end_pos
                        continue
                elif _is_tool_call_dict(obj):
                    tool_calls.append(_normalize_tool_call(obj, len(tool_calls)))
                    spans_to_remove.append((idx, end_pos))
                    idx = end_pos
                    continue
            except json.JSONDecodeError:
                pass
        idx += 1

    # Remove all tool call spans in reverse order to preserve string indices
    spans_to_remove.sort(key=lambda x: x[0], reverse=True)
    cleaned = text
    for start, end in spans_to_remove:
        cleaned = cleaned[:start] + cleaned[end:]

    # Clean up empty code blocks and redundant whitespace
    cleaned = re.sub(r"```(?:json)?\s*```", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return tool_calls, cleaned


def _parse_tool_calls_from_text(text: str) -> list[dict]:
    """Extract tool calls from model text output (small models)."""
    tool_calls, _ = extract_tool_calls_from_text(text)
    return tool_calls


def _extract_thinking_and_clean_content(full_content: str, full_thinking: str = "") -> tuple[str, str]:
    """Extract <think>...</think> tags and combine with reasoning into (cleaned_content, thinking)."""
    thinking_parts = [full_thinking] if full_thinking else []

    # Extract all closed <think>...</think> blocks
    for match in re.finditer(r"<think>(.*?)</think>", full_content, re.DOTALL):
        th = match.group(1).strip()
        if th:
            thinking_parts.append(th)

    cleaned = re.sub(r"<think>.*?</think>", "", full_content, flags=re.DOTALL)

    # Handle unclosed <think> tag if model output was truncated
    unclosed_match = re.search(r"<think>(.*)$", cleaned, re.DOTALL)
    if unclosed_match:
        th = unclosed_match.group(1).strip()
        if th:
            thinking_parts.append(th)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)

    cleaned = re.sub(r"</think>", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    combined_thinking = "\n".join(tp for tp in thinking_parts if tp).strip()
    return cleaned, combined_thinking


def format_ollama_stats(stats: dict | None) -> str:
    """Format Ollama generation statistics in clean, compact dimmed terminal style."""
    if not stats:
        return ""
    eval_count = stats.get("eval_count") or 0
    eval_rate = stats.get("eval_rate") or 0.0
    total_s = stats.get("total_duration_s") or 0.0
    prompt_count = stats.get("prompt_eval_count")
    prompt_rate = stats.get("prompt_eval_rate")

    parts = []
    if eval_count > 0:
        parts.append(f"{eval_count} tokens")
    if eval_rate > 0:
        parts.append(f"{eval_rate:.1f} tok/s")
    if total_s > 0:
        parts.append(f"{total_s:.2f}s")
    elif stats.get("client_duration_s"):
        parts.append(f"{stats['client_duration_s']:.2f}s")

    prompt_str = ""
    if prompt_count:
        if prompt_rate and prompt_rate > 0:
            prompt_str = f" (prompt: {prompt_count} tok · {prompt_rate:.1f} tok/s)"
        else:
            prompt_str = f" (prompt: {prompt_count} tok)"

    main_info = " · ".join(parts) if parts else (f"{total_s:.2f}s" if total_s > 0 else "")
    if not main_info:
        return ""
    return f"{DIM}• {main_info}{prompt_str}{RESET}"


class LLM:
    def __init__(self, config):
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.api_key = config.api_key
        self.timeout = config.timeout_s
        self.stream = config.stream

    def chat(self, messages: list[dict], tools: list[dict] | None = None, on_delta: Callable | None = None) -> dict:
        """Send messages to LLM, return response dict.

        Returns: {content, tool_calls, finish_reason, thinking, stats}
        on_delta(token, is_thinking) is called for each token chunk during streaming.
        """
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": self.stream,
        }
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.perf_counter()

        # Handle non-streaming mode
        if not self.stream:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        json=body,
                        headers=headers,
                    )
                    elapsed_s = time.perf_counter() - start_time
                    if resp.status_code != 200:
                        msg = _format_http_error(resp.status_code, resp.content, self.base_url, self.model)
                        _emit_delta(on_delta, msg, False)
                        return {"content": msg, "tool_calls": [], "finish_reason": "error", "thinking": "", "stats": {}}

                    data = resp.json()
                    choice = (data.get("choices") or [{}])[0]
                    finish_reason = choice.get("finish_reason", "stop")
                    msg_obj = choice.get("message") or {}

                    raw_content = msg_obj.get("content") or ""
                    raw_thinking = (
                        msg_obj.get("reasoning_content")
                        or msg_obj.get("thinking")
                        or msg_obj.get("reasoning")
                        or ""
                    )

                    tool_calls = []
                    for tc in msg_obj.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        raw_args = fn.get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {"raw": raw_args}
                        tool_calls.append({
                            "id": tc.get("id", f"call_{len(tool_calls)}"),
                            "name": fn.get("name", ""),
                            "arguments": args if isinstance(args, dict) else {"raw": args},
                        })

                    content, thinking = _extract_thinking_and_clean_content(raw_content, raw_thinking)

                    if not tool_calls:
                        parsed_tcs, content = extract_tool_calls_from_text(content)
                        if parsed_tcs:
                            tool_calls = parsed_tcs

                    if on_delta:
                        if thinking:
                            _emit_delta(on_delta, thinking, True)
                        if content:
                            _emit_delta(on_delta, content, False)

                    # Extract stats
                    usage = data.get("usage") or {}
                    eval_count = data.get("eval_count") or usage.get("completion_tokens") or max(len(content.split()), 1)
                    eval_duration_ns = data.get("eval_duration")
                    prompt_eval_count = data.get("prompt_eval_count") or usage.get("prompt_tokens")
                    prompt_eval_duration_ns = data.get("prompt_eval_duration")
                    total_duration_ns = data.get("total_duration")

                    eval_rate = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else (eval_count / max(elapsed_s, 0.001))
                    prompt_rate = (prompt_eval_count / (prompt_eval_duration_ns / 1e9)) if prompt_eval_duration_ns else 0.0
                    total_s = (total_duration_ns / 1e9) if total_duration_ns else elapsed_s

                    stats = {
                        "eval_count": eval_count,
                        "eval_rate": eval_rate,
                        "prompt_eval_count": prompt_eval_count,
                        "prompt_eval_rate": prompt_rate,
                        "total_duration_s": total_s,
                        "client_duration_s": elapsed_s,
                    }

                    return {
                        "content": content,
                        "tool_calls": tool_calls,
                        "finish_reason": finish_reason,
                        "thinking": thinking,
                        "stats": stats,
                    }
            except httpx.TimeoutException:
                msg = f"LLM request timed out after {self.timeout}s"
                _emit_delta(on_delta, msg, False)
                return {"content": msg, "tool_calls": [], "finish_reason": "error", "thinking": "", "stats": {}}
            except httpx.ConnectError as exc:
                msg = f"Cannot connect to LLM at {self.base_url}: {exc}"
                _emit_delta(on_delta, msg, False)
                return {"content": msg, "tool_calls": [], "finish_reason": "error", "thinking": "", "stats": {}}
            except Exception as exc:
                msg = f"LLM error ({type(exc).__name__}): {exc}"
                _emit_delta(on_delta, msg, False)
                return {"content": msg, "tool_calls": [], "finish_reason": "error", "thinking": "", "stats": {}}

        # Handle streaming mode
        reasoning_parser = StreamReasoningParser()
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls_map: dict[int, dict] = {}
        finish_reason = "stop"
        server_metrics: dict = {}
        token_count = 0

        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        error_body = resp.read()
                        msg = _format_http_error(resp.status_code, error_body, self.base_url, self.model)
                        _emit_delta(on_delta, msg, False)
                        return {"content": msg, "tool_calls": [], "finish_reason": "error", "thinking": "", "stats": {}}

                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        # Capture any top-level metrics or usage from chunk
                        for k in ("usage", "eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration", "total_duration"):
                            if k in chunk and chunk[k] is not None:
                                server_metrics[k] = chunk[k]

                        choice = (chunk.get("choices") or [None])[0]
                        if not choice:
                            continue

                        delta = choice.get("delta") or {}
                        finish_reason = choice.get("finish_reason") or finish_reason

                        # 1. Direct reasoning content (e.g. DeepSeek-R1 / Ollama reasoning_content / thinking)
                        reasoning_chunk = (
                            delta.get("reasoning_content")
                            or delta.get("thinking")
                            or delta.get("reasoning")
                        )
                        if reasoning_chunk:
                            token_count += 1
                            thinking_parts.append(reasoning_chunk)
                            _emit_delta(on_delta, reasoning_chunk, True)

                        # 2. Content chunk (may contain embedded <think>...</think> tags)
                        text = delta.get("content")
                        if text:
                            token_count += 1
                            segments = reasoning_parser.feed(text)
                            for seg_text, is_think in segments:
                                if is_think:
                                    thinking_parts.append(seg_text)
                                else:
                                    content_parts.append(seg_text)
                                _emit_delta(on_delta, seg_text, is_think)

                        # 3. Streaming structured tool calls
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {
                                    "id": tc.get("id", ""),
                                    "name": "",
                                    "arguments": "",
                                }
                            entry = tool_calls_map[idx]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                entry["name"] = fn["name"]
                            if fn.get("id"):
                                entry["id"] = fn["id"]
                            if fn.get("arguments"):
                                entry["arguments"] += fn["arguments"]

                    # Flush stream reasoning parser buffer
                    for seg_text, is_think in reasoning_parser.flush():
                        if is_think:
                            thinking_parts.append(seg_text)
                        else:
                            content_parts.append(seg_text)
                        _emit_delta(on_delta, seg_text, is_think)

        except httpx.TimeoutException:
            msg = f"LLM request timed out after {self.timeout}s"
            _emit_delta(on_delta, msg, False)
            return {"content": msg, "tool_calls": [], "finish_reason": "error", "thinking": "", "stats": {}}
        except httpx.ConnectError as exc:
            msg = f"Cannot connect to LLM at {self.base_url}: {exc}"
            _emit_delta(on_delta, msg, False)
            return {"content": msg, "tool_calls": [], "finish_reason": "error", "thinking": "", "stats": {}}
        except Exception as exc:
            msg = f"LLM error ({type(exc).__name__}): {exc}"
            _emit_delta(on_delta, msg, False)
            return {"content": msg, "tool_calls": [], "finish_reason": "error", "thinking": "", "stats": {}}

        elapsed_s = time.perf_counter() - start_time
        full_content = "".join(content_parts)
        full_thinking = "".join(thinking_parts)

        # Clean remaining tags and extract thinking from full text
        full_content, full_thinking = _extract_thinking_and_clean_content(full_content, full_thinking)

        # Parse tool calls from text if structured tool calls are empty
        tool_calls = []
        if not tool_calls_map:
            parsed_tcs, full_content = extract_tool_calls_from_text(full_content)
            if parsed_tcs:
                tool_calls = parsed_tcs
        else:
            for entry in tool_calls_map.values():
                try:
                    args = json.loads(entry["arguments"]) if entry["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"raw": entry["arguments"]}
                tool_calls.append({
                    "id": entry["id"] or f"call_{len(tool_calls)}",
                    "name": entry["name"],
                    "arguments": args if isinstance(args, dict) else {"raw": args},
                })

        # Calculate statistics
        usage = server_metrics.get("usage") or {}
        eval_count = server_metrics.get("eval_count") or usage.get("completion_tokens") or max(token_count, len(full_content.split()))
        eval_duration_ns = server_metrics.get("eval_duration")
        prompt_eval_count = server_metrics.get("prompt_eval_count") or usage.get("prompt_tokens")
        prompt_eval_duration_ns = server_metrics.get("prompt_eval_duration")
        total_duration_ns = server_metrics.get("total_duration")

        eval_rate = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns else (eval_count / max(elapsed_s, 0.001))
        prompt_rate = (prompt_eval_count / (prompt_eval_duration_ns / 1e9)) if prompt_eval_duration_ns else 0.0
        total_s = (total_duration_ns / 1e9) if total_duration_ns else elapsed_s

        stats = {
            "eval_count": eval_count,
            "eval_rate": eval_rate,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_rate": prompt_rate,
            "total_duration_s": total_s,
            "client_duration_s": elapsed_s,
        }

        return {
            "content": full_content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "thinking": full_thinking,
            "stats": stats,
        }


class TypewriterStreamer:
    """Stateful streaming typewriter managing thinking lifecycle, completions, and content headers."""

    def __init__(
        self,
        delay: float = 0.02,
        dark_color_code: str = DARK_GRAY,
        stream: io.TextIOBase | None = None,
    ):
        self.delay = 0.03 if delay >= 0.1 else (delay if delay > 0 else 0.0)
        self.dark_color_code = dark_color_code
        self.stream = stream or sys.stdout
        self.start_time = time.time()
        self._in_thinking = False
        self._had_thinking = False
        self._thinking_done = False
        self._in_content = False
        self._has_trailing_newline = False
        self._thinking_lines = 0
        self._thinking_col = 0
        try:
            import shutil
            self._term_cols = shutil.get_terminal_size((80, 24)).columns
        except Exception:
            self._term_cols = 80

    def _safe_write(self, text: str) -> None:
        """Write string to stream handling encoding errors gracefully."""
        if not text:
            return
        try:
            self.stream.write(text)
        except UnicodeEncodeError:
            safe_text = text.replace("✔", "v").replace("✓", "v")
            try:
                self.stream.write(safe_text)
            except UnicodeEncodeError:
                self.stream.write(safe_text.encode("ascii", errors="replace").decode("ascii"))
        self.stream.flush()

    def _write_chars(self, text: str) -> None:
        """Write text character-by-character with delay, writing ANSI codes immediately."""
        parts = ANSI_ESCAPE_RE.split(text)
        for part in parts:
            if not part:
                continue
            if ANSI_ESCAPE_RE.fullmatch(part):
                # Write ANSI escape sequences immediately with zero delay
                self._safe_write(part)
            else:
                for ch in part:
                    self._safe_write(ch)
                    self._has_trailing_newline = (ch == "\n")
                    if self.delay > 0:
                        time.sleep(self.delay)

    def _write_thinking_chars(self, text: str) -> None:
        """Write thinking text character-by-character, tracking line and column counts for cleanup."""
        parts = ANSI_ESCAPE_RE.split(text)
        for part in parts:
            if not part:
                continue
            if ANSI_ESCAPE_RE.fullmatch(part):
                self._safe_write(part)
            else:
                for ch in part:
                    if ch == "\n":
                        self._safe_write(ch)
                        self._thinking_lines += 1
                        self._thinking_col = 0
                    else:
                        self._safe_write(ch)
                        self._thinking_col += 1
                        if self._thinking_col >= self._term_cols:
                            self._thinking_lines += 1
                            self._thinking_col = 0
                    if self.delay > 0:
                        time.sleep(self.delay)

    def _collapse_thinking(self) -> None:
        """Collapse multi-line thinking block into a single clean timing line."""
        if not self._in_thinking:
            return
        elapsed = max(time.time() - self.start_time, 0.1)
        # Erase current line and previous thinking lines
        self._safe_write("\r\x1b[2K")
        for _ in range(self._thinking_lines):
            self._safe_write("\x1b[1A\x1b[2K")
        self._safe_write(f"{GREEN}✔{RESET} {self.dark_color_code}Thought in {elapsed:.1f}s{RESET}\n")
        self._in_thinking = False
        self._thinking_done = True
        self._has_trailing_newline = True

    def on_delta(self, token: str, is_thinking: bool = False) -> None:
        """Process a streamed token with thinking and assistant lifecycle management."""
        if not token:
            return

        if is_thinking:
            if not self._in_thinking:
                if self._in_content:
                    self._safe_write("\n")
                    self._in_content = False
                self._safe_write(f"\r\x1b[2K{self.dark_color_code}Thinking: ")
                self._thinking_lines = 0
                self._thinking_col = 10
                self._in_thinking = True
                self._had_thinking = True

            self._write_thinking_chars(token)
        else:
            if self._in_thinking:
                self._collapse_thinking()

            if not self._in_content:
                self._safe_write(f"{BOLD}{GREEN}Assistant:{RESET} ")
                self._in_content = True

            self._write_chars(token)

    def close(self) -> None:
        """Finalize stream styling and complete any remaining thinking lifecycle."""
        if self._in_thinking:
            self._collapse_thinking()

        if self._in_content and not self._has_trailing_newline:
            self._safe_write("\n")
            self._has_trailing_newline = True

        self._safe_write(RESET)


def typewriter(
    text: str,
    delay: float = 0.02,
    is_thinking: bool = False,
    dark_color: bool = False,
    stream: io.TextIOBase | None = None,
) -> None:
    """Print text character by character with smooth delay (<0.1s).

    ANSI escape sequences are written immediately without sleeping to prevent flickering.
    If is_thinking or dark_color is True, text is styled in dark gray/dim and reset to normal.
    """
    if stream is None:
        stream = sys.stdout

    if not text:
        return

    if delay >= 0.1:
        delay = 0.03

    use_dark = is_thinking or dark_color
    if use_dark:
        stream.write(DARK_GRAY)
        stream.flush()

    try:
        parts = ANSI_ESCAPE_RE.split(text)
        for part in parts:
            if not part:
                continue
            if ANSI_ESCAPE_RE.fullmatch(part):
                stream.write(part)
                stream.flush()
            else:
                for ch in part:
                    try:
                        stream.write(ch)
                    except UnicodeEncodeError:
                        safe_ch = ch.replace("✔", "v").replace("✓", "v")
                        try:
                            stream.write(safe_ch)
                        except UnicodeEncodeError:
                            stream.write("?")
                    stream.flush()
                    if delay > 0:
                        time.sleep(delay)
    finally:
        if use_dark:
            stream.write(RESET)
            stream.flush()


def safe_print(text: str, stream: io.TextIOBase | None = None) -> None:
    """Print text that might contain unicode, fallback to ascii."""
    if stream is None:
        stream = sys.stdout
    try:
        stream.write(text + "\n")
        stream.flush()
    except UnicodeEncodeError:
        safe_text = text.replace("✔", "v").replace("✓", "v")
        try:
            stream.write(safe_text + "\n")
            stream.flush()
        except UnicodeEncodeError:
            stream.write(safe_text.encode("ascii", errors="replace").decode("ascii") + "\n")
            stream.flush()


def thinking_spinner(stop_event: threading.Event, text: str = "Thinking", stream: io.TextIOBase | None = None) -> None:
    """Show animated spinner while stop_event is not set."""
    if stream is None:
        stream = sys.stdout
    idx = 0
    while not stop_event.is_set():
        frame = SPINNER_FRAMES[idx % len(SPINNER_FRAMES)]
        try:
            stream.write(f"\r{DIM}  {frame} {text}...{RESET}")
            stream.flush()
        except Exception:
            pass
        idx += 1
        time.sleep(0.1)
    try:
        stream.write(f"\r{' ' * 60}\r")
        stream.flush()
    except Exception:
        pass



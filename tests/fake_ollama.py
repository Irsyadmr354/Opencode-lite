"""Threaded fake Ollama (OpenAI-compatible) server for offline tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _split(text: str, parts: int = 2) -> list[str]:
    """Split text into ``parts`` pieces (>=2 elements always, padding with '')."""
    if len(text) < parts:
        chars = list(text)
        while len(chars) < parts:
            chars.append("")
        return chars
    size = len(text) // parts
    pieces = [text[i * size:(i + 1) * size] for i in range(parts - 1)]
    pieces.append(text[(parts - 1) * size:])
    return pieces


def _chunk(delta: dict, finish_reason: str | None, model: str) -> dict:
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse_payload(entry: dict, model: str) -> str:
    chunks = [_chunk({"role": "assistant"}, None, model)]
    content = entry.get("content")
    if content:
        for piece in _split(content, 2):
            chunks.append(_chunk({"content": piece}, None, model))
    for index, call in enumerate(entry.get("tool_calls") or []):
        func = call.get("function") or {}
        head, tail = _split(func.get("arguments") or "")
        chunks.append(_chunk(
            {"tool_calls": [{
                "index": index,
                "id": call.get("id") or f"call_{index}",
                "type": "function",
                "function": {"name": func.get("name", ""), "arguments": head},
            }]}, None, model))
        chunks.append(_chunk(
            {"tool_calls": [{"index": index, "function": {"arguments": tail}}]},
            None, model))
    finish = entry.get("finish_reason") or ("tool_calls" if entry.get("tool_calls") else "stop")
    chunks.append(_chunk({}, finish, model))
    frames = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
    return frames + "data: [DONE]\n\n"


def _json_payload(entry: dict, model: str) -> dict:
    message: dict = {"role": "assistant", "content": entry.get("content")}
    if entry.get("tool_calls"):
        message["tool_calls"] = entry["tool_calls"]
    finish = entry.get("finish_reason") or ("tool_calls" if entry.get("tool_calls") else "stop")
    return {
        "id": "chatcmpl-fake",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


class FakeOllama:
    """Scripted /v1/chat/completions server.

    Script entries are OpenAI-style assistant messages:
      {"content": "..."} and/or
      {"content": None, "tool_calls": [{"id","type","function":{"name","arguments"}}]}
      optional "finish_reason".
    Special raw entry: {"status": 500, "body": "..."} for HTTP error testing.
    Entries are consumed in order; the last one repeats when exhausted (so a
    single tool_call entry yields an infinite tool-calling loop).
    """

    def __init__(self, script: list[dict], model: str = "fake") -> None:
        self.script = list(script)
        self.model = model
        self.last_request: dict | None = None
        self.port: int | None = None
        self.base_url: str | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._cursor = 0
        self._lock = threading.Lock()

    def _next_entry(self) -> dict:
        if not self.script:
            raise RuntimeError("FakeOllama script is empty")
        with self._lock:
            index = min(self._cursor, len(self.script) - 1)
            self._cursor += 1
        return self.script[index]

    def start(self, port: int = 0):
        """Start serving on 127.0.0.1:<port>; returns (server, port)."""
        if self._server is not None:
            return self._server, self.port
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    outer.last_request = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    outer.last_request = None
                    self._send_json(400, {"error": "invalid JSON body"})
                    return
                try:
                    entry = outer._next_entry()
                except RuntimeError as exc:
                    self._send_json(500, {"error": str(exc)})
                    return
                if "status" in entry:
                    self._send_raw(entry.get("status", 500), entry.get("body", ""),
                                   "application/json")
                    return
                if outer.last_request.get("stream"):
                    body = _sse_payload(entry, outer.model).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._send_json(200, _json_payload(entry, outer.model))

            def _send_json(self, status: int, obj: dict) -> None:
                self._send_raw(status, json.dumps(obj), "application/json")

            def _send_raw(self, status: int, text: str, ctype: str) -> None:
                body = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):  # keep pytest output clean
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={"poll_interval": 0.05}, daemon=True)
        self._thread.start()
        self.port = self._server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        return self._server, self.port

    def shutdown(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

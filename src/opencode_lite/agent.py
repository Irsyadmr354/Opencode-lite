"""Core agent loop: wires streamed LLM turns to duck-typed tools."""

from __future__ import annotations

import json
import time

from .config import Config
from .llm import LLMClient, LLMError, ToolCall  # noqa: F401  (re-exported for typing)

SYSTEM_PROMPT = (
    "You are Assistant, coding agent in workspace. Tools: websearch, webfetch, "
    "read_file, write_file, delete_file, list_files, shell. Always use tools, "
    "never fabricate. Straight and concise. No intro/outro/chit-chat or headers "
    "(#, ##, ###)."
)


#: Maps tool names to their ``Config.permissions`` attribute. Tools absent
#: here are governed purely by their ``danger`` flag (default policy: allow).
_PERMISSION_KEY = {
    "write_file": "write",
    "delete_file": "delete",
    "shell": "shell",
    "webfetch": "webfetch",
    "websearch": "websearch",
}


class Hooks:
    """No-op lifecycle hooks; the UI subclasses this."""

    def on_start(self) -> None: ...

    def on_delta(self, text: str) -> None: ...

    def on_reasoning(self, text: str) -> None: ...

    def on_assistant_done(self, turn) -> None: ...

    def on_tool_start(self, name: str, args: dict) -> None: ...

    def on_tool_result(self, name: str, res) -> None: ...

    def on_permission(self, name: str, args: dict) -> bool:
        return False

    def on_status(self, info: dict) -> None: ...

    def on_error(self, msg: str) -> None: ...


class _CancelFlag:
    """threading.Event-compatible read-only view of ``Agent.cancelled``.

    Lets another thread flip the plain bool attribute and have in-flight
    streaming observe it via ``is_set()``.
    """

    __slots__ = ("_agent",)

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent

    def is_set(self) -> bool:
        return bool(self._agent.cancelled)


class Agent:
    """Blocking tool-loop agent. Tools are duck-typed objects with
    ``name/description/parameters/danger/fn``; ``fn(args)`` returns something
    with ``ok`` bool and ``output`` str."""

    def __init__(self, client: LLMClient, tools: list, config: Config,
                 hooks: Hooks | None = None) -> None:
        self.client = client
        self.tools = list(tools)
        self.config = config
        self.hooks = hooks if hooks is not None else Hooks()
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.cancelled: bool = False

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.cancelled = False

    # -- internals -----------------------------------------------------------

    def _tools_schema(self) -> list[dict]:
        schema = []
        for tool in self.tools:
            schema.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters",
                                          {"type": "object", "properties": {}}),
                },
            })
        return schema

    def _approx_tokens(self) -> int:
        total = 0
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content) // 4
            for call in message.get("tool_calls") or []:
                total += len(json.dumps(call)) // 4
        return total

    def _assistant_message(self, turn) -> dict:
        msg: dict = {"role": "assistant"}
        if turn.content:
            msg["content"] = turn.content
        if turn.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name,
                                 "arguments": json.dumps(call.arguments)},
                }
                for call in turn.tool_calls
            ]
        return msg

    def _execute_tool(self, call: ToolCall, _timings: list | None = None) -> dict:
        tool = self.tools_by_name.get(call.name)
        if tool is None:
            if _timings is not None:
                _timings.append({"name": call.name, "duration_s": 0.0, "ok": False})
            return {"role": "tool", "tool_call_id": call.id,
                    "content": f"ERROR: unknown tool '{call.name}'"}
        perm_key = _PERMISSION_KEY.get(call.name)
        perm = getattr(self.config.permissions, perm_key,
                       "allow") if perm_key else "allow"
        if perm == "deny":
            if _timings is not None:
                _timings.append({"name": call.name, "duration_s": 0.0, "ok": False})
            return {"role": "tool", "tool_call_id": call.id,
                    "content": "DENIED by policy (config [permissions])"}
        needs_ask = bool(getattr(tool, "danger", False)) or perm == "ask"
        if needs_ask and not self.hooks.on_permission(call.name, call.arguments):
            if _timings is not None:
                _timings.append({"name": call.name, "duration_s": 0.0, "ok": False})
            return {"role": "tool", "tool_call_id": call.id, "content": "DENIED by user"}
        self.hooks.on_tool_start(call.name, call.arguments)
        t0 = time.monotonic()
        try:
            res = tool.fn(call.arguments)
        except Exception as exc:  # noqa: BLE001 - tool crashes become tool messages
            res = {"ok": False, "output": f"ERROR: {exc}"}
        duration = time.monotonic() - t0
        # Determine ok for timing record
        if isinstance(res, dict):
            ok = bool(res.get("ok", True))
        else:
            ok = bool(getattr(res, "ok", True))
        if _timings is not None:
            _timings.append({"name": call.name, "duration_s": duration, "ok": ok})
        self.hooks.on_tool_result(call.name, res)
        if isinstance(res, dict):
            output = str(res.get("output", ""))
        else:
            output = str(res)
        return {"role": "tool", "tool_call_id": call.id, "content": output}

    def _prune_context(self, max_tokens: int = 32000) -> None:
        """Prune oldest conversation units when over the token budget.

        A removable UNIT is either a single non-tool message, or an assistant
        message carrying tool_calls together with ALL immediately-following
        contiguous ``role:"tool"`` results — so pruning can never orphan a
        tool message (OpenAI-compatible servers reject those with HTTP 400).
        The system prompt (index 0) and the newest user turn are never
        removed.
        """
        while len(self.messages) > 2 and self._approx_tokens() > max_tokens:
            newest_user = 0
            for idx, message in enumerate(self.messages):
                if message.get("role") == "user":
                    newest_user = idx
            if newest_user <= 1:
                break  # only the newest user turn remains; nothing safely prunable
            end = 2
            head = self.messages[1]
            if head.get("role") == "assistant" and head.get("tool_calls"):
                while (end < len(self.messages)
                       and self.messages[end].get("role") == "tool"):
                    end += 1
            del self.messages[1:end]

    # -- public API ----------------------------------------------------------

    def submit(self, user_text: str) -> None:
        """Run the full agent loop for one user request (blocking)."""
        self._prune_context()
        self.messages.append({"role": "user", "content": user_text})
        self.cancelled = False
        # Immediate feedback: show spinner before any network wait (TTFT)
        try:
            self.hooks.on_start()
        except Exception:
            pass
        flag = _CancelFlag(self)
        max_rounds = max(1, int(self.config.max_tool_rounds))

        for rnd in range(1, max_rounds + 1):
            turn = None
            t0 = time.monotonic()
            t_first_token: float | None = None
            approx_before = self._approx_tokens()
            try:
                for event in self.client.chat_stream(self.messages,
                                                     tools_schema=self._tools_schema(),
                                                     cancel=flag):
                    if event["type"] in ("delta", "reasoning") and t_first_token is None:
                        t_first_token = time.monotonic()
                    if event["type"] == "delta":
                        self.hooks.on_delta(event["text"])
                    elif event["type"] == "reasoning":
                        self.hooks.on_reasoning(event["text"])
                    elif event["type"] == "final":
                        turn = event["turn"]
            except LLMError as exc:
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
                self.hooks.on_error(str(exc))
                return
            t_done = time.monotonic()
            if turn is None:
                if self.messages and self.messages[-1].get("role") == "user":
                    self.messages.pop()
                self.hooks.on_error("stream ended without final turn")
                return

            # Merge agent timings into turn.stats (preserve llm wall_duration_s etc.)
            if not hasattr(turn, "stats") or turn.stats is None:
                turn.stats = {}
            if not isinstance(turn.stats, dict):
                turn.stats = {"_raw": turn.stats}
            # wall_duration_s: prefer llm's value, fallback to agent measurement
            if turn.stats.get("wall_duration_s") is None:
                turn.stats["wall_duration_s"] = t_done - t0
            # ttft
            turn.stats["ttft_s"] = (t_first_token - t0) if t_first_token is not None else None
            turn.stats["approx_tokens_before"] = approx_before
            # placeholder for after (updated after assistant message appended)
            turn.stats["approx_tokens_after"] = self._approx_tokens()
            # tool timings list shared with hook reference
            tool_timings: list[dict] = []
            # Preserve any existing tool_calls stats? Use our list
            turn.stats["tool_calls"] = tool_timings
            turn.stats["tool_total_s"] = 0.0
            turn.stats.setdefault("ollama", None)
            turn.stats.setdefault("usage", None)

            self.hooks.on_assistant_done(turn)
            self.messages.append(self._assistant_message(turn))
            # Update approx after assistant message
            turn.stats["approx_tokens_after"] = self._approx_tokens()

            if not turn.tool_calls:
                return
            if self.cancelled:  # stop between rounds; skip pending tool execution
                return
            for call in turn.tool_calls:
                if self.cancelled:
                    break
                self.messages.append(self._execute_tool(call, _timings=tool_timings))
            # Update total after all tools
            turn.stats["tool_total_s"] = sum(t.get("duration_s", 0.0) for t in tool_timings)
            # Also keep tool_calls list live (already updated via shared reference)
            self.hooks.on_status({"round": rnd, "max": max_rounds,
                                  "approx_tokens": self._approx_tokens()})
            if self.cancelled:
                return
            self._prune_context()

        self.hooks.on_error("max tool rounds reached")

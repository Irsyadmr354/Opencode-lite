"""Core agent loop: wires streamed LLM turns to duck-typed tools."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .config import Config
from .llm import LLMClient, LLMError, ToolCall  # noqa: F401  (re-exported for typing)

DEFAULT_SYSTEM_PROMPT = (
    "You are Assistant, coding agent in workspace. Tools: get_current_time, "
    "websearch, webfetch, read_file, write_file, delete_file, list_files, shell. "
    "Use tools only when needed. Match user language. "
    "Always list_files '.' before read, if vague list immediately. "
    "Be concise, no intro/outro, >20 lines -> file, use tool_calls."
)

# Kept for backwards compat / tests — prefer config.system_prompt at runtime
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def _resolve_system_prompt(config: Config | None) -> str:
    if config is not None and getattr(config, "system_prompt", None):
        sp = str(config.system_prompt).strip()
        if sp:
            return sp
    return DEFAULT_SYSTEM_PROMPT


def _resolve_identity(config: Config | None, default_ws: str | None = None) -> str:
    if config is not None and getattr(config, "identity", None):
        ident = str(config.identity).strip()
        if ident:
            return ident
    ws = default_ws or (str(config.workspace) if config and hasattr(config, "workspace") else "workspace")
    return f"Assistant, coding agent in workspace {ws}."


_HALU_PERSONA_RE = re.compile(
    r"\b(claude|anthropic|openai|gpt-4|gpt-3\.5|chatgpt)\b",
    re.IGNORECASE,
)


def _sanitize_turn_content(content: str | None, config: Config | None = None) -> str | None:
    if not content or not isinstance(content, str):
        return content
    # Strip any leaked raw tool call syntax markers from content
    cleaned = re.sub(r"<\/?(?:tools|tool_calls?|function_calls?|actions?)>.*?(?:<\/(?:tools|tool_calls?|function_calls?|actions?)>|$)", "", content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r">tool_calls?\s*\[[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\/?(?:tools|tool_calls?|function_calls?|actions?)>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r">tool_calls?\s*\{.*?\}", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\{\s*\"type\"\s*:\s*\"function\".*?\}", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()

    # Universal identity replacement: if model hallucinates a vendor/model name (Claude, OpenAI, etc.),
    # replace it with the configured identity name without altering the user's language or sentence structure.
    if _HALU_PERSONA_RE.search(cleaned):
        cleaned = _HALU_PERSONA_RE.sub("Assistant", cleaned)

    return cleaned


class Hooks:
    """No-op lifecycle hooks; the UI subclasses this."""

    def on_start(self) -> None: ...

    def on_delta(self, text: str) -> None: ...

    def on_reasoning(self, text: str) -> None: ...

    def on_assistant_done(self, turn) -> None: ...

    def on_tool_start(self, name: str, args: dict) -> None: ...

    def on_tool_result(self, name: str, res) -> None: ...

    def on_permission(self, name: str, args: dict) -> bool:
        """Called when a tool requires confirmation; returns True if approved."""
        return False

    def on_status(self, info: dict) -> None: ...

    def on_error(self, message: str) -> None: ...


class _CancelFlag:
    """Read-only view of agent.cancelled passed into client.chat_stream."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def is_set(self) -> bool:
        return self._agent.cancelled


class Agent:
    def __init__(self, client: LLMClient, tools: list,
                 config: Config | None = None,
                 hooks: Hooks | None = None) -> None:
        self.client = client
        self.tools = list(tools)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.config = config if config is not None else Config()
        self.hooks = hooks if hooks is not None else Hooks()
        self.cancelled = False
        sys_prompt = _resolve_system_prompt(self.config)
        self.messages: list[dict] = [
            {"role": "system", "content": sys_prompt}
        ]

    # -- session management --------------------------------------------------

    def save_session(self, name: str) -> Path:
        """Save current conversation history to a session file."""
        from . import session
        return session.save_session(name, self.messages)

    def load_session(self, name: str) -> list[dict]:
        """Load conversation history from a named session file."""
        from . import session
        messages = session.load_session(name)
        self.messages = messages
        self.cancelled = False
        return list(self.messages)

    def new_session(self) -> None:
        """Reset conversation history to a new session."""
        self.reset_session()
        self.cancelled = False

    def get_session_messages(self) -> list[dict]:
        """Return the current conversation history (for saving)."""
        return list(self.messages)

    def load_session_messages(self, messages: list[dict]) -> None:
        """Replace the current conversation history (from a loaded session).

        Preserves the current system prompt if the loaded messages do not
        contain one.
        """
        if not messages:
            sys_prompt = _resolve_system_prompt(self.config)
            self.messages = [{"role": "system", "content": sys_prompt}]
            return
        # Ensure system prompt is present as first message
        if messages[0].get("role") != "system":
            sys_prompt = _resolve_system_prompt(self.config)
            self.messages = [{"role": "system", "content": sys_prompt}] + list(messages)
        else:
            self.messages = list(messages)

    def reset_session(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        sys_prompt = _resolve_system_prompt(self.config)
        self.messages = [{"role": "system", "content": sys_prompt}]

    # -- tool / context helpers ----------------------------------------------

    def _tools_schema(self) -> list[dict]:
        schema: list[dict] = []
        for t in self.tools:
            schema.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": (getattr(t, "parameters", None) or
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
        # Always set content as string - Ollama rejects null/missing content (Go <nil>)
        content = _sanitize_turn_content(turn.content, self.config) if isinstance(turn.content, str) else ""
        # If model dumped tool JSON as content but also provided tool_calls, hide the JSON
        if turn.tool_calls and content:
            stripped = content.strip()
            fenced = re.sub(r"^```[a-z]*\s*\n?", "", stripped, flags=re.IGNORECASE)
            fenced = re.sub(r"\n?```\s*$", "", fenced).strip()
            if fenced.startswith("{") and '"name"' in fenced and '"arguments"' in fenced:
                content = ""
            elif '"name"' in content and '"arguments"' in content and (content.strip().startswith("{") or content.strip().startswith("```")):
                content = ""
        msg: dict = {"role": "assistant", "content": content}
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
        perm_key = getattr(tool, "permission_key", None)
        if perm_key is not None and hasattr(self.config.permissions, perm_key):
            perm = getattr(self.config.permissions, perm_key)
        else:
            perm = "ask" if tool.danger else "allow"
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
        if isinstance(res, dict):
            ok = bool(res.get("ok", True))
        else:
            ok = bool(getattr(res, "ok", True))
        if _timings is not None:
            _timings.append({"name": call.name, "duration_s": duration, "ok": ok})
        self.hooks.on_tool_result(call.name, res)
        if isinstance(res, dict):
            body = res.get("output", "")
        else:
            body = getattr(res, "output", "")
        return {"role": "tool", "tool_call_id": call.id, "content": str(body)}

    def _prune_context(self, max_tokens: int | None = None) -> None:
        """Prune oldest conversation turns when context budget is exceeded.

        Pruning removes turns in complete atomic UNITs: (a) a user turn
        along with its immediately following assistant response and any
        tool results, or (b) an assistant tool_calls turn along with its
        contiguous role:tool results - so pruning can never orphan a tool
        message (OpenAI-compatible servers reject those with HTTP 400). The
        system prompt (index 0) and the newest user turn are never removed.
        """
        if max_tokens is None:
            max_tokens = int(getattr(self.config, "max_context_tokens", 12000))
        while len(self.messages) > 2 and self._approx_tokens() > max_tokens:
            newest_user = 0
            for idx, message in enumerate(self.messages):
                if message.get("role") == "user":
                    newest_user = idx
            if newest_user <= 1:
                break  # only the newest user turn remains; nothing safely prunable
            head = self.messages[1]
            end = 2
            if head.get("role") == "user":
                if end < len(self.messages) and self.messages[end].get("role") == "assistant":
                    end += 1
                    while end < len(self.messages) and self.messages[end].get("role") == "tool":
                        end += 1
                if newest_user < end:
                    end = 2
                    if 1 <= newest_user < end:
                        break
            elif head.get("role") == "assistant" and head.get("tool_calls"):
                while (end < len(self.messages)
                       and self.messages[end].get("role") == "tool"):
                    end += 1
            if 1 <= newest_user < end:
                break
            del self.messages[1:end]

    # -- public API ----------------------------------------------------------

    def submit(self, user_text: str) -> None:
        """Run the full agent loop for one user request (blocking)."""
        self._prune_context()
        for m in self.messages:
            if "content" not in m or m["content"] is None:
                m["content"] = ""
            elif not isinstance(m["content"], str):
                m["content"] = str(m["content"])
        self.messages.append({"role": "user", "content": user_text})
        self.cancelled = False
        try:
            self.hooks.on_start()
        except Exception:
            pass
        flag = _CancelFlag(self)
        max_rounds = max(1, int(self.config.max_tool_rounds))

        for rnd in range(1, max_rounds + 1):
            if rnd > 1:
                try:
                    self.hooks.on_start()
                except Exception:
                    pass
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

            if not hasattr(turn, "stats") or turn.stats is None:
                turn.stats = {}
            if not isinstance(turn.stats, dict):
                turn.stats = {"_raw": turn.stats}
            if turn.stats.get("wall_duration_s") is None:
                turn.stats["wall_duration_s"] = t_done - t0
            turn.stats["ttft_s"] = (t_first_token - t0) if t_first_token is not None else None
            turn.stats["approx_tokens_before"] = approx_before
            turn.stats["approx_tokens_after"] = self._approx_tokens()
            tool_timings: list[dict] = []
            turn.stats["tool_calls"] = tool_timings
            turn.stats["tool_total_s"] = 0.0
            turn.stats.setdefault("ollama", None)
            turn.stats.setdefault("usage", None)

            if turn.content:
                turn.content = _sanitize_turn_content(turn.content, self.config)

            self.hooks.on_assistant_done(turn)
            self.messages.append(self._assistant_message(turn))
            turn.stats["approx_tokens_after"] = self._approx_tokens()

            if not turn.tool_calls:
                return
            if self.cancelled:
                return
            for call in turn.tool_calls:
                if self.cancelled:
                    break
                self.messages.append(self._execute_tool(call, _timings=tool_timings))
            turn.stats["tool_total_s"] = sum(t.get("duration_s", 0.0) for t in tool_timings)
            self.hooks.on_status({"round": rnd, "max": max_rounds,
                                  "approx_tokens": self._approx_tokens()})
            if self.cancelled:
                return
            self._prune_context()

        self.hooks.on_error("max tool rounds reached")

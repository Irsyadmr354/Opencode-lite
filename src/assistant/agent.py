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
    "Never call tools for simple greetings (hi/hello). Only for news/search: "
    "call get_current_time then websearch. Always list_files '.' before read, if vague list immediately. "
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


_HALU_RE = re.compile(
    r"\b(anthropic|openai|gpt-4|gpt-3\.5|chatgpt|as an ai language model|i am an ai language model|i'm an ai language model|created by anthropic|created by openai|based on openai)\b",
    re.IGNORECASE,
)


def _sanitize_turn_content(
    content: str | None,
    config: Config | None,
    is_greeting: bool = False,
    user_text: str = "",
) -> str | None:
    if not content or not isinstance(content, str):
        if is_greeting:
            u = user_text.lower()
            if any(k in u for k in ("halo", "hai", "selamat", "pagi", "siang", "malam", "siapa", "apa")):
                return "Halo! Ada yang bisa saya bantu di workspace ini?"
            return "Hello! How can I assist you today?"
        return content
    stripped = content.strip()
    # Check for raw tool call leak in content
    if (
        stripped.startswith(">tool_call")
        or stripped.startswith("<tool_call")
        or ">tool_calls" in stripped
        or re.match(r"^>tool_calls?\s*\[.*\]$", stripped)
        or (stripped.startswith("[") and stripped.endswith("]") and "get_current_time" in stripped)
    ):
        if is_greeting:
            u = user_text.lower()
            if any(k in u for k in ("halo", "hai", "selamat", "pagi", "siang", "malam", "siapa", "apa")):
                return "Halo! Ada yang bisa saya bantu di workspace ini?"
            return "Hello! How can I assist you today?"
        return ""
    # Strip any inline >tool_calls or <tool_call>...</tool_call> fragments
    cleaned = re.sub(r">tool_calls?\s*\[[^\]]*\]", "", content, flags=re.IGNORECASE)
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\/?tool_calls?>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r">tool_calls?\s*\{.*?\}", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()

    if not cleaned and is_greeting:
        u = user_text.lower()
        if any(k in u for k in ("halo", "hai", "selamat", "pagi", "siang", "malam", "siapa", "apa")):
            return "Halo! Ada yang bisa saya bantu di workspace ini?"
        return "Hello! How can I assist you today?"

    if _HALU_RE.search(cleaned or content):
        ws = str(config.workspace) if config and hasattr(config, "workspace") else "workspace"
        ident = _resolve_identity(config, ws)
        lowered = (cleaned or content).lower()
        if any(k in lowered for k in ("saya", "adalah", "model", "bahasa", "dibuat", "oleh")):
            return f"Assistant, coding agent di workspace {ws}. Bisa bantu baca/tulis/hapus file, list file, shell, dan websearch/webfetch."
        return f"{ident} Capabilities: read/write/delete files, list files, shell commands, websearch/webfetch."
    return cleaned if cleaned else content


_GREETING_TOKENS = {"hi", "hello", "hey", "yo", "sup", "hiya", "howdy", "halo", "hai", "p"}


def _is_simple_greeting(text: str) -> bool:
    t = text.strip().lower().rstrip("!.,? \t\r\n")
    if t in _GREETING_TOKENS:
        return True
    words = t.split()
    if len(words) == 1 and words[0] in _GREETING_TOKENS:
        return True
    if len(words) == 2 and words[0] in _GREETING_TOKENS and words[1] in {"there", "bot", "assistant", "bro", "kawan", "teman"}:
        return True
    return False


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
        self.messages: list[dict] = [{"role": "system", "content": _resolve_system_prompt(config)}]
        self.cancelled: bool = False

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": _resolve_system_prompt(self.config)}]
        self.cancelled = False

    # -- session persistence -------------------------------------------------

    def save_session(self, name: str):
        """Persist current conversation history to ``~/.assistant/sessions/<name>.json``."""
        from . import session as _session  # lazy import to avoid circular

        return _session.save_session(name, self.messages)

    def load_session(self, name: str) -> None:
        """Load conversation history from a saved session, replacing current messages."""
        from . import session as _session  # lazy import to avoid circular

        msgs = _session.load_session(name)
        # Sanitize loaded messages (old files may have content=null)
        for m in msgs:
            if "content" not in m or m["content"] is None:
                m["content"] = ""
            elif not isinstance(m["content"], str):
                m["content"] = str(m["content"])
        self.messages = msgs
        self.cancelled = False

    def new_session(self) -> None:
        """Start a fresh session (alias for reset)."""
        self.reset()

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
        # Always set content as string - Ollama rejects null/missing content (Go <nil>)
        content = (
            _sanitize_turn_content(
                turn.content,
                self.config,
                is_greeting=getattr(self, "_is_greeting", False),
                user_text=getattr(self, "_last_user_text", ""),
            )
            if isinstance(turn.content, str)
            else ""
        )
        # If model dumped tool JSON as content but also provided tool_calls, hide the JSON
        if turn.tool_calls and content:
            stripped = content.strip()
            # strip fences for check
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

    def _prune_context(self, max_tokens: int | None = None) -> None:
        """Prune oldest conversation units when over the token budget.

        Prevents hallucination by dropping the oldest turns first while
        preserving conversational coherence.

        A removable UNIT is either a single non-tool message, or an assistant
        message carrying tool_calls together with ALL immediately-following
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
            # Determine removable UNIT starting at 1, never including the newest user
            head = self.messages[1]
            end = 2
            if head.get("role") == "user":
                # User turn: include the following assistant (with or without tool_calls) and its tools
                # so we don't leave a dangling assistant/tool without its user query
                # which triggers Jinja "No user query found" on some templates
                if end < len(self.messages) and self.messages[end].get("role") == "assistant":
                    end += 1
                    # Include any immediately following tool results
                    while end < len(self.messages) and self.messages[end].get("role") == "tool":
                        end += 1
                # Safety: don't delete the newest user
                if newest_user < end:
                    # The unit would include the newest user, so just delete the single old user
                    # (should not happen since newest_user is last, but keep safe)
                    end = 2
                    if 1 <= newest_user < end:
                        break
            elif head.get("role") == "assistant" and head.get("tool_calls"):
                while (end < len(self.messages)
                       and self.messages[end].get("role") == "tool"):
                    end += 1
            # Final safety: never delete the newest user
            if 1 <= newest_user < end:
                break
            del self.messages[1:end]

    # -- public API ----------------------------------------------------------

    def submit(self, user_text: str) -> None:
        """Run the full agent loop for one user request (blocking)."""
        self._prune_context()
        # Sanitize existing messages: Ollama Go rejects content=null (<nil>)
        for m in self.messages:
            if "content" not in m or m["content"] is None:
                m["content"] = ""
            elif not isinstance(m["content"], str):
                m["content"] = str(m["content"])
        self.messages.append({"role": "user", "content": user_text})
        is_greeting = _is_simple_greeting(user_text)
        self._last_user_text = user_text
        self._is_greeting = is_greeting
        self.cancelled = False
        # Immediate feedback: show spinner before any network wait (TTFT)
        try:
            self.hooks.on_start()
        except Exception:
            pass
        flag = _CancelFlag(self)
        max_rounds = max(1, int(self.config.max_tool_rounds))

        for rnd in range(1, max_rounds + 1):
            # Show instant spinner for every LLM round (tool rounds have TTFT too)
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
                tools_schema = None if is_greeting else self._tools_schema()
                for event in self.client.chat_stream(self.messages,
                                                     tools_schema=tools_schema,
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

            if is_greeting:
                turn.tool_calls = []
                turn.content = _sanitize_turn_content(
                    turn.content,
                    self.config,
                    is_greeting=True,
                    user_text=user_text,
                )
            elif turn.content:
                turn.content = _sanitize_turn_content(
                    turn.content,
                    self.config,
                    is_greeting=False,
                    user_text=user_text,
                )

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

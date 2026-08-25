"""Core agent loop: wires streamed LLM turns to duck-typed tools."""

from __future__ import annotations

import json

from .config import Config
from .llm import LLMClient, LLMError, ToolCall  # noqa: F401  (re-exported for typing)

SYSTEM_PROMPT = (
    "You are a helpful, terse, and highly capable coding agent working inside a workspace.\n"
    "\n"
    "Tools available:\n"
    "- read_file(path, start_line): read a file's contents.\n"
    "- write_file(path, content): create or overwrite a file.\n"
    "- delete_file(path): remove a file.\n"
    "- list_files(path, pattern): list directory entries.\n"
    "- shell(command): run a shell command.\n"
    "- webfetch(url): fetch text from a URL.\n"
    "- websearch(query): search the web.\n"
    "\n"
    "Rules:\n"
    "1. When greeting or conversing, respond directly in plain text. Do NOT emit empty json or fake tool blocks.\n"
    "2. Always read_file or list_files before editing; never guess file contents.\n"
    "3. Prefer surgical edits; rewrite whole files only when necessary.\n"
    "4. Paths are relative to the workspace root.\n"
    "5. When calling tools, emit valid function calls. When done, summarize findings concisely.\n"
    "6. You may include concise <think> reasoning thoughts </think> before taking action or answering."
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

    def on_delta(self, text: str) -> None: ...

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

    def _execute_tool(self, call: ToolCall) -> dict:
        tool = self.tools_by_name.get(call.name)
        if tool is None:
            return {"role": "tool", "tool_call_id": call.id,
                    "content": f"ERROR: unknown tool '{call.name}'"}
        perm_key = _PERMISSION_KEY.get(call.name)
        perm = getattr(self.config.permissions, perm_key,
                       "allow") if perm_key else "allow"
        if perm == "deny":
            return {"role": "tool", "tool_call_id": call.id,
                    "content": "DENIED by policy (config [permissions])"}
        needs_ask = bool(getattr(tool, "danger", False)) or perm == "ask"
        if needs_ask and not self.hooks.on_permission(call.name, call.arguments):
            return {"role": "tool", "tool_call_id": call.id, "content": "DENIED by user"}
        self.hooks.on_tool_start(call.name, call.arguments)
        try:
            res = tool.fn(call.arguments)
        except Exception as exc:  # noqa: BLE001 - tool crashes become tool messages
            res = {"ok": False, "output": f"ERROR: {exc}"}
        self.hooks.on_tool_result(call.name, res)
        if isinstance(res, dict):
            output = str(res.get("output", ""))
        else:
            output = str(res)
        return {"role": "tool", "tool_call_id": call.id, "content": output}

    # -- public API ----------------------------------------------------------

    def submit(self, user_text: str) -> None:
        """Run the full agent loop for one user request (blocking)."""
        self.messages.append({"role": "user", "content": user_text})
        self.cancelled = False
        flag = _CancelFlag(self)
        max_rounds = max(1, int(self.config.max_tool_rounds))

        for rnd in range(1, max_rounds + 1):
            turn = None
            try:
                for event in self.client.chat_stream(self.messages,
                                                     tools_schema=self._tools_schema(),
                                                     cancel=flag):
                    if event["type"] == "delta":
                        self.hooks.on_delta(event["text"])
                    elif event["type"] == "final":
                        turn = event["turn"]
            except LLMError as exc:
                self.hooks.on_error(str(exc))
                return
            if turn is None:
                self.hooks.on_error("stream ended without final turn")
                return

            self.hooks.on_assistant_done(turn)
            self.messages.append(self._assistant_message(turn))

            if not turn.tool_calls:
                return
            if self.cancelled:  # stop between rounds; skip pending tool execution
                return
            for call in turn.tool_calls:
                self.messages.append(self._execute_tool(call))
            self.hooks.on_status({"round": rnd, "max": max_rounds,
                                  "approx_tokens": self._approx_tokens()})
            if self.cancelled:
                return

        self.hooks.on_error("max tool rounds reached")

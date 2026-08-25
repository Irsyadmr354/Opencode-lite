"""Integration tests: real tools + agent loop + permission enforcement."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fake_ollama import FakeOllama  # noqa: E402

import opencode_lite.ui as ui_mod  # noqa: E402
from opencode_lite.agent import Agent, Hooks  # noqa: E402
from opencode_lite.config import load_config  # noqa: E402
from opencode_lite.llm import LLMClient  # noqa: E402
from opencode_lite.tools import get_tools  # noqa: E402


class TrackingHooks(Hooks):
    def __init__(self, allow: bool = True):
        super().__init__()
        self.allow = allow
        self.asked: list[tuple[str, dict]] = []
        self.tool_results: list[tuple[str, object]] = []

    def on_permission(self, name: str, args: dict) -> bool:
        self.asked.append((name, dict(args)))
        return self.allow

    def on_tool_result(self, name: str, res) -> None:
        self.tool_results.append((name, res))


def _run(script: list[dict], tmp_path: Path, hooks: Hooks,
         permissions: dict | None = None) -> Agent:
    server = FakeOllama(script)
    server.start()
    try:
        cfg = load_config(None, {"workspace": tmp_path})
        if permissions:
            for key, value in permissions.items():
                setattr(cfg.permissions, key, value)
        client = LLMClient(server.base_url, "ollama", "fake")
        agent = Agent(client, get_tools(cfg.workspace, cfg), cfg, hooks)
        agent.submit("do it")
        return agent
    finally:
        server.shutdown()


def test_policy_deny_blocks_write_without_asking(tmp_path):
    """permissions.write='deny' -> write_file refused, never prompts the user."""
    hooks = TrackingHooks()
    script = [
        {"content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "write_file",
                         "arguments": '{"path": "note.txt", "content": "hi"}'}}]},
        {"content": "done"},
    ]
    agent = _run(script, tmp_path, hooks, permissions={"write": "deny"})

    assert not (tmp_path / "note.txt").exists(), "file must NOT be written"
    assert hooks.asked == [], "deny must short-circuit without prompting"
    tool_msgs = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    assert any("DENIED by policy" in c for c in tool_msgs)
    assert agent.messages[-1]["content"] == "done"


def test_policy_ask_prompts_for_non_dangerous_write(tmp_path):
    """permissions.write='ask' -> even safe tools prompt; allow => executed."""
    hooks = TrackingHooks(allow=True)
    script = [
        {"content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "write_file",
                         "arguments": '{"path": "ok.txt", "content": "x"}'}}]},
        {"content": "wrote"},
    ]
    _run(script, tmp_path, hooks, permissions={"write": "ask"})

    assert ("write_file", {"path": "ok.txt", "content": "x"}) in hooks.asked
    assert (tmp_path / "ok.txt").read_text(encoding="utf-8") == "x"


def test_shell_denied_by_user_never_executes(tmp_path):
    """danger tool + user says no -> DENIED by user, command not run."""
    hooks = TrackingHooks(allow=False)
    script = [
        {"content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "shell",
                         "arguments": '{"command": "Write-Output ran"}'}}]},
        {"content": "ok"},
    ]
    agent = _run(script, tmp_path, hooks)

    assert ("shell", {"command": "Write-Output ran"}) in hooks.asked
    tool_msgs = [m["content"] for m in agent.messages if m.get("role") == "tool"]
    assert any(c == "DENIED by user" for c in tool_msgs)
    assert not any("ran" in c and "DENIED" not in c for c in tool_msgs)


# --- TUI smoke (headless pilot, no display needed) ---------------------------

class FakeAgent:
    def __init__(self, ask_permission: bool = False):
        self.hooks = None
        self.messages: list[dict] = []
        self.cancelled = False
        self._ask_permission = ask_permission
        self.permission_answer: bool | None = None

    def reset(self) -> None:
        self.messages.clear()

    def submit(self, text: str) -> None:
        h = self.hooks
        if self._ask_permission:
            self.permission_answer = bool(h.on_permission(
                "shell", {"command": "Write-Output hi"}))
            return
        h.on_delta("hello ")
        h.on_delta("world")
        h.on_assistant_done(SimpleNamespace(content="hello world",
                                            tool_calls=[]))
        h.on_status({"round": 1, "max": 25, "approx_tokens": 42})


async def _drive(app, predicate, timeout_s: float = 8.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        if predicate():
            return True
    return False


def _log_text(app) -> str:
    chat = app.query_one("#chat", ui_mod.RichLog)
    return "\n".join("".join(seg.text for seg in strip) for strip in chat.lines)


def test_tui_smoke_stream_to_log(tmp_path):
    import opencode_lite.ui as ui_mod

    agent = FakeAgent()
    cfg = SimpleNamespace(model="fake", workspace=tmp_path)
    app = ui_mod.ChatApp(agent, cfg)

    async def scenario():
        async with app.run_test(size=(100, 30)) as pilot:
            assert agent.hooks is app.hooks, "ChatApp must auto-attach UiHooks"
            inp = app.query_one("#input", ui_mod.Input)
            inp.value = "say hi"
            await pilot.press("enter")
            ok = await _drive(app, lambda: not app._busy)
            assert ok, "generation never finished"
            text = _log_text(app)
            assert "hello" in text and "world" in text, f"deltas not rendered: {text!r}"
            assert "ERROR" not in text, f"unexpected error path: {text!r}"
            assert not inp.disabled, "input must re-enable after turn"

    asyncio.run(scenario())


def test_tui_permission_modal_allow(tmp_path):
    import opencode_lite.ui as ui_mod

    agent = FakeAgent(ask_permission=True)
    cfg = SimpleNamespace(model="fake", workspace=tmp_path)
    app = ui_mod.ChatApp(agent, cfg)

    async def scenario():
        async with app.run_test(size=(100, 30)) as pilot:
            inp = app.query_one("#input", ui_mod.Input)
            inp.value = "run it"
            await pilot.press("enter")

            def modal_up():
                return isinstance(app.screen, ui_mod.PermissionModal)

            assert await _drive(app, modal_up), "permission modal never appeared"
            await pilot.press("y")
            ok = await _drive(app, lambda: not app._busy)
            assert ok, "turn never finished after modal"
            assert agent.permission_answer is True

    asyncio.run(scenario())

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


# --- Terminal REPL & TerminalHooks integration tests -----------------------

import io


class FakeAgent:
    def __init__(self, ask_permission: bool = False, fail_tool: bool = False):
        self.hooks = None
        self.messages: list[dict] = []
        self.cancelled = False
        self._ask_permission = ask_permission
        self._fail_tool = fail_tool
        self.permission_answer: bool | None = None
        self.submit_calls: list[str] = []

    def reset(self) -> None:
        self.messages.clear()

    def submit(self, text: str) -> None:
        self.submit_calls.append(text)
        h = self.hooks
        if not h:
            return
        if self._ask_permission:
            self.permission_answer = bool(h.on_permission(
                "shell", {"command": "Write-Output hi"}))
            return
        h.on_tool_start("shell", {"command": "echo test"})
        if self._fail_tool:
            h.on_tool_result("shell", SimpleNamespace(ok=False, output="command failed"))
        else:
            h.on_tool_result("shell", SimpleNamespace(ok=True, output="test output"))
        h.on_delta("hello ")
        h.on_delta("world")
        h.on_assistant_done(SimpleNamespace(content="hello world", tool_calls=[]))
        h.on_status({"round": 1, "max": 25, "approx_tokens": 42})


def test_terminal_hooks_streaming_word_by_word():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("Hello ")
    hooks.on_delta("world")
    hooks.on_delta("!")
    hooks.on_assistant_done(SimpleNamespace(content="Hello world!"))

    output = out.getvalue()
    assert "Hello world!\n" in output


def test_terminal_hooks_thinking_tags_ansi_color():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("<think>pondering deeply</think>Here is the answer.")
    hooks.on_assistant_done(SimpleNamespace(content="<think>pondering deeply</think>Here is the answer."))

    output = out.getvalue()
    assert ui_mod.ANSI_THINKING in output
    assert "pondering deeply" in output
    assert "Here is the answer." in output


def test_terminal_hooks_tool_start_and_result():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)

    # Success tool
    hooks.on_tool_start("shell", {"command": "git status"})
    hooks.on_tool_result("shell", SimpleNamespace(ok=True, output="clean"))
    output = out.getvalue()
    assert "⚡" in output
    assert "[shell]" in output
    assert "git status" in output
    assert "ok" in output

    # Failed tool
    out2 = io.StringIO()
    hooks2 = ui_mod.TerminalHooks(stdout=out2)
    hooks2.on_tool_start("read_file", {"path": "missing.txt"})
    hooks2.on_tool_result("read_file", SimpleNamespace(ok=False, output="ERROR: file not found"))
    output2 = out2.getvalue()
    assert "⚡" in output2
    assert "[read_file]" in output2
    assert "missing.txt" in output2
    assert "ERROR: file not found" in output2


def test_terminal_hooks_permission_prompt_inline():
    prompts: list[str] = []

    def mock_input(p: str) -> str:
        prompts.append(p)
        return "y"

    hooks_allow = ui_mod.TerminalHooks(input_fn=mock_input)
    assert hooks_allow.on_permission("delete_file", {"path": "data.csv"}) is True
    assert len(prompts) == 1
    assert "Allow delete_file(data.csv)? [y/N]:" in prompts[0]
    assert "[permission]" in prompts[0]

    # Deny path
    hooks_deny = ui_mod.TerminalHooks(input_fn=lambda p: "n")
    assert hooks_deny.on_permission("delete_file", {"path": "data.csv"}) is False

    # KeyboardInterrupt path
    def raise_interrupt(p: str) -> str:
        raise KeyboardInterrupt()

    hooks_interrupt = ui_mod.TerminalHooks(input_fn=raise_interrupt)
    assert hooks_interrupt.on_permission("delete_file", {"path": "data.csv"}) is False



def test_terminal_hooks_error_formatting_red():
    err = io.StringIO()
    hooks = ui_mod.TerminalHooks(stderr=err)
    hooks.on_error("Test failure message")
    output = err.getvalue()
    assert ui_mod.ANSI_RED in output
    assert "ERROR:" in output
    assert "Test failure message" in output
    assert hooks.had_error is True


def test_repl_banner_and_exit(tmp_path):
    agent = FakeAgent()
    cfg = SimpleNamespace(model="qwen-test", workspace=tmp_path, base_url="http://127.0.0.1:11434/v1")
    outputs: list[str] = []
    inputs = iter(["/exit"])

    ui_mod.run_repl(
        agent=agent,
        config=cfg,
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
        clear_fn=lambda: None,
    )

    combined = "\n".join(outputs)
    assert "opencode-lite" in combined
    assert "model: qwen-test" in combined
    assert "/help for commands, /exit to quit" in combined
    assert "Goodbye!" in combined


def test_repl_commands_status_model_reset_help(tmp_path):
    agent = FakeAgent()
    agent.client = SimpleNamespace(model="qwen-test")
    agent.messages = [{"role": "user", "content": "hi"}]
    cfg = SimpleNamespace(
        model="qwen-test",
        workspace=tmp_path,
        base_url="http://127.0.0.1:11434/v1",
        permissions=SimpleNamespace(write="allow", shell="ask"),
    )
    outputs: list[str] = []
    inputs = iter([
        "/status",
        "/model",
        "/model deepseek-coder:14b",
        "/reset",
        "/help",
        "/quit",
    ])

    ui_mod.run_repl(
        agent=agent,
        config=cfg,
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
        clear_fn=lambda: None,
    )

    combined = "\n".join(outputs)
    # /status check
    assert "Model:       qwen-test" in combined
    assert "Base URL:    http://127.0.0.1:11434/v1" in combined
    assert "write=allow" in combined
    # /model check
    assert "Active model: qwen-test" in combined
    assert "Model switched to: deepseek-coder:14b" in combined
    assert cfg.model == "deepseek-coder:14b"
    assert agent.client.model == "deepseek-coder:14b"
    # /reset check
    assert "Session memory reset." in combined
    assert agent.messages == []
    # /help check
    assert "OPENCODE-LITE GUIDE" in combined


def test_repl_clear_screen(tmp_path):
    agent = FakeAgent()
    cfg = SimpleNamespace(model="fake", workspace=tmp_path)
    cleared = []
    inputs = iter(["/clear", "/exit"])

    ui_mod.run_repl(
        agent=agent,
        config=cfg,
        input_fn=lambda prompt: next(inputs),
        output_fn=lambda msg: None,
        clear_fn=lambda: cleared.append(True),
    )

    # Launch clear + /clear command = 2 clears
    assert len(cleared) == 2


def test_repl_user_prompt_execution(tmp_path):
    agent = FakeAgent()
    cfg = SimpleNamespace(model="fake", workspace=tmp_path)
    inputs = iter(["build me a feature", "/exit"])
    outputs: list[str] = []

    ui_mod.run_repl(
        agent=agent,
        config=cfg,
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
        clear_fn=lambda: None,
    )

    assert agent.submit_calls == ["build me a feature"]


def test_repl_ctrl_c_cancels_active_generation(tmp_path):
    class InterruptAgent(FakeAgent):
        def submit(self, text: str) -> None:
            self.messages.append({"role": "user", "content": text})
            raise KeyboardInterrupt()

    agent = InterruptAgent()
    cfg = SimpleNamespace(model="fake", workspace=tmp_path)
    outputs: list[str] = []
    inputs = iter(["long running task", "/exit"])

    ui_mod.run_repl(
        agent=agent,
        config=cfg,
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
        clear_fn=lambda: None,
    )

    assert agent.cancelled is True
    combined = "\n".join(outputs)
    assert "[Cancelled]" in combined
    # Unfulfilled user message popped
    assert agent.messages == []


def test_repl_ctrl_c_idle_notice(tmp_path):
    agent = FakeAgent()
    cfg = SimpleNamespace(model="fake", workspace=tmp_path)
    outputs: list[str] = []

    count = 0

    def sim_input(prompt):
        nonlocal count
        count += 1
        if count == 1:
            raise KeyboardInterrupt()
        elif count == 2:
            raise KeyboardInterrupt()
        return "/exit"

    ui_mod.run_repl(
        agent=agent,
        config=cfg,
        input_fn=sim_input,
        output_fn=outputs.append,
        clear_fn=lambda: None,
    )

    combined = "\n".join(outputs)
    assert "Type /exit or press Ctrl+C again to quit." in combined
    assert "Goodbye!" in combined


def test_repl_eof_clean_exit(tmp_path):
    agent = FakeAgent()
    cfg = SimpleNamespace(model="fake", workspace=tmp_path)
    outputs: list[str] = []

    def sim_input(prompt):
        raise EOFError()

    ui_mod.run_repl(
        agent=agent,
        config=cfg,
        input_fn=sim_input,
        output_fn=outputs.append,
        clear_fn=lambda: None,
    )

    combined = "\n".join(outputs)
    assert "Goodbye!" in combined


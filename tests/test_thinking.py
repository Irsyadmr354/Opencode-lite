"""Thinking/reasoning support: field capture (llm) + split-tag-safe UI (single-line \\r)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fake_ollama import FakeOllama

import assistant.ui as ui_mod
from assistant.agent import Agent, Hooks
from assistant.config import load_config
from assistant.llm import LLMClient
from assistant.tools import get_tools


def _collect(base_url: str, **kw):
    client = LLMClient(base_url, "ollama", "fake", **kw)
    events = list(client.chat_stream([{"role": "user", "content": "hi"}]))
    final = [e for e in events if e["type"] == "final"][0]["turn"]
    return events, final


def test_stream_reasoning_field():
    server = FakeOllama([{"reasoning": "pondering deeply", "content": "final answer"}])
    server.start()
    try:
        events, turn = _collect(server.base_url)
    finally:
        server.shutdown()
    reasoning = "".join(e["text"] for e in events if e["type"] == "reasoning")
    content = "".join(e["text"] for e in events if e["type"] == "delta")
    assert reasoning == "pondering deeply"
    assert content == "final answer"
    assert turn.reasoning == "pondering deeply"


def test_stream_reasoning_content_alias():
    server = FakeOllama([{"reasoning_content": "hmm ok", "content": "yes"}])
    server.start()
    try:
        events, turn = _collect(server.base_url)
    finally:
        server.shutdown()
    assert "".join(e["text"] for e in events if e["type"] == "reasoning") == "hmm ok"
    assert turn.reasoning == "hmm ok"


def test_nonstream_fallback_reasoning():
    raw = {
        "id": "x", "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi there",
                        "reasoning": "slow careful thought"},
            "finish_reason": "stop",
        }],
    }
    server = FakeOllama([{"raw_json": raw}])
    server.start()
    try:
        events, turn = _collect(server.base_url)
    finally:
        server.shutdown()
    assert any(e["type"] == "reasoning" for e in events)
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "hi there"
    assert turn.reasoning == "slow careful thought"


class RecordingHooks(Hooks):
    def __init__(self):
        super().__init__()
        self.reasoning: list[str] = []
        self.deltas: list[str] = []

    def on_reasoning(self, text: str) -> None:
        self.reasoning.append(text)

    def on_delta(self, text: str) -> None:
        self.deltas.append(text)


def test_agent_forwards_reasoning_events(tmp_path):
    server = FakeOllama([{"reasoning": "step one two", "content": "done"}])
    server.start()
    try:
        cfg = load_config(None, {"workspace": tmp_path})
        hooks = RecordingHooks()
        agent = Agent(LLMClient(server.base_url, "ollama", "fake"),
                      get_tools(cfg.workspace, cfg), cfg, hooks)
        agent.submit("go")
    finally:
        server.shutdown()
    assert "".join(hooks.reasoning) == "step one two"
    assert "".join(hooks.deltas) == "done"


def test_tag_holdback_units():
    f = ui_mod._tag_holdback
    assert f("<thi", False) == 4
    assert f("hello</thin", True) == 6
    assert f("</think>", True) == 0
    assert f("<think>", False) == 0
    assert f("abc<", False) == 1
    assert f("plain text!", False) == 0
    assert f("[thinkin", False) == 8


import io
import os


def _term(monkeypatch, cols: int, lines: int) -> None:
    monkeypatch.setattr(ui_mod.shutil, "get_terminal_size", lambda: os.terminal_size((cols, lines)))


def test_ui_renders_reasoning_field():
    """Native reasoning deltas stream as single-line preview, header collapses to static."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    frame0 = ui_mod.SPINNER_FRAMES[0]

    hooks.on_reasoning("thinking hard\nabout it")
    output = out.getvalue()
    assert f"{frame0} Thinking" in output
    # preview is inline on header line, newlines become spaces
    assert "thinking hard" in output
    assert "about it" in output

    hooks.on_delta("answer here")
    output = out.getvalue()
    # collapse writes static checkmark header with newline, no vertical moves
    assert "✓ Thinking" in output
    assert "\033[1A" not in output  # no vertical cursor moves
    assert "answer here" in output
    # spinner frozen after done
    hooks.on_assistant_done(SimpleNamespace(content="answer here", reasoning="thinking hard\nabout it"))
    assert hooks._spinner_frozen is True


def test_ui_split_tag_no_leak():
    """'<thi|nk>' split across deltas must still parse as a think block."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    for chunk in ("I wonder <thi", "nk>secret plan</thin", "k>so, proceed"):
        hooks.on_delta(chunk)
    hooks.on_assistant_done(SimpleNamespace(content="I wonder <think>secret plan</think>so, proceed"))

    output = out.getvalue()
    assert "I wonder " in output
    assert "secret plan" in output  # preview contains it
    assert "✓ Thinking" in output  # header collapsed
    assert ui_mod.AI_PREFIX in output
    assert "so, proceed" in output
    assert "<think>" not in output
    assert "</think>" not in output


def test_ui_collapses_thinking_on_transition_to_content():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_reasoning("analyzing the code\nfinding bugs")
    assert "analyzing the code" in out.getvalue()
    assert "finding bugs" in out.getvalue()

    hooks.on_delta("Here is the fix.")
    hooks.on_assistant_done(SimpleNamespace(content="Here is the fix."))

    output = out.getvalue()
    assert "✓ Thinking" in output
    assert "\033[1A" not in output
    assert "Here is the fix." in output


def test_ui_single_chunk_think_no_plain_leak():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("<think>secret plan</think>Visible answer.")
    hooks.on_assistant_done(SimpleNamespace(content="<think>secret plan</think>Visible answer."))

    output = out.getvalue()
    assert "secret plan" in output
    # preview appears once via header, not as plain answer
    assert output.count("secret plan") == 1
    assert "✓ Thinking" in output
    assert ui_mod.AI_PREFIX in output
    assert "Visible answer." in output


def test_ui_physical_rows_wrap_erase(monkeypatch):
    """Long reasoning is truncated to terminal width, not multi-row erase."""
    _term(monkeypatch, cols=40, lines=24)
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("<think>" + "x" * 100 + "</think>Answer.")
    hooks.on_assistant_done(SimpleNamespace(content="<think>" + "x" * 100 + "</think>Answer."))

    output = out.getvalue()
    # preview truncated with ellipsis, not full 100 x's
    assert "…" in output
    assert "✓ Thinking" in output
    assert "Answer." in output
    assert "\033[1A" not in output


def test_ui_collapses_bracket_thinking_on_close():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("[thinking: quick thoughts]Ready.")
    hooks.on_assistant_done(SimpleNamespace(content="[thinking: quick thoughts]Ready."))

    output = out.getvalue()
    assert "quick thoughts" in output
    assert "✓ Thinking" in output
    assert ui_mod.AI_PREFIX in output
    assert "Ready." in output


def test_ui_spinner_guard_tiny_viewport(monkeypatch):
    """Tiny viewport must not crash; single-line spinner never needs vertical moves."""
    _term(monkeypatch, cols=40, lines=5)
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_reasoning("a\nb\nc\nd\ne\nf")
    hooks.on_reasoning("more")
    output = out.getvalue()
    # header appears, but no vertical moves ever
    assert "Thinking" in output
    assert "\033[1A" not in output
    assert "\033[s" not in output

    hooks.on_delta("A")
    hooks.on_assistant_done(SimpleNamespace(content="A"))
    output = out.getvalue()
    assert "✓ Thinking" in output
    assert "\033[1A" not in output
    assert "A" in output


def test_ui_spinner_freezes_on_done():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_reasoning("x")
    hooks.on_delta("ans")
    output = out.getvalue()
    assert "✓ Thinking" in output
    # single-line mode: no save/restore, just \r
    assert "\r\033[2K" in output

    hooks.on_assistant_done(SimpleNamespace(content="ans"))
    snapshot = out.getvalue()
    assert hooks._spinner_frozen is True
    hooks._update_header()
    assert out.getvalue() == snapshot


def test_ui_plain_model_output_identical():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("Just answer.")
    hooks.on_assistant_done(SimpleNamespace(content="Just answer."))

    output = out.getvalue()
    assert output == ui_mod.AI_PREFIX + "Just answer." + "\n"
    assert "Thinking" not in output
    assert "\033[1A" not in output


def test_ui_reset_stream_semantics():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_reasoning("t")
    hooks.on_delta("a")
    hooks.on_assistant_done(SimpleNamespace(content="a"))
    snapshot = out.getvalue()
    hooks.reset_stream()
    assert out.getvalue() == snapshot
    assert hooks._header_active is False
    assert hooks._spinner_frozen is False
    assert hooks._buffer == ""

    out2 = io.StringIO()
    hooks2 = ui_mod.TerminalHooks(stdout=out2)
    hooks2.on_reasoning("t")
    hooks2.reset_stream()
    output2 = out2.getvalue()
    assert "\r\033[2K" in output2
    assert hooks2._header_active is False


def test_ui_hides_tool_json_leaks():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta('```json\n{"name": "write_file", "arguments": {"path": "a.txt", "content": "123"}}\n```')
    hooks.on_assistant_done(SimpleNamespace(content='```json\n{"name": "write_file", "arguments": {"path": "a.txt", "content": "123"}}\n```', tool_calls=[SimpleNamespace(id="1", name="write_file", arguments={"path": "a.txt"})]))
    output = out.getvalue()
    assert '"name"' not in output
    assert '"write_file"' not in output
    assert '"arguments"' not in output


def test_ui_hides_raw_tool_call_syntax():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta(">tool_calls [get_current_time]")
    hooks.on_assistant_done(SimpleNamespace(content=">tool_calls [get_current_time]"))
    output = out.getvalue()
    assert ">tool_calls" not in output
    assert "get_current_time" not in output

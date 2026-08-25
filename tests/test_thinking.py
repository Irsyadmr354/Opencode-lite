"""Thinking/reasoning support: field capture (llm) + split-tag-safe UI."""

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


# --- split-tag holdback ------------------------------------------------------

def test_tag_holdback_units():
    f = ui_mod._tag_holdback
    assert f("<thi", False) == 4            # partial opener held back
    assert f("hello</thin", True) == 6      # partial closer while inside think
    assert f("</think>", True) == 0         # complete token -> parser handles it
    assert f("<think>", False) == 0         # complete token
    assert f("abc<", False) == 1
    assert f("plain text!", False) == 0
    assert f("[thinkin", False) == 8        # partial bracket opener


# --- TerminalHooks UI rendering ----------------------------------------------

import io
import os


def _term(monkeypatch, cols: int, lines: int) -> None:
    """Force a deterministic terminal geometry for physical-row accounting."""
    monkeypatch.setattr(ui_mod.shutil, "get_terminal_size", lambda: os.terminal_size((cols, lines)))


def test_ui_renders_reasoning_field():
    """Native reasoning deltas must stream TEXT live below a persistent header."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    frame0 = ui_mod.SPINNER_FRAMES[0]

    hooks.on_reasoning("thinking hard\nabout it")
    output = out.getvalue()
    assert f"{frame0} Thinking" in output                    # header line
    assert ui_mod.ANSI_THINKING + "thinking hard" in output  # reasoning text streamed
    assert "about it" in output

    hooks.on_delta("answer here")
    output = out.getvalue()
    # Exactly the 2 physical reasoning rows are erased; the header stays.
    assert "\r\033[2K\033[1A\033[2K\r" in output
    assert ui_mod.SPINNER_FRAMES[2] + " Thinking" in output  # spinner advanced on delta
    assert "answer here" in output
    assert "\033[u" in output                                # cursor restored after redraw

    hooks.on_assistant_done(
        SimpleNamespace(content="answer here", reasoning="thinking hard\nabout it"))
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
    assert ui_mod.ANSI_THINKING + "secret plan" in output    # think body rendered styled
    assert "\r\033[2K\r" in output                           # its single physical row erased
    assert ui_mod.AI_PREFIX in output                        # answer follows persisted header
    assert ui_mod.SPINNER_FRAMES[0] + " Thinking" in output  # header survives collapse
    assert "so, proceed" in output
    # Must not have raw tag literals leaked in output
    assert "<think>" not in output
    assert "</think>" not in output


def test_ui_collapses_thinking_on_transition_to_content():
    """Reasoning text is rendered live, then erased physically when content starts."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_reasoning("analyzing the code\nfinding bugs")
    assert "analyzing the code" in out.getvalue()            # live reasoning TEXT
    assert "finding bugs" in out.getvalue()

    hooks.on_delta("Here is the fix.")
    hooks.on_assistant_done(SimpleNamespace(content="Here is the fix."))

    output = out.getvalue()
    assert "\r\033[2K\033[1A\033[2K\r" in output             # both physical rows erased
    assert ui_mod.SPINNER_FRAMES[0] + " Thinking" in output  # header kept, not wiped
    assert ui_mod.SPINNER_FRAMES[2] + " Thinking" in output  # and still animating
    assert "Here is the fix." in output


def test_ui_single_chunk_think_no_plain_leak():
    """<think>...</think> inside ONE chunk must never re-flush as plain answer."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("<think>secret plan</think>Visible answer.")
    hooks.on_assistant_done(SimpleNamespace(content="<think>secret plan</think>Visible answer."))

    output = out.getvalue()
    assert ui_mod.ANSI_THINKING + "secret plan" in output    # flushed styled BEFORE collapse
    assert output.count("secret plan") == 1                  # exactly once, never plain
    assert "\033[2K" in output                               # reasoning rows erased
    assert ui_mod.AI_PREFIX in output                        # real answer starts fresh
    assert "Visible answer." in output


def test_ui_physical_rows_wrap_erase(monkeypatch):
    """Soft-wrapped long reasoning consumes ceil(chars/columns) physical rows."""
    _term(monkeypatch, cols=40, lines=24)
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    # 100 chars at 40 cols -> 1 leading row + 2 wrapped rows = 3 physical rows.
    hooks.on_delta("<think>" + "x" * 100 + "</think>Answer.")
    hooks.on_assistant_done(SimpleNamespace(content="<think>" + "x" * 100 + "</think>Answer."))

    output = out.getvalue()
    assert ui_mod.ANSI_THINKING + "x" * 100 in output
    # 3-row erase = two (clear+up) steps plus final clear, ending on answer row.
    assert "\r\033[2K\033[1A\033[2K\033[1A\033[2K\r" in output
    assert "Answer." in output


def test_ui_collapses_bracket_thinking_on_close():
    """[thinking: ...] streams styled text, erases it, keeps the header on ']'."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("[thinking: quick thoughts]Ready.")
    hooks.on_assistant_done(SimpleNamespace(content="[thinking: quick thoughts]Ready."))

    output = out.getvalue()
    assert ui_mod.ANSI_THINKING + " quick thoughts" in output  # opener tag consumed w/ space
    assert "\r\033[2K\r" in output
    assert ui_mod.SPINNER_FRAMES[0] + " Thinking" in output  # header survives
    assert ui_mod.AI_PREFIX in output
    assert "Ready." in output


def test_ui_spinner_guard_tiny_viewport(monkeypatch):
    """Up-moves that would meet/exceed screen height are suppressed, erases clamped."""
    _term(monkeypatch, cols=40, lines=5)
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_reasoning("a\nb\nc\nd\ne\nf")                  # 7 physical rows > 5-line screen
    hooks.on_reasoning("more")                              # would need an up-move -> guarded
    output = out.getvalue()
    assert output.count(ui_mod.SPINNER_FRAMES[0] + " Thinking") == 1  # no rewrite emitted
    assert "\033[s" not in output                                     # cursor untouched

    hooks.on_delta("A")
    hooks.on_assistant_done(SimpleNamespace(content="A"))
    output = out.getvalue()
    # Erase clamped to lines-2 = 3 rows: exactly two up-moves, never more.
    assert "\r\033[2K\033[1A\033[2K\033[1A\033[2K\r" in output
    assert output.count("\033[1A") == 2
    assert "A" in output


def test_ui_spinner_freezes_on_done():
    """Spinner advances while the answer streams, freezes permanently on done."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_reasoning("x")
    hooks.on_delta("ans")
    output = out.getvalue()
    assert ui_mod.SPINNER_FRAMES[2] + " Thinking" in output  # spun during answer phase
    assert "\033[s" in output and "\033[u" in output         # in-place header redraws

    hooks.on_assistant_done(SimpleNamespace(content="ans"))
    snapshot = out.getvalue()
    assert hooks._spinner_frozen is True
    hooks._rewrite_header()                                  # frozen: must be a no-op
    assert out.getvalue() == snapshot                        # zero further rewrites


def test_ui_plain_model_output_identical():
    """Models that never reason produce byte-identical output to the legacy path."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_delta("Just answer.")
    hooks.on_assistant_done(SimpleNamespace(content="Just answer."))

    output = out.getvalue()
    assert output == ui_mod.AI_PREFIX + "Just answer." + "\n"
    assert "Thinking" not in output
    assert "\033[s" not in output
    assert "\033[1A" not in output
    assert "\033[2K" not in output


def test_ui_reset_stream_semantics():
    """reset_stream: completed turn keeps its frozen header; aborted stream cleans up."""
    # Completed turn: frozen header persists, state cleared, screen untouched.
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

    # Mid-stream abort (e.g. Ctrl+C): erase reasoning rows AND the header itself.
    out2 = io.StringIO()
    hooks2 = ui_mod.TerminalHooks(stdout=out2)
    hooks2.on_reasoning("t")
    hooks2.reset_stream()
    output2 = out2.getvalue()
    assert "\r\033[2K\r" in output2                          # reasoning row erased
    assert "\033[1A\r\033[2K" in output2                     # header line erased too
    assert hooks2._header_active is False



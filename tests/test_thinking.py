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


def test_ui_renders_reasoning_field():
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    hooks.on_reasoning("thinking hard\nabout it")
    hooks.on_delta("answer here")
    hooks.on_assistant_done(SimpleNamespace(content="answer here", reasoning="thinking hard\nabout it"))

    output = out.getvalue()
    assert ui_mod.ANSI_THINKING in output
    assert "thinking hard" in output
    assert "about it" in output
    assert "answer here" in output


def test_ui_split_tag_no_leak():
    """'<thi|nk>' split across deltas must still parse as a think block."""
    out = io.StringIO()
    hooks = ui_mod.TerminalHooks(stdout=out)
    for chunk in ("I wonder <thi", "nk>secret plan</thin", "k>so, proceed"):
        hooks.on_delta(chunk)
    hooks.on_assistant_done(SimpleNamespace(content="I wonder <think>secret plan</think>so, proceed"))

    output = out.getvalue()
    assert "I wonder " in output
    assert ui_mod.ANSI_THINKING in output
    assert "secret plan" in output
    assert "so, proceed" in output
    # Must not have raw tag literals leaked in output
    assert "<think>" not in output
    assert "</think>" not in output


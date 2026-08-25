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


# --- UI rendering ------------------------------------------------------------

class ReasonFakeAgent:
    def __init__(self, split_tags: bool = False):
        self.hooks = None
        self.messages: list[dict] = []
        self.cancelled = False
        self._split_tags = split_tags

    def reset(self) -> None:
        self.messages.clear()

    def submit(self, text: str) -> None:
        h = self.hooks
        if self._split_tags:
            for chunk in ("I wonder <thi", "nk>secret plan</thin", "k>so, proceed"):
                h.on_delta(chunk)
            h.on_assistant_done(SimpleNamespace(
                content="I wonder <think>secret plan</think>so, proceed",
                tool_calls=[]))
            return
        h.on_reasoning("thinking hard\nabout it")
        h.on_delta("answer here")
        h.on_assistant_done(SimpleNamespace(
            content="answer here", tool_calls=[], reasoning="thinking hard\nabout it"))


def _log_text(app) -> str:
    chat = app.query_one("#chat", ui_mod.RichLog)
    return "\n".join("".join(seg.text for seg in strip) for strip in chat.lines)


def test_ui_renders_reasoning_field(tmp_path):
    agent = ReasonFakeAgent()
    app = ui_mod.ChatApp(agent, SimpleNamespace(model="fake", workspace=tmp_path))

    async def scenario():
        async with app.run_test(size=(100, 30)) as pilot:
            inp = app.query_one("#input", ui_mod.Input)
            inp.value = "go"
            await pilot.press("enter")
            deadline = asyncio.get_event_loop().time() + 8
            while asyncio.get_event_loop().time() < deadline and app._busy:
                await asyncio.sleep(0.05)
            assert not app._busy
            text = _log_text(app)
            assert "thinking hard" in text and "about it" in text
            assert "answer here" in text

    asyncio.run(scenario())


def test_ui_split_tag_no_leak(tmp_path):
    """'<thi|nk>' split across deltas must still parse as a think block."""
    agent = ReasonFakeAgent(split_tags=True)
    app = ui_mod.ChatApp(agent, SimpleNamespace(model="fake", workspace=tmp_path))

    async def scenario():
        async with app.run_test(size=(100, 30)) as pilot:
            inp = app.query_one("#input", ui_mod.Input)
            inp.value = "go"
            await pilot.press("enter")
            deadline = asyncio.get_event_loop().time() + 8
            while asyncio.get_event_loop().time() < deadline and app._busy:
                await asyncio.sleep(0.05)
            assert not app._busy
            text = _log_text(app)
            assert "secret plan" in text
            assert "proceed" in text

    asyncio.run(scenario())

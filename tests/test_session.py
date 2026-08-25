"""Tests for session persistence + pruning integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import assistant.session as session
from assistant.agent import Agent
from assistant.config import Config
from assistant.llm import LLMClient

from fake_ollama import FakeOllama  # noqa: E402


# --- helpers ---------------------------------------------------------------

def _use_tmp_sessions(monkeypatch, tmp_path: Path):
    """Redirect SESSION_DIR to a temp directory."""
    tmp_sess = tmp_path / "sessions"
    monkeypatch.setattr(session, "SESSION_DIR", tmp_sess)
    # Ensure _session_dir creates it lazily
    return tmp_sess


# --- session module -------------------------------------------------------

def test_save_load_roundtrip(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    p = session.save_session("my-work_1.test", msgs)
    assert p.exists()
    assert p.name == "my-work_1.test.json"
    # Verify file structure
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["messages"] == msgs
    assert "saved_at" in data
    # indent 2 check: file contains two-space indent
    assert '  "messages"' in p.read_text(encoding="utf-8")

    loaded = session.load_session("my-work_1.test")
    assert loaded == msgs

    # list
    assert session.list_sessions() == ["my-work_1.test"]
    # overwrite
    session.save_session("a", msgs)
    session.save_session("b", msgs)
    assert session.list_sessions() == ["a", "b", "my-work_1.test"]


def test_list_sorted_and_empty(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    assert session.list_sessions() == []
    for name in ["zebra", "apple", "middle"]:
        session.save_session(name, [])
    assert session.list_sessions() == ["apple", "middle", "zebra"]


def test_invalid_names_rejected(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    bad = ["", "bad/name", "a/b", "../evil", "has space", "bad*char", "slash\\back", "/abs", "name/json"]
    for name in bad:
        with pytest.raises(ValueError):
            session.save_session(name, [])
        with pytest.raises(ValueError):
            session.load_session(name)
        with pytest.raises(ValueError):
            session.delete_session(name)
        with pytest.raises(ValueError):
            session.session_path(name)
    # valid names should pass
    for good in ["ok", "ok_123", "my.session-1", "A", "123", "a.b_c-d"]:
        p = session.save_session(good, [])
        assert p.exists()
        session.delete_session(good)


def test_save_validates_messages_type(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        session.save_session("good", "not-a-list")  # type: ignore
    with pytest.raises(ValueError):
        session.save_session("good2", {"role": "user"})  # type: ignore


def test_load_missing_raises(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        session.load_session("nope")


def test_load_invalid_json(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    p = session.session_path("badjson")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        session.load_session("badjson")


def test_load_invalid_structure(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    p = session.session_path("badstruct")
    p.parent.mkdir(parents=True, exist_ok=True)
    # missing messages key
    p.write_text(json.dumps({"oops": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        session.load_session("badstruct")
    # messages not a list
    p.write_text(json.dumps({"messages": "hi"}), encoding="utf-8")
    with pytest.raises(ValueError):
        session.load_session("badstruct")
    # messages contains non-dicts
    p.write_text(json.dumps({"messages": ["hi"]}), encoding="utf-8")
    with pytest.raises(ValueError):
        session.load_session("badstruct")


def test_delete_roundtrip(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    session.save_session("todelete", [{"role": "user", "content": "x"}])
    assert "todelete" in session.list_sessions()
    session.delete_session("todelete")
    assert "todelete" not in session.list_sessions()
    assert not session.session_path("todelete").exists()


def test_delete_missing_raises(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        session.delete_session("ghost")


def test_session_path_helper(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    p = session.session_path("alpha")
    assert p.name == "alpha.json"
    assert str(tmp_path) in str(p)
    with pytest.raises(ValueError):
        session.session_path("../traversal")


def test_session_dir_creates(tmp_path, monkeypatch):
    tmp = tmp_path / "newdir" / "sessions"
    monkeypatch.setattr(session, "SESSION_DIR", tmp)
    assert not tmp.exists()
    d = session._session_dir()
    assert d.exists()
    assert d == tmp


# --- agent integration ----------------------------------------------------

def test_agent_save_load_new_session(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    # Minimal fake client (not used for pruning tests)
    srv = FakeOllama([{"content": "hi"}])
    srv.start()
    try:
        client = LLMClient(base_url=srv.base_url, api_key="k", model="m", timeout_s=5)
        cfg = Config()
        agent = Agent(client=client, tools=[], config=cfg)
        # populate messages
        agent.messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        orig = list(agent.messages)
        # save
        path = agent.save_session("agenttest")
        assert Path(path).exists()
        # mutate
        agent.messages.append({"role": "user", "content": "extra"})
        assert len(agent.messages) == 4
        # load restores
        agent.cancelled = True
        agent.load_session("agenttest")
        assert agent.messages == orig
        assert agent.cancelled is False
        # new_session resets
        agent.new_session()
        assert len(agent.messages) == 1
        assert agent.messages[0]["role"] == "system"
        assert agent.cancelled is False
        # save again and verify delete via session
        agent.messages = orig
        agent.save_session("todelete2")
        session.delete_session("todelete2")
        assert "todelete2" not in session.list_sessions()
    finally:
        srv.shutdown()


def test_agent_pruning_still_type_aware(tmp_path, monkeypatch):
    """Ensure pruning hasn't regressed: type-aware, keeps system + newest user."""
    _use_tmp_sessions(monkeypatch, tmp_path)
    srv = FakeOllama([{"content": "fresh answer"}])
    srv.start()
    try:
        client = LLMClient(base_url=srv.base_url, api_key="k", model="m", timeout_s=5)
        cfg = Config()
        agent = Agent(client=client, tools=[], config=cfg)
        agent.messages = [{"role": "system", "content": "system"}]
        for i in range(40):
            agent.messages.append({"role": "user", "content": f"OLD_USER_{i} " + "x" * 2000})
            agent.messages.append({"role": "assistant", "content": f"OLD_ASSIST_{i} " + "y" * 2000})
        # Also test tool pair atomic
        agent.messages.insert(1, {"role": "tool", "tool_call_id": "bad", "content": "orphan"})
        # But that orphan would be invalid; ensure pruning handles normally without orphan insertion
        # Reset to valid shape for pruning test
        agent.messages = [{"role": "system", "content": "system"}]
        for i in range(40):
            agent.messages.append({"role": "user", "content": f"OLD_USER_{i} " + "x" * 2000})
            agent.messages.append({"role": "assistant", "content": f"OLD_ASSIST_{i} " + "y" * 2000})
        agent.submit("fresh question")
        # pruning assertions
        assert agent.messages[0]["role"] == "system"
        flat = json.dumps(agent.messages)
        assert "OLD_USER_0" not in flat
        assert "fresh question" in flat
        # No orphaned tool message
        for idx, m in enumerate(agent.messages):
            if m.get("role") == "tool":
                prev = agent.messages[idx - 1]
                assert prev.get("role") == "assistant" and prev.get("tool_calls")
    finally:
        srv.shutdown()


# --- ui /session integration ----------------------------------------------

def _make_fake_agent():
    class FakeAgent:
        def __init__(self):
            self.messages: list[dict] = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
            self.cancelled = False
            self.hooks = None
            self.client = SimpleNamespace(model="fake")
        def reset(self):
            self.messages = [{"role": "system", "content": "sys"}]
            self.cancelled = False
        def save_session(self, name):
            return session.save_session(name, self.messages)
        def load_session(self, name):
            self.messages = session.load_session(name)
            self.cancelled = False
        def new_session(self):
            self.reset()
        def _approx_tokens(self):
            return sum(len(str(m.get("content", ""))) // 4 for m in self.messages)
        def submit(self, text):
            self.messages.append({"role": "user", "content": text})
    return FakeAgent()


def test_ui_session_commands(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    import assistant.ui as ui_mod

    agent = _make_fake_agent()
    cfg = SimpleNamespace(model="fake", workspace=tmp_path, base_url="http://127.0.0.1:11434/v1", verbose=False, permissions=SimpleNamespace(write="allow"))

    # helper to run repl with sequence of inputs
    def run_inputs(inputs):
        outputs = []
        errs = []

        class ErrHooks:
            def on_error(self, msg):
                errs.append(msg)

        # patch agent.hooks to capture errors via ui's hooks after run_repl sets it
        it = iter(inputs)
        def inp(prompt):
            return next(it)
        # collect _print outputs via output_fn
        ui_mod.run_repl(agent=agent, config=cfg, input_fn=inp, output_fn=outputs.append, clear_fn=lambda: None)
        return outputs, errs

    # Sequence covering all subcommands
    inputs = [
        "/session save test1",
        "/session list",
        "/session",  # status
        "/session save bad/name",  # invalid -> error but not crash
        "/session load test1",
        "/session delete test1",
        "/session list",
        "/session new",
        "/session help",
        "/exit",
    ]
    outputs, _ = run_inputs(inputs)
    combined = "\n".join(outputs)

    assert "Session saved: test1" in combined
    assert "test1" in combined  # list shows it
    assert "Session: " in combined  # status
    assert "Saved sessions:" in combined
    assert "Session loaded: test1" in combined
    assert "Session deleted: test1" in combined
    assert "(no saved sessions)" in combined
    assert "New session started" in combined
    assert "Usage:" in combined
    # After new, agent should have 1 message (system)
    assert len(agent.messages) == 1

    # Test alias /sessions and error handling doesn't crash
    agent2 = _make_fake_agent()
    outputs2 = []
    inputs2 = iter(["/sessions list", "/session load ghost", "/session delete ghost", "/session save", "/session unknownsub", "/exit"])
    errs2 = []
    # Monkey patch hooks.on_error to capture
    original_on_error = ui_mod.TerminalHooks.on_error
    captured = []
    def cap(self, msg):
        captured.append(msg)
        original_on_error(self, msg)
    with mock.patch.object(ui_mod.TerminalHooks, "on_error", cap):
        ui_mod.run_repl(agent=agent2, config=cfg, input_fn=lambda p: next(inputs2), output_fn=outputs2.append, clear_fn=lambda: None)
    # Should have captured errors for missing sessions but still exited cleanly
    assert any("ghost" in c or "not found" in c.lower() for c in captured)
    assert any("Usage" in c for c in captured) or "Usage: /session save" in "\n".join(outputs2)


def test_ui_session_help_and_cls(tmp_path, monkeypatch):
    _use_tmp_sessions(monkeypatch, tmp_path)
    import assistant.ui as ui_mod
    agent = _make_fake_agent()
    cfg = SimpleNamespace(model="fake", workspace=tmp_path)
    # Ensure help text includes /session
    txt = ui_mod._get_help_text()
    assert "/session" in txt
    assert "Manage sessions" in txt
    assert "~/.assistant/sessions/" in txt

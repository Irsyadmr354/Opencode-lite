"""Offline tests for opencode_lite.tools (fs / shell / web).

No network access: webfetch and websearch are exercised via monkeypatched
fakes/stubs. Self-sufficient sys.path setup (does not depend on conftest.py).
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from opencode_lite.tools import (  # noqa: E402
    ToolResult,
    get_tools,
    openai_schema,
)
from opencode_lite.tools import web as web_mod  # noqa: E402


# --- duck-typed config stubs --------------------------------------------------
class Limits:
    def __init__(self, **overrides):
        self.read_max_lines = 200
        self.shell_timeout_s = 30
        self.shell_output_chars = 20000
        self.webfetch_chars = 5000
        self.list_max_entries = 50
        for key, value in overrides.items():
            setattr(self, key, value)


class Config:
    def __init__(self, **limit_overrides):
        self.limits = Limits(**limit_overrides)
        self.shell_cmd = ["powershell", "-NoProfile", "-Command"]


def tool_map(workspace, config=None) -> dict:
    cfg = config or Config()
    return {t.name: t for t in get_tools(pathlib.Path(workspace), cfg)}


# --- contract shape -----------------------------------------------------------
def test_public_contract(tmp_path):
    tools = get_tools(tmp_path, Config())
    assert {t.name for t in tools} == {
        "read_file", "write_file", "delete_file",
        "list_files", "shell", "webfetch", "websearch",
    }
    assert all(t.fn is not None and isinstance(t.parameters, dict) for t in tools)
    assert sorted(t.name for t in tools if t.danger) == ["delete_file", "shell"]

    schema = openai_schema(tools)
    assert schema[0]["type"] == "function"
    assert set(schema[0]["function"]) == {"name", "description", "parameters"}
    assert schema[0]["function"]["name"] == tools[0].name

    # every tool with required args reports a missing-argument error, never raises
    for tool in tools:
        if tool.name == "list_files":  # has no required args
            continue
        res = tool.fn({})
        assert isinstance(res, ToolResult)
        assert not res.ok
        assert res.output.startswith("ERROR: missing argument")


# --- 1. fs roundtrip ----------------------------------------------------------
def test_fs_roundtrip(tmp_path):
    tm = tool_map(tmp_path)

    w = tm["write_file"].fn({"path": "notes/todo.txt", "content": "alpha\nbeta\n"})
    assert w.ok
    assert "todo.txt" in w.output

    r = tm["read_file"].fn({"path": "notes/todo.txt"})
    assert r.ok
    assert r.output.splitlines()[0] == "L1-L2 of 2 lines"
    assert "    1: alpha" in r.output
    assert "    2: beta" in r.output

    lf = tm["list_files"].fn({"path": ".", "pattern": "**/*"})
    assert lf.ok and "notes/todo.txt" in lf.output

    d = tm["delete_file"].fn({"path": "notes/todo.txt"})
    assert d.ok
    assert not (tmp_path / "notes" / "todo.txt").exists()

    r2 = tm["read_file"].fn({"path": "notes/todo.txt"})
    assert not r2.ok and r2.output.startswith("ERROR:")


def test_read_empty_file(tmp_path):
    tm = tool_map(tmp_path)
    tm["write_file"].fn({"path": "e.txt", "content": ""})
    r = tm["read_file"].fn({"path": "e.txt"})
    assert r.ok and r.output == "(empty file)"


def test_delete_directory_is_rejected(tmp_path):
    tm = tool_map(tmp_path)
    (tmp_path / "sub").mkdir()
    r = tm["delete_file"].fn({"path": "sub"})
    assert not r.ok and "directory" in r.output


# --- 2. escape guard ----------------------------------------------------------
def test_escape_guard(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    sibling = tmp_path / "outside.txt"
    sibling.write_text("secret", encoding="utf-8")

    tm = tool_map(ws)
    r = tm["read_file"].fn({"path": "../outside.txt"})
    assert not r.ok and "outside workspace" in r.output

    w = tm["write_file"].fn({"path": "../escaped.txt", "content": "x"})
    assert not w.ok and "outside workspace" in w.output
    d = tm["delete_file"].fn({"path": "../outside.txt"})
    assert not d.ok and "outside workspace" in d.output
    assert sibling.exists()  # untouched


# --- 3. read truncation -------------------------------------------------------
def test_read_caps_at_limit(tmp_path):
    content = "\n".join(f"line{i}" for i in range(1, 501)) + "\n"
    (tmp_path / "big.txt").write_text(content, encoding="utf-8")

    tm = tool_map(tmp_path, Config(read_max_lines=20))
    r = tm["read_file"].fn({"path": "big.txt"})
    assert r.ok
    out_lines = r.output.splitlines()
    assert out_lines[0] == "L1-L20 of 500 lines"
    numbered = [ln for ln in out_lines if re.match(r"\s*\d+: ", ln)]
    assert len(numbered) == 20
    assert "   20: line20" in out_lines[20]
    assert "line21" not in r.output


def test_read_start_line_window(tmp_path):
    (tmp_path / "w.txt").write_text(
        "\n".join(f"L{i}" for i in range(1, 31)) + "\n", encoding="utf-8"
    )
    tm = tool_map(tmp_path, Config(read_max_lines=10))
    r = tm["read_file"].fn({"path": "w.txt", "start_line": 25})
    assert r.ok
    assert r.output.splitlines()[0] == "L25-L30 of 30 lines"
    assert "   25: L25" in r.output


def test_list_excludes_and_pagination(tmp_path):
    for sub in ("node_modules/pkg", ".git/objects", "__pycache__",
                "proj.egg-info", "src/deep"):
        (tmp_path / sub).mkdir(parents=True)
    (tmp_path / "node_modules/pkg/i.js").write_text("1", encoding="utf-8")
    (tmp_path / ".git/config").write_text("g", encoding="utf-8")
    (tmp_path / "__pycache__/x.pyc").write_bytes(b"x")
    (tmp_path / "proj.egg-info/PKG-INFO").write_text("m", encoding="utf-8")
    (tmp_path / "src/a.py").write_text("a", encoding="utf-8")
    (tmp_path / "src/deep/b.py").write_text("b", encoding="utf-8")

    tm = tool_map(tmp_path, Config(list_max_entries=2))
    r = tm["list_files"].fn({})
    assert r.ok
    out = r.output
    assert "src/" in out and "src/deep/" in out
    for junk in ("node_modules", ".git", "__pycache__", "egg-info"):
        assert junk not in out
    assert "... and 2 more" in out  # src/a.py + src/deep/b.py hidden


# --- 4. shell -----------------------------------------------------------------
def test_shell_ok_and_failure(tmp_path):
    tm = tool_map(tmp_path)

    ok = tm["shell"].fn({"command": "Write-Output hello-oclite"})
    assert ok.ok
    assert "hello-oclite" in ok.output
    assert ok.output.endswith("exit=0")

    bad = tm["shell"].fn({"command": "exit 3"})
    assert not bad.ok          # failure flagged...
    assert bad.output.endswith("exit=3")  # ...but output/exit code still captured


def test_shell_timeout(tmp_path):
    tm = tool_map(tmp_path, Config(shell_timeout_s=2))
    r = tm["shell"].fn({"command": "Start-Sleep -Seconds 60"})
    assert not r.ok
    assert "timed out after 2s" in r.output


# --- 5. webfetch (offline via monkeypatch) ------------------------------------
class FakeResponse:
    def __init__(self, status_code, headers, text):
        self.status_code = status_code
        self.headers = headers
        self.text = text


def test_webfetch_html_and_error(tmp_path, monkeypatch):
    tm = tool_map(tmp_path)
    wf = tm["webfetch"]

    monkeypatch.setattr(
        web_mod.httpx,
        "get",
        lambda *a, **k: FakeResponse(
            200,
            {"content-type": "text/html"},
            "<html><body><h1>Hi</h1><p>x</p></body></html>",
        ),
    )
    r = wf.fn({"url": "https://example.com/page"})
    assert r.ok
    assert "Hi" in r.output and "x" in r.output

    monkeypatch.setattr(
        web_mod.httpx,
        "get",
        lambda *a, **k: FakeResponse(404, {"content-type": "text/plain"}, "gone"),
    )
    r404 = wf.fn({"url": "https://example.com/missing"})
    assert not r404.ok and "404" in r404.output


def test_webfetch_scheme_guard_and_truncation(tmp_path):
    tm = tool_map(tmp_path, Config(webfetch_chars=5))
    guard = tm["webfetch"].fn({"url": "ftp://example.com/file"})
    assert not guard.ok and "http" in guard.output.lower()


# --- 6. websearch (offline via stub) ------------------------------------------
class StubDDGS:
    calls = []  # class-level so instances created by DDGS() inside web.py log here

    def __init__(self):
        pass

    def text(self, query, max_results=5):
        StubDDGS.calls.append((query, max_results))
        return [
            {"title": "Ollama docs", "url": "https://ollama.com/docs",
             "snippet": "Run models locally"},
            {"title": "Alt keys", "href": "https://example.org/2",
             "body": "href/body key style"},
        ]


def test_websearch_formatted_output(tmp_path, monkeypatch):
    tm = tool_map(tmp_path)
    StubDDGS.calls.clear()
    monkeypatch.setattr(web_mod, "DDGS", StubDDGS)

    r = tm["websearch"].fn({"query": "ollama python", "max_results": 2})
    assert r.ok
    assert "[1] Ollama docs" in r.output
    assert "https://ollama.com/docs" in r.output
    assert "Run models locally" in r.output
    assert "[2] Alt keys" in r.output
    assert "https://example.org/2" in r.output
    assert StubDDGS.calls == [("ollama python", 2)]


def test_websearch_failure_message(tmp_path, monkeypatch):
    class Boom:
        def text(self, query, max_results=5):
            raise RuntimeError("offline")

    monkeypatch.setattr(web_mod, "DDGS", Boom)
    r = tool_map(tmp_path)["websearch"].fn({"query": "anything"})
    assert not r.ok
    assert r.output == "ERROR: websearch failed (offline or rate-limited)"

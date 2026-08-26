"""Tests for config, session, tools, and agent."""

import json
import pathlib
import tempfile

from assistant.config import Config, Limits, Permissions, load_config
from assistant.session import Session
from assistant.tools import get_tools, openai_schema, Tool, ToolResult


# --- Config ---

def test_config_defaults():
    cfg = Config()
    assert cfg.model == "qwen2.5-coder-3b-abliterated"
    assert cfg.max_rounds == 12
    assert cfg.permissions.shell == "ask"
    assert cfg.limits.read_max_lines == 200


def test_config_load_missing_file():
    cfg = load_config(pathlib.Path("/nonexistent/config.json"))
    assert cfg.model == "qwen2.5-coder-3b-abliterated"


def test_config_load_valid_json(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"model": "test-model", "max_rounds": 5}))
    cfg = load_config(config_file)
    assert cfg.model == "test-model"
    assert cfg.max_rounds == 5


def test_config_invalid_permission(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"permissions": {"shell": "invalid"}}))
    try:
        load_config(config_file)
        assert False, "should have raised"
    except ValueError as e:
        assert "permissions" in str(e)


# --- Session ---

def test_session_save_load(tmp_path):
    path = tmp_path / "session.json"
    s = Session(path)
    s.append("user", "hello")
    s.append("assistant", "hi")
    s.save()

    s2 = Session.load(path)
    assert len(s2.messages) == 2
    assert s2.messages[0]["role"] == "user"
    assert s2.messages[1]["content"] == "hi"


def test_session_load_missing():
    s = Session.load(pathlib.Path("/nonexistent/session.json"))
    assert s.messages == []


def test_session_clear(tmp_path):
    path = tmp_path / "session.json"
    s = Session(path)
    s.append("user", "hello")
    s.clear()
    assert s.messages == []


# --- Tools ---

def test_get_tools():
    cfg = Config()
    tools = get_tools(cfg.workspace, cfg)
    names = [t.name for t in tools]
    assert "read_file" in names
    assert "write_file" in names
    assert "list_files" in names
    assert "delete_file" in names
    assert "shell" in names
    assert "get_current_time" in names
    assert "webfetch" in names
    assert "websearch" in names
    assert len(tools) == 8


def test_openai_schema():
    cfg = Config()
    tools = get_tools(cfg.workspace, cfg)
    schema = openai_schema(tools)
    assert len(schema) == 8
    assert schema[0]["type"] == "function"
    assert "name" in schema[0]["function"]
    assert "parameters" in schema[0]["function"]


def test_tool_result():
    r = ToolResult(True, "ok")
    assert r.ok is True
    assert r.output == "ok"


# --- Tool execution ---

def test_time_tool():
    cfg = Config()
    tools = get_tools(cfg.workspace, cfg)
    time_tool = next(t for t in tools if t.name == "get_current_time")
    result = time_tool.fn({})
    assert result.ok is True
    data = json.loads(result.output)
    assert "iso" in data
    assert "year" in data


def test_list_files_tool():
    cfg = Config()
    tools = get_tools(cfg.workspace, cfg)
    list_tool = next(t for t in tools if t.name == "list_files")
    result = list_tool.fn({"path": "."})
    assert result.ok is True
    assert "src" in result.output


def test_write_read_file(tmp_path):
    cfg = Config(workspace=tmp_path)
    tools = get_tools(tmp_path, cfg)

    write_tool = next(t for t in tools if t.name == "write_file")
    result = write_tool.fn({"path": "test.txt", "content": "hello world"})
    assert result.ok is True

    read_tool = next(t for t in tools if t.name == "read_file")
    result = read_tool.fn({"path": "test.txt"})
    assert result.ok is True
    assert "hello world" in result.output


def test_delete_file(tmp_path):
    cfg = Config(workspace=tmp_path)
    tools = get_tools(tmp_path, cfg)

    write_tool = next(t for t in tools if t.name == "write_file")
    write_tool.fn({"path": "to_delete.txt", "content": "delete me"})

    delete_tool = next(t for t in tools if t.name == "delete_file")
    result = delete_tool.fn({"path": "to_delete.txt"})
    assert result.ok is True

    read_tool = next(t for t in tools if t.name == "read_file")
    result = read_tool.fn({"path": "to_delete.txt"})
    assert result.ok is False

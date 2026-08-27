"""Comprehensive tests for assistant.tools (time, web, shell, fs)."""

import json
import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from assistant.config import Config
from assistant.tools import get_tools, ToolResult
from assistant.tools.shell import _resolve_shell_cmd, get_default_shell_cmd
from assistant.tools.time import time_tool
from assistant.tools.web import webfetch_tool, websearch_tool
from assistant.tools.fs import read_file_tool, write_file_tool, delete_file_tool, list_files_tool


# --- Time tool tests ---

def test_time_tool_complete_fields():
    cfg = Config()
    tool = time_tool(cfg.workspace, cfg)
    res = tool.fn({})
    assert res.ok is True
    data = json.loads(res.output)
    expected_keys = {"iso", "utc", "local", "year", "month", "day", "weekday", "tz_name"}
    assert expected_keys.issubset(data.keys())
    assert isinstance(data["year"], int)
    assert isinstance(data["month"], int)
    assert isinstance(data["day"], int)
    assert isinstance(data["weekday"], str)
    assert "get_current_time" in tool.description
    assert "websearch" in tool.description or "webfetch" in tool.description


# --- Web tool tests ---

def test_webfetch_date_context_header(tmp_path):
    cfg = Config(workspace=tmp_path)
    tool = webfetch_tool(tmp_path, cfg)
    assert "get_current_time" in tool.description

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/plain"}
    mock_resp.iter_text.return_value = ["Hello web world"]

    with patch("assistant.tools.web._hop_blocked_reason", return_value=None), \
         patch("httpx.stream", return_value=mock_resp):
        mock_resp.__enter__.return_value = mock_resp
        res = tool.fn({"url": "https://example.com"})
        assert res.ok is True
        assert res.output.startswith("[context: current datetime")
        assert "prefer sources/results dated closest to this date]" in res.output
        assert "Hello web world" in res.output


def test_websearch_date_context_header_and_recency(tmp_path):
    cfg = Config(workspace=tmp_path)
    tool = websearch_tool(tmp_path, cfg)
    assert "get_current_time" in tool.description

    mock_results = [
        {"title": "Example Result", "url": "https://example.com", "snippet": "Example snippet"}
    ]

    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = mock_results

    with patch("assistant.tools.web.DDGS", return_value=mock_ddgs_instance):
        res = tool.fn({"query": "python", "recency": "week"})
        assert res.ok is True
        assert res.output.startswith("[context: current datetime")
        assert "prefer sources/results dated closest to this date]" in res.output
        assert "Example Result" in res.output

        # Verify invalid recency
        invalid_res = tool.fn({"query": "python", "recency": "century"})
        assert invalid_res.ok is False
        assert "recency must be one of" in invalid_res.output


def test_websearch_missing_package_graceful(tmp_path):
    cfg = Config(workspace=tmp_path)
    tool = websearch_tool(tmp_path, cfg)

    with patch("assistant.tools.web.DDGS", None):
        res = tool.fn({"query": "python"})
        assert res.ok is False
        assert "duckduckgo search package is not installed" in res.output


# --- Shell tool tests ---

def test_shell_tool_platform_resolution(tmp_path):
    cfg = Config(workspace=tmp_path)
    default_cmd = get_default_shell_cmd()
    assert isinstance(default_cmd, list)
    assert len(default_cmd) >= 1

    # Custom shell_cmd list
    custom_cfg = Config(workspace=tmp_path, shell_cmd=["python", "-c"])
    resolved = _resolve_shell_cmd(custom_cfg)
    assert resolved == ["python", "-c"]


def test_shell_tool_execution(tmp_path):
    cfg = Config(workspace=tmp_path)
    tools = get_tools(tmp_path, cfg)
    shell_t = next(t for t in tools if t.name == "shell")

    # Run a simple echo command
    res = shell_t.fn({"command": 'echo "hello_shell"'})
    assert res.ok is True
    assert "hello_shell" in res.output
    assert "exit=0" in res.output


# --- FS tool tests ---

def test_fs_read_robust_start_line_and_errors(tmp_path):
    cfg = Config(workspace=tmp_path)
    f = tmp_path / "sample.txt"
    f.write_text("line1\nline2\nline3\n", encoding="utf-8")

    r_tool = read_file_tool(tmp_path, cfg)

    # Normal read with quotes/whitespace
    res = r_tool.fn({"path": "  'sample.txt'  ", "start_line": 2})
    assert res.ok is True
    assert "L2-L3 of 3 lines" in res.output
    assert "line2" in res.output

    # start_line beyond EOF
    res_eof = r_tool.fn({"path": "sample.txt", "start_line": 10})
    assert res_eof.ok is False
    assert "beyond end of file" in res_eof.output

    # Read non-existent
    res_missing = r_tool.fn({"path": "does_not_exist.txt"})
    assert res_missing.ok is False
    assert "not found" in res_missing.output


def test_fs_write_and_delete_protections(tmp_path):
    cfg = Config(workspace=tmp_path)
    w_tool = write_file_tool(tmp_path, cfg)
    d_tool = delete_file_tool(tmp_path, cfg)

    # Subdirectory write
    res = w_tool.fn({"path": "nested/dir/file.txt", "content": "deep content"})
    assert res.ok is True
    assert (tmp_path / "nested" / "dir" / "file.txt").exists()

    # Attempt to write to existing directory
    res_dir = w_tool.fn({"path": "nested/dir", "content": "fails"})
    assert res_dir.ok is False
    assert "existing directory" in res_dir.output

    # Attempt to delete directory via delete_file
    res_del_dir = d_tool.fn({"path": "nested/dir"})
    assert res_del_dir.ok is False
    assert "is a directory" in res_del_dir.output


def test_fs_list_files_filtering_and_errors(tmp_path):
    cfg = Config(workspace=tmp_path)
    l_tool = list_files_tool(tmp_path, cfg)

    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("c")

    # Filter pattern
    res = l_tool.fn({"path": ".", "pattern": "**/*.py"})
    assert res.ok is True
    assert "a.py" in res.output
    assert "sub/c.py" in res.output
    assert "b.txt" not in res.output

    # List non-existent directory
    res_err = l_tool.fn({"path": "non_existent_folder"})
    assert res_err.ok is False
    assert "not found" in res_err.output


def test_view_image_tool(tmp_path):
    from assistant.tools.fs import view_image_tool

    cfg = Config(workspace=tmp_path)
    v_tool = view_image_tool(tmp_path, cfg)

    # Valid PNG image mock
    img_file = tmp_path / "photo.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtest_binary_data")

    res = v_tool.fn({"path": "photo.png"})
    assert res.ok is True
    assert "photo.png" in res.output
    assert "data:image_base64;image/png;" in res.output

    # Unsupported extension
    txt_file = tmp_path / "note.pdf"
    txt_file.write_text("dummy")
    res_unsupp = v_tool.fn({"path": "note.pdf"})
    assert res_unsupp.ok is False
    assert "unsupported image extension" in res_unsupp.output

    # Non-existent file
    res_not_found = v_tool.fn({"path": "missing.jpg"})
    assert res_not_found.ok is False
    assert "not found" in res_not_found.output

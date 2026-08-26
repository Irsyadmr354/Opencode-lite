"""Tests for assistant.agent: multi-round loop, date refresh, permissions, reasoning, and tools."""

import io
import json
import pathlib
import sys
import unittest.mock as mock
from datetime import datetime, timezone
import pytest

from assistant.agent import (
    CHAR_DELAY,
    Agent,
    ThinkingSpinner,
    _char_by_char,
    _get_live_datetime_str,
    build_system_prompt,
)
from assistant.config import Config, Limits, Permissions
from assistant.tools import Tool, ToolResult


# --- 1. System Prompt & Live Datetime Tests ---


def test_get_live_datetime_str():
    dt_str = _get_live_datetime_str()
    assert str(datetime.now().year) in dt_str


def test_build_system_prompt_date_and_workspace():
    # Default tools
    prompt = build_system_prompt("/test/workspace")
    assert prompt.startswith("Be concise.")
    assert "/test/workspace" in prompt
    assert "read_file" in prompt
    assert "shell" in prompt
    assert "get_current_time" in prompt
    assert "websearch" in prompt
    assert "webfetch" in prompt
    assert "Use tools for actions, not raw text." in prompt
    assert "Before websearch/webfetch, call get_current_time." in prompt

    # Dynamic custom tools
    custom_prompt = build_system_prompt("/custom", ["tool_a", "tool_b"])
    assert "Tools: tool_a, tool_b" in custom_prompt


def test_system_prompt_refreshed_on_every_turn(tmp_path):
    cfg = Config(workspace=tmp_path)
    agent = Agent(cfg)

    # Initial prompt
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert str(tmp_path) in agent.messages[0]["content"]

    # Workspace change is refreshed on turn
    new_ws = tmp_path / "new_workspace"
    agent.workspace = new_ws
    with mock.patch.object(agent.llm, "chat", return_value={"content": "Hello!", "tool_calls": []}):
        agent.handle("hi")

    assert str(new_ws) in agent.messages[0]["content"]


def test_clear_context_resets_history_and_refreshes_prompt(tmp_path):
    cfg = Config(workspace=tmp_path)
    agent = Agent(cfg)

    agent.messages.append({"role": "user", "content": "test"})
    agent.messages.append({"role": "assistant", "content": "response"})
    agent._recent_calls.append("call1")

    agent.clear_context()

    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"
    assert str(tmp_path) in agent.messages[0]["content"]
    assert len(agent._recent_calls) == 0


# --- 2. Date Requirement in Prompt and Tool Specs ---


def test_date_requirement_in_prompt_and_all_tools(tmp_path):
    cfg = Config(workspace=tmp_path)
    agent = Agent(cfg)

    # Rule in system prompt
    sys_prompt = agent.messages[0]["content"]
    assert "get_current_time" in sys_prompt
    assert "websearch" in sys_prompt or "webfetch" in sys_prompt

    # Tool descriptions
    websearch = agent.tool_map["websearch"]
    webfetch = agent.tool_map["webfetch"]
    get_time = agent.tool_map["get_current_time"]

    assert "get_current_time" in websearch.description
    assert "get_current_time" in webfetch.description
    assert "websearch" in get_time.description or "webfetch" in get_time.description


# --- 3. Multi-Round Agent Execution Tests ---


def test_multi_round_execution_tool_call_then_final_response(tmp_path):
    cfg = Config(workspace=tmp_path, stream=False)
    agent = Agent(cfg)

    # Create a test file
    test_file = tmp_path / "hello.txt"
    test_file.write_text("file content 42", encoding="utf-8")

    # Round 1: LLM decides to call read_file
    # Round 2: LLM receives tool output and responds with final answer
    llm_responses = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read_1",
                    "name": "read_file",
                    "arguments": {"path": "hello.txt"},
                }
            ],
            "finish_reason": "tool_calls",
            "thinking": "Need to read file first",
        },
        {
            "content": "The file contains: file content 42",
            "tool_calls": [],
            "finish_reason": "stop",
            "thinking": "Now I can answer",
        },
    ]

    with mock.patch.object(agent.llm, "chat", side_effect=llm_responses) as mock_chat:
        resp = agent.handle("Read hello.txt")

    assert mock_chat.call_count == 2
    assert resp == "The file contains: file content 42"

    # Verify conversation history structure
    # 0: system, 1: user, 2: assistant (tool call), 3: tool result, 4: assistant (final)
    assert len(agent.messages) == 5
    assert agent.messages[1]["role"] == "user"
    assert agent.messages[1]["content"] == "Read hello.txt"

    assert agent.messages[2]["role"] == "assistant"
    assert agent.messages[2]["tool_calls"][0]["function"]["name"] == "read_file"

    assert agent.messages[3]["role"] == "tool"
    assert "file content 42" in agent.messages[3]["content"]
    assert agent.messages[3]["tool_call_id"] == "call_read_1"

    assert agent.messages[4]["role"] == "assistant"
    assert agent.messages[4]["content"] == "The file contains: file content 42"


def test_multi_round_execution_multiple_tools(tmp_path):
    cfg = Config(workspace=tmp_path, stream=False)
    agent = Agent(cfg)

    llm_responses = [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_time_1",
                    "name": "get_current_time",
                    "arguments": {},
                }
            ],
            "finish_reason": "tool_calls",
        },
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_write_1",
                    "name": "write_file",
                    "arguments": {"path": "log.txt", "content": "logged"},
                }
            ],
            "finish_reason": "tool_calls",
        },
        {
            "content": "Successfully checked time and wrote log.",
            "tool_calls": [],
            "finish_reason": "stop",
        },
    ]

    with mock.patch.object(agent.llm, "chat", side_effect=llm_responses) as mock_chat:
        resp = agent.handle("Get time and log it")

    assert mock_chat.call_count == 3
    assert resp == "Successfully checked time and wrote log."
    assert (tmp_path / "log.txt").read_text(encoding="utf-8") == "logged"


def test_max_rounds_limit(tmp_path):
    cfg = Config(workspace=tmp_path, max_rounds=2, stream=False)
    agent = Agent(cfg)

    infinite_tool_calls = {
        "content": "still working...",
        "tool_calls": [{"id": "call_inf", "name": "get_current_time", "arguments": {}}],
        "finish_reason": "tool_calls",
    }

    with mock.patch.object(agent.llm, "chat", return_value=infinite_tool_calls) as mock_chat:
        resp = agent.handle("Infinite loop test")

    assert mock_chat.call_count == 2
    assert resp == "still working..."


def test_consecutive_identical_tool_call_loop_breaker(tmp_path):
    cfg = Config(workspace=tmp_path, stream=False)
    agent = Agent(cfg)

    identical_call = {
        "content": "",
        "tool_calls": [{"id": "call_same", "name": "get_current_time", "arguments": {}}],
        "finish_reason": "tool_calls",
    }
    final_resp = {
        "content": "Stopped loop.",
        "tool_calls": [],
        "finish_reason": "stop",
    }

    with mock.patch.object(agent.llm, "chat", side_effect=[identical_call, identical_call, identical_call, final_resp]):
        resp = agent.handle("Repeat test")

    assert resp == "Stopped loop."
    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert any("Loop breaker" in m["content"] or "3 times in a row" in m["content"] for m in tool_messages)


def test_unknown_tool_handling(tmp_path):
    cfg = Config(workspace=tmp_path, stream=False)
    agent = Agent(cfg)

    unknown_call = {
        "content": "",
        "tool_calls": [{"id": "call_unk", "name": "non_existent_tool", "arguments": {"foo": "bar"}}],
        "finish_reason": "tool_calls",
    }
    final_resp = {
        "content": "Handled unknown tool.",
        "tool_calls": [],
        "finish_reason": "stop",
    }

    with mock.patch.object(agent.llm, "chat", side_effect=[unknown_call, final_resp]):
        resp = agent.handle("Run unknown")

    assert resp == "Handled unknown tool."
    tool_msg = next(m for m in agent.messages if m.get("role") == "tool")
    assert "Unknown tool" in tool_msg["content"]


# --- 4. Permission Gating in Agent Loop ---


def test_permission_allow_executes_silently(tmp_path):
    cfg = Config(workspace=tmp_path, permissions=Permissions(shell="allow"), stream=False)
    agent = Agent(cfg)

    call = {
        "content": "",
        "tool_calls": [{"id": "call_sh", "name": "shell", "arguments": {"command": "echo allow_test"}}],
    }
    final_resp = {"content": "Done shell.", "tool_calls": []}

    with mock.patch.object(agent.llm, "chat", side_effect=[call, final_resp]):
        resp = agent.handle("Run shell")

    assert resp == "Done shell."
    tool_msg = next(m for m in agent.messages if m.get("role") == "tool")
    assert "allow_test" in tool_msg["content"]


def test_permission_deny_blocks_execution(tmp_path):
    cfg = Config(workspace=tmp_path, permissions=Permissions(shell="deny"), stream=False)
    agent = Agent(cfg)

    call = {
        "content": "",
        "tool_calls": [{"id": "call_sh", "name": "shell", "arguments": {"command": "echo should_not_run"}}],
    }
    final_resp = {"content": "Understood denial.", "tool_calls": []}

    with mock.patch.object(agent.llm, "chat", side_effect=[call, final_resp]):
        resp = agent.handle("Run shell")

    assert resp == "Understood denial."
    tool_msg = next(m for m in agent.messages if m.get("role") == "tool")
    assert "Permission denied for 'shell'" in tool_msg["content"]


def test_permission_ask_user_accepts(tmp_path):
    cfg = Config(workspace=tmp_path, permissions=Permissions(delete="ask"), stream=False)
    agent = Agent(cfg)

    (tmp_path / "target.txt").write_text("delete this")

    call = {
        "content": "",
        "tool_calls": [{"id": "call_del", "name": "delete_file", "arguments": {"path": "target.txt"}}],
    }
    final_resp = {"content": "Deleted file.", "tool_calls": []}

    with mock.patch("builtins.input", return_value="y"), \
         mock.patch.object(agent.llm, "chat", side_effect=[call, final_resp]):
        resp = agent.handle("Delete target.txt")

    assert resp == "Deleted file."
    assert not (tmp_path / "target.txt").exists()


def test_permission_ask_user_declines(tmp_path):
    cfg = Config(workspace=tmp_path, permissions=Permissions(delete="ask"), stream=False)
    agent = Agent(cfg)

    (tmp_path / "target.txt").write_text("keep this")

    call = {
        "content": "",
        "tool_calls": [{"id": "call_del", "name": "delete_file", "arguments": {"path": "target.txt"}}],
    }
    final_resp = {"content": "Cancelled delete.", "tool_calls": []}

    with mock.patch("builtins.input", return_value="n"), \
         mock.patch.object(agent.llm, "chat", side_effect=[call, final_resp]):
        resp = agent.handle("Delete target.txt")

    assert resp == "Cancelled delete."
    assert (tmp_path / "target.txt").exists()


# --- 5. Streaming & Reasoning Display ---


def test_agent_streaming_mode(tmp_path):
    cfg = Config(workspace=tmp_path, stream=True)
    agent = Agent(cfg)

    def mock_chat_stream(messages, tools=None, on_delta=None):
        if on_delta:
            on_delta("Thinking step 1... ", True)
            on_delta("Here is answer.", False)
        return {
            "content": "Here is answer.",
            "tool_calls": [],
            "finish_reason": "stop",
            "thinking": "Thinking step 1... ",
        }

    with mock.patch.object(agent.llm, "chat", side_effect=mock_chat_stream):
        resp = agent.handle("What is 1+1?")

    assert resp == "Here is answer."


def test_agent_thinking_flow_reasoning_and_completion(tmp_path):
    cfg = Config(workspace=tmp_path, stream=True)
    agent = Agent(cfg)

    def mock_stream(messages, tools=None, on_delta=None):
        if on_delta:
            on_delta("Analyzing query deeply...", True)
            on_delta("The computed result is 42.", False)
        return {
            "content": "The computed result is 42.",
            "tool_calls": [],
            "finish_reason": "stop",
            "thinking": "Analyzing query deeply...",
        }

    fake_out = io.StringIO()
    with mock.patch("sys.stdout", fake_out), \
         mock.patch.object(agent.llm, "chat", side_effect=mock_stream):
        resp = agent.handle("Compute answer")

    output = fake_out.getvalue()
    assert resp == "The computed result is 42."
    assert "Thinking:" in output
    assert "Analyzing query deeply..." in output
    assert "✔" in output and "Thinking" in output
    assert "Assistant:" in output
    assert "The computed result is 42." in output


def test_agent_non_reasoning_direct_response(tmp_path):
    cfg = Config(workspace=tmp_path, stream=True)
    agent = Agent(cfg)

    def mock_stream(messages, tools=None, on_delta=None):
        if on_delta:
            on_delta("Direct greeting without thinking.", False)
        return {
            "content": "Direct greeting without thinking.",
            "tool_calls": [],
            "finish_reason": "stop",
            "thinking": "",
        }

    fake_out = io.StringIO()
    with mock.patch("sys.stdout", fake_out), \
         mock.patch.object(agent.llm, "chat", side_effect=mock_stream):
        resp = agent.handle("hello")

    output = fake_out.getvalue()
    assert resp == "Direct greeting without thinking."
    assert "Thinking:" not in output
    assert "✔" not in output
    assert "Assistant:" in output
    assert "Direct greeting without thinking." in output


def test_agent_multi_round_tool_turns_with_thinking(tmp_path):
    cfg = Config(workspace=tmp_path, stream=True)
    agent = Agent(cfg)

    def mock_round_1(messages, tools=None, on_delta=None):
        if on_delta:
            on_delta("Checking time tool first...", True)
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_time_r1",
                    "name": "get_current_time",
                    "arguments": {},
                }
            ],
            "finish_reason": "tool_calls",
            "thinking": "Checking time tool first...",
        }

    def mock_round_2(messages, tools=None, on_delta=None):
        if on_delta:
            on_delta("Synthesizing current time...", True)
            on_delta("The current date is available.", False)
        return {
            "content": "The current date is available.",
            "tool_calls": [],
            "finish_reason": "stop",
            "thinking": "Synthesizing current time...",
        }

    call_count = 0

    def mock_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return mock_round_1(*args, **kwargs)
        return mock_round_2(*args, **kwargs)

    fake_out = io.StringIO()
    with mock.patch("sys.stdout", fake_out), \
         mock.patch.object(agent.llm, "chat", side_effect=mock_chat):
        resp = agent.handle("What is the date?")

    output = fake_out.getvalue()
    assert resp == "The current date is available."
    # Both thinking rounds should have executed
    assert "Thinking: Checking time tool first..." in output
    assert "Thinking: Synthesizing current time..." in output
    assert "get_current_time" in output
    assert "Assistant:" in output
    assert "The current date is available." in output


def test_thinking_spinner_lifecycle():
    spinner = ThinkingSpinner(prompt_prefix="")
    spinner.start()
    assert spinner._started is True
    spinner.stop()
    assert spinner._stopped is True

    # Double start/stop safe
    spinner.start()
    spinner.stop()


def test_char_by_char_helper():
    out = io.StringIO()
    with mock.patch("sys.stdout", out):
        _char_by_char("hi", delay=0.001)
    assert out.getvalue() == "hi"


def test_agent_non_streaming_thinking_lifecycle(tmp_path):
    cfg = Config(workspace=tmp_path, stream=False)
    agent = Agent(cfg)

    mock_resp = {
        "content": "Direct 42 answer.",
        "tool_calls": [],
        "finish_reason": "stop",
        "thinking": "Deep thinking in non-streaming mode.",
    }

    fake_out = io.StringIO()
    with mock.patch.object(agent.llm, "chat", return_value=mock_resp), \
         mock.patch("sys.stdout", fake_out):
        resp = agent.handle("What is the meaning of life?")

    assert resp == "Direct 42 answer."
    output = fake_out.getvalue()
    assert "Thinking: Deep thinking in non-streaming mode." in output
    assert "Thought in" in output
    assert "Assistant:" in output
    assert "Direct 42 answer." in output



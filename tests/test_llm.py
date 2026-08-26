"""Comprehensive tests for assistant.llm."""

import io
import json
import threading
import time
import unittest.mock as mock
import httpx
import pytest

from assistant.config import Config
from assistant.llm import (
    BOLD,
    CYAN,
    DARK_GRAY,
    DIM,
    GREEN,
    LLM,
    RED,
    RESET,
    YELLOW,
    StreamReasoningParser,
    TypewriterStreamer,
    _extract_thinking_and_clean_content,
    _format_http_error,
    _is_tool_call_dict,
    _normalize_tool_call,
    _parse_tool_calls_from_text,
    extract_tool_calls_from_text,
    safe_print,
    thinking_spinner,
    typewriter,
)


class MockStream:
    """Mock stream recording writes and timestamps."""

    def __init__(self):
        self.writes: list[str] = []

    def write(self, s: str):
        self.writes.append(s)

    def flush(self):
        pass

    def getvalue(self) -> str:
        return "".join(self.writes)


# --- 1. StreamReasoningParser & Reasoning Streaming Tests ---


def test_stream_reasoning_parser_basic():
    parser = StreamReasoningParser()
    chunks = ["Hello ", "<think>this is thought</think>", " world!"]
    results = []
    for c in chunks:
        results.extend(parser.feed(c))
    results.extend(parser.flush())

    assert results == [
        ("Hello ", False),
        ("this is thought", True),
        (" world!", False),
    ]


def test_stream_reasoning_parser_split_tags():
    parser = StreamReasoningParser()
    chunks = [
        "Pre-text <th",
        "ink>First part of thought ",
        "second part of thought</th",
        "ink> Post-text",
    ]
    results = []
    for c in chunks:
        results.extend(parser.feed(c))
    results.extend(parser.flush())

    assert results == [
        ("Pre-text ", False),
        ("First part of thought ", True),
        ("second part of thought", True),
        (" Post-text", False),
    ]


def test_stream_reasoning_parser_multiple_think_blocks():
    parser = StreamReasoningParser()
    chunks = [
        "<think>thought 1</think>answer 1",
        "<think>thought 2</think>answer 2",
    ]
    results = []
    for c in chunks:
        results.extend(parser.feed(c))
    results.extend(parser.flush())

    assert results == [
        ("thought 1", True),
        ("answer 1", False),
        ("thought 2", True),
        ("answer 2", False),
    ]


def test_stream_reasoning_parser_unclosed_tag():
    parser = StreamReasoningParser()
    results = parser.feed("<think>unclosed thought")
    results.extend(parser.flush())

    assert results == [("unclosed thought", True)]


def test_stream_reasoning_parser_non_think_brackets():
    parser = StreamReasoningParser()
    results = []
    results.extend(parser.feed("5 < 10 and 2 < 3"))
    results.extend(parser.flush())

    assert results == [("5 < 10 and 2 < 3", False)]


def test_extract_thinking_and_clean_content():
    text = "Intro <think>reasoning step 1\nstep 2</think> Answer text <think>step 3</think> Final."
    cleaned, thinking = _extract_thinking_and_clean_content(text, "initial reasoning")

    assert "initial reasoning" in thinking
    assert "reasoning step 1" in thinking
    assert "step 3" in thinking
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "Intro" in cleaned
    assert "Answer text" in cleaned
    assert "Final." in cleaned


# --- 2. Typewriter & ANSI Streaming Tests ---


def test_typewriter_ansi_immediate_write():
    mock_out = MockStream()
    # ANSI escape code \033[36m should be written atomically
    typewriter("A \033[36mB\033[0m C", delay=0.001, stream=mock_out)

    writes = mock_out.writes
    assert "\033[36m" in writes
    assert "\033[0m" in writes
    assert "A" in writes
    assert "B" in writes
    assert "C" in writes


def test_typewriter_thinking_dim_color():
    mock_out = MockStream()
    typewriter("Reasoning...", delay=0.001, is_thinking=True, stream=mock_out)

    writes = mock_out.writes
    assert writes[0] == DARK_GRAY
    assert writes[-1] == RESET


def test_typewriter_delay_capping():
    mock_out = MockStream()
    start = time.perf_counter()
    typewriter("Hi", delay=0.5, stream=mock_out)
    duration = time.perf_counter() - start
    assert duration < 0.4


def test_typewriter_streamer_transitions():
    mock_out = MockStream()
    streamer = TypewriterStreamer(delay=0.001, stream=mock_out)

    # Thinking token
    streamer.on_delta("Thinking 1...", is_thinking=True)
    # Transition to normal response
    streamer.on_delta("Here is response.", is_thinking=False)
    streamer.close()

    val = mock_out.getvalue()
    assert DARK_GRAY in val
    assert RESET in val
    assert "Thinking 1..." in val
    assert "Here is response." in val


def test_safe_print_and_spinner():
    safe_print("test safe print \u2713")

    stop_event = threading.Event()
    stop_event.set()
    thinking_spinner(stop_event, "Testing")


# --- 3. Robust Tool Call Parsing Tests ---


def test_parse_tool_calls_markdown_fence():
    text = (
        "I will read the file.\n"
        "```json\n"
        "{\n"
        '  "name": "read_file",\n'
        '  "arguments": {"path": "src/assistant/llm.py", "nested": {"a": [1, 2]}}\n'
        "}\n"
        "```\n"
        "Let me know if you need anything else."
    )
    tool_calls, cleaned = extract_tool_calls_from_text(text)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "read_file"
    assert tool_calls[0]["arguments"]["path"] == "src/assistant/llm.py"
    assert tool_calls[0]["arguments"]["nested"]["a"] == [1, 2]
    assert "I will read the file." in cleaned
    assert "Let me know if you need anything else." in cleaned
    assert "```" not in cleaned


def test_parse_tool_calls_multiple_in_array():
    text = (
        "Executing both tools:\n"
        "```json\n"
        "[\n"
        '  {"name": "read_file", "arguments": {"path": "a.txt"}},\n'
        '  {"name": "write_file", "arguments": {"path": "b.txt", "content": "hello"}}\n'
        "]\n"
        "```"
    )
    tool_calls, cleaned = extract_tool_calls_from_text(text)

    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "read_file"
    assert tool_calls[0]["arguments"]["path"] == "a.txt"
    assert tool_calls[1]["name"] == "write_file"
    assert tool_calls[1]["arguments"]["content"] == "hello"
    assert cleaned == "Executing both tools:"


def test_parse_tool_calls_xml_tags():
    text = '<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>'
    tool_calls, cleaned = extract_tool_calls_from_text(text)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_current_time"
    assert tool_calls[0]["arguments"] == {}
    assert cleaned == ""


def test_parse_tool_calls_function_object_format():
    text = (
        '{"function": {"name": "shell", "arguments": "{\\"command\\": \\"ls -la\\"}"}}'
    )
    tool_calls, cleaned = extract_tool_calls_from_text(text)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "shell"
    assert tool_calls[0]["arguments"] == {"command": "ls -la"}
    assert cleaned == ""


def test_parse_tool_calls_tool_parameters_format():
    text = '{"tool": "websearch", "parameters": {"query": "python latest"}}'
    tool_calls, cleaned = extract_tool_calls_from_text(text)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "websearch"
    assert tool_calls[0]["arguments"] == {"query": "python latest"}
    assert cleaned == ""


def test_parse_tool_calls_non_tool_json_preserved():
    text = (
        'Here is the sample config: {"theme": "dark", "version": 2}.\n'
        'Now calling: {"name": "read_file", "arguments": {"path": "main.py"}}'
    )
    tool_calls, cleaned = extract_tool_calls_from_text(text)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "read_file"
    assert '{"theme": "dark", "version": 2}' in cleaned
    assert '{"name": "read_file"' not in cleaned


def test_legacy_parse_tool_calls_wrapper():
    text = '{"name": "delete_file", "arguments": {"path": "tmp.txt"}}'
    tcs = _parse_tool_calls_from_text(text)
    assert len(tcs) == 1
    assert tcs[0]["name"] == "delete_file"


# --- 4. LLM Chat Streaming, Reasoning & Tool Calling Integration ---


def test_llm_streaming_with_reasoning_content():
    cfg = Config()
    cfg.stream = True
    llm = LLM(cfg)

    sse_events = [
        'data: {"choices": [{"delta": {"reasoning_content": "Step 1: check files."}}]}\n\n',
        'data: {"choices": [{"delta": {"reasoning_content": " Step 2: ready."}}]}\n\n',
        'data: {"choices": [{"delta": {"content": "The result is "}}]}\n\n',
        'data: {"choices": [{"delta": {"content": "42."}}]}\n\n',
        "data: [DONE]\n\n",
    ]

    deltas_received = []

    def on_delta(token, is_thinking):
        deltas_received.append((token, is_thinking))

    with mock.patch("httpx.Client.stream") as mock_stream_call:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            line.strip() for line in "".join(sse_events).split("\n") if line.strip()
        ]
        mock_stream_call.return_value.__enter__.return_value = mock_resp

        result = llm.chat([{"role": "user", "content": "hi"}], on_delta=on_delta)

    assert result["content"] == "The result is 42."
    assert result["thinking"] == "Step 1: check files. Step 2: ready."
    assert ("Step 1: check files.", True) in deltas_received
    assert ("The result is ", False) in deltas_received


def test_llm_streaming_with_think_tags():
    cfg = Config()
    cfg.stream = True
    llm = LLM(cfg)

    sse_events = [
        'data: {"choices": [{"delta": {"content": "<think>Processing "}}]}\n\n',
        'data: {"choices": [{"delta": {"content": "request...</think>Here is "}}]}\n\n',
        'data: {"choices": [{"delta": {"content": "the answer."}}]}\n\n',
        "data: [DONE]\n\n",
    ]

    deltas_received = []

    def on_delta(token, is_thinking):
        deltas_received.append((token, is_thinking))

    with mock.patch("httpx.Client.stream") as mock_stream_call:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            line.strip() for line in "".join(sse_events).split("\n") if line.strip()
        ]
        mock_stream_call.return_value.__enter__.return_value = mock_resp

        result = llm.chat([{"role": "user", "content": "test"}], on_delta=on_delta)

    assert result["content"] == "Here is the answer."
    assert result["thinking"] == "Processing request..."
    assert ("Processing ", True) in deltas_received
    assert ("request...", True) in deltas_received
    assert ("Here is ", False) in deltas_received


def test_llm_legacy_on_delta_1_arg():
    cfg = Config()
    cfg.stream = True
    llm = LLM(cfg)

    sse_events = [
        'data: {"choices": [{"delta": {"content": "Hello!"}}]}\n\n',
        "data: [DONE]\n\n",
    ]

    tokens_received = []

    def on_delta(token):
        tokens_received.append(token)

    with mock.patch("httpx.Client.stream") as mock_stream_call:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            line.strip() for line in "".join(sse_events).split("\n") if line.strip()
        ]
        mock_stream_call.return_value.__enter__.return_value = mock_resp

        result = llm.chat([{"role": "user", "content": "hi"}], on_delta=on_delta)

    assert result["content"] == "Hello!"
    assert tokens_received == ["Hello!"]


def test_llm_non_streaming():
    cfg = Config()
    cfg.stream = False
    llm = LLM(cfg)

    response_json = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>My internal thought</think>Direct answer.",
                    "reasoning_content": "Deep reasoning",
                },
                "finish_reason": "stop",
            }
        ]
    }

    with mock.patch("httpx.Client.post") as mock_post:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response_json
        mock_post.return_value = mock_resp

        result = llm.chat([{"role": "user", "content": "hi"}])

    assert result["content"] == "Direct answer."
    assert "Deep reasoning" in result["thinking"]
    assert "My internal thought" in result["thinking"]


# --- 5. Error Handling & HTTP Status Tests ---


def test_format_http_error_known_statuses():
    err_json = json.dumps({"error": {"message": "Invalid API key provided."}}).encode("utf-8")
    msg = _format_http_error(401, err_json, "http://localhost:11434/v1", "test-model")
    assert "401" in msg
    assert "Unauthorized" in msg
    assert "Invalid API key" in msg

    err_404 = json.dumps({"error": "model 'nonexistent' not found"}).encode("utf-8")
    msg_404 = _format_http_error(404, err_404, "http://localhost:11434/v1", "nonexistent")
    assert "404" in msg_404
    assert "Not Found" in msg_404


def test_llm_http_error_response_streaming():
    cfg = Config()
    cfg.stream = True
    llm = LLM(cfg)

    with mock.patch("httpx.Client.stream") as mock_stream_call:
        mock_resp = mock.MagicMock()
        mock_resp.status_code = 404
        mock_resp.read.return_value = b'{"error": "model not found"}'
        mock_stream_call.return_value.__enter__.return_value = mock_resp

        result = llm.chat([{"role": "user", "content": "hi"}])

    assert result["finish_reason"] == "error"
    assert "404" in result["content"]
    assert "model not found" in result["content"]


def test_llm_timeout_exception():
    cfg = Config()
    cfg.stream = True
    llm = LLM(cfg)

    with mock.patch("httpx.Client.stream", side_effect=httpx.TimeoutException("timed out")):
        result = llm.chat([{"role": "user", "content": "hi"}])

    assert result["finish_reason"] == "error"
    assert "timed out" in result["content"]


def test_llm_connect_error():
    cfg = Config()
    cfg.stream = True
    llm = LLM(cfg)

    with mock.patch("httpx.Client.stream", side_effect=httpx.ConnectError("connection refused")):
        result = llm.chat([{"role": "user", "content": "hi"}])

    assert result["finish_reason"] == "error"
    assert "Cannot connect" in result["content"]


# --- 6. Exact Thinking Lifecycle & Unicode Fallback Tests ---


def test_typewriter_streamer_exact_thinking_lifecycle():
    mock_out = MockStream()
    streamer = TypewriterStreamer(delay=0.001, stream=mock_out)

    # 1. First token with is_thinking=True triggers header "  | Thinking: "
    streamer.on_delta("Step 1: calculate 2+2.", is_thinking=True)
    # 2. Second thinking token continues without duplicate header
    streamer.on_delta(" Step 2: result is 4.", is_thinking=True)
    # 3. Transition to content token triggers completion mark and assistant prompt
    streamer.on_delta("The final answer is 4.", is_thinking=False)
    # 4. Second content token continues normally
    streamer.on_delta(" Have a nice day!", is_thinking=False)
    streamer.close()

    out = mock_out.getvalue()
    # Check thinking header
    assert "Thinking: Step 1: calculate 2+2. Step 2: result is 4." in out
    # Check reasoning completion mark
    assert "Thought in" in out
    assert "✔" in out
    # Check assistant prompt
    assert "Assistant:" in out
    assert "The final answer is 4. Have a nice day!" in out


def test_typewriter_streamer_direct_content_no_reasoning():
    mock_out = MockStream()
    streamer = TypewriterStreamer(delay=0.001, stream=mock_out)

    streamer.on_delta("Hello directly!", is_thinking=False)
    streamer.close()

    out = mock_out.getvalue()
    assert "Thinking" not in out
    assert "Assistant:" in out
    assert "Hello directly!" in out


def test_typewriter_streamer_thinking_only_no_content():
    mock_out = MockStream()
    streamer = TypewriterStreamer(delay=0.001, stream=mock_out)

    streamer.on_delta("Only thinking about tools...", is_thinking=True)
    streamer.close()

    out = mock_out.getvalue()
    assert "Thinking: Only thinking about tools..." in out
    assert "✔" in out and "Thought in" in out
    assert "Assistant:" not in out


def test_typewriter_streamer_unicode_fallback():
    class FallbackStream:
        def __init__(self):
            self.writes = []

        def write(self, s: str):
            if "✔" in s:
                raise UnicodeEncodeError("charmap", s, 0, 1, "character maps to <undefined>")
            self.writes.append(s)

        def flush(self):
            pass

        def getvalue(self):
            return "".join(self.writes)

    stream = FallbackStream()
    streamer = TypewriterStreamer(delay=0.001, stream=stream)

    streamer.on_delta("Thinking step...", is_thinking=True)
    streamer.on_delta("Answer text", is_thinking=False)
    streamer.close()

    out = stream.getvalue()
    assert "v" in out
    assert "Thinking" in out
    assert "Assistant:" in out
    assert "Answer text" in out


def test_format_ollama_stats():
    from assistant.llm import format_ollama_stats

    # Empty stats
    assert format_ollama_stats(None) == ""
    assert format_ollama_stats({}) == ""

    # Complete stats
    stats = {
        "eval_count": 42,
        "eval_rate": 25.5,
        "total_duration_s": 1.65,
        "prompt_eval_count": 120,
        "prompt_eval_rate": 180.0,
    }
    formatted = format_ollama_stats(stats)
    assert "42 tokens" in formatted
    assert "25.5 tok/s" in formatted
    assert "1.65s" in formatted
    assert "120 tok" in formatted
    assert "180.0 tok/s" in formatted


"""Offline tests: agent loop, LLM streaming client, config loading."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fake_ollama import FakeOllama  # noqa: E402

from assistant.agent import SYSTEM_PROMPT, Agent, Hooks  # noqa: E402
from assistant.config import Config, load_config  # noqa: E402
from assistant.llm import LLMClient, LLMError  # noqa: E402


# --- helpers -----------------------------------------------------------------

class StubTool:
    """Duck-typed tool recording calls; returns ok/output like the real ones."""

    def __init__(self, name: str = "echo", danger: bool = False, raises: bool = False):
        self.name = name
        self.description = "stub tool for tests"
        self.parameters = {"type": "object", "properties": {}, "required": []}
        self.danger = danger
        self.raises = raises
        self.calls: list[dict] = []

    def fn(self, args: dict) -> dict:
        self.calls.append(dict(args))
        if self.raises:
            raise RuntimeError("boom")
        return {"ok": True, "output": "ran:" + json.dumps(args, sort_keys=True)}


class RecordingHooks(Hooks):
    def __init__(self, allow: bool = True):
        super().__init__()
        self.allow = allow
        self.deltas: list[str] = []
        self.turns: list = []
        self.statuses: list[dict] = []
        self.errors: list[str] = []
        self.permission_requests: list[tuple[str, dict]] = []
        self.cancel_after_done = 0  # >0: cancel agent after N assistant turns
        self.agent_ref: dict = {}

    def on_delta(self, text: str) -> None:
        self.deltas.append(text)

    def on_assistant_done(self, turn) -> None:
        self.turns.append(turn)
        if self.cancel_after_done and len(self.turns) >= self.cancel_after_done:
            self.agent_ref["agent"].cancelled = True

    def on_permission(self, name: str, args: dict) -> bool:
        self.permission_requests.append((name, dict(args)))
        return self.allow

    def on_status(self, info: dict) -> None:
        self.statuses.append(dict(info))

    def on_error(self, msg: str) -> None:
        self.errors.append(msg)


CONTENT_REPLY = {"content": "Hello from the fake model.", "finish_reason": "stop"}


def tool_call_reply(name: str, args: dict | None = None, call_id: str = "call_42") -> dict:
    return {
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})},
        }],
        "finish_reason": "tool_calls",
    }


@pytest.fixture
def harness():
    servers: list[FakeOllama] = []

    def make(script, tools=(), hooks=None, max_tool_rounds: int = 25) -> Agent:
        srv = FakeOllama(script)
        srv.start()
        servers.append(srv)
        client = LLMClient(base_url=srv.base_url, api_key="ollama",
                           model="fake-model", timeout_s=10)
        cfg = Config(max_tool_rounds=max_tool_rounds)
        return Agent(client=client, tools=list(tools), config=cfg, hooks=hooks)

    make.servers = servers  # expose for client-received payload inspection
    yield make
    for srv in servers:
        srv.shutdown()


# --- agent loop ---------------------------------------------------------------

def test_plain_reply(harness):
    hooks = RecordingHooks()
    agent = harness([CONTENT_REPLY], hooks=hooks)

    agent.submit("hi there")

    assert [m["role"] for m in agent.messages] == ["system", "user", "assistant"]
    assert agent.messages[1] == {"role": "user", "content": "hi there"}
    assert agent.messages[2]["content"] == "Hello from the fake model."
    assert "tool_calls" not in agent.messages[2]
    assert "".join(hooks.deltas) == "Hello from the fake model."
    assert len(hooks.deltas) >= 2, "deltas must arrive incrementally (>=2 chunks)"
    assert len(hooks.turns) == 1
    assert hooks.turns[0].finish_reason == "stop"
    assert hooks.errors == []


def test_tool_roundtrip(harness):
    echo = StubTool(name="echo")
    hooks = RecordingHooks()
    agent = harness([tool_call_reply("echo", {"msg": "ping"}), CONTENT_REPLY],
                    tools=[echo], hooks=hooks)

    agent.submit("please echo")

    assert echo.calls == [{"msg": "ping"}], "fn must receive parsed args dict"
    assistants = [m for m in agent.messages if m["role"] == "assistant"]
    assert len(assistants) == 2
    assert assistants[0]["tool_calls"][0]["id"] == "call_42"
    assert json.loads(assistants[0]["tool_calls"][0]["function"]["arguments"]) == {"msg": "ping"}
    assert assistants[1]["content"] == "Hello from the fake model."
    assert assistants[1].get("tool_calls") is None and "tool_calls" not in assistants[1]
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_42"
    assert "ping" in tool_msgs[0]["content"]
    assert hooks.errors == []


def test_danger_denied(harness):
    guard = StubTool(name="guard", danger=True)
    hooks = RecordingHooks(allow=False)
    agent = harness([tool_call_reply("guard", {"path": "x.txt"}), CONTENT_REPLY],
                    tools=[guard], hooks=hooks)

    agent.submit("delete stuff")

    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "DENIED by user"
    assert guard.calls == [], "denied tools must not execute"
    assert hooks.permission_requests == [("guard", {"path": "x.txt"})]
    assert hooks.turns[-1].content == "Hello from the fake model."  # loop continued
    assert hooks.errors == []


def test_unknown_tool(harness):
    hooks = RecordingHooks()
    agent = harness([tool_call_reply("nope", {"a": 1}), CONTENT_REPLY],
                    tools=[], hooks=hooks)

    agent.submit("do it")

    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert "unknown tool" in tool_msgs[0]["content"]
    assert hooks.errors == []


def test_cancel(harness):
    hooks = RecordingHooks()
    hooks.cancel_after_done = 1
    # single entry repeats forever -> infinite tool-calling script
    agent = harness([tool_call_reply("echo", {"n": 1})],
                    tools=[StubTool(name="echo")], hooks=hooks,
                    max_tool_rounds=50)
    hooks.agent_ref["agent"] = agent

    agent.submit("loop forever")  # must return without raising or error spam

    assert agent.cancelled is True
    assert len(hooks.turns) == 1, "loop must stop right after round 1"
    assistants = [m for m in agent.messages if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert all(m["role"] != "tool" for m in agent.messages), "no tools ran post-cancel"
    assert hooks.errors == [], "cancel must not trigger on_error"


def test_tool_exception_becomes_error_message(harness):
    broken = StubTool(name="broken", raises=True)
    hooks = RecordingHooks()
    agent = harness([tool_call_reply("broken", {}), CONTENT_REPLY],
                    tools=[broken], hooks=hooks)

    agent.submit("crash it")

    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("ERROR: ")
    assert "boom" in tool_msgs[0]["content"]
    assert hooks.errors == []


def test_system_prompt_contract():
    assert "Assistant" in SYSTEM_PROMPT
    assert "workspace" in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT.split()) < 50


def test_llm_error_cleans_unfulfilled_user_message(harness):
    hooks = RecordingHooks()
    # Script that raises 500 error on chat_stream
    agent = harness([{"status": 500, "body": '{"error":"fail"}'}], hooks=hooks)
    agent.submit("failed prompt")
    assert hooks.errors != []
    # User message must have been popped so state is not corrupted
    assert [m["role"] for m in agent.messages] == ["system"]


# --- context pruning (exercised end-to-end through submit) ---------------------

def _assert_api_valid_shape(messages: list[dict]) -> None:
    """No role:'tool' message may lack its assistant(tool_calls) parent."""
    for i, message in enumerate(messages):
        if message.get("role") == "tool":
            prev = messages[i - 1] if i else None
            assert (prev is not None and prev.get("role") == "assistant"
                    and prev.get("tool_calls")), f"orphaned tool message at index {i}"


def test_context_pruning(harness):
    hooks = RecordingHooks()
    agent = harness([CONTENT_REPLY], hooks=hooks)
    agent.messages = [{"role": "system", "content": "system"}]
    for i in range(40):
        agent.messages.append({"role": "user", "content": f"OLD_USER_{i} " + "x" * 2000})
        agent.messages.append({"role": "assistant", "content": f"OLD_ASSIST_{i} " + "y" * 2000})
    assert agent._approx_tokens() > 32000

    agent.submit("fresh question")

    assert agent.messages[0] == {"role": "system", "content": "system"}
    _assert_api_valid_shape(agent.messages)
    assert agent._approx_tokens() <= 33000, "history must be pruned to the budget"
    flat = json.dumps(agent.messages)
    assert "OLD_USER_0" not in flat and "OLD_ASSIST_0" not in flat, "oldest turns pruned"
    assert "fresh question" in flat, "newest user turn must survive"
    assert agent.messages[-2] == {"role": "user", "content": "fresh question"}
    assert agent.messages[-1]["content"] == "Hello from the fake model."
    assert hooks.errors == []


def test_pruning_keeps_tool_pair_atomic(harness):
    """Pruning must remove assistant(tool_calls)+tool results as ONE unit so
    no orphaned tool message ever reaches an OpenAI-compatible server."""
    echo = StubTool(name="echo")
    hooks = RecordingHooks()
    agent = harness([CONTENT_REPLY], tools=[echo], hooks=hooks)
    pad = json.dumps({"pad": "P" * 30000})
    agent.messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant",
         "content": None,
         "tool_calls": [{"id": "call_old", "type": "function",
                         "function": {"name": "echo", "arguments": pad}}]},
        {"role": "tool", "tool_call_id": "call_old", "content": "R" * 30000},
    ]
    for i in range(12):
        agent.messages.append({"role": "user", "content": f"FILLER_U_{i} " + "f" * 4000})
        agent.messages.append({"role": "assistant", "content": f"FILLER_A_{i} " + "g" * 4000})
    assert agent._approx_tokens() > 32000

    agent.submit("and now?")

    _assert_api_valid_shape(agent.messages)
    flat = json.dumps(agent.messages)
    assert '"call_old"' not in flat and '"RRR' not in flat, \
        "assistant(tool_calls) and its tool result must vanish atomically"
    assert agent._approx_tokens() <= 34000, "pruning must have run to the budget"
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[-2] == {"role": "user", "content": "and now?"}
    assert agent.messages[-1]["content"] == "Hello from the fake model."
    assert echo.calls == [], "scripted reply only; no tool may fire during submit"
    assert hooks.errors == []


# --- inline <think> stripping from stored history -------------------------------

@pytest.mark.parametrize("raw,expected_content,expected_reasoning", [
    ("<think>secret</think>answer", "answer", "secret"),
    ("A<think>x</think>B", "AB", "x"),
    ("<think>leaked tail", None, "leaked tail"),          # unclosed trailing opener
    ("[thinking: quick look]Ready.", "Ready.", "quick look"),
    ("<THINK>case</THINK>ok", "ok", "case"),
    ("just text", "just text", None),
])
def test_inline_thinking_split(harness, raw, expected_content, expected_reasoning):
    hooks = RecordingHooks()
    agent = harness([{"content": raw, "finish_reason": "stop"}], hooks=hooks)

    agent.submit("go")

    turn = hooks.turns[0]
    assert turn.content == expected_content
    assert turn.reasoning == expected_reasoning
    # Wire format must never be null (Go <nil>) - empty string is used instead
    expected_msg = expected_content if expected_content is not None else ""
    assert agent.messages[-1].get("content") == expected_msg
    assert "<think>" not in json.dumps(agent.messages)


def test_think_tags_stripped_from_replayed_history(harness):
    hooks = RecordingHooks()
    think_reply = {"content": "<think>secret plan</think>The answer is 42.",
                   "finish_reason": "stop"}
    agent = harness([think_reply, CONTENT_REPLY], hooks=hooks)

    agent.submit("question one")
    agent.submit("question two")

    # Round 2 request payload as received by the fake server
    replay = harness.servers[-1].last_request["messages"]
    assert hooks.turns[0].content == "The answer is 42."
    assert hooks.turns[0].reasoning == "secret plan"
    assert agent.messages[2]["content"] == "The answer is 42."
    assert "<think>" not in json.dumps(replay), "clean history must be replayed"
    assert "secret plan" not in json.dumps(replay)
    assert hooks.errors == []


def test_native_reasoning_field_wins_over_inline_think(harness):
    hooks = RecordingHooks()
    agent = harness([{"reasoning": "native thought",
                      "content": "<think>junk</think>final",
                      "finish_reason": "stop"}], hooks=hooks)

    agent.submit("go")

    assert hooks.turns[0].reasoning == "native thought"
    assert hooks.turns[0].content == "final"
    assert "junk" not in json.dumps(agent.messages), \
        "inline think text discarded when native reasoning exists"


# --- llm client -----------------------------------------------------------------

def test_llm_error_on_unreachable_server():
    srv = FakeOllama([])
    _, port = srv.start()
    srv.shutdown()
    client = LLMClient(base_url=f"http://127.0.0.1:{port}/v1", api_key="k",
                       model="m", timeout_s=5)
    with pytest.raises(LLMError):
        list(client.chat_stream([{"role": "user", "content": "hi"}]))


def test_llm_error_on_http_status():
    srv = FakeOllama([{"status": 500, "body": '{"error":"kaboom"}'}])
    srv.start()
    try:
        client = LLMClient(base_url=srv.base_url, api_key="k", model="m", timeout_s=5)
        with pytest.raises(LLMError) as excinfo:
            list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert "500" in str(excinfo.value)
        assert "kaboom" in str(excinfo.value)
    finally:
        srv.shutdown()


# --- config ----------------------------------------------------------------------

def test_load_config_toml(tmp_path, monkeypatch):
    monkeypatch.delenv("OCLITE_MODEL", raising=False)
    monkeypatch.delenv("OCLITE_BASE_URL", raising=False)

    cfg_file = tmp_path / "oclite.toml"
    cfg_file.write_text(
        'model = "llama3:8b"\n'
        "max_tool_rounds = 7\n"
        "\n[limits]\n"
        "read_max_lines = 50\n"
        "\n[permissions]\n"
        'shell = "allow"\n',
        encoding="utf-8",
    )

    cfg = load_config(cfg_file)
    assert cfg.model == "llama3:8b"
    assert cfg.max_tool_rounds == 7
    assert cfg.limits.read_max_lines == 50
    assert cfg.limits.shell_timeout_s == 120, "untouched limits keep defaults"
    assert cfg.permissions.shell == "allow"
    assert cfg.permissions.delete == "ask", "untouched permissions keep defaults"
    assert cfg.workspace == Path.cwd()
    assert cfg.stream is True

    monkeypatch.setenv("OCLITE_MODEL", "env-model")
    assert load_config(cfg_file).model == "env-model", "env overrides file"

    overrides = {"model": "override-model"}
    assert load_config(cfg_file, overrides).model == "override-model"

    with pytest.raises(ValueError):
        load_config(None, {"not_a_real_key": 1})


def test_tool_syntax_sanitized_from_content(harness):
    hooks = RecordingHooks()
    leak_reply = {"content": ">tool_calls [echo] hello", "finish_reason": "stop"}
    echo_tool = StubTool(name="echo")
    agent = harness([leak_reply, CONTENT_REPLY], tools=[echo_tool], hooks=hooks)
    agent.submit("do echo")

    assert len(echo_tool.calls) == 1
    assert ">tool_calls" not in agent.messages[-1]["content"]


def test_truncated_tool_json_fallback(harness):
    write_tool = StubTool(name="write_file")
    hooks = RecordingHooks()
    truncated_reply = {
        "content": '```json\n{"name": "write_file", "arguments": {"path": "sample.txt", "content": "hello"',
        "finish_reason": "stop",
    }
    agent = harness([truncated_reply, CONTENT_REPLY], tools=[write_tool], hooks=hooks)
    agent.submit("create file")

    assert len(write_tool.calls) == 1
    assert write_tool.calls[0]["path"] == "sample.txt"
    assert write_tool.calls[0]["content"] == "hello"


def test_tools_xml_fallback_and_persona_sanitization(harness):
    time_tool = StubTool(name="get_current_time")
    hooks = RecordingHooks()
    tools_reply = {
        "content": '<tools>\n{"type": "function", "function": {"name": "get_current_time", "arguments": {}}}\n</tools>',
        "finish_reason": "stop",
    }
    claude_reply = {
        "content": "I'm a large language model called Claude created by Anthropic.",
        "finish_reason": "stop",
    }
    agent = harness([tools_reply, claude_reply], tools=[time_tool], hooks=hooks)
    agent.submit("what time is it?")

    assert len(time_tool.calls) == 1
    final_content = agent.messages[-1]["content"]
    assert "Claude" not in final_content
    assert "Anthropic" not in final_content
    assert "Assistant" in final_content

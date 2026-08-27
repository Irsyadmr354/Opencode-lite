"""End-to-end and integration tests: config, session, CLI commands, headless mode, Ctrl+C, and typewriter."""

import io
import json
import os
import pathlib
import sys
import time
import unittest.mock as mock
import pytest

from assistant.__main__ import SESSIONS_DIR, clear_screen, cmd_sessions, main, print_banner, print_help
from assistant.config import Config, Limits, Permissions, get_default_shell_cmd, load_config
from assistant.llm import DARK_GRAY, RESET, TypewriterStreamer, typewriter
from assistant.session import Session


# --- 1. Config Loading, Env Overrides & Validations ---


def test_config_default_platform_shell():
    cfg = Config()
    default_shell = get_default_shell_cmd()
    assert cfg.shell_cmd == default_shell
    assert isinstance(cfg.shell_cmd, list)
    assert len(cfg.shell_cmd) >= 1
    if sys.platform == "win32" or os.name == "nt":
        assert any(sh in cfg.shell_cmd[0].lower() for sh in ("powershell", "cmd", "bash"))
    else:
        assert any(sh in cfg.shell_cmd[0].lower() for sh in ("bash", "sh"))


def test_config_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("ASSISTANT_MODEL", "custom-env-model")
    monkeypatch.setenv("ASSISTANT_BASE_URL", "http://custom:1234/v1")
    monkeypatch.setenv("ASSISTANT_API_KEY", "custom-key")
    monkeypatch.setenv("ASSISTANT_MAX_ROUNDS", "20")
    monkeypatch.setenv("ASSISTANT_TIMEOUT_S", "300")
    monkeypatch.setenv("ASSISTANT_WORKSPACE", str(tmp_path / "env_ws"))
    monkeypatch.setenv("ASSISTANT_STREAM", "false")
    monkeypatch.setenv("ASSISTANT_VERBOSE", "true")
    monkeypatch.setenv("ASSISTANT_SHELL_CMD", '["sh", "-c"]')

    cfg = load_config(tmp_path / "nonexistent.json")

    assert cfg.model == "custom-env-model"
    assert cfg.base_url == "http://custom:1234/v1"
    assert cfg.api_key == "custom-key"
    assert cfg.max_rounds == 20
    assert cfg.timeout_s == 300
    assert cfg.workspace == tmp_path / "env_ws"
    assert cfg.stream is False
    assert cfg.verbose is True
    assert cfg.shell_cmd == ["sh", "-c"]


def test_config_integer_validation(tmp_path):
    config_file = tmp_path / "bad_int.json"
    config_file.write_text(json.dumps({"max_rounds": -1}), encoding="utf-8")
    with pytest.raises(ValueError, match="max_rounds must be a positive integer"):
        load_config(config_file)

    config_file.write_text(json.dumps({"limits": {"read_max_lines": 0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="limits.read_max_lines must be a positive integer"):
        load_config(config_file)


def test_config_permission_validation(tmp_path):
    config_file = tmp_path / "bad_perm.json"
    config_file.write_text(json.dumps({"permissions": {"shell": "superuser"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="permissions.shell must be one of"):
        load_config(config_file)


# --- 2. Session Management Tests ---


def test_session_lifecycle(tmp_path):
    sess_path = tmp_path / "test_session.json"
    sess = Session(sess_path)
    assert sess.messages == []

    sess.append("user", "What is the date?")
    sess.append("assistant", "2026-08-27")
    sess.save()

    assert sess_path.exists()
    loaded = Session.load(sess_path)
    assert len(loaded.messages) == 2
    assert loaded.messages[0]["role"] == "user"
    assert loaded.messages[1]["content"] == "2026-08-27"

    sess.clear()
    assert sess.messages == []


# --- 3. Typewriter Speed & Dimmed Reasoning Rendering ---


def test_typewriter_speed_clamped_under_tenth_second():
    out = io.StringIO()
    start = time.perf_counter()
    typewriter("Test", delay=0.5, stream=out)  # Should clamp delay to < 0.1s
    elapsed = time.perf_counter() - start
    assert elapsed < 0.3
    assert out.getvalue() == "Test"


def test_typewriter_dimmed_reasoning_and_reset():
    out = io.StringIO()
    typewriter("Reasoning...", is_thinking=True, stream=out)
    val = out.getvalue()
    assert val.startswith(DARK_GRAY)
    assert val.endswith(RESET)


def test_typewriter_streamer_chunked_ansi():
    out = io.StringIO()
    streamer = TypewriterStreamer(delay=0.001, stream=out)
    streamer.on_delta("Step 1", is_thinking=True)
    streamer.on_delta("Answer", is_thinking=False)
    streamer.close()

    result = out.getvalue()
    assert DARK_GRAY in result
    assert "Step 1" in result
    assert RESET in result
    assert "Answer" in result


# --- 4. Headless CLI Mode Tests ---


def test_headless_command_flag(tmp_path):
    mock_resp = {"content": "42", "tool_calls": []}

    with mock.patch("assistant.llm.LLM.chat", return_value=mock_resp), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
        main(["-c", "what is 2+2", "--workspace", str(tmp_path)])
        output = fake_out.getvalue()
        assert "42" in output


def test_headless_prompt_flag(tmp_path):
    mock_resp = {"content": "Paris", "tool_calls": []}

    with mock.patch("assistant.llm.LLM.chat", return_value=mock_resp), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
        main(["-p", "capital of France", "--workspace", str(tmp_path), "--no-stream"])
        output = fake_out.getvalue()
        assert "Paris" in output


def test_headless_keyboard_interrupt(tmp_path):
    with mock.patch("assistant.agent.Agent.handle", side_effect=KeyboardInterrupt), \
         mock.patch("sys.stderr", new=io.StringIO()) as fake_err:
        with pytest.raises(SystemExit) as exc_info:
            main(["-c", "long task", "--workspace", str(tmp_path)])
        assert exc_info.value.code == 130
        assert "Cancelled" in fake_err.getvalue()


# --- 5. Interactive CLI Commands & Menu Tests ---


def test_print_help(capsys):
    print_help()
    captured = capsys.readouterr().out
    assert "/sessions" in captured
    assert "/clear" in captured
    assert "/c-context" in captured
    assert "/help" in captured
    assert "exit" in captured


def test_print_banner(capsys):
    cfg = Config()
    print_banner(cfg)
    captured = capsys.readouterr().out
    assert "Assistant" in captured
    assert cfg.model in captured


def test_clear_screen():
    with mock.patch("os.system") as mock_sys:
        clear_screen()
        assert mock_sys.called


def test_interactive_commands_exit(tmp_path):
    # Simulates entering 'exit' in interactive loop
    with mock.patch("builtins.input", side_effect=["exit"]), \
         mock.patch("sys.stdout", new=io.StringIO()):
        main(["--workspace", str(tmp_path)])


def test_interactive_commands_help_and_clear_context(tmp_path):
    # Simulates /help, /c-context, then exit
    with mock.patch("builtins.input", side_effect=["/help", "/c-context", "exit"]), \
         mock.patch("sys.stdout", new=io.StringIO()):
        main(["--workspace", str(tmp_path)])


def test_interactive_turn_ctrl_c_cancel(tmp_path):
    # Simulates KeyboardInterrupt during agent.handle, then exit
    with mock.patch("builtins.input", side_effect=["test query", "exit"]), \
         mock.patch("assistant.agent.Agent.handle", side_effect=KeyboardInterrupt), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
        main(["--workspace", str(tmp_path)])
        output = fake_out.getvalue()
        assert "Turn cancelled" in output


def test_cmd_sessions_menu_save_and_load(tmp_path, monkeypatch):
    monkeypatch.setattr("assistant.__main__.SESSIONS_DIR", tmp_path / "sessions")
    sess_path = tmp_path / "current_session.json"
    session = Session(sess_path)
    session.append("user", "Hello world")

    cfg = Config(workspace=tmp_path)
    from assistant.agent import Agent
    agent = Agent(cfg)

    # 1. Save session
    with mock.patch("builtins.input", side_effect=["s", "my_session"]):
        cmd_sessions(session, agent)

    saved_file = tmp_path / "sessions" / "my_session.json"
    assert saved_file.exists()

    # Clear current session in memory
    session.clear()
    assert len(session.messages) == 0

    # 2. Load session
    with mock.patch("builtins.input", side_effect=["l", "1"]):
        cmd_sessions(session, agent)

    assert len(session.messages) == 1
    assert session.messages[0]["content"] == "Hello world"
    assert any(m.get("content") == "Hello world" for m in agent.messages)

    # 3. Delete session
    with mock.patch("builtins.input", side_effect=["d", "1", "y"]):
        cmd_sessions(session, agent)

    assert not saved_file.exists()


def test_interactive_chat_turn_success(tmp_path):
    # Simulates entering user query, receiving response, then exiting
    mock_resp = {"content": "I am doing well!", "tool_calls": []}
    with mock.patch("builtins.input", side_effect=["how are you?", "exit"]), \
         mock.patch("assistant.llm.LLM.chat", return_value=mock_resp), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
        main(["--workspace", str(tmp_path)])
        output = fake_out.getvalue()
        assert "Assistant:" in output
        assert "I am doing well!" in output


def test_interactive_user_prompt_formatting(tmp_path):
    prompts_captured = []

    def fake_input(prompt=""):
        prompts_captured.append(prompt)
        return "exit"

    with mock.patch("builtins.input", side_effect=fake_input), \
         mock.patch("sys.stdout", new=io.StringIO()):
        main(["--workspace", str(tmp_path)])

    assert len(prompts_captured) >= 1
    assert "\033[1mYou:\033[0m " in prompts_captured[0]


def test_verbose_cli_flags(tmp_path):
    mock_resp = {
        "content": "Test response",
        "tool_calls": [],
        "stats": {"eval_count": 10, "eval_rate": 20.0, "total_duration_s": 0.5},
    }

    # 1. Default verbose is enabled
    with mock.patch("builtins.input", side_effect=["test", "exit"]), \
         mock.patch("assistant.llm.LLM.chat", return_value=mock_resp), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
        main(["--workspace", str(tmp_path)])
        assert "10 tokens" in fake_out.getvalue()

    # 2. Explicit --no-verbose disables stats
    with mock.patch("builtins.input", side_effect=["test", "exit"]), \
         mock.patch("assistant.llm.LLM.chat", return_value=mock_resp), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out_no_verb:
        main(["--workspace", str(tmp_path), "--no-verbose"])
        assert "10 tokens" not in fake_out_no_verb.getvalue()


def test_interactive_list_and_load_model_commands(tmp_path):
    mock_models = ["qwen2.5-coder-3b", "llama3.2:3b"]

    # 1. Test /list-model
    with mock.patch("builtins.input", side_effect=["/list-model", "exit"]), \
         mock.patch("assistant.llm.LLM.list_models", return_value=mock_models), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
        main(["--workspace", str(tmp_path)])
        out = fake_out.getvalue()
        assert "Available Models" in out
        assert "qwen2.5-coder-3b" in out
        assert "llama3.2:3b" in out

    # 2. Test /load-model with argument
    with mock.patch("builtins.input", side_effect=["/load-model llama3.2:3b", "exit"]), \
         mock.patch("assistant.llm.LLM.list_models", return_value=mock_models), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
        main(["--workspace", str(tmp_path)])
        out = fake_out.getvalue()
        assert "model switched to: \033[1mllama3.2:3b\033[0m" in out

    # 3. Test interactive /load-model pick by number
    with mock.patch("builtins.input", side_effect=["/load-model", "2", "exit"]), \
         mock.patch("assistant.llm.LLM.list_models", return_value=mock_models), \
         mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
        main(["--workspace", str(tmp_path)])
        out = fake_out.getvalue()
        assert "model switched to: \033[1mllama3.2:3b\033[0m" in out

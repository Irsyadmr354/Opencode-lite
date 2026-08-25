"""Offline tests: headless console hooks — piped-stdin deny + interactive prompt.

Covers the SEC-3 fix: when stdin is not a tty (e.g. ``opencode-lite -p`` with a
piped script containing "y"), on_permission must DENY without ever calling
input() or writing the prompt to stdout.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opencode_lite.agent import Hooks  # noqa: E402
from opencode_lite.__main__ import build_console_hooks  # noqa: E402


class FakeStdin:
    def __init__(self, is_a_tty: bool) -> None:
        self._is_a_tty = is_a_tty

    def isatty(self) -> bool:
        return self._is_a_tty


@pytest.fixture()
def hooks_cls() -> type:
    return build_console_hooks(Hooks)


def _set_stdin(monkeypatch: pytest.MonkeyPatch, is_a_tty: bool) -> None:
    monkeypatch.setattr(sys, "stdin", FakeStdin(is_a_tty))


def _forbid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> str:
        raise AssertionError("input() must not be called")

    monkeypatch.setattr(builtins, "input", _boom)


# --- piped / non-interactive stdin: deny, no prompt, stdout stays pure --------


@pytest.mark.parametrize("name,args", [("shell", {"command": "echo hi"}), ("delete_file", {"path": "x.txt"})])
def test_piped_stdin_denies_danger_tools_without_prompt(
    hooks_cls: type, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    name: str, args: dict,
) -> None:
    _set_stdin(monkeypatch, False)
    _forbid_input(monkeypatch)
    hooks = hooks_cls()

    assert hooks.on_permission(name, args) is False
    captured = capsys.readouterr()
    assert captured.out == ""  # prompt text never reaches piped stdout


# --- interactive stdin: legacy [y/N] prompt behavior preserved ----------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", True), ("yes", True), ("Y", True), ("YES", True), (" y ", True),
     ("n", False), ("no", False), ("N", False), ("", False), ("yeah", False)],
)
def test_interactive_answers(
    hooks_cls: type,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    expected: bool,
) -> None:
    _set_stdin(monkeypatch, True)
    seen_prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        seen_prompts.append(prompt)
        return answer

    monkeypatch.setattr(builtins, "input", fake_input)
    hooks = hooks_cls()

    assert hooks.on_permission("delete_file", {"path": "x.txt"}) is expected
    assert seen_prompts == ["Allow delete_file? [y/N] "]


def test_interactive_eof_denies(hooks_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_stdin(monkeypatch, True)

    def raise_eof(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    assert hooks_cls().on_permission("shell", {}) is False


def test_interactive_ctrl_c_denies(hooks_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_stdin(monkeypatch, True)

    def raise_kbdint(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", raise_kbdint)
    assert hooks_cls().on_permission("shell", {}) is False

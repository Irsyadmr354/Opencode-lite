"""Tool registry and shared tool contracts for assistant.

Contract (consumed by core/ and ui/):
    ToolResult(ok: bool, output: str)
    Tool(name, description, parameters, danger, fn)
    get_tools(workspace, config) -> list[Tool]
    openai_schema(tools) -> list[dict]

``config`` is duck-typed: ``config.limits.<field>`` and ``config.shell_cmd``.
This package deliberately never imports ``assistant.config``.

NOTE: the dataclasses below are intentionally declared *before* the submodule
imports; fs/shell/web bind ``Tool``/``ToolResult`` from the partially
initialized package object at import time (deterministic, stdlib-safe).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Callable

__all__ = ["ToolResult", "Tool", "get_tools", "openai_schema"]


@dataclass
class ToolResult:
    ok: bool
    output: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    danger: bool
    fn: Callable[[dict], ToolResult]


# --- imported after shared types so submodules can bind them -----------------
from .fs import build_tools as _build_fs_tools  # noqa: E402
from .shell import build_tools as _build_shell_tools  # noqa: E402
from .time import build_tools as _build_time_tools  # noqa: E402
from .web import build_tools as _build_web_tools  # noqa: E402


def get_tools(workspace: pathlib.Path, config) -> list[Tool]:
    """Return every built-in tool bound to ``workspace``/``config``.

    Safe tools come first, dangerous ones last (UI groups/paints by ``danger``).
    Time tool kept near web for 'date before search' flow but not first to avoid bias on greetings.
    """
    return (
        _build_fs_tools(workspace, config)
        + _build_web_tools(workspace, config)
        + _build_time_tools(workspace, config)
        + _build_shell_tools(workspace, config)
    )


def openai_schema(tools: list[Tool]) -> list[dict]:
    """OpenAI chat-completions ``tools=`` payload for the given tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]

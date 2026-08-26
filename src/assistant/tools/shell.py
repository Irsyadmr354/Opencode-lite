"""Shell tool: runs commands through the configured shell with cwd=workspace.

Output always includes stdout+stderr (even on failure — the model needs the
stderr to recover) plus a trailing ``exit=<returncode>`` line. Tool.fn never
raises.
"""
from __future__ import annotations

import pathlib
import subprocess

from assistant.tools import Tool, ToolResult

# Model-supplied timeouts are clamped into [1, config.limits.shell_timeout_s]:
# a model may lower the timeout but never raise it. On Windows, subprocess.run
# with a negative timeout waits ~forever, hence the hard floor of 1 second.
_MIN_TIMEOUT_S = 1


def shell_tool(workspace: pathlib.Path, config) -> Tool:
    default_timeout = int(config.limits.shell_timeout_s)
    output_chars = int(config.limits.shell_output_chars)

    def fn(args: dict) -> ToolResult:
        try:
            command = args.get("command")
            if command is None:
                return ToolResult(False, "ERROR: missing argument 'command'")
            raw_timeout = args.get("timeout_s")
            try:
                requested = int(raw_timeout) if raw_timeout else default_timeout
            except (TypeError, ValueError):
                return ToolResult(False, "ERROR: timeout_s must be an integer")
            limit = max(default_timeout, _MIN_TIMEOUT_S)
            timeout = min(max(requested, _MIN_TIMEOUT_S), limit)

            try:
                proc = subprocess.run(
                    [*config.shell_cmd, str(command)],
                    cwd=str(workspace),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                )
            except subprocess.TimeoutExpired:
                # subprocess.run already killed the child process on timeout.
                return ToolResult(False, f"ERROR: timed out after {timeout}s")

            output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if len(output) > output_chars:
                output = output[:output_chars] + "\n...[truncated]"
            output += f"\nexit={proc.returncode}"
            return ToolResult(proc.returncode == 0, output)
        except Exception as exc:  # noqa: BLE001 - tool boundary must never raise
            return ToolResult(False, f"ERROR: {exc}")

    return Tool(
        name="shell",
        description=(
            "Run a command via the configured shell inside the workspace root. "
            "stdout and stderr are returned along with the exit code."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command line to execute.",
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Timeout in seconds "
                    "(default: config.limits.shell_timeout_s).",
                },
            },
            "required": ["command"],
        },
        danger=True,
        fn=fn,
        permission_key="shell",
    )


def build_tools(workspace: pathlib.Path, config) -> list[Tool]:
    return [shell_tool(workspace, config)]

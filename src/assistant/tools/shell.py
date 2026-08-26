"""Shell tool: runs commands through the configured shell with cwd=workspace.

Output always includes stdout+stderr (even on failure — the model needs the
stderr to recover) plus a trailing ``exit=<returncode>`` line. Tool.fn never
raises.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

from assistant.tools import Tool, ToolResult

# Model-supplied timeouts are clamped into [1, config.limits.shell_timeout_s]:
# a model may lower the timeout but never raise it. On Windows, subprocess.run
# with a negative timeout waits ~forever, hence the hard floor of 1 second.
_MIN_TIMEOUT_S = 1


def get_default_shell_cmd() -> list[str]:
    """Return the default platform-aware shell command."""
    if sys.platform == "win32" or os.name == "nt":
        if shutil.which("powershell"):
            return ["powershell", "-NoProfile", "-Command"]
        if shutil.which("cmd.exe") or shutil.which("cmd"):
            return ["cmd.exe", "/c"]
        if shutil.which("bash"):
            return ["bash", "-c"]
        return ["powershell", "-NoProfile", "-Command"]
    else:
        if shutil.which("bash"):
            return ["bash", "-c"]
        if shutil.which("sh"):
            return ["sh", "-c"]
        return ["sh", "-c"]


def _is_working_shell(cmd_list: list[str]) -> bool:
    try:
        proc = subprocess.run(
            [*cmd_list, "exit 0"],
            capture_output=True,
            timeout=2,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _resolve_shell_cmd(config) -> list[str]:
    """Resolve shell command respecting user config with platform-aware fallback."""
    raw = getattr(config, "shell_cmd", None)
    if raw:
        if isinstance(raw, str):
            cmd_list = [raw]
        else:
            cmd_list = list(raw)
        if cmd_list:
            exe = cmd_list[0]
            exe_lower = pathlib.Path(exe).stem.lower()

            # On Windows, if bash is configured but missing or non-functional (e.g. WSL stub), fall back
            if (sys.platform == "win32" or os.name == "nt") and exe_lower in ("bash", "sh"):
                if not shutil.which(exe) or not _is_working_shell([exe, "-c"]):
                    return get_default_shell_cmd()

            # Check if configured shell executable exists or is accessible
            if shutil.which(exe) or pathlib.Path(exe).is_file():
                if len(cmd_list) == 1:
                    if exe_lower in ("bash", "sh", "zsh", "dash", "ksh"):
                        return [exe, "-c"]
                    if exe_lower in ("powershell", "pwsh"):
                        return [exe, "-NoProfile", "-Command"]
                    if exe_lower == "cmd":
                        return [exe, "/c"]
                return cmd_list

    return get_default_shell_cmd()


def shell_tool(workspace: pathlib.Path, config) -> Tool:
    default_timeout = int(getattr(getattr(config, "limits", None), "shell_timeout_s", 120))
    output_chars = int(getattr(getattr(config, "limits", None), "shell_output_chars", 6000))

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

            shell_cmd = _resolve_shell_cmd(config)

            try:
                proc = subprocess.run(
                    [*shell_cmd, str(command)],
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
            except FileNotFoundError as exc:
                return ToolResult(False, f"ERROR: shell executable not found: {exc}")

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

"""Filesystem tools: read_file, write_file, delete_file, list_files.

Every path argument is resolved against ``workspace.resolve()``; anything that
escapes the workspace root is rejected with ``ERROR: path outside workspace``.
All tool bodies are wrapped in try/except — a Tool.fn never raises.
"""
from __future__ import annotations

import heapq
import os
import pathlib
import re
from typing import Callable, Generator, Iterable

from assistant.tools import Tool, ToolResult

_EXCLUDED_PARTS = {".git", "__pycache__", "node_modules", ".venv", ".pytest_cache"}

# Refuse to buffer files larger than this into memory (read_file).
_MAX_READ_BYTES = 10 * 1024 * 1024
# Upper bound on entries enumerated per list_files traversal (early stop).
_MAX_WALK_ENTRIES = 100_000

_MISSING = "ERROR: missing argument '{}'"
_OUTSIDE = "ERROR: path outside workspace"


def _resolve_in_workspace(
    workspace: pathlib.Path, raw: str
) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    """Resolve *raw* against the workspace; return (ws, target) or (None, None)."""
    ws = workspace.resolve()
    candidate = (ws / str(raw)).resolve()
    if not candidate.is_relative_to(ws):
        return None, None
    return ws, candidate


def _is_excluded(parts: Iterable[str]) -> bool:
    for part in parts:
        if part in _EXCLUDED_PARTS or part.endswith(".egg-info"):
            return True
    return False


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob pattern to a regex over '/'-separated relative paths.

    ``*``/``?`` never cross directory separators; ``**`` does (``**/`` also
    matches zero directories), mirroring pathlib glob semantics closely enough
    for the patterns this tool accepts.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            j = i
            while j < n and pattern[j] == "*":
                j += 1
            if j - i >= 2:
                if j < n and pattern[j] == "/":
                    out.append(r"(?:[^/]+/)*")
                    j += 1
                else:
                    out.append(r".*")
            else:
                out.append(r"[^/]*")
            i = j
        elif char == "?":
            out.append(r"[^/]")
            i += 1
        else:
            out.append(re.escape(char))
            i += 1
    return re.compile("".join(out), re.DOTALL)


def _iter_entries(base: pathlib.Path, cap: int) -> Generator[pathlib.Path]:
    """Yield files and dirs under *base*, pruning excluded dirs before descent.

    Unlike ``Path.glob("**/*")`` this never materializes the full tree and
    stops after *cap* raw entries, so huge junk trees cannot exhaust memory.
    """
    scanned = 0
    for root, dirs, names in os.walk(base):
        dirs[:] = sorted(d for d in dirs if not _is_excluded((d,)))
        root_path = pathlib.Path(root)
        for name in sorted(names):
            if scanned >= cap:
                return
            scanned += 1
            yield root_path / name
        for name in dirs:
            if scanned >= cap:
                return
            scanned += 1
            yield root_path / name


# --- read_file ---------------------------------------------------------------
def read_file_tool(workspace: pathlib.Path, config) -> Tool:
    max_lines = int(config.limits.read_max_lines)

    def fn(args: dict) -> ToolResult:
        try:
            raw = args.get("path")
            if raw is None:
                return ToolResult(False, _MISSING.format("path"))
            ws, target = _resolve_in_workspace(workspace, raw)
            if target is None:
                return ToolResult(False, _OUTSIDE)
            if target.is_dir():
                return ToolResult(False, f"ERROR: '{raw}' is a directory. Call list_files with path '{raw}' to see contents.")
            if not target.is_file():
                return ToolResult(False, f"ERROR: not a file: {raw}")
            size = target.stat().st_size
            if size > _MAX_READ_BYTES:
                return ToolResult(
                    False,
                    f"ERROR: file too large: {size} bytes "
                    f"(limit {_MAX_READ_BYTES} bytes)",
                )
            try:
                start_line = int(args.get("start_line") or 1)
            except (TypeError, ValueError):
                start_line = 1
            start_line = max(start_line, 1)

            text = target.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if not lines:
                return ToolResult(True, "(empty file)")
            selected = lines[start_line - 1 : start_line - 1 + max_lines]
            if not selected:
                return ToolResult(
                    False,
                    f"ERROR: start_line {start_line} beyond end of file ({len(lines)} lines)",
                )
            end = start_line + len(selected) - 1
            body = "\n".join(
                f"{n:5d}: {lines[n - 1]}" for n in range(start_line, end + 1)
            )
            header = f"L{start_line}-L{end} of {len(lines)} lines"
            return ToolResult(True, f"{header}\n{body}")
        except Exception as exc:  # noqa: BLE001 - tool boundary must never raise
            return ToolResult(False, f"ERROR: {exc}")

    return Tool(
        name="read_file",
        description="Read a text file from the workspace with numbered lines.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace root.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based first line to read (default 1).",
                },
            },
            "required": ["path"],
        },
        danger=False,
        fn=fn,
        permission_key=None,
    )


# --- write_file --------------------------------------------------------------
def write_file_tool(workspace: pathlib.Path, config) -> Tool:
    def fn(args: dict) -> ToolResult:
        try:
            raw = args.get("path")
            if raw is None:
                return ToolResult(False, _MISSING.format("path"))
            content = args.get("content")
            if content is None:
                return ToolResult(False, _MISSING.format("content"))
            ws, target = _resolve_in_workspace(workspace, raw)
            if target is None:
                return ToolResult(False, _OUTSIDE)
            data = str(content).encode("utf-8")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)  # exact utf-8 bytes, no newline translation
            rel = target.relative_to(ws).as_posix()
            return ToolResult(True, f"OK: wrote {len(data)} bytes to {rel}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"ERROR: {exc}")

    return Tool(
        name="write_file",
        description="Create or overwrite a file in the workspace with utf-8 content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace root.",
                },
                "content": {
                    "type": "string",
                    "description": "Full new content of the file.",
                },
            },
            "required": ["path", "content"],
        },
        danger=False,
        fn=fn,
        permission_key="write",
    )


# --- delete_file -------------------------------------------------------------
def delete_file_tool(workspace: pathlib.Path, config) -> Tool:
    def fn(args: dict) -> ToolResult:
        try:
            raw = args.get("path")
            if raw is None:
                return ToolResult(False, _MISSING.format("path"))
            ws, target = _resolve_in_workspace(workspace, raw)
            if target is None:
                return ToolResult(False, _OUTSIDE)
            if not target.exists():
                return ToolResult(False, f"ERROR: not found: {raw}")
            if target.is_dir():
                return ToolResult(
                    False,
                    f"ERROR: '{raw}' is a directory",
                )
            os.remove(target)
            rel = target.relative_to(ws).as_posix()
            return ToolResult(True, f"OK: deleted {rel}")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"ERROR: {exc}")

    return Tool(
        name="delete_file",
        description="Permanently delete a single file inside the workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workspace root.",
                }
            },
            "required": ["path"],
        },
        danger=True,
        fn=fn,
        permission_key="delete",
    )


# --- list_files --------------------------------------------------------------
def list_files_tool(workspace: pathlib.Path, config) -> Tool:
    max_entries = int(config.limits.list_max_entries)

    def fn(args: dict) -> ToolResult:
        try:
            raw = args.get("path") or "."
            pattern = str(args.get("pattern") or "**/*")
            ws, base = _resolve_in_workspace(workspace, raw)
            if base is None:
                return ToolResult(False, _OUTSIDE)
            if not base.is_dir():
                return ToolResult(False, f"ERROR: not a directory: {raw}")

            regex = _glob_to_regex(pattern)

            total_matched = 0

            def _counting(iterable: Iterable[pathlib.Path]) -> Generator[pathlib.Path]:
                nonlocal total_matched
                for item in iterable:
                    total_matched += 1
                    yield item

            candidates = (
                p
                for p in _iter_entries(base, _MAX_WALK_ENTRIES)
                if not _is_excluded(p.relative_to(base).parts)
                and regex.fullmatch(p.relative_to(base).as_posix())
            )
            # Streaming top-k: identical result to sorted(all)[:max_entries]
            # while holding only max_entries paths in memory.
            matches = heapq.nsmallest(
                max_entries,
                _counting(candidates),
                key=lambda p: (
                    0 if p.is_dir() else 1,
                    p.relative_to(ws).as_posix().lower(),
                ),
            )

            lines = [
                p.relative_to(ws).as_posix() + ("/" if p.is_dir() else "")
                for p in matches
            ]
            output = "\n".join(lines)
            hidden = total_matched - len(matches)
            if hidden > 0:
                output += f"\n... and {hidden} more"
            if not output:
                output = "(no matching entries)"
            return ToolResult(True, output)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"ERROR: {exc}")

    return Tool(
        name="list_files",
        description="List files/dirs under a workspace folder using glob patterns.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to search, relative to the "
                    "workspace root (default '.').",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern relative to 'path' "
                    "(default '**/*').",
                },
            },
            "required": [],
        },
        danger=False,
        fn=fn,
        permission_key=None,
    )


def build_tools(workspace: pathlib.Path, config) -> list[Tool]:
    return [
        read_file_tool(workspace, config),
        write_file_tool(workspace, config),
        list_files_tool(workspace, config),
        delete_file_tool(workspace, config),
    ]

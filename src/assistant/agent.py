"""Core agent loop: conversation + multi-round tool dispatch + smooth streaming."""

from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from assistant.llm import (
    BOLD,
    CYAN,
    DARK_GRAY,
    DIM,
    GREEN,
    LLM,
    RESET,
    YELLOW,
    StreamReasoningParser,
    TypewriterStreamer,
    format_ollama_stats,
    safe_print,
    typewriter,
)
from assistant.tools import Tool, get_tools, openai_schema

CHAR_DELAY = 0.02


def _get_live_datetime_str() -> str:
    """Return formatted live date and time with local timezone info."""
    now = datetime.now(timezone.utc)
    local = now.astimezone()
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} ({local.strftime('%A')})"


def build_system_prompt(workspace: Path | str, tools: list[Tool] | list[str] | None = None) -> str:
    """Build ultra-concise dynamic system prompt with dynamic tools and workspace."""
    if tools:
        tool_names = ", ".join(t.name if hasattr(t, "name") else str(t) for t in tools)
    else:
        tool_names = "read_file, write_file, delete_file, list_files, shell, websearch, webfetch, get_current_time"
    return (
        f"Be concise. Assistant in {workspace}. Tools: {tool_names}.\n"
        "Before websearch/webfetch, call get_current_time."
    )


DEFAULT_PROMPT = build_system_prompt(".")


class ThinkingSpinner:
    """Phase 1: animated spinner while waiting for LLM response or first streaming token."""

    def __init__(self, prompt_prefix: str = "", stream: io.TextIOBase | None = None):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prompt_prefix = prompt_prefix
        self.stream = stream or sys.stdout
        self._started = False
        self._stopped = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while not self._stop.is_set():
            f = frames[idx % len(frames)]
            try:
                self.stream.write(f"\r{DIM}  {f} Thinking...{RESET}")
                self.stream.flush()
            except (UnicodeEncodeError, Exception):
                try:
                    alt_frames = ["|", "/", "-", "\\"]
                    af = alt_frames[idx % len(alt_frames)]
                    self.stream.write(f"\r{DIM}  {af} Thinking...{RESET}")
                    self.stream.flush()
                except Exception:
                    pass
            idx += 1
            time.sleep(0.08)
        self._clear_line()

    def _clear_line(self) -> None:
        try:
            self.stream.write(f"\r\x1b[2K\r{' ' * 40}\r")
            self.stream.flush()
        except Exception:
            pass

    def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._clear_line()


# Backward compatibility alias
ThinkingIndicator = ThinkingSpinner


def _char_by_char(text: str, delay: float = CHAR_DELAY):
    """Print text character by character (legacy helper)."""
    typewriter(text, delay=delay)


class Agent:
    def __init__(self, config):
        self.config = config
        self.llm = LLM(config)
        self.workspace = config.workspace
        self.tools = get_tools(self.workspace, config)
        self.tool_map = {t.name: t for t in self.tools}
        self.tool_schema = openai_schema(self.tools)
        self.messages: list[dict] = []
        self._recent_calls: deque[str] = deque(maxlen=6)
        self._refresh_system_prompt()

    def _refresh_system_prompt(self) -> None:
        """Update system prompt with dynamic workspace and tools on each turn."""
        prompt = build_system_prompt(self.workspace, self.tools)
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def clear_context(self) -> None:
        """Clear conversation history and reset system prompt."""
        self.messages.clear()
        self._recent_calls.clear()
        self._refresh_system_prompt()

    def handle(self, user_input: str) -> str:
        """Process one user turn through the multi-round agent loop.

        Executes tool calls iteratively up to `config.max_rounds` until
        the LLM produces a final response or max rounds is reached.
        Returns the final assistant response text.
        """
        self._refresh_system_prompt()
        self.messages.append({"role": "user", "content": user_input})

        round_count = 0
        final_content = ""

        while round_count < self.config.max_rounds:
            round_count += 1

            spinner = ThinkingSpinner()
            streamer = TypewriterStreamer(delay=CHAR_DELAY, dark_color_code=DARK_GRAY)
            streamed_any = False

            def on_token(token: str, is_thinking: bool = False) -> None:
                nonlocal streamed_any
                if not streamed_any:
                    streamed_any = True
                    spinner.stop()
                streamer.on_delta(token, is_thinking)

            spinner.start()
            try:
                if self.config.stream:
                    response = self.llm.chat(
                        self.messages,
                        tools=self.tool_schema if self.tool_schema else None,
                        on_delta=on_token,
                    )
                else:
                    response = self.llm.chat(
                        self.messages,
                        tools=self.tool_schema if self.tool_schema else None,
                    )
            finally:
                spinner.stop()
                if self.config.stream:
                    streamer.close()

            content = response.get("content", "") or ""
            tool_calls = response.get("tool_calls", []) or []
            thinking = response.get("thinking", "") or ""

            # Fallback printing if tokens weren't streamed incrementally
            if not streamed_any:
                if thinking:
                    streamer.on_delta(thinking, is_thinking=True)
                if content:
                    streamer.on_delta(content, is_thinking=False)
                streamer.close()

            # Ensure newline before tool calls if content was streamed without trailing newline
            if content and tool_calls and not content.endswith("\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()

            # No tool calls -> final response reached
            if not tool_calls:
                self.messages.append({"role": "assistant", "content": content})
                if getattr(self.config, "verbose", False):
                    stats_str = format_ollama_stats(response.get("stats"))
                    if stats_str:
                        safe_print(stats_str)
                return content

            # Format assistant tool call request for conversation history
            tc_msg_parts = []
            for i, tc in enumerate(tool_calls):
                tc_id = tc.get("id") or f"call_{i}"
                args = tc.get("arguments", {})
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                tc_msg_parts.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": args_str,
                    },
                })
            self.messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": tc_msg_parts,
            })

            # Execute tool calls
            for i, tc in enumerate(tool_calls):
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                tc_id = tc.get("id") or f"call_{i}"

                # Loop breaker check: 3 consecutive identical calls
                sig = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
                self._recent_calls.append(sig)
                if len(self._recent_calls) >= 3 and list(self._recent_calls)[-3:] == [sig, sig, sig]:
                    result_text = f"Tool '{name}' called 3 times in a row with identical arguments. Stopping repeat call."
                    safe_print(f"{YELLOW}  ! Loop breaker: {result_text}{RESET}")
                    self.messages.append({"role": "tool", "content": result_text, "tool_call_id": tc_id})
                    continue

                tool = self.tool_map.get(name)
                if not tool:
                    result_text = f"Unknown tool: {name}"
                    safe_print(f"{DIM}  > {name}({self._preview_args(args)}){RESET}")
                    safe_print(f"{DIM}  -> {result_text}{RESET}")
                    self.messages.append({"role": "tool", "content": result_text, "tool_call_id": tc_id})
                    continue

                # Permission check
                if tool.permission_key and not self._check_permission(tool):
                    result_text = f"Permission denied for '{name}'."
                    safe_print(f"{YELLOW}  Permission denied for '{name}'.{RESET}")
                    self.messages.append({"role": "tool", "content": result_text, "tool_call_id": tc_id})
                    continue

                safe_print(f"{DIM}  > {name}({self._preview_args(args)}){RESET}")
                try:
                    result = tool.fn(args if isinstance(args, dict) else {})
                    result_text = result.output
                    status = "ok" if result.ok else "FAIL"
                    safe_print(f"{DIM}  -> {status} ({len(result_text)} chars){RESET}")
                except Exception as exc:
                    result_text = f"Error executing tool '{name}': {exc}"
                    safe_print(f"{DIM}  -> FAIL: {result_text}{RESET}")

                self.messages.append({"role": "tool", "content": result_text, "tool_call_id": tc_id})

        # Max rounds reached without natural stop
        fallback = content if content else "Reached maximum tool execution rounds limit."
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback

    def _check_permission(self, tool: Tool) -> bool:
        key = tool.permission_key
        if not key:
            return True
        setting = getattr(self.config.permissions, key, "allow")
        if setting == "allow":
            return True
        if setting == "deny":
            return False
        try:
            answer = input(f"  Allow {tool.name}? [y/N] ").strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    @staticmethod
    def _preview_args(args: dict | Any) -> str:
        try:
            s = json.dumps(args, ensure_ascii=False, default=str)
        except Exception:
            s = str(args)
        if len(s) > 80:
            return s[:77] + "..."
        return s

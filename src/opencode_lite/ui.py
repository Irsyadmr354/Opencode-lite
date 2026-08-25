"""Pure Native Terminal REPL and UI for opencode-lite."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, TextIO

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, RichLog, Static

try:
    from opencode_lite.agent import Hooks
except ImportError:
    class Hooks:  # type: ignore[no-redef]
        def on_delta(self, text: str) -> None: ...
        def on_reasoning(self, text: str) -> None: ...
        def on_assistant_done(self, turn: Any) -> None: ...
        def on_tool_start(self, name: str, args: dict) -> None: ...
        def on_tool_result(self, name: str, res: Any) -> None: ...
        def on_permission(self, name: str, args: dict) -> bool:
            return False
        def on_status(self, info: dict) -> None: ...
        def on_error(self, msg: str) -> None: ...


VERSION = "0.1.0"
RESULT_PREVIEW_CHARS = 300
ARGS_PREVIEW_LINES = 30

THINKING_STYLE = "italic dim cyan"
MAX_LINE_CHARS = 80
_TAG_OPENERS = ("<think>", "[thinking:")
_TAG_CLOSER_THINK = "</think>"

# ANSI Color and Style Codes
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_ITALIC = "\033[3m"
ANSI_RED = "\033[31m"
ANSI_BOLD_RED = "\033[1;31m"
ANSI_GREEN = "\033[32m"
ANSI_DIM_GREEN = "\033[2;32m"
ANSI_YELLOW = "\033[33m"
ANSI_BOLD_YELLOW = "\033[1;33m"
ANSI_CYAN = "\033[36m"
ANSI_BOLD_CYAN = "\033[1;36m"
ANSI_DIM_CYAN = "\033[2;36m"
ANSI_WHITE = "\033[37m"
ANSI_BOLD_WHITE = "\033[1;37m"
ANSI_THINKING = "\033[3;2;36m"  # italic dim cyan
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

PROMPT_PREFIX = f"{ANSI_BOLD_CYAN}⬢ You:{ANSI_RESET} "
AI_PREFIX = f"{ANSI_BOLD_WHITE}⬡ Assistant:{ANSI_RESET} "
THOUGHTS_PREFIX = f"{ANSI_THINKING}⠋ Thinking:{ANSI_RESET} "


def _compact_json(args: Any) -> str:
    try:
        return json.dumps(args, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(args)


def _format_args_summary(name: str, args: dict | Any) -> str:
    if not isinstance(args, dict):
        return str(args)
    if name == "shell" and "command" in args:
        return str(args["command"])
    if name in ("read_file", "delete_file") and "path" in args:
        start = args.get("start_line")
        if start:
            return f"{args['path']}:{start}"
        return str(args["path"])
    if name == "write_file" and "path" in args:
        return str(args["path"])
    if name == "list_files":
        path = args.get("path", ".")
        pattern = args.get("pattern")
        return f"{path} [{pattern}]" if pattern else str(path)
    if name == "webfetch" and "url" in args:
        return str(args["url"])
    if name == "websearch" and "query" in args:
        return str(args["query"])
    return _compact_json(args)


def _pretty_json(args: Any, max_lines: int = ARGS_PREVIEW_LINES) -> str:
    try:
        rendered = json.dumps(args, indent=2, default=str)
    except (TypeError, ValueError):
        rendered = str(args)
    lines = rendered.splitlines() or ["{}"]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (+{len(lines) - max_lines} more lines)"]
    return "\n".join(lines)


def clean_roleplay_asterisks(text: str) -> str:
    """Clean roleplay asterisks from text (e.g. *Assistant is ready* -> Assistant is ready).
    Preserves markdown bold (**text**) and bullet lists (* item)."""
    if not text:
        return text
    lines = text.splitlines(keepends=True)
    cleaned_lines = []
    for line in lines:
        newline = ""
        if line.endswith("\r\n"):
            newline = "\r\n"
            content = line[:-2]
        elif line.endswith("\n"):
            newline = "\n"
            content = line[:-1]
        else:
            content = line

        stripped = content.strip()
        # Skip empty lines, markdown bold (**...), and standard bullet lists (* item)
        if not stripped or stripped.startswith("**"):
            cleaned_lines.append(line)
            continue

        if stripped.startswith("* ") and not (stripped.endswith("*") and not stripped.endswith("**")):
            cleaned_lines.append(line)
            continue

        # Check for full line wrapped in single *...*
        if stripped.startswith("*") and not stripped.startswith("**") and stripped.endswith("*") and not stripped.endswith("**") and len(stripped) > 1:
            indent = content[:len(content) - len(content.lstrip())]
            inner = stripped[1:-1].strip()
            cleaned_lines.append(f"{indent}{inner}{newline}")
            continue

        # Check for leading roleplay action: *Action* Rest of sentence
        m = re.match(r"^\*([^*\r\n]+)\*\s*(.*)$", stripped)
        if m and not stripped.startswith("**"):
            indent = content[:len(content) - len(content.lstrip())]
            action = m.group(1).strip()
            rest = m.group(2).strip()
            if rest:
                cleaned_lines.append(f"{indent}{action} {rest}{newline}")
            else:
                cleaned_lines.append(f"{indent}{action}{newline}")
            continue

        cleaned_lines.append(line)

    return "".join(cleaned_lines)


def _tag_holdback(buffer: str, in_think: bool, at_line_start: bool = False) -> int:
    """Length of buffer's trailing fragment that might be a PARTIAL tag token.

    Prevents flushing text that ends in e.g. ``"<thi"`` — if that fragment
    were rendered now, the completed ``"<think>"`` arriving next chunk would
    no longer be recognised. Returns how many trailing characters to keep
    buffered (0 when nothing is at risk).
    """
    candidates: tuple[str, ...] = (_TAG_CLOSER_THINK,) if in_think else _TAG_OPENERS
    limit = min(max(len(c) for c in candidates) - 1, len(buffer))
    for h in range(limit, 0, -1):
        tail = buffer[-h:].lower()
        if any(c.startswith(tail) and c != tail for c in candidates):
            return h
    if not in_think and at_line_start and buffer == "*":
        return 1
    return 0


def _parse_thinking_text(text: str, state: dict[str, bool]) -> Text:
    """Parse text into a Rich Text object, highlighting <think>...</think>
    and [Thinking:...] sections in italic dim cyan/gray."""
    res = Text()
    i = 0
    n = len(text)
    cur = ""

    while i < n:
        if not state.get("in_think") and not state.get("in_bracket"):
            lower = text[i:].lower()
            if lower.startswith("<think>"):
                if cur:
                    res.append(cur)
                    cur = ""
                state["in_think"] = True
                cur += text[i : i + 7]
                i += 7
            elif lower.startswith("[thinking:"):
                if cur:
                    res.append(cur)
                    cur = ""
                state["in_bracket"] = True
                cur += text[i : i + 10]
                i += 10
            else:
                cur += text[i]
                i += 1
        elif state.get("in_think"):
            lower = text[i:].lower()
            if lower.startswith("</think>"):
                cur += text[i : i + 8]
                res.append(cur, style=THINKING_STYLE)
                cur = ""
                state["in_think"] = False
                i += 8
            else:
                cur += text[i]
                i += 1
        elif state.get("in_bracket"):
            if text[i] == "]":
                cur += "]"
                res.append(cur, style=THINKING_STYLE)
                cur = ""
                state["in_bracket"] = False
                i += 1
            else:
                cur += text[i]
                i += 1

    if cur:
        if state.get("in_think") or state.get("in_bracket"):
            res.append(cur, style=THINKING_STYLE)
        else:
            res.append(cur)

    return res


class TerminalHooks(Hooks):
    """Pure native terminal hooks for real-time word-by-word streaming, ANSI colors,
    clean inline tool execution, inline permission prompts, thinking collapse/clear,
    and red error messages."""

    def __init__(
        self,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        input_fn: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__()
        self._stdout: TextIO = stdout if stdout is not None else sys.stdout
        self._stderr: TextIO = stderr if stderr is not None else sys.stderr
        self._input_fn: Callable[[str], str] = input_fn if input_fn is not None else input
        self._in_think: bool = False
        self._in_bracket: bool = False
        self._buffer: str = ""
        self._saw_reasoning: bool = False
        self._thinking_active: bool = False
        self._thinking_written: bool = False
        self._thinking_lines: int = 0
        self._spinner_idx: int = 0
        self._current_tool: tuple[str, str] | None = None
        self._last_char_was_newline: bool = True
        self._at_line_start: bool = True
        self._in_roleplay_asterisk: bool = False
        self._ai_prefix_printed: bool = False
        self.had_error: bool = False

    def _write_stdout(self, text: str) -> None:
        if not text:
            return
        self._stdout.write(text)
        self._stdout.flush()
        self._last_char_was_newline = text.endswith("\n")

    def _write_stderr(self, text: str) -> None:
        if not text:
            return
        self._stderr.write(text)
        self._stderr.flush()

    def _collapse_thinking(self) -> None:
        """Collapse and erase streamed thinking from the terminal screen."""
        if not self._thinking_active and not self._saw_reasoning:
            return
        if self._thinking_written:
            # Clear current line
            erase_seq = "\r\033[2K"
            # For each previous line written during thinking, move up and clear
            if self._thinking_lines > 0:
                erase_seq += "\033[1A\033[2K" * self._thinking_lines
            erase_seq += "\r"
            self._write_stdout(f"{ANSI_RESET}{erase_seq}")
        else:
            self._write_stdout(ANSI_RESET)

        self._thinking_active = False
        self._saw_reasoning = False
        self._thinking_written = False
        self._thinking_lines = 0
        self._last_char_was_newline = True
        self._at_line_start = True

    def reset_stream(self) -> None:
        """Reset internal streaming state cleanly."""
        self._collapse_thinking()
        self._in_think = False
        self._in_bracket = False
        self._buffer = ""
        self._saw_reasoning = False
        self._thinking_active = False
        self._thinking_written = False
        self._thinking_lines = 0
        self._current_tool = None
        self._at_line_start = True
        self._in_roleplay_asterisk = False
        self._ai_prefix_printed = False

    def on_reasoning(self, text: str) -> None:
        """Real-time streaming for dedicated reasoning/reasoning_content field in italic dim cyan."""
        if not text:
            return
        if not self._saw_reasoning:
            self._saw_reasoning = True
            self._thinking_active = True
            frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
            self._write_stdout(f"{ANSI_THINKING}{frame} Thinking:{ANSI_RESET}\n")
            self._thinking_written = True
            self._thinking_lines += 1
        self._write_stdout(f"{ANSI_THINKING}{text}{ANSI_RESET}")
        self._thinking_written = True
        self._thinking_lines += text.count("\n")

    def on_delta(self, text: str) -> None:
        """Real-time word-by-word streaming directly to stdout, collapsing thinking on transition."""
        if not text:
            return
        if self._saw_reasoning or (self._thinking_active and not self._in_think and not self._in_bracket):
            self._collapse_thinking()

        self._buffer += text
        hold = _tag_holdback(self._buffer, self._in_think, self._at_line_start)
        if hold > 0:
            to_process = self._buffer[:-hold]
            self._buffer = self._buffer[-hold:]
        else:
            to_process = self._buffer
            self._buffer = ""

        if not to_process:
            return

        self._render_chunk(to_process)

    def _render_chunk(self, chunk: str) -> None:
        i = 0
        n = len(chunk)
        cur: list[str] = []

        def flush_cur(style: str | None = None) -> None:
            if cur:
                joined = "".join(cur)
                if style:
                    self._write_stdout(f"{style}{joined}{ANSI_RESET}")
                    if self._thinking_active:
                        self._thinking_written = True
                        self._thinking_lines += joined.count("\n")
                else:
                    if not self._ai_prefix_printed and not self._in_think and not self._in_bracket:
                        # Print [AI]: prefix before first content token
                        clean_check = joined.lstrip()
                        if clean_check:
                            self._write_stdout(AI_PREFIX)
                            self._ai_prefix_printed = True
                    self._write_stdout(joined)
                cur.clear()

        while i < n:
            ch = chunk[i]

            if not self._in_think and not self._in_bracket:
                lower = chunk[i:].lower()
                if lower.startswith("<think>"):
                    flush_cur()
                    self._in_think = True
                    self._thinking_active = True
                    frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
                    self._write_stdout(f"{ANSI_THINKING}{frame} Thinking:{ANSI_RESET}\n")
                    self._thinking_written = True
                    self._thinking_lines += 1
                    i += 7
                    continue
                elif lower.startswith("[thinking:"):
                    flush_cur()
                    self._in_bracket = True
                    self._thinking_active = True
                    frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
                    self._write_stdout(f"{ANSI_THINKING}{frame} Thinking:{ANSI_RESET}\n")
                    self._thinking_written = True
                    self._thinking_lines += 1
                    i += 10
                    continue

                # Strip roleplay asterisks at line start
                if self._at_line_start and ch == "*":
                    if i + 1 < n:
                        next_ch = chunk[i + 1]
                        if next_ch == "*":
                            # Markdown bold "**"
                            self._at_line_start = False
                            cur.append("**")
                            i += 2
                            continue
                        elif next_ch == " ":
                            # Markdown bullet "* "
                            self._at_line_start = False
                            cur.append("* ")
                            i += 2
                            continue
                        elif next_ch not in ("\n", "\r"):
                            # Roleplay asterisk start -> skip leading asterisk
                            self._at_line_start = False
                            self._in_roleplay_asterisk = True
                            i += 1
                            continue
                    else:
                        self._at_line_start = False
                        cur.append(ch)
                        i += 1
                        continue

                # Strip matching roleplay closing asterisk
                if self._in_roleplay_asterisk and ch == "*":
                    self._in_roleplay_asterisk = False
                    i += 1
                    continue

                if ch == "\n":
                    self._at_line_start = True
                    self._in_roleplay_asterisk = False
                elif ch != " " and ch != "\r":
                    self._at_line_start = False

                cur.append(ch)
                i += 1

            elif self._in_think:
                lower = chunk[i:].lower()
                if lower.startswith("</think>"):
                    flush_cur(ANSI_THINKING)
                    self._in_think = False
                    self._collapse_thinking()
                    i += 8
                else:
                    cur.append(ch)
                    i += 1
            elif self._in_bracket:
                if ch == "]":
                    flush_cur(ANSI_THINKING)
                    self._in_bracket = False
                    self._collapse_thinking()
                    i += 1
                else:
                    cur.append(ch)
                    i += 1

        if cur:
            if self._in_think or self._in_bracket:
                flush_cur(ANSI_THINKING)
            else:
                flush_cur()

    def on_assistant_done(self, turn: Any) -> None:
        """Called when the assistant turn completes. Flush buffers, collapse active thinking, ensure newline."""
        if self._buffer:
            buf = self._buffer
            self._buffer = ""
            self._render_chunk(buf)

        if self._in_think or self._in_bracket or self._saw_reasoning or self._thinking_active:
            self._collapse_thinking()
            self._in_think = False
            self._in_bracket = False

        self._at_line_start = True
        self._in_roleplay_asterisk = False
        self._ai_prefix_printed = False

        if not self._last_char_was_newline:
            self._write_stdout("\n")

    def on_tool_start(self, name: str, args: dict) -> None:
        """Clean inline tool execution indication: ⚡ [tool_name] args_summary."""
        summary = _format_args_summary(name, args)
        self._current_tool = (name, summary)
        if not self._last_char_was_newline:
            self._write_stdout("\n")
        self._write_stdout(f"{ANSI_YELLOW}⚡{ANSI_RESET} {ANSI_BOLD}[{name}]{ANSI_RESET} {summary}")

    def on_tool_result(self, name: str, res: Any) -> None:
        """Clean inline tool result: -> ok / -> ERROR: <msg>."""
        ok = bool(getattr(res, "ok", True) if not isinstance(res, dict) else res.get("ok", True))
        out = str(getattr(res, "output", "") if not isinstance(res, dict) else res.get("output", ""))
        out_clean = " ".join(out.split())
        preview = f"{out_clean[:RESULT_PREVIEW_CHARS]}{'...' if len(out_clean) > RESULT_PREVIEW_CHARS else ''}"

        if self._current_tool is None:
            summary = _format_args_summary(name, {})
            if not self._last_char_was_newline:
                self._write_stdout("\n")
            self._write_stdout(f"{ANSI_YELLOW}⚡{ANSI_RESET} {ANSI_BOLD}[{name}]{ANSI_RESET} {summary}")

        if ok:
            if preview and preview != "ok" and len(preview) <= 60:
                self._write_stdout(f" -> {ANSI_GREEN}ok{ANSI_RESET} ({preview})\n")
            else:
                self._write_stdout(f" -> {ANSI_GREEN}ok{ANSI_RESET}\n")
        else:
            if preview.startswith("ERROR:"):
                preview = preview[6:].strip()
            err_text = preview or "failed"
            self._write_stdout(f" -> {ANSI_RED}ERROR: {err_text}{ANSI_RESET}\n")

        self._current_tool = None

    def on_permission(self, name: str, args: dict) -> bool:
        """Inline permission prompt: [permission] Allow <tool_name>(<args_summary>)? [y/N]: """
        summary = _format_args_summary(name, args)
        if not self._last_char_was_newline:
            self._write_stdout("\n")
        prompt = f"{ANSI_BOLD_YELLOW}[permission]{ANSI_RESET} Allow {name}({summary})? [y/N]: "
        try:
            answer = self._input_fn(prompt).strip().lower()
        except (KeyboardInterrupt, EOFError):
            self._write_stdout("\n")
            return False
        return answer in ("y", "yes")

    def on_status(self, info: dict) -> None:
        pass

    def on_error(self, msg: str) -> None:
        """Error messages formatted in red."""
        self.had_error = True
        self._write_stderr(f"{ANSI_BOLD_RED}ERROR:{ANSI_RESET} {ANSI_RED}{msg}{ANSI_RESET}\n")


UiHooks = TerminalHooks


def _get_help_text() -> str:
    return (
        f"\n"
        f"{ANSI_BOLD}============================== OPENCODE-LITE GUIDE =============================={ANSI_RESET}\n"
        f"\n"
        f"  {ANSI_BOLD}COMMANDS:{ANSI_RESET}\n"
        f"    {ANSI_CYAN}/help{ANSI_RESET}            Display this comprehensive reference guide\n"
        f"    {ANSI_CYAN}/clear{ANSI_RESET}           Clear screen & reset conversation context memory\n"
        f"    {ANSI_CYAN}/cls{ANSI_RESET}             Clear terminal screen view only\n"
        f"    {ANSI_CYAN}/model [name]{ANSI_RESET}    View active model or switch to another Ollama model on the fly\n"
        f"    {ANSI_CYAN}/status{ANSI_RESET}          Show current workspace, model, base URL, and permissions\n"
        f"    {ANSI_CYAN}/exit, /quit{ANSI_RESET}     Exit opencode-lite\n"
        f"\n"
        f"  {ANSI_BOLD}KEYBOARD SHORTCUTS:{ANSI_RESET}\n"
        f"    {ANSI_BOLD}Enter{ANSI_RESET}            Send prompt to the agent\n"
        f"    {ANSI_BOLD}Ctrl+C{ANSI_RESET}           Cancel active streaming generation / reset prompt\n"
        f"    {ANSI_BOLD}Ctrl+D / Ctrl+Z{ANSI_RESET}  Exit opencode-lite (EOF)\n"
        f"\n"
        f"  {ANSI_BOLD}BUILT-IN TOOLS (AUTONOMOUS):{ANSI_RESET}\n"
        f"    * {ANSI_BOLD}read_file(path, start_line){ANSI_RESET}   Read files in workspace with line numbers\n"
        f"    * {ANSI_BOLD}write_file(path, content){ANSI_RESET}     Create or update files in workspace\n"
        f"    * {ANSI_BOLD}delete_file(path){ANSI_RESET}             Permanently delete a file (gated by confirmation)\n"
        f"    * {ANSI_BOLD}list_files(path, pattern){ANSI_RESET}     List workspace directory entries via glob\n"
        f"    * {ANSI_BOLD}shell(command){ANSI_RESET}                Execute shell commands (gated by confirmation)\n"
        f"    * {ANSI_BOLD}webfetch(url){ANSI_RESET}                 Fetch webpage text content directly\n"
        f"    * {ANSI_BOLD}websearch(query){ANSI_RESET}              Search the web via DuckDuckGo\n"
        f"\n"
        f"  {ANSI_BOLD}PERMISSIONS & CONFIGURATION:{ANSI_RESET}\n"
        f"    Configuration file: ~/.opencode-lite/config.toml\n"
        f"    Tool permissions can be set to: 'allow' | 'ask' | 'deny'\n"
        f"{ANSI_BOLD}=================================================================================={ANSI_RESET}\n"
    )


def run_repl(
    agent: Any,
    config: Any = None,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    clear_fn: Callable[[], None] | None = None,
) -> None:
    """Run the pure native terminal REPL loop."""
    if config is None:
        config = getattr(agent, "config", None)

    # Enable ANSI escape sequences on Windows if possible
    if os.name == "nt":
        os.system("")

    def _clear_screen() -> None:
        if clear_fn is not None:
            clear_fn()
        else:
            os.system("cls" if os.name == "nt" else "clear")

    def _print(msg: str = "") -> None:
        if output_fn is not None:
            output_fn(msg)
        else:
            print(msg)

    _read_input = input_fn if input_fn is not None else input

    # Clear terminal screen on launch
    _clear_screen()

    model = getattr(config, "model", getattr(getattr(agent, "client", None), "model", "default"))
    workspace = getattr(config, "workspace", Path.cwd())

    # Minimal banner in dim styling
    _print(f"{ANSI_DIM}opencode-lite {VERSION} | model: {model} | workspace: {workspace}{ANSI_RESET}")
    _print(f"{ANSI_DIM}Type /help for commands, /exit to quit.{ANSI_RESET}\n")

    hooks = TerminalHooks(input_fn=_read_input)
    if hasattr(agent, "hooks"):
        agent.hooks = hooks

    last_was_sigint = False

    while True:
        try:
            user_input = _read_input(PROMPT_PREFIX).strip()
            last_was_sigint = False
        except KeyboardInterrupt:
            if last_was_sigint:
                _print(f"\n{ANSI_DIM}Goodbye!{ANSI_RESET}")
                return
            last_was_sigint = True
            _print(f"\n{ANSI_DIM}Type /exit or press Ctrl+C again to quit.{ANSI_RESET}")
            continue
        except EOFError:
            _print(f"\n{ANSI_DIM}Goodbye!{ANSI_RESET}")
            return

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit", "/q"):
                _print(f"{ANSI_DIM}Goodbye!{ANSI_RESET}")
                break
            elif cmd in ("/help", "/?", "/h"):
                _print(_get_help_text())
                continue
            elif cmd in ("/cls",):
                _clear_screen()
                continue
            elif cmd in ("/clear",):
                _clear_screen()
                if hasattr(agent, "reset"):
                    agent.reset()
                elif hasattr(agent, "messages"):
                    agent.messages.clear()
                continue
            elif cmd == "/model":
                if arg:
                    if config is not None and hasattr(config, "model"):
                        config.model = arg
                    if hasattr(agent, "client") and hasattr(agent.client, "model"):
                        agent.client.model = arg
                    _print(f"{ANSI_DIM}Model switched to: {arg}{ANSI_RESET}\n")
                else:
                    cur_model = getattr(config, "model", getattr(getattr(agent, "client", None), "model", "?"))
                    _print(f"{ANSI_DIM}Active model: {cur_model}{ANSI_RESET}\n")
                continue
            elif cmd == "/status":
                ws = getattr(config, "workspace", getattr(agent, "config", None) and getattr(agent.config, "workspace", "."))
                mdl = getattr(config, "model", getattr(getattr(agent, "client", None), "model", "?"))
                url = getattr(config, "base_url", "http://127.0.0.1:11434/v1")
                perms = getattr(config, "permissions", None)
                perm_str = ""
                if perms:
                    if hasattr(perms, "__dict__"):
                        perm_str = ", ".join(f"{k}={v}" for k, v in perms.__dict__.items() if not k.startswith("_"))
                    elif isinstance(perms, dict):
                        perm_str = ", ".join(f"{k}={v}" for k, v in perms.items())
                _print(f"Workspace:   {ws}")
                _print(f"Model:       {mdl}")
                _print(f"Base URL:    {url}")
                if perm_str:
                    _print(f"Permissions: {perm_str}")
                _print()
                continue
            else:
                _print(f"{ANSI_RED}Unknown command '{cmd}'. Type /help for available commands.{ANSI_RESET}\n")
                continue

        # Normal prompt submission to agent
        try:
            agent.submit(user_input)
        except KeyboardInterrupt:
            agent.cancelled = True
            hooks.reset_stream()
            _print(f"\n{ANSI_DIM}[Cancelled]{ANSI_RESET}\n")
            if hasattr(agent, "messages") and agent.messages and agent.messages[-1].get("role") == "user":
                agent.messages.pop()
        except Exception as exc:
            hooks.on_error(f"{type(exc).__name__}: {exc}")


# --- Legacy Compatibility Classes (Textual TUI) ------------------------------

class PermissionModal(ModalScreen[bool]):
    """Compatibility modal screen for Textual."""

    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    DEFAULT_CSS = """
    PermissionModal {
        align: center middle;
        background: $background 70%;
    }
    #perm-dialog {
        width: 60;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: heavy $warning;
        padding: 1 2;
    }
    #perm-title {
        text-style: bold;
        color: $warning;
    }
    #perm-name {
        margin-top: 1;
        text-style: bold;
    }
    #perm-args {
        margin-top: 1;
        color: $text-muted;
    }
    #perm-buttons {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    PermissionModal Button {
        margin-left: 2;
    }
    """

    def __init__(self, tool_name: str, args: Any) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args

    def compose(self) -> ComposeResult:
        with Vertical(id="perm-dialog"):
            yield Static("Confirm Tool Execution", id="perm-title")
            yield Static(f"Tool: {self._tool_name}", id="perm-name")
            yield Static(_pretty_json(self._args), id="perm-args")
            with Horizontal(id="perm-buttons"):
                yield Button("Allow (y)", variant="success", id="allow")
                yield Button("Deny (n)", variant="error", id="deny")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


class ChatApp(App[None]):
    """Compatibility Textual App export."""

    TITLE = "opencode-lite"

    def __init__(self, agent: Any, config: Any) -> None:
        super().__init__()
        self.agent = agent
        self.config = config
        self._busy = False
        self.hooks = TerminalHooks()
        if hasattr(agent, "hooks"):
            agent.hooks = self.hooks

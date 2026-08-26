"""Pure Native Terminal REPL and UI for assistant."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, TextIO

# Heavy UI deps removed for lightweight startup – pure ANSI by default
# Legacy Textual/Rich support is opt-in; keep import <0.1s
Text = None  # type: ignore
App = ComposeResult = Binding = Horizontal = Vertical = ModalScreen = Button = Input = RichLog = Static = None  # type: ignore

try:
    from assistant.agent import Hooks
except ImportError:
    class Hooks:  # type: ignore[no-redef]
        def on_start(self) -> None: ...
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
    while isinstance(args, dict) and "arguments" in args and len(args) == 1:
        inner = args["arguments"]
        if isinstance(inner, dict):
            args = inner
        else:
            break
    if "command" in args:
        return str(args["command"])
    if "path" in args:
        start = args.get("start_line")
        pattern = args.get("pattern")
        if start:
            return f"{args['path']}:{start}"
        if pattern:
            return f"{args['path']} [{pattern}]"
        return str(args["path"])
    if "url" in args:
        return str(args["url"])
    if "query" in args:
        return str(args["query"])
    for v in args.values():
        if isinstance(v, str) and len(v) < 60:
            return v
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


_UI_TOOL_LEAK_RE = re.compile(
    r"<\/?(?:tools|tool_calls?|function_calls?|actions?)>|"
    r"\[\/?(?:TOOL_CALLS?|TOOLS|ACTIONS?)\]|"
    r">\s*tool_calls?\b|"
    r"\btool_calls?\s*:\s*\[\s*\{|"
    r"\{\s*\"type\"\s*:\s*\"function\"|"
    r"\{\s*\"n+a+m+e+\"\s*:\s*\"[a-zA-Z0-9_]+\"\s*,\s*\"arguments\"",
    re.IGNORECASE,
)


def _is_tool_json_display(s: str) -> bool:
    if not s:
        return False
    if _UI_TOOL_LEAK_RE.search(s):
        return True
    t = s.strip()
    if t.startswith(">"):
        t = t.lstrip("> \t").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-z]*\s*\n?", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\n?```\s*$", "", t).strip()
    if t.startswith("{") and ('"name":' in t or '"name" :' in t or '"naame":' in t) and ('"arguments"' in t or '"parameters"' in t):
        return True
    return False


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
    # Always consider both openers and closer, so a partial closer split
    # across the boundary like "</thin" + "k>" is correctly held even when
    # the current chunk also contains an opener at its start.
    candidates: tuple[str, ...] = _TAG_OPENERS + (_TAG_CLOSER_THINK,)
    limit = min(max(len(c) for c in candidates) - 1, len(buffer))
    for h in range(limit, 0, -1):
        tail = buffer[-h:].lower()
        if any(c.startswith(tail) and c != tail for c in candidates):
            return h
    if not in_think and at_line_start and buffer == "*":
        return 1
    return 0


def _parse_thinking_text(text: str, state: dict[str, bool]):  # type: ignore[no-untyped-def]
    """Parse text into a Rich Text object, highlighting <think>...</think>
    and [Thinking:...] sections in italic dim cyan/gray. Lazy: returns str if rich not available."""
    if Text is None:
        return text  # fallback for lightweight mode
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
    """Pure terminal hooks: single-line \\r spinner, no vertical cursor moves.
    Reasoning preview is shown inline on the spinner line itself (truncated),
    so the header never needs to be rewritten while the answer streams below.
    This is the only animation model that is 100% stable on Windows conhost,
    PowerShell and dumb terminals."""

    def __init__(
        self,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        input_fn: Callable[[str], str] | None = None,
        verbose: bool = False,
        config: Any | None = None,
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
        self._spinner_idx: int = 0
        self._spinner_frozen: bool = False
        self._header_active: bool = False
        self._thinking_text: str = ""
        self._current_tool: tuple[str, str] | None = None
        self._last_char_was_newline: bool = True
        self._at_line_start: bool = True
        self._in_roleplay_asterisk: bool = False
        self._ai_prefix_printed: bool = False
        self.had_error: bool = False
        self._workspace: Path | None = None
        if config is not None and hasattr(config, "workspace"):
            try:
                self._workspace = Path(config.workspace)
            except Exception:
                self._workspace = None
        self._current_args: dict | None = None
        self._pending_thread: threading.Thread | None = None
        self._pending_stop = threading.Event()
        self._pending_lock = threading.Lock()
        # Typewriter smoothing: char-by-char <0.1s eye-comfort, stdlib only
        self._tw_queue: deque[str] = deque()
        self._tw_lock = threading.Lock()
        self._tw_cond = threading.Condition(self._tw_lock)
        self._tw_thread: threading.Thread | None = None
        self._tw_stop = threading.Event()
        self._tw_enabled: bool = False
        try:
            # Enable only for real TTY (StringIO tests stay sync for determinism)
            self._tw_enabled = bool(getattr(self._stdout, "isatty", lambda: False)())
        except Exception:
            self._tw_enabled = False
        if config is not None and hasattr(config, "verbose"):
            self.verbose: bool = bool(config.verbose)
        else:
            self.verbose: bool = bool(verbose)

    def _format_verbose(self, turn) -> str:
        stats = getattr(turn, "stats", None)
        if stats is None and isinstance(turn, dict):
            stats = turn.get("stats")
        if not stats:
            return "(no stats)"

        def _get(key: str, default=None):
            if isinstance(stats, dict):
                if key in stats:
                    return stats[key]
                ollama = stats.get("ollama")
                if isinstance(ollama, dict) and key in ollama:
                    return ollama[key]
                return default
            else:
                if hasattr(stats, key):
                    return getattr(stats, key)
                ollama = getattr(stats, "ollama", None)
                if ollama is not None:
                    if isinstance(ollama, dict) and key in ollama:
                        return ollama[key]
                    if hasattr(ollama, key):
                        return getattr(ollama, key)
                return default

        parts = ["verbose"]
        wall = _get("wall_duration_s")
        if wall is not None:
            try:
                parts.append(f"wall {float(wall):.1f}s")
            except Exception:
                pass
        prompt_count = _get("prompt_eval_count")
        prompt_dur = _get("prompt_eval_duration")
        if prompt_count is not None:
            try:
                pc = int(prompt_count)
                if prompt_dur is not None:
                    pd_s = float(prompt_dur) / 1e9 if float(prompt_dur) > 1e6 else float(prompt_dur)
                    parts.append(f"prompt {pc} tok ({pd_s:.2f}s)")
                else:
                    parts.append(f"prompt {pc} tok")
            except Exception:
                pass
        eval_count = _get("eval_count")
        eval_dur = _get("eval_duration")
        if eval_count is not None:
            try:
                ec = int(eval_count)
                if eval_dur is not None:
                    ed_s = float(eval_dur) / 1e9 if float(eval_dur) > 1e6 else float(eval_dur)
                    if ed_s > 0:
                        tok_per_s = ec / ed_s
                        parts.append(f"eval {ec} tok {tok_per_s:.2f}tok/s")
                    else:
                        parts.append(f"eval {ec} tok")
                else:
                    parts.append(f"eval {ec} tok")
            except Exception:
                pass
        total_dur = _get("total_duration")
        if total_dur is not None:
            try:
                td = float(total_dur)
                td_s = td / 1e9 if td > 1e6 else td
                if td_s >= 1:
                    parts.append(f"total {td_s:.1f}s")
                else:
                    parts.append(f"total {td_s*1000:.1f}ms")
            except Exception:
                pass
        load_dur = _get("load_duration")
        if load_dur is not None:
            try:
                ld = float(load_dur)
                if ld > 1e6:
                    parts.append(f"load {ld/1e6:.1f}ms")
                elif ld > 1e3:
                    parts.append(f"load {ld:.1f}ms")
                else:
                    parts.append(f"load {ld:.2f}s")
            except Exception:
                pass
        if prompt_count is None and eval_count is None:
            approx = _get("approx_tokens")
            if approx is None:
                approx = _get("approx_tokens_before")
            if approx is None:
                approx = _get("approx_tokens_after")
            if approx is not None:
                try:
                    parts.append(f"~{int(approx)} tok")
                except Exception:
                    pass
        if len(parts) == 1:
            return "(no stats)"
        return " | ".join(parts)

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

    # -- typewriter smoothing (char-by-char <0.1s, eye-comfort, lightweight) --
    def _tw_ensure_thread(self) -> None:
        if not self._tw_enabled or self._tw_thread is not None and self._tw_thread.is_alive():
            return
        self._tw_stop.clear()
        self._tw_thread = threading.Thread(target=self._tw_loop, daemon=True)
        self._tw_thread.start()

    def _tw_loop(self) -> None:
        while not self._tw_stop.is_set():
            chunk = None
            with self._tw_cond:
                while not self._tw_queue and not self._tw_stop.is_set():
                    self._tw_cond.wait(timeout=0.05)
                if self._tw_stop.is_set() and not self._tw_queue:
                    break
                if self._tw_queue:
                    # adaptive chars per tick to keep <0.1s per char but drain bursts fast
                    qlen = len(self._tw_queue)
                    if qlen > 1000:
                        take = min(40, qlen // 50)
                    elif qlen > 200:
                        take = min(20, (qlen // 100) + 1)
                    else:
                        take = 1
                    chunk = "".join(self._tw_queue.popleft() for _ in range(min(take, len(self._tw_queue))))
            if chunk:
                try:
                    self._stdout.write(chunk)
                    self._stdout.flush()
                    self._last_char_was_newline = chunk.endswith("\n")
                except Exception:
                    pass
            time.sleep(0.015)  # ~15ms -> <0.1s per char, smooth

    def _tw_enqueue(self, text: str) -> None:
        if not text:
            return
        if not self._tw_enabled:
            # sync path for tests / non-TTY
            self._write_stdout(text)
            return
        self._tw_ensure_thread()
        with self._tw_cond:
            for ch in text:
                self._tw_queue.append(ch)
            self._tw_cond.notify_all()

    def _tw_drain_sync(self) -> None:
        if not self._tw_enabled:
            return
        # flush remaining queue synchronously
        while True:
            with self._tw_lock:
                if not self._tw_queue:
                    break
                chunk = "".join(self._tw_queue.popleft() for _ in range(len(self._tw_queue)))
            if chunk:
                try:
                    self._stdout.write(chunk)
                    self._stdout.flush()
                    self._last_char_was_newline = chunk.endswith("\n")
                except Exception:
                    pass

    def _tw_stop_thread(self) -> None:
        try:
            self._tw_stop.set()
            with self._tw_cond:
                self._tw_cond.notify_all()
            thr = self._tw_thread
            if thr is not None and thr.is_alive():
                thr.join(timeout=0.2)
        except Exception:
            pass
        self._tw_thread = None

    def _begin_header(self) -> None:
        if self._header_active:
            return
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        dots = "." * ((self._spinner_idx % 3) + 1)
        self._spinner_idx += 1
        self._write_stdout(f"\r\033[2K{ANSI_THINKING}{frame} Thinking{dots}{ANSI_RESET}")
        self._header_active = True
        self._thinking_text = ""

    def _update_header(self) -> None:
        if self._spinner_frozen or not self._header_active:
            return
        frame = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]
        dots = "." * ((self._spinner_idx % 3) + 1)
        self._spinner_idx += 1
        preview = self._thinking_text.replace("\n", " ").strip()
        cols = shutil.get_terminal_size().columns
        if cols <= 0:
            cols = 80
        # header "⠋ Thinking...: " ~ 14 chars + preview
        max_preview = max(0, cols - 24)
        if preview and max_preview:
            if len(preview) > max_preview:
                preview = "…" + preview[-(max_preview-1):]
            self._write_stdout(f"\r\033[2K{ANSI_THINKING}{frame} Thinking{dots}: {preview}{ANSI_RESET}")
        else:
            self._write_stdout(f"\r\033[2K{ANSI_THINKING}{frame} Thinking{dots}{ANSI_RESET}")

    def _collapse_thinking(self, keep_header: bool = True) -> None:
        if not self._header_active:
            self._saw_reasoning = False
            self._thinking_active = False
            return
        if keep_header:
            # Freeze to a static dim line above the answer
            self._write_stdout(f"\r\033[2K{ANSI_DIM}✓ Thinking{ANSI_RESET}\n")
        else:
            self._write_stdout(f"\r\033[2K")
        self._thinking_active = False
        self._saw_reasoning = False
        self._header_active = False
        self._thinking_text = ""
        self._last_char_was_newline = True
        self._at_line_start = True

    def _stop_pending(self) -> None:
        try:
            self._pending_stop.set()
            thr = self._pending_thread
            if thr is not None and thr.is_alive():
                thr.join(timeout=0.3)
        except Exception:
            pass
        self._pending_thread = None

    def _pending_loop(self) -> None:
        # Animates the single-line header while waiting for first token (TTFT)
        while not self._pending_stop.is_set():
            time.sleep(0.08)
            if self._pending_stop.is_set():
                break
            with self._pending_lock:
                if self._header_active and not self._spinner_frozen:
                    self._update_header()

    def on_start(self) -> None:
        # Immediate feedback: show spinner instantly, animate in background
        if self._header_active:
            return
        self._spinner_frozen = False
        self._saw_reasoning = True
        self._thinking_active = True
        self._begin_header()
        try:
            self._pending_stop.clear()
            self._pending_thread = threading.Thread(target=self._pending_loop, daemon=True)
            self._pending_thread.start()
        except Exception:
            pass

    def reset_stream(self) -> None:
        self._tw_drain_sync()
        self._tw_stop_thread()
        self._stop_pending()
        if self._header_active and not self._spinner_frozen:
            self._collapse_thinking(keep_header=False)
        elif self._header_active and self._spinner_frozen:
            # Completed turn's frozen header is already a normal line; just clear state
            self._header_active = False
        self._in_think = False
        self._in_bracket = False
        self._buffer = ""
        self._saw_reasoning = False
        self._thinking_active = False
        self._thinking_text = ""
        self._spinner_frozen = False
        self._current_tool = None
        self._at_line_start = True
        self._in_roleplay_asterisk = False
        self._ai_prefix_printed = False

    def on_reasoning(self, text: str) -> None:
        if not text:
            return
        self._stop_pending()
        self._spinner_frozen = False
        self._saw_reasoning = True
        self._thinking_active = True
        if not self._header_active:
            self._begin_header()
        # accumulate and refresh single-line preview
        self._thinking_text += text
        # cap memory
        if len(self._thinking_text) > 2000:
            self._thinking_text = self._thinking_text[-2000:]
        self._update_header()

    def on_delta(self, text: str) -> None:
        if not text:
            return
        # First real token ends the pending TTFT animation
        if self._pending_thread is not None:
            self._stop_pending()
        if self._saw_reasoning or (self._thinking_active and not self._in_think and not self._in_bracket):
            self._collapse_thinking(keep_header=True)

        self._buffer += text
        if ('"name"' in self._buffer and '"arguments"' in self._buffer) or _is_tool_json_display(self._buffer):
            self._buffer = ""
            return

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
                    # Thinking text -> update single-line preview, not a new line
                    self._thinking_text += joined
                    if len(self._thinking_text) > 2000:
                        self._thinking_text = self._thinking_text[-2000:]
                    self._update_header()
                else:
                    # hide leaked tool JSON (model dumped tool call as text)
                    if _is_tool_json_display(joined):
                        cur.clear()
                        return
                    if not self._ai_prefix_printed and not self._in_think and not self._in_bracket:
                        clean_check = joined.lstrip("\r\n")
                        if clean_check:
                            # also check after stripping AI prefix
                            if _is_tool_json_display(clean_check):
                                cur.clear()
                                return
                            self._write_stdout(AI_PREFIX)
                            self._ai_prefix_printed = True
                            joined = clean_check
                    if _is_tool_json_display(joined):
                        cur.clear()
                        return
                    self._tw_enqueue(joined)
                cur.clear()

        while i < n:
            ch = chunk[i]
            if not self._in_think and not self._in_bracket:
                lower = chunk[i:].lower()
                if lower.startswith("<think>"):
                    flush_cur()
                    self._in_think = True
                    self._thinking_active = True
                    if not self._header_active:
                        self._begin_header()
                    i += 7
                    continue
                elif lower.startswith("[thinking:"):
                    flush_cur()
                    self._in_bracket = True
                    self._thinking_active = True
                    if not self._header_active:
                        self._begin_header()
                    i += 10
                    continue
                if ch == "*":
                    if i + 1 < n and chunk[i + 1] == "*":
                        i += 2
                        continue
                    elif self._at_line_start and i + 1 < n and chunk[i + 1] == " ":
                        cur.append("- ")
                        self._at_line_start = False
                        i += 2
                        continue
                    else:
                        i += 1
                        continue
                if ch == "\n":
                    self._at_line_start = True
                elif ch != " " and ch != "\r":
                    self._at_line_start = False
                cur.append(ch)
                i += 1
            elif self._in_think:
                lower = chunk[i:].lower()
                if lower.startswith("</think>"):
                    flush_cur(ANSI_THINKING)
                    self._in_think = False
                    self._collapse_thinking(keep_header=True)
                    i += 8
                else:
                    cur.append(ch)
                    i += 1
            elif self._in_bracket:
                if ch == "]":
                    flush_cur(ANSI_THINKING)
                    self._in_bracket = False
                    self._collapse_thinking(keep_header=True)
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
        self._stop_pending()
        if self._buffer:
            buf = self._buffer
            self._buffer = ""
            self._render_chunk(buf)
        self._tw_drain_sync()
        if self._in_think or self._in_bracket or self._saw_reasoning or self._thinking_active:
            self._collapse_thinking(keep_header=True)
            self._in_think = False
            self._in_bracket = False
        self._spinner_frozen = True
        self._at_line_start = True
        self._in_roleplay_asterisk = False
        self._ai_prefix_printed = False
        if not self._last_char_was_newline:
            self._write_stdout("\n")
        if getattr(self, "verbose", False):
            try:
                line = self._format_verbose(turn)
            except Exception:
                line = "(no stats)"
            self._write_stdout(f"{ANSI_DIM}⏱ {line}{ANSI_RESET}\n")

    def on_tool_start(self, name: str, args: dict) -> None:
        """Clean inline tool execution indication: ⚡ [tool_name] args_summary."""
        summary = _format_args_summary(name, args)
        self._current_tool = (name, summary)
        try:
            self._current_args = dict(args) if isinstance(args, dict) else None
        except Exception:
            self._current_args = None
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

        # Show file preview when path & content are modified (lightweight, no model cost)
        if ok and self._current_args is not None and isinstance(self._current_args, dict) and "path" in self._current_args and "content" in self._current_args:
            try:
                pth = self._current_args.get("path")
                if pth:
                    base = self._workspace if self._workspace is not None else Path.cwd()
                    # Resolve safely inside workspace if possible
                    try:
                        fp = (base / str(pth)).resolve()
                        # Only show if inside workspace or workspace unknown
                        show = True
                        if self._workspace is not None:
                            try:
                                fp.relative_to(base.resolve())
                            except Exception:
                                show = False
                        if show and fp.is_file():
                            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
                            preview_n = min(20, len(lines))
                            if preview_n:
                                self._write_stdout(f"{ANSI_DIM}── {fp.name} preview ({preview_n}/{len(lines)} lines) ──{ANSI_RESET}\n")
                                for idx, line in enumerate(lines[:20], start=1):
                                    # Truncate long lines
                                    if len(line) > 120:
                                        line = line[:117] + "..."
                                    self._write_stdout(f"{ANSI_DIM}{idx:>3}│{ANSI_RESET} {line}\n")
                                if len(lines) > 20:
                                    self._write_stdout(f"{ANSI_DIM}... +{len(lines)-20} more lines{ANSI_RESET}\n")
                    except Exception:
                        pass
            except Exception:
                pass
        self._current_args = None
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
        self._stop_pending()
        # Ensure Thinking header doesn't stay stuck next to ERROR
        if self._header_active:
            try:
                self._collapse_thinking(keep_header=False)
            except Exception:
                pass
            # Ensure next output starts on fresh line
            if not self._last_char_was_newline:
                self._write_stdout("\n")
        self.had_error = True
        self._write_stderr(f"{ANSI_BOLD_RED}ERROR:{ANSI_RESET} {ANSI_RED}{msg}{ANSI_RESET}\n")


UiHooks = TerminalHooks


def _get_help_text() -> str:
    return (
        f"\n"
        f"{ANSI_BOLD}================================ ASSISTANT GUIDE ================================{ANSI_RESET}\n"
        f"\n"
        f"  {ANSI_BOLD}COMMANDS:{ANSI_RESET}\n"
        f"    {ANSI_CYAN}/help{ANSI_RESET}            Display this comprehensive reference guide\n"
        f"    {ANSI_CYAN}/clear{ANSI_RESET}           Clear screen & reset conversation context memory\n"
        f"    {ANSI_CYAN}/cls{ANSI_RESET}             Clear terminal screen view only\n"
        f"    {ANSI_CYAN}/model [name]{ANSI_RESET}    View active model or switch to another Ollama model on the fly\n"
        f"    {ANSI_CYAN}/status{ANSI_RESET}          Show current workspace, model, base URL, and permissions\n"
        f"    {ANSI_CYAN}/verbose [on|off]{ANSI_RESET} Toggle verbose stats display (on/off/status)\n"
        f"    {ANSI_CYAN}/session [list|save <name>|load <name>|new|delete <name>]{ANSI_RESET}  Manage sessions (pruned history saved to ~/.assistant/sessions/)\n"
        f"    {ANSI_CYAN}/exit, /quit{ANSI_RESET}     Exit assistant\n"
        f"\n"
        f"  {ANSI_BOLD}KEYBOARD SHORTCUTS:{ANSI_RESET}\n"
        f"    {ANSI_BOLD}Enter{ANSI_RESET}            Send prompt to the agent\n"
        f"    {ANSI_BOLD}Ctrl+C{ANSI_RESET}           Cancel active streaming generation / reset prompt\n"
        f"    {ANSI_BOLD}Ctrl+D / Ctrl+Z{ANSI_RESET}  Exit assistant (EOF)\n"
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
        f"    Configuration file: ~/.assistant/config.toml\n"
        f"    Tool permissions can be set to: 'allow' | 'ask' | 'deny'\n"
        f"    Verbose stats use Ollama built-in when available (otherwise local estimate)\n"
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
        from assistant.config import Config
        config = Config.load()

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
    _print(f"{ANSI_DIM}assistant {VERSION} | model: {model} | workspace: {workspace}{ANSI_RESET}")
    _print(f"{ANSI_DIM}Type /help for commands, /exit to quit.{ANSI_RESET}\n")

    hooks = TerminalHooks(input_fn=_read_input, config=config, verbose=bool(getattr(config, "verbose", False)) if config is not None else False)
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
            elif cmd == "/verbose":
                cur = bool(getattr(hooks, "verbose", False) if hasattr(hooks, "verbose") else getattr(config, "verbose", False) if config is not None and hasattr(config, "verbose") else False)
                if not arg:
                    new_val = not cur
                    if config is not None and hasattr(config, "verbose"):
                        config.verbose = new_val
                    hooks.verbose = new_val
                    _print(f"{ANSI_DIM}Verbose: {'on' if new_val else 'off'}{ANSI_RESET}")
                elif arg.lower() in ("on", "true", "1", "yes", "enable"):
                    if config is not None and hasattr(config, "verbose"):
                        config.verbose = True
                    hooks.verbose = True
                    _print(f"{ANSI_DIM}Verbose: on{ANSI_RESET}")
                elif arg.lower() in ("off", "false", "0", "no", "disable"):
                    if config is not None and hasattr(config, "verbose"):
                        config.verbose = False
                    hooks.verbose = False
                    _print(f"{ANSI_DIM}Verbose: off{ANSI_RESET}")
                elif arg.lower() in ("status", "show"):
                    _print(f"{ANSI_DIM}Verbose: {'on' if cur else 'off'}{ANSI_RESET}")
                else:
                    _print(f"{ANSI_RED}Usage: /verbose [on|off|status]{ANSI_RESET}")
                continue
            elif cmd in ("/session", "/sessions"):
                try:
                    from assistant import session as _sess  # type: ignore
                except ImportError:
                    try:
                        import assistant.session as _sess  # type: ignore
                    except ImportError as _e:
                        hooks.on_error(f"session module not available: {_e}")
                        continue
                # Parse subcommand
                sub_arg = arg.strip()
                if not sub_arg:
                    # Status view
                    try:
                        cnt = len(getattr(agent, "messages", []))
                        try:
                            toks = agent._approx_tokens() if hasattr(agent, "_approx_tokens") else 0
                        except Exception:
                            toks = 0
                        saved = _sess.list_sessions()
                        _print(f"{ANSI_DIM}Session: {cnt} messages, ~{toks} tokens{ANSI_RESET}")
                        _print(f"{ANSI_DIM}Saved sessions: {len(saved)} on disk{ANSI_RESET}")
                        if saved:
                            _print(f"{ANSI_DIM}Saved: {', '.join(saved)}{ANSI_RESET}")
                        _print(f"{ANSI_DIM}Hint: /session save <name> | load <name> | list | new | delete <name>{ANSI_RESET}")
                    except Exception as exc:
                        hooks.on_error(str(exc))
                    continue
                # split subcommand and remainder name
                _parts = sub_arg.split(maxsplit=1)
                sub = _parts[0].lower()
                rest = _parts[1].strip() if len(_parts) > 1 else ""
                if sub in ("list", "ls"):
                    try:
                        sessions = _sess.list_sessions()
                        if not sessions:
                            _print(f"{ANSI_DIM}(no saved sessions){ANSI_RESET}")
                        else:
                            for s in sessions:
                                _print(f"{ANSI_DIM}{s}{ANSI_RESET}")
                    except Exception as exc:
                        hooks.on_error(str(exc))
                    continue
                elif sub == "save":
                    if not rest:
                        hooks.on_error("Usage: /session save <name>")
                        _print(f"{ANSI_RED}Usage: /session save <name>{ANSI_RESET}")
                        continue
                    try:
                        if hasattr(agent, "save_session"):
                            agent.save_session(rest)
                        else:
                            _sess.save_session(rest, getattr(agent, "messages", []))
                        n = len(getattr(agent, "messages", []))
                        _print(f"{ANSI_DIM}Session saved: {rest} ({n} messages){ANSI_RESET}")
                    except (ValueError, FileNotFoundError, OSError) as exc:
                        hooks.on_error(str(exc))
                    except Exception as exc:
                        hooks.on_error(f"{type(exc).__name__}: {exc}")
                    continue
                elif sub == "load":
                    if not rest:
                        hooks.on_error("Usage: /session load <name>")
                        _print(f"{ANSI_RED}Usage: /session load <name>{ANSI_RESET}")
                        continue
                    try:
                        if hasattr(agent, "load_session"):
                            agent.load_session(rest)
                        else:
                            msgs = _sess.load_session(rest)
                            agent.messages = msgs
                            if hasattr(agent, "cancelled"):
                                agent.cancelled = False
                        n = len(getattr(agent, "messages", []))
                        _print(f"{ANSI_DIM}Session loaded: {rest} ({n} messages){ANSI_RESET}")
                    except FileNotFoundError as exc:
                        hooks.on_error(str(exc))
                    except ValueError as exc:
                        hooks.on_error(str(exc))
                    except Exception as exc:
                        hooks.on_error(f"{type(exc).__name__}: {exc}")
                    continue
                elif sub in ("new", "clear"):
                    try:
                        _clear_screen()
                        if hasattr(agent, "new_session"):
                            agent.new_session()
                        elif hasattr(agent, "reset"):
                            agent.reset()
                        elif hasattr(agent, "messages"):
                            try:
                                agent.messages.clear()
                            except Exception:
                                agent.messages = []
                        _print(f"{ANSI_DIM}New session started{ANSI_RESET}")
                    except Exception as exc:
                        hooks.on_error(str(exc))
                    continue
                elif sub in ("delete", "rm", "remove"):
                    if not rest:
                        hooks.on_error("Usage: /session delete <name>")
                        _print(f"{ANSI_RED}Usage: /session delete <name>{ANSI_RESET}")
                        continue
                    try:
                        _sess.delete_session(rest)
                        _print(f"{ANSI_DIM}Session deleted: {rest}{ANSI_RESET}")
                    except FileNotFoundError as exc:
                        hooks.on_error(str(exc))
                    except ValueError as exc:
                        hooks.on_error(str(exc))
                    except Exception as exc:
                        hooks.on_error(f"{type(exc).__name__}: {exc}")
                    continue
                elif sub == "help":
                    _print(f"{ANSI_DIM}Usage:{ANSI_RESET}")
                    _print(f"  {ANSI_CYAN}/session{ANSI_RESET}               Show current session status")
                    _print(f"  {ANSI_CYAN}/session list{ANSI_RESET}          List saved sessions")
                    _print(f"  {ANSI_CYAN}/session save <name>{ANSI_RESET}   Save current session")
                    _print(f"  {ANSI_CYAN}/session load <name>{ANSI_RESET}   Load saved session")
                    _print(f"  {ANSI_CYAN}/session new{ANSI_RESET}           Start new session (clear)")
                    _print(f"  {ANSI_CYAN}/session delete <name>{ANSI_RESET} Delete saved session")
                    _print(f"  {ANSI_DIM}Sessions stored in ~/.assistant/sessions/{ANSI_RESET}")
                    continue
                else:
                    hooks.on_error(f"Unknown /session subcommand '{sub}'. Try /session help")
                    _print(f"{ANSI_RED}Unknown /session subcommand '{sub}'. Try /session help{ANSI_RESET}")
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
# Only define if textual is available; otherwise provide lightweight stubs
try:
    _has_textual = ModalScreen is not None and App is not None
except Exception:
    _has_textual = False

if _has_textual:
    class PermissionModal(ModalScreen[bool]):  # type: ignore
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

        def compose(self) -> ComposeResult:  # type: ignore
            with Vertical(id="perm-dialog"):  # type: ignore
                yield Static("Confirm Tool Execution", id="perm-title")  # type: ignore
                yield Static(f"Tool: {self._tool_name}", id="perm-name")  # type: ignore
                yield Static(_pretty_json(self._args), id="perm-args")  # type: ignore
                with Horizontal(id="perm-buttons"):  # type: ignore
                    yield Button("Allow (y)", variant="success", id="allow")  # type: ignore
                    yield Button("Deny (n)", variant="error", id="deny")  # type: ignore

        def action_allow(self) -> None:
            self.dismiss(True)

        def action_deny(self) -> None:
            self.dismiss(False)

        def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore
            self.dismiss(event.button.id == "allow")

    class ChatApp(App[None]):  # type: ignore
        """Compatibility Textual App export."""

        TITLE = "assistant"

        def __init__(self, agent: Any, config: Any) -> None:
            super().__init__()
            self.agent = agent
            self.config = config
            self._busy = False
            self.hooks = TerminalHooks()
            if hasattr(agent, "hooks"):
                agent.hooks = self.hooks
else:
    class PermissionModal:  # type: ignore
        pass

    class ChatApp:  # type: ignore
        pass
        pass

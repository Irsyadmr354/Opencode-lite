"""Textual TUI for opencode-lite (targets Textual >= 1.0)."""

from __future__ import annotations

import datetime
import json
import threading
import time
from typing import Any

from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, RichLog, Static

try:  # sibling module built in parallel; fall back so this file imports standalone
    from opencode_lite.agent import Hooks  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - only before the agent module lands
    class Hooks:  # type: ignore[no-redef]
        """Duck-typed stand-in matching opencode_lite.agent.Hooks callbacks."""

        def on_delta(self, text: str) -> None: ...
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

TOOL_ICONS: dict[str, str] = {
    "shell": "⚙️",
    "bash": "⚙️",
    "powershell": "⚙️",
    "read_file": "📄",
    "view_file": "📄",
    "write_file": "✍️",
    "edit_file": "✍️",
    "delete_file": "🗑️",
    "rm": "🗑️",
    "list_files": "📂",
    "search_files": "🔍",
    "websearch": "🔍",
    "webfetch": "🌐",
    "fetch": "🌐",
}


def _compact_json(args: Any) -> str:
    try:
        return json.dumps(args, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(args)


def _pretty_json(args: Any, max_lines: int = ARGS_PREVIEW_LINES) -> str:
    try:
        rendered = json.dumps(args, indent=2, default=str)
    except (TypeError, ValueError):
        rendered = str(args)
    lines = rendered.splitlines() or ["{}"]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (+{len(lines) - max_lines} more lines)"]
    return "\n".join(lines)


def _get_tool_icon(name: str) -> str:
    return TOOL_ICONS.get(name.lower(), "🔧")


class PermissionModal(ModalScreen[bool]):
    """Ask the user to allow or deny a tool call. Dismisses with a bool."""

    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    DEFAULT_CSS = """
    PermissionModal {
        align: center middle;
        background: rgba(10, 15, 29, 0.85);
    }
    #perm-dialog {
        width: 72;
        max-width: 95%;
        height: auto;
        max-height: 85%;
        background: #0f172a;
        border: heavy #ef4444;
        padding: 1 2;
    }
    #perm-title {
        text-style: bold;
        color: #f87171;
        text-align: center;
        margin-bottom: 1;
    }
    #perm-name {
        color: #fbbf24;
        text-style: bold;
        margin-bottom: 1;
    }
    #perm-args {
        color: #cbd5e1;
        background: #090d16;
        border: solid #334155;
        padding: 0 1;
        max-height: 14;
        overflow-y: auto;
        margin-bottom: 1;
    }
    #perm-buttons {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    PermissionModal Button {
        margin-left: 2;
        min-width: 12;
    }
    #allow {
        background: #059669;
        color: #ffffff;
    }
    #allow:focus {
        background: #10b981;
    }
    #deny {
        background: #dc2626;
        color: #ffffff;
    }
    #deny:focus {
        background: #ef4444;
    }
    """

    def __init__(self, tool_name: str, args: Any) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args

    def compose(self) -> ComposeResult:
        icon = _get_tool_icon(self._tool_name)
        with Vertical(id="perm-dialog"):
            yield Static("⚠️  SECURITY PERMISSION REQUIRED", id="perm-title")
            yield Static(f"{icon} Tool: {self._tool_name}  [DANGEROUS OPERATION]", id="perm-name")
            json_str = _pretty_json(self._args)
            try:
                syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
                yield Static(syntax, id="perm-args")
            except Exception:
                yield Static(json_str, id="perm-args")
            with Horizontal(id="perm-buttons"):
                yield Button("Allow (y)", variant="success", id="allow")
                yield Button("Deny (n/Esc)", variant="error", id="deny")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


class ChatApp(App[None]):
    """Modern Cyberpunk / Terminal chat UI over an opencode_lite.agent.Agent."""

    TITLE = "⚡ OPENCODE-LITE"

    CSS = """
    Screen {
        background: #0b0f19;
        color: #f3f4f6;
    }
    #top-bar {
        height: 3;
        background: #111827;
        border-bottom: heavy #3b82f6;
        padding: 0 1;
        align: left middle;
    }
    #banner-logo {
        text-style: bold;
        color: #60a5fa;
        width: auto;
    }
    #banner-model {
        color: #34d399;
        background: #064e3b;
        padding: 0 1;
        margin-left: 2;
        text-style: bold;
    }
    #banner-workspace {
        color: #94a3b8;
        margin-left: 2;
        width: 1fr;
    }
    #chat {
        height: 1fr;
        padding: 1 1;
        background: #0b0f19;
        border: none;
        scrollbar-gutter: stable;
        scrollbar-color: #3b82f6 #111827;
    }
    #status {
        height: 1;
        padding: 0 1;
        color: #93c5fd;
        background: #1e293b;
        text-style: bold;
    }
    #input-container {
        height: auto;
        padding: 0 1 1 1;
        background: #111827;
        border-top: solid #1f2937;
    }
    #input {
        background: #1f2937;
        color: #f9fafb;
        border: tall #374151;
    }
    #input:focus {
        border: tall #3b82f6;
    }
    Footer {
        background: #111827;
        color: #9ca3af;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_or_quit", "Cancel/Quit", priority=True),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+l", "clear_log", "Clear Log"),
        Binding("ctrl+r", "reset_agent", "Reset Session"),
        Binding("f1", "show_help", "Help"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: Any, config: Any) -> None:
        super().__init__()
        self.agent = agent
        self.config = config
        self._busy = False
        self._req_start_time: float | None = None
        self._tokens_count: int = 0
        self._current_round: int = 0
        self._max_rounds: int = 0
        outer = self

        class UiHooks(Hooks):
            """Translate worker-thread callbacks into UI-thread updates."""

            def on_delta(self, text: str) -> None:
                outer._ui(lambda: outer._chat().write(text))

            def on_assistant_done(self, turn: Any) -> None:
                outer._ui(lambda: outer._chat().write(Text("")))

            def on_tool_start(self, name: str, args: dict) -> None:
                icon = _get_tool_icon(name)
                line = Text(f"{icon} >> {name} {_compact_json(args)}", style="bold cyan")
                outer._ui(lambda: outer._chat().write(line))

            def on_tool_result(self, name: str, res: Any) -> None:
                ok = bool(getattr(res, "ok", True))
                out = " ".join(str(getattr(res, "output", "")).split())
                preview = f"{out[:RESULT_PREVIEW_CHARS]}{'...' if len(out) > RESULT_PREVIEW_CHARS else ''}"
                prefix = "" if ok else "ERROR: "
                icon = "✨" if ok else "💥"
                line = Text(
                    f"{icon} <- {name}: {prefix}{preview}",
                    style="bold bright_red" if not ok else "dim bright_green",
                )
                outer._ui(lambda: outer._chat().write(line))

            def on_status(self, info: dict) -> None:
                model = getattr(outer.config, "model", "?")
                round_no = info.get("round", "?")
                max_rounds = info.get("max", "?")
                approx_tokens = info.get("approx_tokens", "?")
                try:
                    outer._tokens_count = int(approx_tokens)
                except (ValueError, TypeError):
                    pass
                outer._ui(
                    lambda: outer._set_status(
                        f"{model} | round {round_no}/{max_rounds} | ~{approx_tokens} tok"
                    )
                )

            def on_error(self, msg: str) -> None:
                outer._ui(
                    lambda: outer._chat().write(
                        Text(f"❌ ERROR: {msg}", style="bold bright_red")
                    )
                )

            def on_permission(self, name: str, args: dict) -> bool:
                done = threading.Event()
                box: dict[str, bool] = {"allow": False}

                def push_modal() -> None:
                    def cb(allowed: bool | None) -> None:
                        box["allow"] = bool(allowed)
                        done.set()

                    outer.push_screen(PermissionModal(name, args), cb)

                try:
                    outer.call_from_thread(push_modal)
                except Exception:  # app shutting down under us -> deny safely
                    return False
                done.wait()  # block the worker thread; UI stays responsive
                return box["allow"]

        self.UiHooks = UiHooks  # expose inner class (tests may introspect it)
        self.hooks: Hooks = UiHooks()
        # Auto-attach when possible so the agent is usable immediately;
        # __main__.py performs the same assignment defensively.
        if hasattr(agent, "hooks"):
            try:
                agent.hooks = self.hooks
            except Exception:  # noqa: BLE001 - read-only agent -> caller must wire
                pass

    # ------------------------------------------------------------------ UI

    def compose(self) -> ComposeResult:
        model = str(getattr(self.config, "model", "default"))
        workspace = str(getattr(self.config, "workspace", "."))
        with Horizontal(id="top-bar"):
            yield Static("⚡ OPENCODE-LITE", id="banner-logo")
            yield Static(f"📦 {model}", id="banner-model")
            yield Static(f"📁 {workspace}", id="banner-workspace")
        yield RichLog(id="chat", markup=False, highlight=False, wrap=True)
        yield Static("", id="status")
        with Vertical(id="input-container"):
            yield Input(placeholder="Type a message or /help, /clear, /reset, /exit...", id="input")
        yield Footer()

    def on_mount(self) -> None:
        model = getattr(self.config, "model", "?")
        workspace = getattr(self.config, "workspace", "?")
        log = self._chat()
        
        welcome_banner = (
            f"╔═══════════════════════════════════════════════════════════════════════════╗\n"
            f"║  ⚡ opencode-lite {VERSION:<7} | model: {str(model):<18} | workspace: {str(workspace):<14} ║\n"
            f"║  Commands: /help, /clear, /reset, /model, /exit | ctrl+c cancel | ctrl+q  ║\n"
            f"╚═══════════════════════════════════════════════════════════════════════════╝"
        )
        log.write(Text(welcome_banner, style="cyan bold"))
        log.write(Text(f"opencode-lite {VERSION} | model: {model} | workspace: {workspace}", style="dim"))
        log.write(Text("ctrl+c cancel | ctrl+q quit | ctrl+l clear view", style="dim"))
        self._set_status(f"{model} | idle")
        self.query_one("#input", Input).focus()

    def _chat(self) -> RichLog:
        return self.query_one("#chat", RichLog)

    def _status(self) -> Static:
        return self.query_one("#status", Static)

    def _input(self) -> Input:
        return self.query_one("#input", Input)

    def _set_status(self, text: str) -> None:
        self._status().update(text)

    def _model_label(self) -> str:
        return str(getattr(self.config, "model", "?"))

    def _ui(self, fn: Any) -> None:
        """Run fn on the UI thread; drop updates if the app already exited."""
        try:
            self.call_from_thread(fn)
        except Exception:  # noqa: BLE001 - app torn down mid-generation
            pass

    # ------------------------------------------------------------ actions

    def action_cancel_or_quit(self) -> None:
        if not self._cancel_generation():
            self.exit()

    def action_cancel_generation(self) -> None:
        self._cancel_generation()

    def action_clear_log(self) -> None:
        self._chat().clear()

    def action_reset_agent(self) -> None:
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        elif hasattr(self.agent, "messages"):
            self.agent.messages.clear()
        self._chat().write(Text("🔄 Session & conversation history reset.", style="bold magenta"))

    def action_show_help(self) -> None:
        help_text = (
            "💡 Available Commands & Shortcuts:\n"
            "  /help          - Show this help summary\n"
            "  /clear         - Clear chat viewport (ctrl+l)\n"
            "  /reset         - Reset agent memory & messages (ctrl+r)\n"
            "  /model <name>  - View or switch active model config\n"
            "  /exit, /quit   - Exit opencode-lite (ctrl+q)\n"
            "  ctrl+c         - Cancel active LLM generation or quit"
        )
        self._chat().write(Text(help_text, style="bold yellow"))

    def _cancel_generation(self) -> bool:
        """Flag the running generation as cancelled; returns True if one was active."""
        if not self._busy:
            return False
        cancelled = getattr(self.agent, "cancelled", None)
        if isinstance(cancelled, bool):
            self.agent.cancelled = True
        self._set_status(f"{self._model_label()} | cancelling...")
        return True

    # ----------------------------------------------------------- generate

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            event.input.value = ""
            return
        event.input.value = ""

        # Slash commands handling
        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/help", "/?"):
                self.action_show_help()
                return
            elif cmd in ("/clear", "/cls"):
                self.action_clear_log()
                return
            elif cmd in ("/reset", "/restart"):
                self.action_reset_agent()
                return
            elif cmd in ("/exit", "/quit"):
                self.exit()
                return
            elif cmd == "/model":
                if arg:
                    if hasattr(self.config, "model"):
                        self.config.model = arg
                    if hasattr(self.agent, "client") and hasattr(self.agent.client, "model"):
                        self.agent.client.model = arg
                    try:
                        self.query_one("#banner-model", Static).update(f"📦 {arg}")
                    except Exception:
                        pass
                    self._chat().write(Text(f"🎯 Model set to: {arg}", style="bold green"))
                else:
                    self._chat().write(Text(f"📦 Current model: {self._model_label()}", style="cyan"))
                return

        # Render styled user prompt
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        user_line = Text(f"\n💬 [{timestamp}] USER:\n{text}\n", style="bold bright_cyan")
        self._chat().write(user_line)

        self._start_generation(text)

    def _start_generation(self, text: str) -> None:
        self._busy = True
        inp = self._input()
        inp.disabled = True
        self._req_start_time = time.time()
        self._set_status(f"{self._model_label()} | thinking... (ctrl+c to cancel)")
        thread = threading.Thread(
            target=self._run_submit, args=(text,), name="agent-submit", daemon=True
        )
        thread.start()

    def _run_submit(self, text: str) -> None:
        try:
            self.agent.submit(text)
        except Exception as exc:  # noqa: BLE001 - surface any failure in the log
            try:
                self.hooks.on_error(f"{type(exc).__name__}: {exc}")
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._busy = False
            self._ui(self._finish_turn)

    def _finish_turn(self) -> None:
        inp = self._input()
        inp.disabled = False
        inp.focus()
        elapsed = ""
        if self._req_start_time:
            dur = time.time() - self._req_start_time
            elapsed = f" in {dur:.2f}s"
            self._req_start_time = None
        self._set_status(f"{self._model_label()} | idle{elapsed}")

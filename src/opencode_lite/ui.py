"""Ultra-minimal terminal interface for opencode-lite (pure, transparent CLI feel)."""

from __future__ import annotations

import json
import threading
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, RichLog, Static

try:
    from opencode_lite.agent import Hooks
except ImportError:
    class Hooks:
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


class PermissionModal(ModalScreen[bool]):
    """Minimal confirmation modal for dangerous tool operations."""

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
    """Pure minimal, raw CLI-feeling interface with comprehensive /help."""

    TITLE = "opencode-lite"

    CSS = """
    Screen {
        background: transparent;
        color: $text;
        layout: vertical;
    }
    #chat {
        height: 1fr;
        padding: 0 1;
        background: transparent;
        border: none;
        scrollbar-size: 0 0;
    }
    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: transparent;
    }
    #input {
        background: transparent;
        color: $text;
        border: none;
        border-top: solid $surface-lighten-1;
        padding: 0 1;
    }
    #input:focus {
        border: none;
        border-top: solid $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_or_quit", "Cancel/Quit", priority=True),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+r", "reset_agent", "Reset"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: Any, config: Any) -> None:
        super().__init__()
        self.agent = agent
        self.config = config
        self._busy = False
        outer = self

        class UiHooks(Hooks):
            def __init__(self) -> None:
                super().__init__()
                self._current_delta_parts: list[str] = []

            def on_delta(self, text: str) -> None:
                self._current_delta_parts.append(text)

            def on_assistant_done(self, turn: Any) -> None:
                full_text = "".join(self._current_delta_parts).strip()
                self._current_delta_parts.clear()
                if full_text:
                    outer._ui(lambda: outer._chat().write(Text(full_text + "\n")))

            def on_tool_start(self, name: str, args: dict) -> None:
                line = Text(f">> {name} {_compact_json(args)}", style="dim cyan")
                outer._ui(lambda: outer._chat().write(line))

            def on_tool_result(self, name: str, res: Any) -> None:
                ok = bool(getattr(res, "ok", True))
                out = " ".join(str(getattr(res, "output", "")).split())
                preview = f"{out[:RESULT_PREVIEW_CHARS]}{'...' if len(out) > RESULT_PREVIEW_CHARS else ''}"
                prefix = "" if ok else "ERROR: "
                line = Text(
                    f"<- {name}: {prefix}{preview}",
                    style="red" if not ok else "dim green",
                )
                outer._ui(lambda: outer._chat().write(line))

            def on_status(self, info: dict) -> None:
                model = getattr(outer.config, "model", "?")
                round_no = info.get("round", "?")
                max_rounds = info.get("max", "?")
                approx_tokens = info.get("approx_tokens", "?")
                outer._ui(
                    lambda: outer._set_status(
                        f"{model} | round {round_no}/{max_rounds} | ~{approx_tokens} tok"
                    )
                )

            def on_error(self, msg: str) -> None:
                outer._ui(lambda: outer._chat().write(Text(f"ERROR: {msg}", style="bold red")))

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
                except Exception:
                    return False
                done.wait()
                return box["allow"]

        self.UiHooks = UiHooks
        self.hooks: Hooks = UiHooks()
        if hasattr(agent, "hooks"):
            try:
                agent.hooks = self.hooks
            except Exception:
                pass

    def compose(self) -> ComposeResult:
        yield RichLog(id="chat", markup=False, highlight=False, wrap=True)
        yield Static("", id="status")
        yield Input(placeholder="> Type a prompt or /help...", id="input")

    def on_mount(self) -> None:
        model = getattr(self.config, "model", "?")
        workspace = getattr(self.config, "workspace", "?")
        log = self._chat()
        log.write(Text(f"opencode-lite {VERSION} | model: {model} | workspace: {workspace}", style="dim"))
        log.write(Text("Type /help for full guide and commands.\n", style="dim"))
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
        try:
            self.call_from_thread(fn)
        except Exception:
            pass

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
        self._chat().write(Text("Session memory reset.\n", style="dim cyan"))

    def action_show_help(self) -> None:
        help_guide = (
            "\n"
            "============================== OPENCODE-LITE GUIDE ==============================\n"
            "\n"
            "  COMMANDS:\n"
            "    /help            Display this comprehensive reference guide\n"
            "    /clear           Clear current chat viewport history\n"
            "    /reset           Reset conversation memory & start fresh session\n"
            "    /model <name>    View active model or switch to another Ollama model on the fly\n"
            "    /status          Show current workspace, server base URL, and configuration\n"
            "    /exit, /quit     Exit opencode-lite\n"
            "\n"
            "  KEYBOARD SHORTCUTS:\n"
            "    Enter            Send user prompt to the agent\n"
            "    Ctrl+C           Cancel active streaming generation (or quit if idle)\n"
            "    Ctrl+L           Clear chat screen viewport\n"
            "    Ctrl+R           Reset agent conversation memory\n"
            "    Ctrl+Q           Quit opencode-lite immediately\n"
            "\n"
            "  BUILT-IN TOOLS (AUTONOMOUS):\n"
            "    * read_file(path, start_line)   Read files in workspace with line numbers\n"
            "    * write_file(path, content)     Create or update files in workspace\n"
            "    * delete_file(path)             Permanently delete a file (gated by confirmation)\n"
            "    * list_files(path, pattern)     List workspace directory entries via glob\n"
            "    * shell(command)                Execute shell commands (gated by confirmation)\n"
            "    * webfetch(url)                 Fetch webpage text content directly\n"
            "    * websearch(query)              Search the web via DuckDuckGo\n"
            "\n"
            "  PERMISSIONS & CONFIGURATION:\n"
            "    Configuration lives at: ~/.opencode-lite/config.toml\n"
            "    Tool permissions can be set to: 'allow' | 'ask' | 'deny'\n"
            "==================================================================================\n"
        )
        self._chat().write(Text(help_guide, style="bright_cyan"))

    def _cancel_generation(self) -> bool:
        if not self._busy:
            return False
        cancelled = getattr(self.agent, "cancelled", None)
        if isinstance(cancelled, bool):
            self.agent.cancelled = True
        self._set_status(f"{self._model_label()} | cancelling...")
        return True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._busy:
            event.input.value = ""
            return
        event.input.value = ""

        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/help", "/?", "/h"):
                self.action_show_help()
                return
            elif cmd in ("/clear", "/cls"):
                self.action_clear_log()
                return
            elif cmd in ("/reset", "/restart"):
                self.action_reset_agent()
                return
            elif cmd in ("/exit", "/quit", "/q"):
                self.exit()
                return
            elif cmd == "/status":
                ws = getattr(self.config, "workspace", ".")
                mdl = self._model_label()
                url = getattr(self.config, "base_url", "http://127.0.0.1:11434/v1")
                self._chat().write(Text(f"Model: {mdl} | Base URL: {url}\nWorkspace: {ws}\n", style="dim"))
                return
            elif cmd == "/model":
                if arg:
                    if hasattr(self.config, "model"):
                        self.config.model = arg
                    if hasattr(self.agent, "client") and hasattr(self.agent.client, "model"):
                        self.agent.client.model = arg
                    self._chat().write(Text(f"Model switched to: {arg}\n", style="green"))
                    self._set_status(f"{arg} | idle")
                else:
                    self._chat().write(Text(f"Active model: {self._model_label()}\n", style="dim"))
                return

        self._chat().write(Text(f"> {text}\n", style="bold"))
        self._start_generation(text)

    def _start_generation(self, text: str) -> None:
        self._busy = True
        inp = self._input()
        inp.disabled = True
        self._set_status(f"{self._model_label()} | thinking...")
        thread = threading.Thread(
            target=self._run_submit, args=(text,), name="agent-submit", daemon=True
        )
        thread.start()

    def _run_submit(self, text: str) -> None:
        try:
            self.agent.submit(text)
        except Exception as exc:
            try:
                self.hooks.on_error(f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
        finally:
            self._busy = False
            self._ui(self._finish_turn)

    def _finish_turn(self) -> None:
        inp = self._input()
        inp.disabled = False
        inp.focus()
        self._set_status(f"{self._model_label()} | idle")

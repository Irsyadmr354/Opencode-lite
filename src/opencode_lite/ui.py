"""Textual TUI for opencode-lite (targets Textual >= 1.0)."""

from __future__ import annotations

import json
import threading
from typing import Any

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
    """Ask the user to allow or deny a tool call. Dismisses with a bool."""

    BINDINGS = [
        Binding("y", "allow", "Allow"),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny"),
    ]

    DEFAULT_CSS = """
    PermissionModal {
        align: center middle;
        background: $background 60%;
    }
    #perm-dialog {
        width: 64;
        max-width: 92%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #perm-title {
        text-style: bold;
    }
    #perm-name {
        color: $warning;
        margin-top: 1;
    }
    #perm-args {
        color: $text-muted;
        margin-top: 1;
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
            yield Static("Allow tool?", id="perm-title")
            yield Static(self._tool_name, id="perm-name")
            yield Static(_pretty_json(self._args), id="perm-args")
            with Horizontal(id="perm-buttons"):
                yield Button("Allow", variant="success", id="allow")
                yield Button("Deny", variant="error", id="deny")

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


class ChatApp(App[None]):
    """Minimal chat UI over an opencode_lite.agent.Agent."""

    TITLE = "opencode-lite"

    CSS = """
    #chat {
        height: 1fr;
        padding: 0 1;
    }
    #status {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $panel;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_or_quit", "Cancel/Quit", priority=True),
        Binding("escape", "cancel_generation", "Cancel", show=False),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: Any, config: Any) -> None:
        super().__init__()
        self.agent = agent
        self.config = config
        self._busy = False
        outer = self

        class UiHooks(Hooks):
            """Translate worker-thread callbacks into UI-thread updates."""

            def on_delta(self, text: str) -> None:
                outer._ui(lambda: outer._chat().write(text))

            def on_assistant_done(self, turn: Any) -> None:
                outer._ui(lambda: outer._chat().write(Text("")))

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
        yield Header(show_clock=False)
        yield RichLog(id="chat", markup=False, highlight=False, wrap=True)
        yield Static("", id="status")
        yield Input(placeholder="Ask, or describe a task...", id="input")
        yield Footer()

    def on_mount(self) -> None:
        model = getattr(self.config, "model", "?")
        workspace = getattr(self.config, "workspace", "?")
        log = self._chat()
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
        self._start_generation(text)

    def _start_generation(self, text: str) -> None:
        self._busy = True
        inp = self._input()
        inp.disabled = True
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
        self._set_status(f"{self._model_label()} | idle")

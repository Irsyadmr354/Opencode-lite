"""Command-line entry point for opencode-lite (TUI + headless modes)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

VERSION = "0.1.0"
DEFAULT_CONFIG_PATH = Path.home() / ".opencode-lite" / "config.toml"

SAMPLE_CONFIG = '''\
# opencode-lite configuration (sample, written once)
# All keys optional; unset values fall back to built-in defaults.

model = "qwen2.5-coder:7b"              # any model available in Ollama
base_url = "http://127.0.0.1:11434/v1"  # Ollama OpenAI-compatible endpoint
api_key = "ollama"                      # placeholder; local servers ignore it
max_tool_rounds = 12                    # default: 25
stream = true                           # stream assistant tokens
# workspace = ""                        # empty -> directory you launch from
# shell_cmd = ["powershell", "-NoProfile", "-Command"]  # Windows default

[permissions]
# one of: "allow" | "ask" | "deny"
write = "allow"                         # set "ask" to confirm every file write
delete = "ask"
shell = "ask"
webfetch = "allow"
websearch = "allow"

[limits]
# resource caps applied to tool outputs
# read_max_lines = 200
# shell_timeout_s = 120
# shell_output_chars = 6000
# webfetch_chars = 8000
# list_max_entries = 200
'''


def build_console_hooks(hooks_base: type) -> type:
    """Create the concrete ConsoleHooks class bound to the given Hooks base."""

    class ConsoleHooksImpl(hooks_base):
        def __init__(self) -> None:
            self.had_error = False

        def on_delta(self, text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        def on_reasoning(self, text: str) -> None:
            # keep stdout clean for the actual answer; thinking goes to stderr
            sys.stderr.write(text)
            sys.stderr.flush()

        def on_assistant_done(self, turn: Any) -> None:
            if isinstance(turn, dict):
                content = turn.get("content")
            elif isinstance(turn, str):
                content = turn
            else:
                content = getattr(turn, "text", "")
            if isinstance(content, str) and content and not content.endswith("\n"):
                print()

        def on_tool_start(self, name: str, args: dict) -> None:
            try:
                compact = json.dumps(args, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                compact = str(args)
            print(f"-> tool {name} {compact}", flush=True)

        def on_tool_result(self, name: str, res: Any) -> None:
            out = " ".join(str(getattr(res, "output", "")).split())
            marker = "" if getattr(res, "ok", True) else " [ERROR]"
            print(f"<- {name}: {out[:200]}{marker}", flush=True)

        def on_status(self, info: dict) -> None:
            round_no = info.get("round", "?")
            max_rounds = info.get("max", "?")
            approx_tokens = info.get("approx_tokens", "?")
            print(
                f"[status] round {round_no}/{max_rounds} ~{approx_tokens} tok",
                file=sys.stderr,
                flush=True,
            )

        def on_permission(self, name: str, args: dict) -> bool:
            if not sys.stdin.isatty():
                # Piped/non-interactive stdin (-p mode): never auto-authorize,
                # and keep stdout pure by not emitting the prompt at all.
                return False
            try:
                answer = input(f"Allow {name}? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
            return answer in {"y", "yes"}

        def on_error(self, msg: str) -> None:
            self.had_error = True
            print(f"ERROR: {msg}", file=sys.stderr, flush=True)

    return ConsoleHooksImpl


def load_runtime():
    """Import sibling modules; returns tuple or raises ImportError."""
    from opencode_lite.agent import Agent, Hooks
    from opencode_lite.config import load_config
    from opencode_lite.llm import LLMClient
    from opencode_lite.tools import get_tools

    return load_config, LLMClient, Agent, Hooks, get_tools


def ensure_sample_config(config_path: Path) -> None:
    """Write a commented sample config.toml once if the location is writable."""
    if config_path.exists():
        return
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if os.access(config_path.parent, os.W_OK):
            config_path.write_text(SAMPLE_CONFIG, encoding="utf-8")
            print(f"[info] sample config written: {config_path}", file=sys.stderr)
    except OSError:
        pass  # read-only home etc. -> keep built-in defaults silently


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencode-lite",
        description="Minimal TUI coding agent for local models (Ollama).",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print version and exit",
    )
    parser.add_argument("--model", help="model name, e.g. qwen2.5-coder:7b")
    parser.add_argument("--base-url", dest="base_url", metavar="URL", help="Ollama OpenAI-compatible base URL")
    parser.add_argument("--workspace", metavar="DIR", help="workspace directory (default: current directory)")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        metavar="PATH",
        help=f"config file path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print_mode",
        metavar="TEXT",
        help="headless mode: run TEXT non-interactively, print result, exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.version:
        print(f"opencode-lite {VERSION}")
        return 0

    # Sibling modules are built in parallel; fail with a clear message, never at import time.
    try:
        load_config, LLMClient, Agent, Hooks, get_tools = load_runtime()
    except ImportError as exc:
        print(f"ERROR: core modules not available yet ({exc}). Run 'pip install -e .' "
              f"and ensure all opencode_lite modules are present.", file=sys.stderr)
        return 2

    config_path = Path(ns.config).expanduser()
    if config_path == DEFAULT_CONFIG_PATH:
        ensure_sample_config(config_path)

    overrides: dict[str, Any] = {}
    if ns.model:
        overrides["model"] = ns.model
    if ns.base_url:
        overrides["base_url"] = ns.base_url
    overrides["workspace"] = Path(ns.workspace or ".").expanduser().resolve()

    try:
        config = load_config(str(config_path) if config_path.exists() else None, overrides or None)
    except Exception as exc:  # noqa: BLE001 - surface config errors cleanly
        print(f"ERROR: failed to load config {config_path}: {exc}", file=sys.stderr)
        return 2

    client = LLMClient(config.base_url, config.api_key, config.model)
    tools = get_tools(config.workspace, config)

    if ns.print_mode is not None:
        console_hooks = build_console_hooks(Hooks)()
        agent = Agent(client, tools, config, console_hooks)
        agent.submit(ns.print_mode)
        return 1 if console_hooks.had_error else 0

    from opencode_lite.ui import run_repl

    agent = Agent(client, tools, config, Hooks())
    try:
        run_repl(agent, config)
    except (KeyboardInterrupt, EOFError):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

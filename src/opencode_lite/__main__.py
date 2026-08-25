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
verbose = false                         # show Ollama performance stats after each turn
timeout_s = 600                         # request timeout in seconds (potato: 600)
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
        def __init__(self, verbose: bool = False) -> None:
            self.had_error = False
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
                if approx is not None:
                    try:
                        parts.append(f"~{int(approx)} tok")
                    except Exception:
                        pass
            if len(parts) == 1:
                return "(no stats)"
            return " | ".join(parts)

        def on_start(self) -> None:
            # Immediate TTFT feedback for headless mode
            print("… thinking…", file=sys.stderr, flush=True)

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
                if content is None:
                    content = getattr(turn, "content", "")
            if isinstance(content, str) and content and not content.endswith("\n"):
                print()
            if getattr(self, "verbose", False):
                try:
                    line = self._format_verbose(turn)
                except Exception:
                    line = "(no stats)"
                print(f"\033[2m⏱ {line}\033[0m", file=sys.stderr, flush=True)

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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="show Ollama performance stats after each response",
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
    if getattr(ns, "verbose", False):
        overrides["verbose"] = True
    overrides["workspace"] = Path(ns.workspace or ".").expanduser().resolve()

    try:
        config = load_config(str(config_path) if config_path.exists() else None, overrides or None)
    except Exception as exc:  # noqa: BLE001 - surface config errors cleanly
        print(f"ERROR: failed to load config {config_path}: {exc}", file=sys.stderr)
        return 2

    client = LLMClient(config.base_url, config.api_key, config.model, timeout_s=int(getattr(config, "timeout_s", 600)))
    tools = get_tools(config.workspace, config)

    if ns.print_mode is not None:
        console_hooks = build_console_hooks(Hooks)(verbose=bool(getattr(config, "verbose", False)))
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

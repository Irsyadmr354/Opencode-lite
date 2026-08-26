"""CLI entry point for assistant."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from assistant.config import load_config
from assistant.session import Session

VERSION = "0.2.0"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

SESSIONS_DIR = Path.home() / ".assistant" / "sessions"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner(config):
    clear_screen()
    print(f"{CYAN}{BOLD}Assistant{RESET} {DIM}v{VERSION} · {config.model} · {config.workspace}{RESET}")
    print(f"{DIM}commands: /sessions /clear /c-context /help exit{RESET}\n")


def cmd_sessions(session, agent):
    """Interactive session management."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{CYAN}--- Sessions ---{RESET}")
    print(f"  {BOLD}[s]{RESET} Save current session")
    print(f"  {BOLD}[l]{RESET} Load session")
    print(f"  {BOLD}[d]{RESET} Delete session")
    print(f"  {BOLD}[b]{RESET} Back to chat")
    print()

    try:
        choice = input(f"{DIM}choice: {RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(f"{RESET}\n{DIM}cancelled{RESET}")
        return

    if choice == "s":
        try:
            name = input(f"{DIM}session name: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"{RESET}\n{DIM}cancelled{RESET}")
            return
        if not name:
            print(f"{DIM}cancelled{RESET}")
            return
        sess_path = SESSIONS_DIR / f"{name}.json"
        session.path = sess_path
        session.save()
        print(f"{GREEN}saved to {sess_path}{RESET}")

    elif choice == "l":
        files = list(SESSIONS_DIR.glob("*.json"))
        if not files:
            print(f"{DIM}no saved sessions{RESET}")
            return
        print(f"\n{CYAN}Saved sessions:{RESET}")
        for i, f in enumerate(files, 1):
            print(f"  {BOLD}[{i}]{RESET} {f.stem}")
        print()
        try:
            idx = input(f"{DIM}pick number: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"{RESET}\n{DIM}cancelled{RESET}")
            return
        try:
            chosen = files[int(idx) - 1]
        except (ValueError, IndexError):
            print(f"{DIM}invalid{RESET}")
            return
        session.path = chosen
        loaded = Session.load(chosen)
        session.messages = loaded.messages
        # Reload agent context
        agent.clear_context()
        for msg in session.messages:
            if msg.get("role") in ("user", "assistant"):
                agent.messages.append(msg)
        print(f"{GREEN}loaded {chosen.stem} ({len(session.messages)} messages){RESET}")

    elif choice == "d":
        files = list(SESSIONS_DIR.glob("*.json"))
        if not files:
            print(f"{DIM}no saved sessions{RESET}")
            return
        print(f"\n{CYAN}Saved sessions:{RESET}")
        for i, f in enumerate(files, 1):
            print(f"  {BOLD}[{i}]{RESET} {f.stem}")
        print()
        try:
            idx = input(f"{DIM}pick number to delete: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"{RESET}\n{DIM}cancelled{RESET}")
            return
        try:
            chosen = files[int(idx) - 1]
        except (ValueError, IndexError):
            print(f"{DIM}invalid{RESET}")
            return
        try:
            confirm = input(f"{YELLOW}delete '{chosen.stem}'? [y/N]: {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"{RESET}\n{DIM}cancelled{RESET}")
            return
        if confirm == "y":
            chosen.unlink()
            print(f"{GREEN}deleted{RESET}")
        else:
            print(f"{DIM}cancelled{RESET}")


def print_help():
    print(f"\n{CYAN}--- Commands ---{RESET}")
    print(f"  {BOLD}/sessions{RESET}  save, load, delete sessions")
    print(f"  {BOLD}/clear{RESET}     clear screen only")
    print(f"  {BOLD}/c-context{RESET} clear conversation context + screen")
    print(f"  {BOLD}/help{RESET}      show this help")
    print(f"  {BOLD}exit{RESET}       quit")
    print()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="assistant — lightweight terminal coding agent")
    parser.add_argument("--version", action="version", version=f"assistant {VERSION}")
    parser.add_argument("--config", type=Path, default=None, help="path to config.json")
    parser.add_argument("--model", type=str, default=None, help="override model name")
    parser.add_argument("--workspace", type=Path, default=None, help="override workspace path")
    parser.add_argument("--no-stream", action="store_true", help="disable streaming")
    parser.add_argument("--verbose", action="store_true", default=None, help="enable Ollama generation stats")
    parser.add_argument("--no-verbose", action="store_true", help="disable Ollama generation stats")
    parser.add_argument("-c", "--command", type=str, default=None, help="headless: run one command and exit")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="headless: run one prompt and exit")
    args = parser.parse_args(argv)

    # Load config
    config_path = args.config or (Path.home() / ".assistant" / "config.json")
    config = load_config(config_path)

    # CLI overrides
    if args.model:
        config.model = args.model
    if args.workspace:
        config.workspace = args.workspace.resolve()
    if args.no_stream:
        config.stream = False
    if args.no_verbose:
        config.verbose = False
    elif args.verbose:
        config.verbose = True

    # Ensure workspace exists
    config.workspace.mkdir(parents=True, exist_ok=True)

    from assistant.agent import Agent
    agent = Agent(config)

    # Headless mode
    prompt = args.command if args.command is not None else args.prompt
    if prompt is not None:
        try:
            response = agent.handle(prompt)
        except KeyboardInterrupt:
            print(f"{RESET}\n{YELLOW}[Cancelled]{RESET}", file=sys.stderr)
            sys.exit(130)
        return

    # Interactive mode
    session = Session.load(config.workspace / ".assistant_session.json")
    print_banner(config)

    try:
        while True:
            try:
                user_input = input(f"{BOLD}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                try:
                    print(f"{RESET}\n{DIM}Goodbye!{RESET}")
                except Exception:
                    pass
                break
            if not user_input:
                continue

            # Commands
            cmd = user_input.lower()
            if cmd in ("exit", "quit", "/exit", "/quit"):
                try:
                    print(f"{RESET}\n{DIM}Goodbye!{RESET}")
                except Exception:
                    pass
                break
            elif cmd == "/clear":
                clear_screen()
                print_banner(config)
                continue
            elif cmd == "/c-context":
                agent.clear_context()
                session.messages = []
                session.save()
                clear_screen()
                print_banner(config)
                print(f"\n{GREEN}Context cleared.{RESET}")
                continue
            elif cmd == "/sessions":
                cmd_sessions(session, agent)
                continue
            elif cmd == "/help":
                print_help()
                continue

            try:
                response = agent.handle(user_input)

                session.append("user", user_input)
                session.append("assistant", response)
                session.save()
            except KeyboardInterrupt:
                try:
                    print(f"{RESET}\n{YELLOW}[Turn cancelled]{RESET}")
                except Exception:
                    pass
                continue

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        try:
            session.save()
            sys.stdout.write(RESET)
            sys.stdout.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()

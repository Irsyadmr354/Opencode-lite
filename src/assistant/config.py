"""JSON config loader for assistant."""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

_DEFAULT_PATH = pathlib.Path.home() / ".assistant" / "config.json"
_VALID_PERMS = {"allow", "ask", "deny"}


def get_default_shell_cmd() -> list[str]:
    """Return the default platform-aware shell command."""
    if sys.platform == "win32" or os.name == "nt":
        if shutil.which("powershell"):
            return ["powershell", "-NoProfile", "-Command"]
        if shutil.which("cmd.exe") or shutil.which("cmd"):
            return ["cmd.exe", "/c"]
        return ["powershell", "-NoProfile", "-Command"]
    else:
        if shutil.which("bash"):
            return ["bash", "-c"]
        if shutil.which("sh"):
            return ["sh", "-c"]
        return ["bash", "-c"]


@dataclass
class Limits:
    read_max_lines: int = 200
    shell_timeout_s: int = 120
    shell_output_chars: int = 6000
    webfetch_chars: int = 8000
    list_max_entries: int = 200


@dataclass
class Permissions:
    write: str = "allow"
    delete: str = "ask"
    shell: str = "ask"
    webfetch: str = "allow"
    websearch: str = "allow"


@dataclass
class Config:
    model: str = "qwen2.5-coder-3b-abliterated"
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = "ollama"
    workspace: pathlib.Path = field(default_factory=pathlib.Path.cwd)
    max_rounds: int = 12
    stream: bool = True
    verbose: bool = True
    timeout_s: int = 600
    shell_cmd: list[str] = field(default_factory=get_default_shell_cmd)
    limits: Limits = field(default_factory=Limits)
    permissions: Permissions = field(default_factory=Permissions)


def _apply_env(cfg: Config) -> None:
    """Override config fields from environment variables."""
    env_map: dict[str, tuple[str, type]] = {
        "ASSISTANT_MODEL": ("model", str),
        "ASSISTANT_BASE_URL": ("base_url", str),
        "ASSISTANT_API_KEY": ("api_key", str),
        "ASSISTANT_MAX_ROUNDS": ("max_rounds", int),
        "ASSISTANT_TIMEOUT_S": ("timeout_s", int),
        "ASSISTANT_WORKSPACE": ("workspace", pathlib.Path),
        "ASSISTANT_STREAM": ("stream", bool),
        "ASSISTANT_VERBOSE": ("verbose", bool),
    }
    for env_key, (attr, typ) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None and val.strip() != "":
            if typ == int:
                try:
                    setattr(cfg, attr, int(val))
                except ValueError:
                    raise ValueError(f"{attr} from {env_key} must be an integer, got {val!r}")
            elif typ == bool:
                setattr(cfg, attr, val.lower() in ("1", "true", "yes", "on"))
            elif typ == pathlib.Path:
                setattr(cfg, attr, pathlib.Path(val))
            else:
                setattr(cfg, attr, str(val))

    raw_shell = os.environ.get("ASSISTANT_SHELL_CMD")
    if raw_shell is not None and raw_shell.strip() != "":
        try:
            parsed = json.loads(raw_shell)
            if isinstance(parsed, list):
                cfg.shell_cmd = [str(x) for x in parsed]
            else:
                cfg.shell_cmd = [str(parsed)]
        except json.JSONDecodeError:
            cfg.shell_cmd = [raw_shell.strip()]


def _validate_int_fields(cfg: Config) -> None:
    """Ensure positive integers."""
    for name in ("max_rounds", "timeout_s"):
        val = getattr(cfg, name)
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive integer, got {val!r}")
    for name in (
        "read_max_lines",
        "shell_timeout_s",
        "shell_output_chars",
        "webfetch_chars",
        "list_max_entries",
    ):
        val = getattr(cfg.limits, name)
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"limits.{name} must be a positive integer, got {val!r}")


def _validate_permissions(perms: Permissions) -> None:
    """Ensure permission values are valid."""
    for name in ("write", "delete", "shell", "webfetch", "websearch"):
        val = getattr(perms, name, None)
        if val not in _VALID_PERMS:
            raise ValueError(
                f"permissions.{name} must be one of {_VALID_PERMS}, got {val!r}"
            )


def _deep_update(data: dict, target: Any, prefix: str = "") -> None:
    """Recursively update dataclass fields from a dict."""
    for key, val in data.items():
        if not hasattr(target, key):
            continue
        cur = getattr(target, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            _deep_update(val, cur, f"{prefix}{key}.")
        elif key == "workspace" and isinstance(val, str):
            setattr(target, key, pathlib.Path(val))
        elif key == "shell_cmd" and isinstance(val, str):
            setattr(target, key, [val])
        else:
            setattr(target, key, val)


def load_config(path: pathlib.Path | None = None) -> Config:
    """Load config from JSON file, with env overrides. Returns defaults if missing."""
    cfg = Config()
    path = path or _DEFAULT_PATH

    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _deep_update(data, cfg)
        except (json.JSONDecodeError, OSError):
            pass

    _apply_env(cfg)
    _validate_int_fields(cfg)
    _validate_permissions(cfg.permissions)

    return cfg

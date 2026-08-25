"""Configuration dataclasses and TOML/env/overrides loading for opencode-lite.

Config is pure data: [permissions] values are normalized and validated here
(allow|ask|deny), but enforcement of those decisions lives in the UI layer.
"""

from __future__ import annotations

import os
import pathlib
import tomllib
from dataclasses import dataclass, field, fields


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
    model: str = "qwen3.5-4b-uncensored"
    base_url: str = "http://127.0.0.1:11434/v1"
    api_key: str = "ollama"
    workspace: pathlib.Path = field(default_factory=pathlib.Path.cwd)
    max_tool_rounds: int = 25
    stream: bool = True
    limits: Limits = field(default_factory=Limits)
    permissions: Permissions = field(default_factory=Permissions)
    shell_cmd: list[str] = field(
        default_factory=lambda: ["powershell", "-NoProfile", "-Command"]
        if os.name == "nt"
        else ["/bin/sh", "-c"]
    )


_TOP_LEVEL_KEYS = {
    "model",
    "base_url",
    "api_key",
    "workspace",
    "max_tool_rounds",
    "stream",
    "shell_cmd",
}


_VALID_PERMISSION_VALUES = frozenset({"allow", "ask", "deny"})


def _normalized_permission(key: str, value: object) -> str:
    """Normalize a [permissions] value; reject anything outside allow|ask|deny.

    A typo like ``shell = "Ask"`` must fail loudly instead of silently
    disabling the safety prompt (consumers treat unknown values as allow).
    """
    normalized = str(value).strip().lower()
    if normalized not in _VALID_PERMISSION_VALUES:
        raise ValueError(
            f"invalid permissions.{key} value {value!r}: "
            'expected "allow" | "ask" | "deny"'
        )
    return normalized


def _check_section(section: dict, cls: type, label: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(f"unknown {label} key(s): {', '.join(unknown)}")


def _split_mapping(raw: dict | None) -> tuple[dict, dict, dict]:
    """Split a mapping into (top-level, limits, permissions), validating keys."""
    data = dict(raw or {})
    limits = data.pop("limits", {}) or {}
    permissions = data.pop("permissions", {}) or {}
    if not isinstance(limits, dict) or not isinstance(permissions, dict):
        raise ValueError("[limits] and [permissions] must be tables")
    unknown_top = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unknown_top:
        raise ValueError(f"unknown config key(s): {', '.join(unknown_top)}")
    _check_section(limits, Limits, "limits")
    _check_section(permissions, Permissions, "permissions")
    return data, limits, permissions


def _apply(cfg: Config, top: dict, limits: dict, permissions: dict) -> None:
    workspace = top.pop("workspace", None)
    for key, value in top.items():
        setattr(cfg, key, value)
    for key, value in limits.items():
        setattr(cfg.limits, key, value)
    for key, value in permissions.items():
        setattr(cfg.permissions, key, _normalized_permission(key, value))
    if workspace is not None:
        cfg.workspace = pathlib.Path(str(workspace)).expanduser().resolve()


def load_config(path: pathlib.Path | None = None, overrides: dict | None = None) -> Config:
    """Build a Config: defaults <- TOML file <- env vars <- overrides dict."""
    cfg = Config()
    if path is not None:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        top, limits, permissions = _split_mapping(data)
        _apply(cfg, top, limits, permissions)

    env_model = os.environ.get("OCLITE_MODEL")
    if env_model:
        cfg.model = env_model
    env_base_url = os.environ.get("OCLITE_BASE_URL")
    if env_base_url:
        cfg.base_url = env_base_url

    if overrides:
        top, limits, permissions = _split_mapping(overrides)
        _apply(cfg, top, limits, permissions)
    return cfg

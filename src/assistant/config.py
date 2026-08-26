"""Configuration dataclasses and TOML/env/overrides loading for assistant.

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
    verbose: bool = False
    timeout_s: int = 600
    max_context_tokens: int = 12000
    system_prompt: str | None = None  # universal: null -> generic fallback, set in config.toml
    identity: str | None = None  # separate identity: set in config.toml [identity] or identity
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
    "verbose",
    "timeout_s",
    "max_context_tokens",
    "shell_cmd",
    "system_prompt",
    "prompt",  # table [prompt] with system key
    "identity",  # identity string or [identity] table
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
    prompt_tbl = data.pop("prompt", None)
    identity_tbl = data.pop("identity", None)
    if not isinstance(limits, dict) or not isinstance(permissions, dict):
        raise ValueError("[limits] and [permissions] must be tables")
    # normalize [prompt] table -> system_prompt
    if prompt_tbl is not None:
        if not isinstance(prompt_tbl, dict):
            raise ValueError("[prompt] must be a table")
        if "system" in prompt_tbl and "system_prompt" not in data:
            data["system_prompt"] = prompt_tbl["system"]
        # also allow prompt.system_prompt
        if "system_prompt" in prompt_tbl and "system_prompt" not in data:
            data["system_prompt"] = prompt_tbl["system_prompt"]
        # ignore other prompt keys for now
    # normalize [identity] table or string -> identity
    if identity_tbl is not None:
        if isinstance(identity_tbl, dict):
            if "system" in identity_tbl and "identity" not in data:
                data["identity"] = str(identity_tbl["system"])
            elif "description" in identity_tbl and "identity" not in data:
                data["identity"] = str(identity_tbl["description"])
            elif "identity" not in data:
                parts = [str(v) for k, v in identity_tbl.items() if v]
                if parts:
                    data["identity"] = ", ".join(parts)
        elif isinstance(identity_tbl, str):
            if "identity" not in data:
                data["identity"] = identity_tbl
    unknown_top = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unknown_top:
        raise ValueError(f"unknown config key(s): {', '.join(unknown_top)}")
    _check_section(limits, Limits, "limits")
    _check_section(permissions, Permissions, "permissions")
    return data, limits, permissions


def _apply(cfg: Config, top: dict, limits: dict, permissions: dict) -> None:
    workspace = top.pop("workspace", None)
    if "verbose" in top:
        v = top["verbose"]
        if not isinstance(v, bool):
            raise ValueError(f"invalid verbose value {v!r}: expected bool")
    if "timeout_s" in top:
        tv = top["timeout_s"]
        if not isinstance(tv, int) or tv <= 0:
            raise ValueError(f"invalid timeout_s value {tv!r}: expected positive int")
    if "max_context_tokens" in top:
        mct = top["max_context_tokens"]
        if not isinstance(mct, int) or mct <= 0:
            raise ValueError(f"invalid max_context_tokens value {mct!r}: expected positive int")
    if "system_prompt" in top:
        sp = top["system_prompt"]
        if sp is not None and not isinstance(sp, str):
            raise ValueError(f"invalid system_prompt value {sp!r}: expected string or null")
    if "identity" in top:
        ident = top["identity"]
        if ident is not None and not isinstance(ident, str):
            raise ValueError(f"invalid identity value {ident!r}: expected string or null")
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
    env_verbose = os.environ.get("OCLITE_VERBOSE")
    if env_verbose is not None:
        cfg.verbose = env_verbose.strip().lower() in ("1", "true", "yes")
    env_timeout = os.environ.get("OCLITE_TIMEOUT_S")
    if env_timeout is not None:
        try:
            cfg.timeout_s = int(env_timeout)
        except ValueError:
            raise ValueError(f"invalid OCLITE_TIMEOUT_S value {env_timeout!r}: expected int")
    env_mct = os.environ.get("OCLITE_MAX_CONTEXT_TOKENS")
    if env_mct is not None:
        try:
            cfg.max_context_tokens = int(env_mct)
        except ValueError:
            raise ValueError(f"invalid OCLITE_MAX_CONTEXT_TOKENS value {env_mct!r}: expected int")

    if overrides:
        top, limits, permissions = _split_mapping(overrides)
        _apply(cfg, top, limits, permissions)
    return cfg

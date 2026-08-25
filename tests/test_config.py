"""Offline tests: [permissions] normalization + validation in config loading.

Covers the SEC-4 fix: a typo like ``shell = "allowed"`` must raise ValueError
instead of silently disabling the safety prompt (fail-open). Case variants of
valid words (``"Ask"`` -> ``"ask"``) normalize and therefore stay valid —
which keeps the conservative safety prompt active rather than failing open.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opencode_lite.config import Config, Permissions, load_config  # noqa: E402


def _write_config(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(body, encoding="utf-8")
    return cfg


# --- valid values & normalization (TOML path) ---------------------------------


def test_defaults_untouched_without_permissions_section(tmp_path: Path) -> None:
    path = _write_config(tmp_path, 'model = "m1"\n')
    cfg = load_config(path)
    assert cfg.permissions == Permissions()
    assert cfg.model == "m1"


def test_valid_values_pass_and_normalize(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "\n".join(
            [
                "[permissions]",
                'write = "ALLOW"',
                'delete = " Ask "',
                'shell = "deny"',
                'webfetch = "allow"',
                'websearch = "Deny"',
            ]
        )
        + "\n",
    )
    cfg = load_config(path)
    assert cfg.permissions == Permissions(
        write="allow", delete="ask", shell="deny", webfetch="allow", websearch="deny"
    )


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [("Ask", "ask"), ("ASK", "ask"), ("Allow", "allow"), ("DENY", "deny")],
)
def test_case_variants_of_valid_words_normalize(tmp_path: Path, raw: str, canonical: str) -> None:
    # "Ask" must resolve to "ask" (safety prompt stays ON), never fail open.
    path = _write_config(tmp_path, f'[permissions]\nshell = "{raw}"\n')
    assert load_config(path).permissions.shell == canonical


# --- invalid values raise ValueError naming key and value ---------------------


@pytest.mark.parametrize("raw", ["allowed", "yes", ""])
@pytest.mark.parametrize("key", ["shell", "write", "delete", "webfetch", "websearch"])
def test_invalid_toml_value_raises_naming_key(tmp_path: Path, key: str, raw: str) -> None:
    path = _write_config(tmp_path, f"[permissions]\n{key} = \"{raw}\"\n")
    with pytest.raises(ValueError, match=key):
        load_config(path)


@pytest.mark.parametrize("raw", ["true", "123", "[]"])
def test_non_string_permission_raises(tmp_path: Path, raw: str) -> None:
    path = _write_config(tmp_path, f"[permissions]\nshell = {raw}\n")
    with pytest.raises(ValueError, match="shell"):
        load_config(path)


def test_error_message_names_key_and_value(tmp_path: Path) -> None:
    path = _write_config(tmp_path, '[permissions]\nshell = "allowed"\n')
    with pytest.raises(ValueError) as excinfo:
        load_config(path)
    msg = str(excinfo.value)
    assert "shell" in msg
    assert "allowed" in msg


# --- overrides dict path is validated identically -----------------------------


def test_overrides_valid_values_normalized() -> None:
    cfg = load_config(None, {"permissions": {"shell": " ASK ", "delete": "DENY"}})
    assert cfg.permissions.shell == "ask"
    assert cfg.permissions.delete == "deny"


@pytest.mark.parametrize("raw", ["allowed", "yes", ""])
def test_overrides_invalid_value_raises(tmp_path: Path, raw: str) -> None:
    with pytest.raises(ValueError, match="delete"):
        load_config(None, {"permissions": {"delete": raw}})
    with pytest.raises(ValueError, match="shell"):
        load_config(None, {"permissions": {"shell": raw}})


def test_overrides_case_variant_normalizes() -> None:
    cfg = load_config(None, {"permissions": {"shell": "Ask"}})
    assert cfg.permissions.shell == "ask"


def test_overrides_do_not_leak_invalid_state_on_reject() -> None:
    # The ValueError must propagate out of load_config, leaving no Config behind.
    with pytest.raises(ValueError):
        load_config(None, {"permissions": {"webfetch": "maybe"}})


# --- defaults sanity ----------------------------------------------------------


def test_dataclass_defaults_are_valid_and_lowercase() -> None:
    defaults = Permissions()
    for f in Permissions.__dataclass_fields__:  # type: ignore[attr-defined]
        assert getattr(defaults, f) in {"allow", "ask", "deny"}
    assert Config().max_tool_rounds == 25  # documented default (SAMPLE_CONFIG drift guard)

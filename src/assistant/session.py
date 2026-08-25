"""Session persistence for assistant conversations.

Provides helpers to save, load, list and delete chat sessions stored as
JSON files under ``~/.assistant/sessions``. Session names are restricted
to alphanumeric characters plus ``_``, ``-`` and ``.`` to prevent path
traversal and ensure portable filenames.
"""

import datetime
import json
import re
from pathlib import Path

SESSION_DIR = Path.home() / ".assistant" / "sessions"

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _session_dir() -> Path:
    """Return the sessions directory, creating it if needed."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR


def session_path(name: str) -> Path:
    """Return the file path for a session name.

    Validates ``name`` against ``^[A-Za-z0-9._-]+$`` and raises
    ``ValueError`` if invalid.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"invalid session name: {name!r} (allowed: alphanumeric + _ - .)")
    return _session_dir() / f"{name}.json"


def list_sessions() -> list[str]:
    """List session names (``*.json`` stems) sorted alphabetically."""
    d = _session_dir()
    return sorted(p.stem for p in d.glob("*.json"))


def save_session(name: str, messages: list[dict]) -> Path:
    """Save ``messages`` to a session file.

    Validates ``name`` via ``^[A-Za-z0-9._-]+$`` and that ``messages`` is a
    list. Writes JSON with ``indent=2`` containing ``{"messages": ..., "saved_at": ...}``
    where ``saved_at`` is an ISO-8601 UTC timestamp. Raises ``ValueError``
    if validation fails. Returns the path written.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(f"invalid session name: {name!r} (allowed: alphanumeric + _ - .)")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    path = session_path(name)
    data = {
        "messages": messages,
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_session(name: str) -> list[dict]:
    """Load and return ``messages`` for ``name``.

    Raises ``FileNotFoundError`` if the session file does not exist and
    ``ValueError`` if the JSON structure is invalid (not a dict with a
    ``messages`` list).

    Validation of ``name`` uses the same ``^[A-Za-z0-9._-]+$`` rule as
    :func:`session_path`.
    """
    path = session_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"session not found: {name!r}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid session file: {name!r} contains invalid JSON") from e
    if not isinstance(data, dict) or "messages" not in data or not isinstance(data["messages"], list):
        raise ValueError(f"invalid session file: {name!r} must contain a dict with a 'messages' list")
    # Ensure messages is a list of dicts when non-empty
    if not all(isinstance(m, dict) for m in data["messages"]):
        raise ValueError(f"invalid session file: {name!r} messages must be a list of dicts")
    return data["messages"]


def delete_session(name: str) -> None:
    """Delete the session file for ``name``.

    Validates ``name`` via ``^[A-Za-z0-9._-]+$``. Raises
    ``FileNotFoundError`` if the file does not exist.
    """
    path = session_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"session not found: {name!r}")
    path.unlink()

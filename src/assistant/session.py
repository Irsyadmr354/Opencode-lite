"""JSON session save/load."""
from __future__ import annotations

import json
import os
import tempfile
import pathlib


class Session:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.messages: list[dict] = []

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def clear(self) -> None:
        self.messages.clear()

    def save(self) -> None:
        """Atomic write: temp file then rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: pathlib.Path) -> Session:
        """Load session from file, empty if missing."""
        s = cls(path)
        if path.is_file():
            try:
                s.messages = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return s

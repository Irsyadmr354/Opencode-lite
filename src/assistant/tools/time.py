"""Lightweight time tool – pure stdlib, no network, instant."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from assistant.tools import Tool, ToolResult

TOOL_SPEC = {
    "name": "get_current_time",
    "description": "Get current date/time - MUST call before web_search to get newest update",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _now_payload() -> dict:
    now = datetime.now(timezone.utc)
    local = now.astimezone()
    return {
        "iso": now.isoformat(),
        "utc": now.isoformat(),
        "local": local.isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "weekday": now.strftime("%A"),
    }


def time_tool(workspace, config) -> Tool:
    def fn(args: dict) -> ToolResult:
        try:
            return ToolResult(True, json.dumps(_now_payload()))
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"ERROR: {exc}")

    return Tool(
        name=TOOL_SPEC["name"],
        description=TOOL_SPEC["description"],
        parameters=TOOL_SPEC["parameters"],
        danger=False,
        fn=fn,
    )


def build_tools(workspace, config) -> list[Tool]:
    return [time_tool(workspace, config)]

"""Web tools: webfetch (HTML -> text) and websearch (DuckDuckGo via ddgs).

Both are offline-tolerant and never raise: any failure becomes
``ToolResult(ok=False, "ERROR: ...")``.
"""
from __future__ import annotations

import pathlib
import re

import httpx
from bs4 import BeautifulSoup

from opencode_lite.tools import Tool, ToolResult

try:  # optional runtime dependency; missing ddgs must not break tool loading
    from ddgs import DDGS
except ImportError:  # pragma: no cover - exercised only without the package
    DDGS = None  # type: ignore[assignment]

_USER_AGENT = "Mozilla/5.0 (compatible; opencode-lite/0.1)"
_FETCH_TIMEOUT_S = 30


# --- webfetch ----------------------------------------------------------------
def webfetch_tool(workspace: pathlib.Path, config) -> Tool:
    default_chars = int(config.limits.webfetch_chars)

    def fn(args: dict) -> ToolResult:
        try:
            url = args.get("url")
            if url is None:
                return ToolResult(False, "ERROR: missing argument 'url'")
            url = str(url)
            if not (url.startswith("http://") or url.startswith("https://")):
                return ToolResult(False, "ERROR: only http/https URLs are supported")
            raw_chars = args.get("max_chars")
            try:
                max_chars = int(raw_chars) if raw_chars else default_chars
            except (TypeError, ValueError):
                return ToolResult(False, "ERROR: max_chars must be an integer")

            resp = httpx.get(
                url,
                follow_redirects=True,
                timeout=_FETCH_TIMEOUT_S,
                headers={"User-Agent": _USER_AGENT},
            )
            if resp.status_code >= 400:
                return ToolResult(False, f"ERROR: HTTP {resp.status_code} fetching {url}")

            text = resp.text
            content_type = ""
            try:
                content_type = str(resp.headers.get("content-type", ""))
            except Exception:  # noqa: BLE001 - tolerate odd response objects
                pass
            if "html" in content_type.lower():
                text = BeautifulSoup(text, "html.parser").get_text("\n")
                text = re.sub(r"[ \t]+\n", "\n", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars] + "\n...[truncated]"
            return ToolResult(True, text)
        except Exception as exc:  # noqa: BLE001 - tool boundary must never raise
            return ToolResult(False, f"ERROR: webfetch failed: {exc}")

    return Tool(
        name="webfetch",
        description="Fetch an http(s) URL and return its text "
        "(HTML is stripped to plain text).",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Absolute http(s) URL."},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters returned "
                    "(default from config).",
                },
            },
            "required": ["url"],
        },
        danger=False,
        fn=fn,
    )


# --- websearch ---------------------------------------------------------------
def websearch_tool(workspace: pathlib.Path, config) -> Tool:
    def fn(args: dict) -> ToolResult:
        try:
            query = args.get("query")
            if query is None:
                return ToolResult(False, "ERROR: missing argument 'query'")
            raw_n = args.get("max_results")
            max_results = int(raw_n) if raw_n else 5
            if DDGS is None:
                raise RuntimeError("ddgs package unavailable")
            results = DDGS().text(str(query), max_results=max_results)

            blocks = []
            for i, item in enumerate(results, start=1):
                title = str(item.get("title", "") or "")
                url = str(item.get("url") or item.get("href") or "")
                snippet = str(item.get("snippet") or item.get("body") or "")
                blocks.append(f"[{i}] {title}\n    {url}\n    {snippet}")
            output = "\n\n".join(blocks)
            return ToolResult(True, output if output else "(no results)")
        except Exception:  # noqa: BLE001 - network failures must degrade softly
            return ToolResult(
                False, "ERROR: websearch failed (offline or rate-limited)"
            )

    return Tool(
        name="websearch",
        description="Search the web (DuckDuckGo) and return titled result snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default 5).",
                },
            },
            "required": ["query"],
        },
        danger=False,
        fn=fn,
    )


def build_tools(workspace: pathlib.Path, config) -> list[Tool]:
    return [webfetch_tool(workspace, config), websearch_tool(workspace, config)]

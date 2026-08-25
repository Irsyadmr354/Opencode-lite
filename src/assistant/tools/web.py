"""Web tools: webfetch (HTML -> text) and websearch (DuckDuckGo via ddgs).

Both are offline-tolerant and never raise: any failure becomes
``ToolResult(ok=False, "ERROR: ...")``.
"""
from __future__ import annotations

import ipaddress
import pathlib
import re
import socket
import urllib.parse

import httpx
from bs4 import BeautifulSoup

from assistant.tools import Tool, ToolResult

try:  # optional runtime dependency; missing ddgs must not break tool loading
    from ddgs import DDGS
except ImportError:  # pragma: no cover - exercised only without the package
    DDGS = None  # type: ignore[assignment]

_USER_AGENT = "Mozilla/5.0 (compatible; assistant/0.1)"
_FETCH_TIMEOUT_S = 30
_MAX_REDIRECT_HOPS = 5
_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
# Read slightly past max_chars so HTML stripping still leaves enough text.
_OVERSHOOT_CHARS = 4096


def _host_blocked_reason(host: str) -> str | None:
    """Return why *host* may not be contacted, or None when it is public.

    Literal IPs are checked directly; names are resolved via ``getaddrinfo``
    and every returned address must be globally reachable (blocks loopback,
    LAN, link-local/metadata, reserved and unspecified targets).
    """
    host = host.strip().lower().rstrip(".")
    if not host:
        return "empty host"
    if host == "localhost" or host.endswith(".localhost"):
        return "loopback hostname 'localhost'"
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [literal]
    else:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError as exc:
            return f"dns resolution failed for '{host}' ({exc})"  # fail closed
        addresses = []
        for info in infos:
            raw = str(info[4][0]).split("%", 1)[0]
            try:
                addresses.append(ipaddress.ip_address(raw))
            except ValueError:
                return f"unparseable dns record '{raw}' for '{host}'"
        if not addresses:
            return f"no dns records for '{host}'"
    for addr in addresses:
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_unspecified
        ):
            return f"'{host}' resolves to non-public address {addr}"
    return None


def _hop_blocked_reason(url: str) -> str | None:
    """SSRF gate applied to every fetched URL, including each redirect hop."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return "only http/https URLs are supported"
    return _host_blocked_reason(parts.hostname or "")


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

            text = ""
            current = url
            # Manual redirect loop: every hop passes the SSRF gate before the
            # request is made; TLS verification stays on (httpx default).
            for hop in range(_MAX_REDIRECT_HOPS + 1):
                reason = _hop_blocked_reason(current)
                if reason is not None:
                    return ToolResult(False, f"ERROR: blocked: {reason}: {current}")
                with httpx.stream(
                    "GET",
                    current,
                    follow_redirects=False,
                    timeout=_FETCH_TIMEOUT_S,
                    headers={"User-Agent": _USER_AGENT},
                ) as resp:
                    if resp.status_code in (301, 302, 303, 307, 308):
                        location = resp.headers.get("location")
                        if location is None:
                            return ToolResult(
                                False,
                                f"ERROR: HTTP {resp.status_code} fetching {url}",
                            )
                        if hop >= _MAX_REDIRECT_HOPS:
                            return ToolResult(
                                False, f"ERROR: too many redirects fetching {url}"
                            )
                        current = urllib.parse.urljoin(current, location)
                        continue
                    if resp.status_code >= 400:
                        return ToolResult(
                            False, f"ERROR: HTTP {resp.status_code} fetching {url}"
                        )
                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit():
                        if int(declared) > _MAX_DOWNLOAD_BYTES:
                            return ToolResult(
                                False,
                                f"ERROR: response too large ({declared} bytes > "
                                f"{_MAX_DOWNLOAD_BYTES} byte limit)",
                            )
                    chunks: list[str] = []
                    received = 0
                    budget = max_chars + _OVERSHOOT_CHARS
                    for chunk in resp.iter_text():
                        chunks.append(chunk)
                        received += len(chunk)
                        if received >= budget:
                            break  # stop buffering early; never read it all
                    text = "".join(chunks)
                    break

            content_type = ""
            try:
                content_type = str(resp.headers.get("content-type", ""))
            except Exception:  # noqa: BLE001 - tolerate odd response objects
                pass
            if "html" in content_type.lower():
                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript", "svg"]):
                    tag.decompose()
                text = soup.get_text("\n")
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

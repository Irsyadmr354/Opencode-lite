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
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from assistant.tools import Tool, ToolResult

try:  # optional runtime dependency; missing ddgs must not break tool loading
    from ddgs import DDGS
except ImportError:  # pragma: no cover - exercised only without the package
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None  # type: ignore[assignment]


def _date_context_header() -> str:
    return (
        f"[context: current datetime {datetime.now(timezone.utc).isoformat()} | "
        f"today = {datetime.now().strftime('%Y-%m-%d')}; prefer sources/results dated closest to this date]"
    )


_USER_AGENT = "Mozilla/5.0 (compatible; assistant/0.1)"
_FETCH_TIMEOUT_S = 30
_MAX_REDIRECT_HOPS = 5
_MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
# Read slightly past max_chars so HTML stripping still leaves enough text.
_OVERSHOOT_CHARS = 4096
# Model-supplied max_chars is clamped into this window; config default is trusted.
_MIN_FETCH_CHARS = 100
_MAX_FETCH_CHARS = 50000
_SEARCH_TIMEOUT_S = 20
_RECENCY_TO_TIMELIMIT = {"day": "d", "week": "w", "month": "m", "year": "y"}


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


def _safe_url(url: str) -> str:
    """Scheme://host/path only; never persist query/fragment (may hold secrets)."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, "", "")
    )[:200]


# --- webfetch ----------------------------------------------------------------
def webfetch_tool(workspace: pathlib.Path, config) -> Tool:
    default_chars = int(config.limits.webfetch_chars)

    def fn(args: dict) -> ToolResult:
        try:
            _allowed = {"url", "max_chars"}
            for _k in args:
                if _k not in _allowed:
                    return ToolResult(
                        False,
                        f"ERROR: webfetch: unexpected argument '{_k}' — allowed: url, max_chars",
                    )
            url = args.get("url")
            if url is None:
                return ToolResult(False, "ERROR: missing argument 'url'")
            url = str(url)
            if not (url.startswith("http://") or url.startswith("https://")):
                return ToolResult(False, "ERROR: only http/https URLs are supported")
            raw_chars = args.get("max_chars")
            try:
                max_chars = int(raw_chars) if raw_chars else default_chars
                if raw_chars:
                    max_chars = min(max(max_chars, _MIN_FETCH_CHARS), _MAX_FETCH_CHARS)
            except (TypeError, ValueError):
                return ToolResult(False, "ERROR: max_chars must be an integer")

            text = ""
            current = url
            # Manual redirect loop: every hop passes the SSRF gate before the
            # request is made; TLS verification stays on (httpx default).
            for hop in range(_MAX_REDIRECT_HOPS + 1):
                reason = _hop_blocked_reason(current)
                if reason is not None:
                    return ToolResult(
                        False, f"ERROR: blocked: {reason}: {_safe_url(current)}"
                    )
                with httpx.stream(
                    "GET",
                    current,
                    follow_redirects=False,
                    timeout=_FETCH_TIMEOUT_S,
                    headers={
                        "User-Agent": _USER_AGENT,
                        "Cache-Control": "no-cache",
                    },
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
            return ToolResult(True, f"{_date_context_header()}\n\n{text}")
        except Exception as exc:  # noqa: BLE001 - tool boundary must never raise
            return ToolResult(False, f"ERROR: webfetch failed: {exc}")

    return Tool(
        name="webfetch",
        description=(
            "Fetch an http(s) URL and return its text (HTML stripped). "
            "The model MUST call get_current_time BEFORE calling this tool to verify the current "
            "date, month, and year. Output includes a current datetime context header."
        ),
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
        permission_key="webfetch",
    )


# --- websearch ---------------------------------------------------------------
def websearch_tool(workspace: pathlib.Path, config) -> Tool:
    def fn(args: dict) -> ToolResult:
        try:
            _allowed = {"query", "max_results", "recency"}
            for _k in args:
                if _k not in _allowed:
                    return ToolResult(
                        False,
                        f"ERROR: websearch: unexpected argument '{_k}' — allowed: query, max_results, recency (day|week|month|year). Use 'query' for search text, not 'url'.",
                    )
            query = args.get("query")
            if query is None:
                return ToolResult(False, "ERROR: missing argument 'query'")
            raw_n = args.get("max_results")
            max_results = min(max(int(raw_n) if raw_n else 5, 1), 10)
            recency = args.get("recency")
            timelimit = None
            if recency:
                timelimit = _RECENCY_TO_TIMELIMIT.get(str(recency).lower().strip())
                if timelimit is None:
                    return ToolResult(
                        False,
                        "ERROR: recency must be one of day|week|month|year",
                    )
            if DDGS is None:
                return ToolResult(
                    False,
                    "ERROR: websearch failed: duckduckgo search package is not installed (ddgs/duckduckgo_search unavailable)",
                )
            try:  # ddgs >=9 moved timeout onto the constructor
                ddgs = DDGS(timeout=_SEARCH_TIMEOUT_S)
            except TypeError:  # pragma: no cover - older signatures / stubs
                ddgs = DDGS()
            kwargs = {"max_results": max_results}
            if timelimit is not None:
                kwargs["timelimit"] = timelimit
            try:
                results = list(ddgs.text(str(query), **kwargs))
            except Exception as exc:
                return ToolResult(False, f"ERROR: websearch failed: {type(exc).__name__}: {exc}")

            blocks = []
            for i, item in enumerate(results, start=1):
                if isinstance(item, dict):
                    title = str(item.get("title") or "").strip()
                    url = str(item.get("url") or item.get("href") or "").strip()
                    snippet = str(item.get("snippet") or item.get("body") or "").strip()
                    blocks.append(f"[{i}] {title}\n    {url}\n    {snippet}")
                elif isinstance(item, str):
                    blocks.append(f"[{i}] {item}")
            output = "\n\n".join(blocks)
            result = output if output else "(no results)"
            return ToolResult(True, f"{_date_context_header()}\n\n{result}")
        except Exception as exc:  # noqa: BLE001 - network failures must degrade softly
            detail = f"{type(exc).__name__}: {exc}"[:200]
            return ToolResult(False, f"ERROR: websearch failed: {detail}")

    return Tool(
        name="websearch",
        description=(
            "Search the web (DuckDuckGo) and return titled result snippets. "
            "Optional recency=day|week|month|year limits freshness. "
            "The model MUST call get_current_time BEFORE calling this tool so search queries "
            "target information freshest relative to today's date. Output includes a current datetime context header."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default 5, clamped 1-10).",
                },
                "recency": {
                    "type": "string",
                    "description": "Freshness window: day|week|month|year "
                    "(omit for any time).",
                },
            },
            "required": ["query"],
        },
        danger=False,
        fn=fn,
        permission_key="websearch",
    )


def build_tools(workspace: pathlib.Path, config) -> list[Tool]:
    return [webfetch_tool(workspace, config), websearch_tool(workspace, config)]

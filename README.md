# assistant

Minimal, opencode-like coding agent for **local models**. It runs a
native-terminal chat REPL (or a headless one-shot mode) against an Ollama
endpoint and lets the model work inside a workspace directory using 8 tools:
read files, list files, write files, delete files, run shell commands, get
the current date/time, fetch web pages, and search the web.
Local-model-first: everything runs on your machine except optional web
fetch/search.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running with at least one model pulled

## Setup

```powershell
# 1. Pull a model (any model works; coder models recommended)
ollama pull qwen2.5-coder:7b      # or qwen3:8b, etc.

# 2. Install assistant from the repo root (core: REPL + file/shell/webfetch tools)
pip install -e .

# Optional: enable websearch (DuckDuckGo via ddgs)
pip install -e ".[web]"
```

`webfetch` works without the extra; `websearch` errors clearly if `ddgs`
is not installed.

## Run

```powershell
assistant                      # interactive REPL (or: python -m assistant)
assistant --model qwen3:8b     # override model
assistant -p "list python files"   # headless: answer, print, exit
```

Headless mode streams the answer to stdout, logs tool calls/results, asks
`[y/N]` for permission-gated tools (auto-denied when stdin is not an
interactive terminal), and exits with code `0` on success / `1` on error /
`2` on startup failure (bad config path, invalid config, missing modules).
Pressing Ctrl+C during headless generation cancels cleanly by design:
it prints `(cancelled)` to stderr and stops without treating it as an error.

## Tools

| Tool | Danger | Purpose |
|---|---|---|
| `read_file` | no | Read a text file from the workspace |
| `list_files` | no | List files/directories under the workspace |
| `write_file` | no* | Create or overwrite a file (*prompts if `permissions.write = "ask"`) |
| `delete_file` | yes | Delete a file (permission prompt) |
| `shell` | yes | Run a shell command in the workspace (permission prompt) |
| `get_current_time` | no | Current date/time (local + UTC) |
| `webfetch` | no | Fetch a URL and extract readable text |
| `websearch` | no | Web search via DuckDuckGo |

Danger=yes tools always prompt before executing. Every tool also honors
`[permissions]` in config: `"deny"` blocks it outright, `"ask"` forces a
prompt even for non-danger tools, `"allow"` runs silently.

### Date-aware web access

Web answers stay current through two layers:

1. The system prompt carries a date line that is refreshed on every submit,
   and it instructs the model to call `get_current_time` **before** any
   `websearch`/`webfetch`. The time tool reports LOCAL calendar fields
   (`year`, `month`, `day`, `weekday`) plus `tz_name`, alongside UTC ISO stamps.
2. Every `webfetch`/`websearch` result automatically gets a context header
   prepended — `[context: current datetime <ISO> | today = YYYY-MM-DD;
   prefer sources/results dated closest to this date]` — so recency judgments
   never rely on stale guesses.

`websearch` also accepts a `recency` argument (`day` | `week` | `month` |
`year`, mapped to DuckDuckGo's time limit); omit it for no filter.
Clamps: `max_results` 1–10 (default 5); `webfetch` `max_chars` 100–50000
(default from config `limits.webfetch_chars`, itself 8000).

## Permissions & config

On first run a commented sample is written to `~/.assistant/config.toml`.
Edit it to set defaults:

```toml
model = "qwen2.5-coder:7b"
base_url = "http://127.0.0.1:11434/v1"
api_key = "ollama"
max_tool_rounds = 12     # default: 25
stream = true
verbose = false          # show Ollama performance stats after each turn
timeout_s = 600
max_context_tokens = 12000   # pruning threshold; default 12000 — lower for small-context models

[permissions]
# "allow" | "ask" | "deny"
write = "allow"          # set "ask" to confirm every file write
delete = "ask"
shell = "ask"
webfetch = "allow"
websearch = "allow"

[limits]
# read_max_lines = 200
# shell_timeout_s = 120
# shell_output_chars = 6000
# webfetch_chars = 8000
# list_max_entries = 200
```

CLI flags (`--model`, `--base-url`, `--workspace`, `--config`, `--verbose`/`-v`) override the
config file, which itself overrides environment variables. Env vars:

- `OCLITE_MODEL` — override model name
- `OCLITE_BASE_URL` — override Ollama endpoint
- `OCLITE_VERBOSE` — `1`/`true`/`yes` → verbose on
- `OCLITE_TIMEOUT_S` — request timeout seconds (int)
- `OCLITE_MAX_CONTEXT_TOKENS` — pruning threshold (int)

(`verbose` can also be set via TOML (`verbose = true`) or toggled with `/verbose` in the REPL.)

## REPL commands & keybindings

Slash commands: `/help`, `/clear`, `/cls`, `/model [name]`, `/status`, `/verbose [on|off|status]`,
`/session [list|save <name>|load <name>|new|delete <name>]`, `/exit` (also `/quit`; short aliases `/q`,
`/h`, `/?`, and `/sessions` also work).

Verbose mode (`verbose = true` or `--verbose`/`-v` or `/verbose on`) shows Ollama's technical performance stats after each turn (uses Ollama built-in `total_duration`, `prompt_eval_count`/`eval_count`, `prompt_eval_duration`/`eval_duration` → tokens/s when available, otherwise local wall time + estimated tokens). Toggle at runtime with `/verbose` (no arg toggles, `on`/`off` explicit, `status` shows current).

### Sessions & pruning

Conversation history is pruned automatically when it exceeds the
pruning threshold — `max_context_tokens` (default **12000**, configurable
via config TOML or `OCLITE_MAX_CONTEXT_TOKENS`). Pruning is type-aware: it drops the oldest turns first while keeping the system prompt and the newest user turn, and removes `assistant(tool_calls)` together with all following `tool` results as one atomic unit so no orphaned tool message ever reaches the server. This prevents hallucination by preserving coherence instead of silently truncating.

Use `/session` to persist the (already pruned) history to disk under `~/.assistant/sessions/`:

```
/session                # status: messages, ~tokens, saved count + hint
/session list           # list saved sessions (alias: ls)
/session save mywork    # save current history as mywork.json
/session load mywork    # restore history (replaces current messages)
/session new            # clear screen & start fresh (alias: clear)
/session delete mywork  # delete saved file (alias: rm)
/session help           # usage
```

Sessions are plain JSON `{"messages": [...], "saved_at": "ISO8601"}` with indent 2. Save/load is not wasteful (no redundant copies) — pruning stays active and saved files contain exactly the pruned history you choose to keep.

| Key | Action |
|---|---|
| `enter` | Send prompt |
| `ctrl+c` | Cancel generation / reset prompt; press twice to quit |
| `ctrl+d` / `ctrl+z` | Exit (EOF) |

## Limitations

- Single session per launch; conversation history lives in memory only.
- No context compaction — long sessions are bounded by truncation caps only.
- `websearch` uses DuckDuckGo and may rate-limit under heavy use; the root
  cause (e.g. rate limit vs network error) is now included in error output.
- File tools restrict paths to the workspace directory.
- The SSRF gate blocks private/link-local targets, but DNS-rebinding TOCTOU
  remains a documented residual risk (no IP pinning by design — keeps deps light).
- The shell timeout kills the direct child process only; grandchildren it
  spawned may outlive it.

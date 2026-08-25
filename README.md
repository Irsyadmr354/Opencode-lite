# opencode-lite

Minimal, opencode-like coding agent for **local models**. It runs a
native-terminal chat REPL (or a headless one-shot mode) against an Ollama
endpoint and lets the model work inside a workspace directory using 7 tools:
read files, list files, write files, delete files, run shell commands,
fetch web pages, and search the web. Local-model-first: everything runs on
your machine except optional web fetch/search.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running with at least one model pulled

## Setup

```powershell
# 1. Pull a model (any model works; coder models recommended)
ollama pull qwen2.5-coder:7b      # or qwen3:8b, etc.

# 2. Install opencode-lite from the repo root
pip install -e .
```

## Run

```powershell
opencode-lite                      # interactive REPL (or: python -m opencode_lite)
opencode-lite --model qwen3:8b     # override model
opencode-lite -p "list python files"   # headless: answer, print, exit
```

Headless mode streams the answer to stdout, logs tool calls/results, asks
`[y/N]` for permission-gated tools (auto-denied when stdin is not an
interactive terminal), and exits 0 on success / 1 on error.

## Tools

| Tool | Danger | Purpose |
|---|---|---|
| `read_file` | no | Read a text file from the workspace |
| `list_files` | no | List files/directories under the workspace |
| `write_file` | no* | Create or overwrite a file (*prompts if `permissions.write = "ask"`) |
| `delete_file` | yes | Delete a file (permission prompt) |
| `shell` | yes | Run a shell command in the workspace (permission prompt) |
| `webfetch` | no | Fetch a URL and extract readable text |
| `websearch` | no | Web search via DuckDuckGo |

Danger=yes tools always prompt before executing. Every tool also honors
`[permissions]` in config: `"deny"` blocks it outright, `"ask"` forces a
prompt even for non-danger tools, `"allow"` runs silently.

## Permissions & config

On first run a commented sample is written to `~/.opencode-lite/config.toml`.
Edit it to set defaults:

```toml
model = "qwen2.5-coder:7b"
base_url = "http://127.0.0.1:11434/v1"
api_key = "ollama"
max_tool_rounds = 12     # default: 25
stream = true

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

CLI flags (`--model`, `--base-url`, `--workspace`, `--config`) override the
config file.

## REPL commands & keybindings

Slash commands: `/help`, `/clear`, `/cls`, `/model [name]`, `/status`,
`/exit` (also `/quit`).

| Key | Action |
|---|---|
| `enter` | Send prompt |
| `ctrl+c` | Cancel generation / reset prompt; press twice to quit |
| `ctrl+d` / `ctrl+z` | Exit (EOF) |

## Limitations

- Single session per launch; conversation history lives in memory only.
- No context compaction — long sessions are bounded by truncation caps only.
- `websearch` uses DuckDuckGo and may rate-limit under heavy use.
- File tools restrict paths to the workspace directory.

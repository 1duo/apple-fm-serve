# apple-fm-serve

Minimal, robust OpenAI-compatible local server for Apple's Foundation Models Python SDK.

## Requirements

- macOS 26+
- Python 3.11+
- `uv` package manager

Apple SDK source: [python-apple-fm-sdk](https://github.com/apple/python-apple-fm-sdk)

## Quick start

```bash
uv sync --extra dev --extra apple
```

Run:

```bash
./serve
```

## Configuration

Environment variables:

- `APPLE_FM_HOST` (default: `127.0.0.1`)
- `APPLE_FM_PORT` (default: `8000`)
- `APPLE_FM_MODEL_ID` (default: `apple.fm.system`)
- `APPLE_FM_API_KEY` (optional, enables bearer auth)
- `APPLE_FM_MAX_CONCURRENCY` (default: `4`)
- `APPLE_FM_ESTIMATE_USAGE` (default: `true`)

## Connect to OpenCode

Add following to your OpenCode config (`opencode.jsonc` in your project or `~/.config/opencode/opencode.jsonc`):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "apple-fm-local": {
      "name": "Apple FM Local",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1"
      },
      "models": {
        "apple.fm.system": {
          "name": "Apple FM System",
          "tool_call": false,
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  },
  "model": "apple-fm-local/apple.fm.system"
}
```

3. Restart OpenCode and select `apple-fm-local/apple.fm.system`.

## Quality gates

```bash
make check
```

Includes Ruff lint/format, mypy strict mode, and pytest (all via `uv run`).

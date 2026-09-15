# apple-fm-serve

Swift-native, OpenAI-compatible local server for Apple's Foundation Models on macOS 27,
serving coding harnesses: **opencode**, **pi**, and **codex** (all verified live).

## Why Swift (from first principles)

Apple's Foundation Models (`FoundationModels.framework`) is a Swift-first API. There is no
Rust or Go SDK — either would have to re-implement Apple's Swift `fm` CLI approach via a
Swift-built C dylib plus FFI, adding a bridging layer with zero capability gain. The Python
SDK (`apple/python-apple-fm-sdk`) is beta, lags the OS (no variant/PCC/usage APIs), and needs
a full Xcode install to build its Swift C bindings.

This server therefore uses Swift 6 directly against `FoundationModels`, with no third-party
dependencies (stdlib + `Foundation` + `FoundationModels` + `Network` only). Apple's own
`/usr/bin/fm serve` (Swift, Chat Completions API) validates the choice — this project matches
its API surface while adding what a coding harness needs: real OpenAI `tool_calls`
round-trips, `temperature`/`max_tokens`/`stop` mapping, and real usage tokens.

## Model selection: the most capable AFM on macOS 27

- **Primary: on-device AFM 3 Core Advanced** — Apple's most powerful on-device model (20B
  sparse MoE, 1–4B active). The system auto-selects it via `SystemLanguageModel.default` on
  capable hardware (M3+ with 12GB+; verified `variant = "AFM 3 Core Advanced"` on M5/16GB).
  Offline, unlimited, privacy-preserving. Context: **8192 tokens** on macOS 27.
- **Fallback: AFM 3 Core** (3B dense) — automatic on other Apple Silicon via the same API.
- **Optional: AFM 3 Cloud via `PrivateCloudComputeLanguageModel`** (32K, reasoning) — only
  with the `com.apple.developer.private-cloud-compute` managed entitlement; without it
  requests fail (`ModelManagerError 1046`). Enable with `APPLE_FM_PREFER_PCC=1`; the server
  advertises `apple.fm.pcc` only when PCC reports available.

## Requirements

- macOS 26+ (27 recommended for 8192 context + Advanced variant)
- Swift toolchain: full Xcode **or** Command Line Tools (`xcode-select --install`)
- Apple Intelligence enabled + on-device model downloaded
  (`fm available` should report `System model available`)
- Agree to the `fm` CLI terms once: `sudo fm license` (same underlying models)

## Quick start

```bash
./serve
```

## Configuration

Environment variables (same `APPLE_FM_*` contract as before, plus three new ones):

- `APPLE_FM_HOST` (default: `127.0.0.1`)
- `APPLE_FM_PORT` (default: `8000`)
- `APPLE_FM_MODEL_ID` (default: `apple.fm.system`)
- `APPLE_FM_API_KEY` (optional, enables bearer auth)
- `APPLE_FM_MAX_CONCURRENCY` (default: `4`, max `128`)
- `APPLE_FM_REQUEST_TIMEOUT_S` (default: `120`, replaces `request_timeout_s`)
- `APPLE_FM_PREFER_PCC` (default: `false`; `1` enables PCC fallback/route)
- `APPLE_FM_USE_CASE` (default: `general`; or `content-tagging`)
- `APPLE_FM_GUARDRAILS` (default: `default`; or `permissive-content-transformations`)
- `APPLE_FM_USE_MOCK` (`1` = mock provider, no model needed; for CI/dev)
- `APPLE_FM_LOG_BODIES` (default: `false`; `1` logs truncated request bodies for
  compat debugging)

Endpoints: `GET /healthz`, `GET /health` (fm-compat), `GET /readyz`,
`GET /v1/models`, `POST /v1/chat/completions` (streaming + non-streaming SSE),
`POST /v1/responses` (Responses API for codex, streaming + non-streaming).

`response_format`: `text` and `json_schema` (guided generation via `GenerationSchema`);
`json_object` is rejected like `fm serve` ("Use `json_schema` instead").

## Connect to opencode

`opencode.jsonc` in your project (verified live against AFM 3 Core Advanced):

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
          "name": "AFM 3 Core Advanced (on-device)",
          "tool_call": true,
          "limit": { "context": 8192, "output": 4096 }
        }
      }
    }
  },
  "model": "apple-fm-local/apple.fm.system"
}
```

Notes from live testing:

- `tool_call: true` works — the server translates OpenAI function tools into AFM
  instructions and parses the structured reply back into `tool_calls` (verified: model
  emitted a `calculator` call; file-writing via opencode tools works).
- The 8192-token on-device window is small for opencode's default agent (full tool
  schemas ≈ 7k tokens + transcript). Long loops return proper
  `context_length_exceeded` (400) so opencode can compact; keep tasks scoped or trim
  enabled tools for best results.

## Connect pi

`~/.pi/agent/models.json` (verified live: chat + real `write`-tool file creation):

```json
{
  "providers": {
    "apple-fm": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local-dummy-key",
      "models": [
        {
          "id": "apple.fm.system",
          "name": "AFM 3 Core Advanced (on-device)",
          "contextWindow": 8192,
          "maxTokens": 4096,
          "compat": { "supportsUsageInStreaming": true, "maxTokensField": "max_tokens" }
        }
      ]
    }
  }
}
```

```bash
pi --provider apple-fm --model apple.fm.system --api-key dummy-key -p "Do something"
```

Notes:

- `apiKey` is a placeholder (the server needs no auth unless `APPLE_FM_API_KEY` is
  set); pi requires auth to be configured before models appear.
- `contextWindow: 8192` is required (pi defaults to 128000 and would otherwise
  never compact in time for the on-device window).
- pi may send `developer`-role messages; the server folds them into AFM
  instructions like `system` messages.

## Connect codex

Codex removed Chat Completions support (Feb 2026) and requires the Responses API,
so this server implements `POST /v1/responses` (verified live with codex 0.154:
multi-turn loop, real `exec` tool execution, context compaction). In
`~/.codex/config.toml` (user-level; project files can't set providers):

```toml
model = "apple.fm.system"
model_provider = "apple_fm"
model_context_window = 8192

[model_providers.apple_fm]
name = "Apple FM (on-device)"
base_url = "http://127.0.0.1:8000/v1"
env_key = "APPLE_FM_DUMMY_KEY"
wire_api = "responses"
```

```bash
export APPLE_FM_DUMMY_KEY=dummy-key
codex exec --skip-git-repo-check -s workspace-write "Do something"
```

Notes:

- `env_key` must point to a set variable (any dummy value works unless the server
  sets `APPLE_FM_API_KEY`); `model_context_window = 8192` silences sizing issues
  (a "Model metadata not found" fallback warning is harmless).
- Responses streaming ends at `response.completed` (no `[DONE]`, per spec);
  failures before the first event return HTTP error statuses.
- Small-model reality: AFM 3 Core Advanced sometimes invents tool names or garbles
  argument JSON for codex's abstract tools. The server salvages malformed envelopes
  and returns the calls so codex reports execution errors back (self-healing loop),
  retries once with stricter instructions when tools are `required`, and logs
  `WARN recovered …` / `WARN tool_calls key present but no calls parsed` lines.
  Keep tasks scoped; the loop converges best on concrete file/shell work.

## How it works (design)

- **Stateless HTTP, faithful transcripts**: each request rebuilds AFM
  `instructions` (system messages + rendered tool definitions) and a single `prompt`
  carrying the full turn history, including `assistant.tool_calls` and `role:tool`
  results. No server-side session affinity.
- **Client-executed tools**: AFM native tools auto-execute server-side, but harnesses
  execute client-side, so the server requests a JSON envelope
  (`{"content", "tool_calls"}`) and returns OpenAI `tool_calls` with `finish_reason:
  tool_calls`. Non-envelope replies degrade gracefully to plain content.
  Malformed envelopes are salvaged by a lenient recovery parser (unknown names
  included, so harness error feedback self-heals the loop); when tools are
  `required` and prose comes back, one stricter retry fires. Both paths log
  `WARN` lines.
- **Responses API**: `/v1/responses` translates items (`message` /
  `function_call` / `function_call_output` / `reasoning`-skipped) into the same
  chat pipeline, preserving call IDs across history replay; output uses
  `resp_`/`msg_`/`fc_` IDs and mapped usage. Stateless (ignores
  `previous_response_id`/`store`).
- **Real usage**: `session.usage` (prompt/completion tokens) on macOS 27, with
  `len//4` fallback only when unavailable. Streaming computes snapshot deltas and
  honors `stream_options.include_usage`.
- **Options mapping**: `temperature` → `GenerationOptions.temperature`,
  `top_p` → `samplingMode.random(probabilityThreshold:)`,
  `max_tokens`/`max_completion_tokens` → `maximumResponseTokens`, `stop` via
  post-truncation, `tool_choice` → envelope forcing (`none` disables, named forces).
- **Hybrid routing**: `SystemLanguageModel` primary; PCC only when preferred,
  available, and entitled.

## Quality gates

```bash
make check
```

`make check` = `swift build` + `swift run apple-fm-verify`.
(`swift test` needs full Xcode for XCTest; the `AppleFMServeTests` target mirrors the
same checks for Xcode environments, while `apple-fm-verify` runs them with only the
Command Line Tools.)

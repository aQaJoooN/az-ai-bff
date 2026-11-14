# Azure DevOps AI Backend-for-Frontend (BFF)

A FastAPI service that sits between the [Azure DevOps Build AI Reviewer](https://github.com/aQaJoooN/azure-devops-build-ai-reviewer) browser extension and your AI backend. It fetches failed build logs from your on-premise TFS/Azure DevOps server, compacts them into a dense, signal-rich format, and forwards them to the AI for root-cause analysis.

---

## Features

| Feature | Description |
|---------|-------------|
| **AI concurrency cap** | Configurable limit on simultaneous AI requests across all endpoints. Excess requests receive 429 immediately. |
| **Build deduplication** | Each build ID is processed at most once at a time. A duplicate in-flight request receives 409. |
| **Completed-run idempotency** | Once a build receives a successful AI analysis, the BFF refuses to analyze it again. Prevents curl/replay abuse. |
| **Client signature** | Every `/api/v1/analyze` request must include a 32-char MD5 signature derived from the build URL. Requests without a valid signature are rejected with 401. |
| **Log compaction** | Before sending to the AI, logs are stripped of ANSI codes, timestamps, boilerplate, and duplicate lines. Only error/warning/exception lines and their surrounding context are kept. |
| **SSE streaming** | Both endpoints support `?stream=true` for server-sent event streaming. |
| **Multi-AI fan-out** | Optional parallel dispatch to multiple AI backends with response synthesis. |
| **Token budget** | Optional character-based token estimation with context trimming. |
| **OpenAI + simple formats** | Works with OpenAI-compatible APIs and custom simple backends. |
| **Reasoning model support** | `BACKEND_REASONING_MODE=true` for models that answer in `reasoning_content` (DeepSeek, Qwen3, etc.). |
| **Structured JSON logs** | Every request and AI call emits one JSON line with stable fields. |

---

## Integration with Azure DevOps AI Analyzer Extension

1. **Install the extension** from the Azure DevOps Marketplace
2. **Configure the extension** with your BFF URL
3. **Deploy this BFF** (see [Deployment](#deployment) below)
4. **Configure CORS** — add your Azure DevOps organization URL to `AZ_AI_BFF_ALLOWED_EXTENSION_ORIGINS`
5. **Enable HTTPS** — if Azure DevOps is on HTTPS, this BFF must also be on HTTPS

---

## Quick Start

### Production (Docker Hub image)

```bash
mkdir az-ai-bff-deployment && cd az-ai-bff-deployment

# Download docker-compose and nginx config
curl -O https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/docker-compose.prod.yml

# HTTP only:
curl -o nginx.conf https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/nginx.conf.example-http
# HTTPS with redirect (recommended):
curl -o nginx.conf https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/nginx.conf.example-https

mkdir -p nginx/ssl nginx/logs
cp .env.example .env   # then edit .env

docker-compose -f docker-compose.prod.yml up -d
```

### Development

```bash
git clone https://github.com/aqajooon/az-ai-bff.git
cd az-ai-bff

cp .env.example .env   # edit .env

# Docker
docker-compose up -d --build

# Or locally
pip install -e .
python -m az_ai_bff
```

---

## API Reference

### `GET /health`

```bash
curl http://localhost/health
```

```json
{"status":"available","version":"1.0.0","uptime_seconds":42.1}
```

---

### `POST /api/v1/chat`

Generic AI chat relay.

```bash
curl http://localhost/api/v1/chat \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a NuGet feed?"}'
```

**Stream:**
```bash
curl -N "http://localhost/api/v1/chat?stream=true" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a NuGet feed?"}'
```

**Response `200`:**
```json
{
  "answer": "A NuGet feed is...",
  "conversation_id": null,
  "backend_id": "primary",
  "request_id": "abc-123"
}
```

---

### `POST /api/v1/analyze`

Fetches failed build logs from Azure DevOps, compacts them, and returns an AI root-cause analysis.

```bash
curl https://your-bff-host/api/v1/analyze \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "azure build pipeline error",
    "data": "https://your-tfs/tfs/Collection/ProjectId/_build/results?buildId=12345",
    "role": "user",
    "type": "normal"
  }'
```

**Decision tree:**

| `type` | `message` | Result |
|--------|-----------|--------|
| `super` | any | `{"answer": "unsupported"}` |
| `normal` | `azure build pipeline improvement` | `{"answer": "unsupported"}` |
| `normal` | `azure build pipeline error` | Log fetch → compact → AI analysis |

**Processing pipeline for `azure build pipeline error`:**

1. Parse and validate the build URL
2. Check the build ID against the in-flight and completed sets (→ 409 if duplicate)
3. Acquire an AI concurrency slot (→ 429 if at the limit)
4. Call Azure DevOps APIs to find all failed tasks and fetch their logs
5. Compact each log (strip noise, keep errors + context, enforce char budget)
6. Send the compacted log to the AI with a structured system prompt
7. Mark the build as completed so future requests for the same run are rejected

**Response `200`:**
```json
{
  "answer": "## Problem\n...\n\n## Solution\n...\n\n## Summary\n...",
  "backend_id": "primary",
  "request_id": "abc-123"
}
```

**Error responses:**

| Status | `code` | Cause |
|--------|--------|-------|
| 400 | `invalid_build_url` | URL is not a valid Azure DevOps build URL |
| 401 | `unauthorized` | Missing or wrong `Authorization` header |
| 401 | `invalid_signature` | `client_signature` is missing or does not match the build URL |
| 409 | `pipeline_already_processing` | A run of this pipeline definition is currently being analyzed |
| 409 | `build_already_analyzed` | This run already has an AI result (`ai-analyzed` tag present in Azure DevOps) |
| 429 | `rate_limit_exceeded` | AI concurrency cap reached — retry later |
| 502 | `azure_auth_error` | Azure DevOps token is invalid or lacks permissions |
| 502 | `azure_log_fetch_error` | Azure DevOps API returned an unexpected error |

#### Client signature algorithm

Every `/api/v1/analyze` request body must include a `client_signature` field. The extension generates it as follows:

**Step 1 — extract the three parts from the build URL:**

| Part | Description | Example |
|------|-------------|---------|
| `azure_host` | Hostname only — lowercase, no protocol, no port, no slash | `stageazure.asax.ir` |
| `project_id` | Project GUID or name from the URL path | `bc24cb70-e692-4049-ae06-d776b7bf4b81` |
| `run_id` | Numeric build/run ID as a plain string | `1116251` |

**Step 2 — concatenate with `|` delimiter:**
```
payload = `${azure_host}|${project_id}|${run_id}`
// → "stageazure.asax.ir|bc24cb70-e692-4049-ae06-d776b7bf4b81|1116251"
```

**Step 3 — MD5 of the UTF-8–encoded payload:**
```
signature = md5(payload)   // 32 lowercase hex characters
// → "3ed106d8182ba132e7ba5998a5ab1ef3"
```

**Rules:**
- Always use `|` as the delimiter
- `azure_host` must be lowercase with no protocol, port, or trailing slash
- `run_id` is always treated as a string (no integer formatting)
- Output must be 32 lowercase hexadecimal characters

---

### SSE Streaming

Add `?stream=true` to either endpoint. The response is a stream of `data: {...}\n\n` lines.

| `type` | Sent when | Key fields |
|--------|-----------|------------|
| `chunk` | Each text delta from the AI | `text`, `backend_id` |
| `done` | Stream finished successfully | `conversation_id`, `backend_id`, `request_id` |
| `error` | Stream failed mid-flight | `code`, `message`, `request_id` |

```
data: {"type":"chunk","text":"Root cause:","backend_id":"primary"}

data: {"type":"chunk","text":" missing dependency","backend_id":"primary"}

data: {"type":"done","conversation_id":null,"backend_id":"primary","request_id":"abc-123"}
```

For `openai` format backends the BFF forwards `stream: true` upstream and parses its SSE natively. For `simple` format backends the response is buffered and emitted as one chunk.

---

## Configuration

All variables are prefixed `AZ_AI_BFF_`. Copy `.env.example` to `.env` and edit.

### Required

| Variable | Description |
|----------|-------------|
| `TRUSTED_PROXY_ADDRESSES` | JSON list of IPs allowed to set proxy headers, e.g. `["nginx"]` |
| `BACKEND_URL` | AI backend URL |
| `BACKEND_MODEL` | Model name passed to the backend |
| `AZURE_TOKEN` | PAT for your on-premise TFS/Azure DevOps server |
| `RESPONSE_TIMEOUT_SECONDS` | Seconds to wait for the AI backend |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `ACCESS_CREDENTIAL` | _(empty)_ | Bearer token callers must supply. Leave empty to disable auth. |

### AI Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_FORMAT` | `simple` | `simple` or `openai` |
| `BACKEND_CREDENTIAL` | _(empty)_ | Bearer token forwarded to the AI backend |
| `BACKEND_ID` | `primary` | Identifier used in logs and responses |
| `BACKEND_REASONING_MODE` | `false` | Set `true` for models that reply in `reasoning_content` (DeepSeek, Qwen3, etc.) |
| `MAX_TOKENS` | _(unset)_ | If set, appended as `max_tokens` on every AI request |

### Concurrency

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_AI_REQUESTS` | `5` | Maximum simultaneous AI backend calls across all endpoints. The next request over the limit receives **429**. Set to `0` to disable. |

The concurrency limit applies globally — `/api/v1/chat` and `/api/v1/analyze` share the same pool. Five concurrent users means the sixth gets a 429 and should retry.

### Token Budget (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_BUDGET_ENABLED` | `false` | Enable character-based token estimation |
| `INPUT_TOKEN_BUDGET` | `4096` | Max estimated input tokens |
| `OUTPUT_TOKEN_LIMIT` | `1024` | Max output tokens sent to the backend |
| `TOKEN_CHARACTERS_PER_TOKEN` | `4.0` | Characters per token for estimation |

### Multi-AI (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `MULTI_AI_ENABLED` | `false` | Fan out to all configured backends in parallel |
| `ADDITIONAL_BACKENDS` | `[]` | JSON list of `AIBackendSettings` objects |

### Server / Network

| Variable | Default | Description |
|----------|---------|-------------|
| `INTERNAL_HOST` | `127.0.0.1` | Bind address |
| `INTERNAL_PORT` | `8000` | Bind port |
| `CONNECT_TIMEOUT_SECONDS` | `5` | HTTP connection timeout |
| `ALLOWED_EXTENSION_ORIGINS` | `[]` | CORS origins for the browser extension |

---

## How the Four Features Work

### 1. AI Concurrency Cap

`AZ_AI_BFF_MAX_CONCURRENT_AI_REQUESTS` sets the maximum number of requests that can be calling the AI backend simultaneously. When the cap is reached any new request is rejected immediately with **429** — the semaphore is never waited on, so there is no hidden queue building up behind slow builds.

The slot is held for the full duration of the AI call, including the entire SSE stream for streaming requests.

### 2. Per-Pipeline In-Flight Lock

The BFF fetches the build object from Azure DevOps to extract the **pipeline definition ID** (the `definition.id` field). This is the identifier for the pipeline definition, not for a specific run.

If any run of that pipeline definition is currently being analyzed (by any client), the new request gets **409 `pipeline_already_processing`** immediately.

The lock is in-memory and always released when the request finishes — whether it succeeds, errors, or times out. This means retries after failures are always allowed.

**Why definition ID, not run ID?**  
The same pipeline can produce many runs. Allowing two different runs of the same pipeline to be analyzed at the same time would flood the AI backend with parallel requests and create a race on writing the completion tag. One at a time per pipeline is the right constraint.

### 3. Completed-Run Idempotency via Azure DevOps Tags

The old in-memory `_completed` set had two problems: it reset on restart, and a curl user bypassing the UI could lock a pipeline forever without the analysis ever appearing in the UI.

The new mechanism writes a **build tag** directly to the Azure DevOps run when analysis completes:

```
PUT /_apis/build/builds/{buildId}/tags/ai-analyzed?api-version=6.0
```

Before processing any request, the BFF reads the tags for that run:

```
GET /_apis/build/builds/{buildId}/tags?api-version=6.0
```

If `ai-analyzed` is present → **409 `build_already_analyzed`**.

This state is:
- **Durable** — survives app restarts
- **Shared** — visible to every client and every BFF instance
- **Source-of-truth is Azure DevOps** — not the BFF process memory

If the tags endpoint returns an error (503, network failure), the check **fails open** — the analysis is allowed to proceed rather than blocking indefinitely.

### 4. Log Compaction

Raw Azure Pipelines logs are noisy. Before the AI sees anything, `compact_log` runs the following pipeline:

1. **Strip ANSI escape codes** — colour codes from coloured output
2. **Strip timestamps** — every line starts with `2024-05-10T12:34:56.1234567Z`; dropped entirely
3. **Drop boilerplate** — `##[section]`, `##[command]`, agent setup lines, task-downloading lines, blank lines
4. **Collapse duplicates** — a run of 12 identical `Warning: retrying` lines becomes `[×12] Warning: retrying`
5. **Keep signal + context** — lines matching `error`, `warning`, `exception`, `traceback`, `fatal`, `fail`, `cannot`, `could not`, `unable to`, `not found`, `denied`, `stack trace`, or stack-frame patterns are kept, plus 3 surrounding context lines
6. **Enforce char budget** — hard-truncates at 24 000 characters with a `[... N lines truncated]` note

The AI receives a compact, signal-dense document instead of thousands of lines of setup scaffolding.

---

## Project Structure

```
src/az_ai_bff/
├── app.py               # Routes, lifespan, feature wiring
├── analyze.py           # Azure DevOps log fetcher, tag read/write, compact_log
├── concurrency.py       # AISemaphore (rate limit) + PipelineInFlightTracker (dedup)
├── backend.py           # AI backend HTTP client (simple + openai formats)
├── chat.py              # ChatService — non-streaming and streaming paths
├── config.py            # Pydantic settings (AZ_AI_BFF_* env vars)
├── models.py            # All request/response Pydantic models
├── orchestration.py     # Multi-AI fan-out and response synthesis
├── security.py          # Bearer token auth dependency
├── middleware.py        # Request ID propagation + access logging
├── token_management.py  # Token budget estimation and context trimming
├── errors.py            # Centralized exception → HTTP response mapping
└── logging_config.py    # JSON structured logging formatter

tests/
├── test_app.py              # Core request/response integration tests
├── test_analyze.py          # URL parsing, log fetching, analyze route
├── test_backend.py          # AI backend HTTP client unit tests
├── test_build_tracker.py    # PipelineInFlightTracker (feat2), tag idempotency (feat3), compact_log (feat4)
├── test_concurrency.py      # AISemaphore (feat1) tests
├── test_config.py           # Settings validation tests
├── test_orchestration.py    # Multi-AI fan-out tests
├── test_streaming.py        # SSE streaming tests
└── test_token_management.py # Token budget tests
```

---

## Running Tests

```bash
pip install -e .
pytest tests/
```

---

## CI/CD

GitHub Actions builds and pushes a multi-architecture Docker image (amd64/arm64) to Docker Hub on every push to `main`. Tests run on every pull request.

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full setup instructions including nginx TLS configuration, Docker Compose options, and environment variable reference.

---

## License

See [LICENSE](LICENSE).

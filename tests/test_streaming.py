"""Tests for SSE streaming on /api/v1/chat and /api/v1/analyze."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from az_ai_bff.app import create_app
from az_ai_bff.config import Settings

AUTH = {"Authorization": "Bearer extension-secret"}

BUILD_URL = "https://stageazure.asax.ir/tfs/AsaProjects/967834e8-8a18-4a43-88aa-02bd7f158589/_build/results?buildId=1116401"
_API_BASE = "https://stageazure.asax.ir/tfs/AsaProjects/967834e8-8a18-4a43-88aa-02bd7f158589/_apis"

BUILD_DETAIL = {"_links": {"timeline": {"href": f"{_API_BASE}/build/builds/1116401/timeline"}}}
TIMELINE = {
    "records": [
        {
            "type": "Task",
            "result": "failed",
            "name": "Compile",
            "log": {"id": 42, "url": f"{_API_BASE}/build/builds/1116401/logs/42"},
        }
    ]
}
LOG_TEXT = "step 1 ok\nstep 2 error: build failed\nstep 3 ok"


def _make_settings() -> Settings:
    return Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential="extension-secret",
        backend_url="https://ai.example.test/v1/chat",
        backend_model="example-model",
        backend_credential="backend-secret",
        backend_format="openai",
        response_timeout_seconds=30,
        azure_token="test-azure-pat",
        require_client_signature=False,
    )


def _sse_chunks(chunks: list[str]) -> bytes:
    """Build a minimal OpenAI-compatible SSE response body from text chunks."""
    lines = []
    for text in chunks:
        payload = json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": None}]})
        lines.append(f"data: {payload}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _azure_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/builds/1116401") and "timeline" not in path and "logs" not in path:
        return httpx.Response(200, json=BUILD_DETAIL)
    if "timeline" in path:
        return httpx.Response(200, json=TIMELINE)
    if path.endswith("/logs/42"):
        return httpx.Response(200, text=LOG_TEXT)
    return httpx.Response(404)


def _parse_sse(raw: str) -> list[dict]:
    """Parse SSE response body into a list of JSON event dicts."""
    events = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


# ---------------------------------------------------------------------------
# /api/v1/chat?stream=true
# ---------------------------------------------------------------------------

def test_chat_stream_yields_chunks_then_done() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_chunks(["Hello", ", ", "world!"]))

    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/api/v1/chat?stream=true",
            json={"message": "hi"},
            headers={**AUTH, "X-Request-ID": "stream-test"},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["x-request-id"] == "stream-test"

    events = _parse_sse(response.text)
    chunks = [e for e in events if e["type"] == "chunk"]
    done = [e for e in events if e["type"] == "done"]

    assert [c["text"] for c in chunks] == ["Hello", ", ", "world!"]
    assert len(done) == 1
    assert done[0]["request_id"] == "stream-test"
    assert done[0]["backend_id"] == "primary"


def test_chat_stream_requires_auth() -> None:
    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(lambda _: httpx.Response(200)))) as client:
        response = client.post("/api/v1/chat?stream=true", json={"message": "hi"})
    assert response.status_code == 401


def test_chat_stream_backend_unavailable_yields_error_event() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/api/v1/chat?stream=true",
            json={"message": "hi"},
            headers={**AUTH, "X-Request-ID": "err-test"},
        )

    assert response.status_code == 200  # HTTP 200 — error is in the SSE body
    events = _parse_sse(response.text)
    errors = [e for e in events if e["type"] == "error"]
    assert errors
    assert errors[0]["code"] == "ai_backend_unavailable"
    assert errors[0]["request_id"] == "err-test"


def test_chat_non_stream_still_works() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "buffered reply"}, "finish_reason": "stop"}]})

    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(handler))) as client:
        response = client.post("/api/v1/chat", json={"message": "hi"}, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["answer"] == "buffered reply"


# ---------------------------------------------------------------------------
# /api/v1/analyze?stream=true
# ---------------------------------------------------------------------------

def test_analyze_stream_fetches_logs_then_streams_ai_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "stageazure" in request.url.host:
            return _azure_handler(request)
        return httpx.Response(200, content=_sse_chunks(["Root cause: ", "missing dep"]))

    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/api/v1/analyze?stream=true",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers={**AUTH, "X-Request-ID": "analyze-stream"},
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    events = _parse_sse(response.text)
    chunks = [e for e in events if e["type"] == "chunk"]
    done = [e for e in events if e["type"] == "done"]

    assert "".join(c["text"] for c in chunks) == "Root cause: missing dep"
    assert done[0]["request_id"] == "analyze-stream"


def test_analyze_stream_unsupported_returns_json_not_sse() -> None:
    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(lambda _: httpx.Response(200)))) as client:
        response = client.post(
            "/api/v1/analyze?stream=true",
            json={"message": "azure build pipeline improvement", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "unsupported"

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from az_ai_bff.analyze import (
    AzureAuthError,
    AzureDevOpsLogFetcher,
    BuildURLParseError,
    grep_errors,
    parse_build_url,
)
from az_ai_bff.app import create_app
from az_ai_bff.config import Settings


AUTH = {"Authorization": "Bearer extension-secret"}

BUILD_URL = "https://stageazure.asax.ir/tfs/AsaProjects/967834e8-8a18-4a43-88aa-02bd7f158589/_build/results?buildId=1116401"

# Mirrors what parse_build_url produces for BUILD_URL
_API_BASE = "https://stageazure.asax.ir/tfs/AsaProjects/967834e8-8a18-4a43-88aa-02bd7f158589/_apis"

BUILD_DETAIL = {
    "_links": {
        "timeline": {"href": f"{_API_BASE}/build/builds/1116401/timeline"}
    }
}

TIMELINE = {
    "records": [
        {
            "type": "Task",
            "result": "failed",
            "name": "Compile",
            "log": {"id": 42, "url": f"{_API_BASE}/build/builds/1116401/logs/42"},
        },
        {
            "type": "Task",
            "result": "succeeded",
            "name": "Test",
            "log": {"id": 43, "url": f"{_API_BASE}/build/builds/1116401/logs/43"},
        },
    ]
}

LOG_TEXT = "\n".join(
    ["step 1 ok", "step 2 ok", "step 3 error: build failed", "step 4 failed", "step 5 ok"]
)


def _azure_handler(request: httpx.Request) -> httpx.Response:
    """Mock all Azure DevOps API calls for the test build."""
    path = request.url.path
    if path.endswith(f"/builds/1116401") and "timeline" not in path and "logs" not in path:
        return httpx.Response(200, json=BUILD_DETAIL)
    if "timeline" in path:
        return httpx.Response(200, json=TIMELINE)
    if path.endswith("/logs/42"):
        return httpx.Response(200, text=LOG_TEXT)
    return httpx.Response(404)


# ---------------------------------------------------------------------------
# parse_build_url
# ---------------------------------------------------------------------------

def test_parse_build_url_extracts_components() -> None:
    parsed = parse_build_url(BUILD_URL)
    assert parsed.build_id == "1116401"
    assert parsed.project == "967834e8-8a18-4a43-88aa-02bd7f158589"
    assert parsed.base_url == "https://stageazure.asax.ir/tfs/AsaProjects"


def test_parse_build_url_accepts_api_url_format() -> None:
    api_url = "https://stageazure.asax.ir/tfs/AsaProjects/967834e8-8a18-4a43-88aa-02bd7f158589/_apis/build/builds/1116401"
    parsed = parse_build_url(api_url)
    assert parsed.build_id == "1116401"
    assert parsed.project == "967834e8-8a18-4a43-88aa-02bd7f158589"
    assert parsed.base_url == "https://stageazure.asax.ir/tfs/AsaProjects"


def test_parse_build_url_rejects_invalid_urls() -> None:
    with pytest.raises(BuildURLParseError):
        parse_build_url("https://example.com/not-a-build-url")

    with pytest.raises(BuildURLParseError):
        parse_build_url("https://example.com/project/_build/results")  # missing buildId


# ---------------------------------------------------------------------------
# grep_errors
# ---------------------------------------------------------------------------

def test_grep_errors_returns_context_lines() -> None:
    lines = [f"line {i}" for i in range(20)]
    lines[10] = "line 10 ERROR happened"
    log = "\n".join(lines)

    result = grep_errors(log, context=2)
    result_lines = result.splitlines()

    assert "line 10 ERROR happened" in result_lines
    assert "line 8" in result_lines    # 2 before
    assert "line 12" in result_lines   # 2 after
    assert "line 0" not in result_lines  # outside context


def test_grep_errors_returns_full_log_when_no_errors() -> None:
    log = "all good\nnothing wrong here"
    assert grep_errors(log) == log


# ---------------------------------------------------------------------------
# AzureDevOpsLogFetcher (mocked HTTP, using asyncio.run like test_backend.py)
# ---------------------------------------------------------------------------

def test_fetcher_returns_grepped_error_log() -> None:
    async def exercise() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_azure_handler)) as client:
            fetcher = AzureDevOpsLogFetcher(client, "my-pat")
            return await fetcher.fetch_error_logs(BUILD_URL)

    result = asyncio.run(exercise())
    assert "error: build failed" in result
    assert "Compile" in result


def test_fetcher_raises_auth_error_on_401() -> None:
    def auth_fail(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(auth_fail)) as client:
            fetcher = AzureDevOpsLogFetcher(client, "bad-token")
            with pytest.raises(AzureAuthError):
                await fetcher.fetch_error_logs(BUILD_URL)

    asyncio.run(exercise())


# ---------------------------------------------------------------------------
# /api/v1/analyze route (integration via TestClient)
# ---------------------------------------------------------------------------

def _make_settings(azure_token: str = "valid-pat") -> Settings:
    return Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential="extension-secret",
        backend_url="https://ai.example.test/v1/chat",
        backend_model="example-model",
        backend_credential="backend-secret",
        response_timeout_seconds=30,
        azure_token=azure_token,
        require_client_signature=False,
    )


def _combined_handler(ai_response: dict | None = None, azure_status: int = 200):
    """Returns a handler that routes Azure calls vs AI backend calls."""
    def handler(request: httpx.Request) -> httpx.Response:
        if "stageazure" in request.url.host:
            if azure_status != 200:
                return httpx.Response(azure_status)
            return _azure_handler(request)
        # AI backend
        return httpx.Response(200, json=ai_response or {"answer": "ai analysis result"})
    return handler


def test_analyze_returns_unsupported_for_type_super() -> None:
    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(_combined_handler()))) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "super"},
            headers=AUTH,
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "unsupported"


def test_analyze_returns_unsupported_for_improvement_message() -> None:
    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(_combined_handler()))) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline improvement", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "unsupported"


def test_analyze_fetches_logs_and_returns_ai_answer() -> None:
    ai_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "stageazure" in request.url.host:
            return _azure_handler(request)
        ai_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "root cause: missing dependency"})

    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "root cause: missing dependency"
    assert "request_id" in response.json()
    # Error log content was sent to the AI backend
    assert ai_payloads
    payload_str = json.dumps(ai_payloads[0])
    assert "error" in payload_str.lower()


def test_analyze_returns_502_when_azure_token_invalid() -> None:
    with TestClient(create_app(_make_settings("bad-token"), http_transport=httpx.MockTransport(_combined_handler(azure_status=401)))) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )
    assert response.status_code == 502
    assert response.json()["code"] == "azure_auth_error"


def test_analyze_returns_503_when_azure_token_not_configured() -> None:
    """azure_token is now required — Settings raises ValidationError at startup if missing."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            trusted_proxy_addresses=["127.0.0.1"],
            access_credential="extension-secret",
            backend_url="https://ai.example.test/v1/chat",
            backend_model="example-model",
            backend_credential="backend-secret",
            response_timeout_seconds=30,
            # azure_token intentionally omitted
        )
    invalid_fields = {item["loc"][0] for item in exc_info.value.errors()}
    assert "azure_token" in invalid_fields


def test_analyze_requires_auth_when_credential_configured() -> None:
    with TestClient(create_app(_make_settings(), http_transport=httpx.MockTransport(_combined_handler()))) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
        )
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"

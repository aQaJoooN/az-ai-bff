"""Tests for MCPClient, run_agentic_analysis, and the analyze route agentic fork."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from az_ai_bff.app import create_app
from az_ai_bff.config import Settings
from az_ai_bff.mcp_client import AgenticLoopError, MCPClient, run_agentic_analysis


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BUILD_URL = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589/_build/results?buildId=1116401"
)
_API_BASE = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589/_apis"
)
AUTH = {"Authorization": "Bearer secret"}

_BUILD_DETAIL = {
    "definition": {"id": 42},
    "_links": {"timeline": {"href": f"{_API_BASE}/build/builds/1116401/timeline"}},
    "status": "completed",
}
_TIMELINE = {
    "records": [
        {
            "type": "Task",
            "result": "failed",
            "name": "Compile",
            "log": {"id": 7, "url": f"{_API_BASE}/build/builds/1116401/logs/7"},
        }
    ]
}
_LOG_TEXT = "step 1 ok\nerror: build failed\nstep 3 ok"

_TOOLS_RESPONSE = {
    "tools": [
        {
            "name": "get_pipeline_run_yaml",
            "description": "Get the expanded YAML for a pipeline run",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "runId": {"type": "integer", "description": "Run ID"},
                    "projectId": {"type": "string", "description": "Project"},
                },
                "required": ["runId"],
            },
        },
        {
            "name": "list_pipeline_runs",
            "description": "List recent runs for a pipeline",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pipelineId": {"type": "integer"},
                    "top": {"type": "integer"},
                    "orderBy": {"type": "string"},
                },
                "required": ["pipelineId", "top", "orderBy"],
            },
        },
    ]
}


def _mcp_rpc_response(result: object) -> httpx.Response:
    return httpx.Response(
        200,
        json={"jsonrpc": "2.0", "id": 1, "result": result},
    )


def _mcp_tool_response(text: str, is_error: bool = False) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        },
    )


def _azure_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/builds/1116401") and "timeline" not in path and "logs" not in path:
        return httpx.Response(200, json=_BUILD_DETAIL)
    if "timeline" in path:
        return httpx.Response(200, json=_TIMELINE)
    if path.endswith("/logs/7"):
        return httpx.Response(200, text=_LOG_TEXT)
    if "tags" in path:
        return httpx.Response(200, json=[])
    return httpx.Response(404)


def _make_settings(mcp_enabled: bool = False, mcp_url: str | None = None, mcp_token: str | None = None) -> Settings:
    return Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential="secret",
        backend_url="http://ai.test/v1/chat/completions",
        backend_model="test-model",
        response_timeout_seconds=30,
        azure_token="valid-pat",
        require_client_signature=False,
        mcp_enabled=mcp_enabled,
        mcp_url=mcp_url or ("http://mcp.test:8888" if mcp_enabled else None),
        mcp_token=mcp_token,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_mcp_disabled_by_default() -> None:
    s = _make_settings()
    assert s.mcp_enabled is False
    assert s.mcp_url is None
    assert s.mcp_token is None


def test_mcp_enabled_requires_url() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="mcp_url is required"):
        Settings(
            trusted_proxy_addresses=["127.0.0.1"],
            backend_url="http://ai.test/chat",
            backend_model="m",
            response_timeout_seconds=30,
            azure_token="tok",
            mcp_enabled=True,
            mcp_url=None,
        )


def test_mcp_token_stored_as_secret() -> None:
    s = _make_settings(mcp_enabled=True, mcp_token="my-tok")
    assert s.mcp_token.get_secret_value() == "my-tok"


def test_blank_mcp_token_treated_as_none() -> None:
    s = _make_settings(mcp_enabled=True, mcp_token="   ")
    assert s.mcp_token is None


# ---------------------------------------------------------------------------
# MCPClient.list_tools
# ---------------------------------------------------------------------------

def test_list_tools_returns_openai_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _mcp_rpc_response(_TOOLS_RESPONSE)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            mc = MCPClient(client, "http://mcp.test")
            return await mc.list_tools()

    tools = asyncio.run(run())
    assert len(tools) == 2
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_pipeline_run_yaml"
    assert "parameters" in tools[0]["function"]


def test_list_tools_returns_empty_on_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            mc = MCPClient(client, "http://mcp.test")
            return await mc.list_tools()

    assert asyncio.run(run()) == []


# ---------------------------------------------------------------------------
# MCPClient.call_tool
# ---------------------------------------------------------------------------

def test_call_tool_returns_text_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _mcp_tool_response("trigger:\n  - main")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            mc = MCPClient(client, "http://mcp.test")
            return await mc.call_tool("get_pipeline_run_yaml", {"runId": 1})

    result = asyncio.run(run())
    assert "trigger:" in result


def test_call_tool_returns_error_string_on_mcp_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _mcp_tool_response("Only available for Azure DevOps Services", is_error=True)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            mc = MCPClient(client, "http://mcp.test")
            return await mc.call_tool("get_me", {})

    result = asyncio.run(run())
    assert "[tool error:" in result


def test_call_tool_returns_error_string_on_network_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            mc = MCPClient(client, "http://mcp.test")
            return await mc.call_tool("get_pipeline_run_yaml", {"runId": 1})

    result = asyncio.run(run())
    assert "[tool error:" in result


def test_call_tool_returns_error_string_on_http_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            mc = MCPClient(client, "http://mcp.test")
            return await mc.call_tool("get_pipeline_run_yaml", {"runId": 1})

    result = asyncio.run(run())
    assert "[tool error:" in result


# ---------------------------------------------------------------------------
# MCPClient auth headers
# ---------------------------------------------------------------------------

def test_bearer_token_sent_when_configured() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("Authorization", ""))
        return _mcp_rpc_response(_TOOLS_RESPONSE)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            mc = MCPClient(client, "http://mcp.test", token="tok-123")
            await mc.list_tools()

    asyncio.run(run())
    assert captured[0] == "Bearer tok-123"


def test_no_auth_header_when_no_token() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers.get("Authorization", ""))
        return _mcp_rpc_response(_TOOLS_RESPONSE)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            mc = MCPClient(client, "http://mcp.test", token=None)
            await mc.list_tools()

    asyncio.run(run())
    assert captured[0] == ""


# ---------------------------------------------------------------------------
# run_agentic_analysis
# ---------------------------------------------------------------------------

def _ai_stop_response(answer: str) -> dict:
    """AI backend response with finish_reason=stop."""
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": answer, "reasoning_content": ""},
            }
        ]
    }


def _ai_tool_call_response(tool_name: str, arguments: dict, call_id: str = "tc1") -> dict:
    """AI backend response with finish_reason=tool_calls."""
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ]
    }


def test_agentic_analysis_single_round_no_tool_calls() -> None:
    """AI answers immediately without calling any tools."""
    mcp_call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "mcp.test" in host:
            mcp_call_count["n"] += 1
            return _mcp_rpc_response(_TOOLS_RESPONSE)
        # AI backend
        return httpx.Response(200, json=_ai_stop_response("## Problem\nRoot cause.\n## Solution\n1. Fix it.\n## Summary\nFixed."))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as ai_client:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mcp_http:
                mc = MCPClient(mcp_http, "http://mcp.test")
                return await run_agentic_analysis(
                    error_logs="error: build failed",
                    mcp_client=mc,
                    ai_base_url="http://ai.test/v1/chat/completions",
                    ai_model="test-model",
                    ai_credential="",
                    http_client=ai_client,
                )

    result = asyncio.run(run())
    assert "Problem" in result
    assert "Solution" in result


def test_agentic_analysis_one_tool_call_then_stop() -> None:
    """AI calls one tool, gets result, then produces final answer."""
    call_sequence: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "mcp.test" in host:
            body = json.loads(request.content)
            method = body.get("method", "")
            if method == "tools/list":
                return _mcp_rpc_response(_TOOLS_RESPONSE)
            # tools/call
            tool = body.get("params", {}).get("name", "")
            call_sequence.append(tool)
            return _mcp_tool_response("trigger:\n  - main\nsteps:\n  - script: echo hello")

        # AI backend — first call returns tool_calls, second returns stop
        if len(call_sequence) == 0:
            return httpx.Response(200, json=_ai_tool_call_response("get_pipeline_run_yaml", {"runId": 1}))
        return httpx.Response(200, json=_ai_stop_response("## Problem\nFound via YAML.\n## Solution\n1. Fix.\n## Summary\nDone."))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared:
            mc = MCPClient(shared, "http://mcp.test")
            return await run_agentic_analysis(
                error_logs="error: nuget restore failed",
                mcp_client=mc,
                ai_base_url="http://ai.test/v1/chat/completions",
                ai_model="test-model",
                ai_credential="",
                http_client=shared,
            )

    result = asyncio.run(run())
    assert "get_pipeline_run_yaml" in call_sequence
    assert "Problem" in result


def test_agentic_analysis_raises_on_ai_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "mcp.test" in request.url.host:
            return _mcp_rpc_response(_TOOLS_RESPONSE)
        raise httpx.ConnectError("refused")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared:
            mc = MCPClient(shared, "http://mcp.test")
            await run_agentic_analysis(
                error_logs="error: build failed",
                mcp_client=mc,
                ai_base_url="http://ai.test/v1/chat/completions",
                ai_model="test-model",
                ai_credential="",
                http_client=shared,
            )

    with pytest.raises(AgenticLoopError):
        asyncio.run(run())


def test_agentic_analysis_tool_error_does_not_abort_loop() -> None:
    """A tool failure is embedded as an error string — the AI still gets a response."""
    ai_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "mcp.test" in host:
            body = json.loads(request.content)
            if body.get("method") == "tools/list":
                return _mcp_rpc_response(_TOOLS_RESPONSE)
            # tool call always fails
            return httpx.Response(503)

        ai_calls["n"] += 1
        if ai_calls["n"] == 1:
            return httpx.Response(200, json=_ai_tool_call_response("get_pipeline_run_yaml", {"runId": 1}))
        return httpx.Response(200, json=_ai_stop_response("## Problem\nTool failed but I answered anyway.\n## Solution\n1. Fix.\n## Summary\nDone."))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared:
            mc = MCPClient(shared, "http://mcp.test")
            return await run_agentic_analysis(
                error_logs="error: build failed",
                mcp_client=mc,
                ai_base_url="http://ai.test/v1/chat/completions",
                ai_model="test-model",
                ai_credential="",
                http_client=shared,
            )

    result = asyncio.run(run())
    assert "Problem" in result


# ---------------------------------------------------------------------------
# Integration: analyze route — mcp_enabled=false (standard path unchanged)
# ---------------------------------------------------------------------------

def test_analyze_standard_path_unchanged_when_mcp_disabled() -> None:
    """With mcp_enabled=false the route behaves exactly as before."""
    ai_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "stageazure" in request.url.host:
            return _azure_handler(request)
        ai_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "standard answer"})

    settings = _make_settings(mcp_enabled=False)
    with TestClient(create_app(settings, http_transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "standard answer"
    # Standard path uses the simple backend format — no "tools" key in payload
    assert "tools" not in json.dumps(ai_payloads)


# ---------------------------------------------------------------------------
# Integration: analyze route — mcp_enabled=true (agentic path)
# ---------------------------------------------------------------------------

def test_analyze_agentic_path_sends_tools_to_ai() -> None:
    """With mcp_enabled=true the AI request includes the MCP tool catalogue."""
    ai_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "stageazure" in host:
            return _azure_handler(request)
        if "mcp.test" in host:
            body = json.loads(request.content)
            if body.get("method") == "tools/list":
                return _mcp_rpc_response(_TOOLS_RESPONSE)
            return _mcp_tool_response("yaml content here")
        # AI backend
        ai_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "## Problem\nP\n## Solution\n1. S\n## Summary\nX.", "reasoning_content": ""},
                }]
            },
        )

    settings = _make_settings(mcp_enabled=True, mcp_url="http://mcp.test:8888")
    with TestClient(create_app(settings, http_transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.json()["answer"] != ""
    # The AI must have received a tools array
    assert any("tools" in p for p in ai_payloads)


def test_analyze_agentic_path_returns_502_on_ai_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "stageazure" in request.url.host:
            return _azure_handler(request)
        if "mcp.test" in request.url.host:
            body = json.loads(request.content)
            if body.get("method") == "tools/list":
                return _mcp_rpc_response(_TOOLS_RESPONSE)
            return _mcp_tool_response("ok")
        # AI backend always fails
        return httpx.Response(503)

    settings = _make_settings(mcp_enabled=True, mcp_url="http://mcp.test:8888")
    with TestClient(create_app(settings, http_transport=httpx.MockTransport(handler))) as client:
        response = client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )

    assert response.status_code == 502
    assert response.json()["code"] == "ai_backend_error"


def test_analyze_agentic_path_sends_mcp_token() -> None:
    """When mcp_token is set, the MCP server receives an Authorization header."""
    mcp_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "stageazure" in request.url.host:
            return _azure_handler(request)
        if "mcp.test" in request.url.host:
            mcp_auth.append(request.headers.get("Authorization", ""))
            body = json.loads(request.content)
            if body.get("method") == "tools/list":
                return _mcp_rpc_response(_TOOLS_RESPONSE)
            return _mcp_tool_response("ok")
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "## Problem\nP\n## Solution\n1. S\n## Summary\nX.", "reasoning_content": ""},
                }]
            },
        )

    settings = _make_settings(mcp_enabled=True, mcp_url="http://mcp.test:8888", mcp_token="mcp-secret")
    with TestClient(create_app(settings, http_transport=httpx.MockTransport(handler))) as client:
        client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )

    assert any("Bearer mcp-secret" in h for h in mcp_auth)


def test_analyze_agentic_path_no_auth_header_when_no_token() -> None:
    mcp_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if "stageazure" in request.url.host:
            return _azure_handler(request)
        if "mcp.test" in request.url.host:
            mcp_auth.append(request.headers.get("Authorization", ""))
            body = json.loads(request.content)
            if body.get("method") == "tools/list":
                return _mcp_rpc_response(_TOOLS_RESPONSE)
            return _mcp_tool_response("ok")
        return httpx.Response(
            200,
            json={
                "choices": [{
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "## Problem\nP\n## Solution\n1. S\n## Summary\nX.", "reasoning_content": ""},
                }]
            },
        )

    settings = _make_settings(mcp_enabled=True, mcp_url="http://mcp.test:8888", mcp_token=None)
    with TestClient(create_app(settings, http_transport=httpx.MockTransport(handler))) as client:
        client.post(
            "/api/v1/analyze",
            json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
            headers=AUTH,
        )

    assert all(h == "" for h in mcp_auth)

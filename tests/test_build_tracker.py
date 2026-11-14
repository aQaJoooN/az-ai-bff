"""Tests for feat2 (PipelineInFlightTracker), feat3 (tag-based idempotency),
and feat4 (compact_log).
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from az_ai_bff.analyze import AI_ANALYZED_TAG, compact_log, grep_errors
from az_ai_bff.app import create_app
from az_ai_bff.concurrency import PipelineInFlightTracker
from az_ai_bff.config import Settings


AUTH = {"Authorization": "Bearer extension-secret"}

BUILD_URL = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589"
    "/_build/results?buildId=1116401"
)

_API_BASE = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589/_apis"
)

# Build detail includes a definition ID so the route can key the pipeline lock.
BUILD_DETAIL = {
    "_links": {"timeline": {"href": f"{_API_BASE}/build/builds/1116401/timeline"}},
    "definition": {"id": 42, "name": "MyPipeline"},
}
BUILD_DETAIL_OTHER_PIPELINE = {
    "_links": {"timeline": {"href": f"{_API_BASE}/build/builds/9999/timeline"}},
    "definition": {"id": 99, "name": "OtherPipeline"},
}
TIMELINE = {
    "records": [
        {
            "type": "Task",
            "result": "failed",
            "name": "Compile",
            "log": {"id": 7, "url": f"{_API_BASE}/build/builds/1116401/logs/7"},
        }
    ]
}
LOG_TEXT = "step 1 ok\nstep 2 ok\nstep 3 error: build failed\nstep 4 failed\nstep 5 ok"


def _make_settings() -> Settings:
    return Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential="extension-secret",
        backend_url="https://ai.example.test/v1/chat",
        backend_model="example-model",
        backend_credential="backend-secret",
        response_timeout_seconds=30,
        azure_token="valid-pat",
    )


def _azure_handler(
    request: httpx.Request,
    *,
    tags: list[str] | None = None,
    build_detail: dict | None = None,
) -> httpx.Response:
    """Mock all Azure DevOps API calls needed for tests.

    ``tags``         — list returned by the GET tags endpoint (default: empty)
    ``build_detail`` — build JSON (default: BUILD_DETAIL with pipeline id=42)
    """
    tags = tags or []
    build_detail = build_detail or BUILD_DETAIL
    path = request.url.path

    # GET build detail
    if path.endswith("/builds/1116401") and "timeline" not in path and "logs" not in path and "tags" not in path:
        return httpx.Response(200, json=build_detail)
    # GET tags
    if path.endswith(f"/builds/1116401/tags") and request.method == "GET":
        return httpx.Response(200, json=tags)
    # PUT tag (mark analyzed)
    if f"/tags/{AI_ANALYZED_TAG}" in path and request.method == "PUT":
        return httpx.Response(200, json=tags + [AI_ANALYZED_TAG])
    # GET timeline
    if "timeline" in path:
        return httpx.Response(200, json=TIMELINE)
    # GET log
    if path.endswith("/logs/7"):
        return httpx.Response(200, text=LOG_TEXT)
    return httpx.Response(404)


def _make_combined_handler(tags: list[str] | None = None, build_detail: dict | None = None):
    """Return a handler that routes Azure vs AI backend calls."""
    def handler(request: httpx.Request) -> httpx.Response:
        if "stageazure" in request.url.host:
            return _azure_handler(request, tags=tags, build_detail=build_detail)
        return httpx.Response(200, json={"answer": "ai result"})
    return handler


# ===========================================================================
# feat2 — PipelineInFlightTracker unit tests
# ===========================================================================

class TestPipelineTrackerUnit:
    def test_first_registration_succeeds(self):
        tracker = PipelineInFlightTracker()
        slot = tracker.try_start("42")
        assert slot.ok
        assert slot.reason == ""

    def test_duplicate_pipeline_rejected(self):
        tracker = PipelineInFlightTracker()
        slot1 = tracker.try_start("42")
        assert slot1.ok
        slot2 = tracker.try_start("42")
        assert not slot2.ok
        assert slot2.reason == "pipeline_already_processing"

    def test_different_pipelines_are_independent(self):
        tracker = PipelineInFlightTracker()
        s1 = tracker.try_start("1")
        s2 = tracker.try_start("2")
        assert s1.ok
        assert s2.ok

    def test_context_manager_releases_on_exit(self):
        tracker = PipelineInFlightTracker()
        with tracker.try_start("42"):
            assert "42" in tracker.in_flight
        assert "42" not in tracker.in_flight

    def test_failed_analysis_releases_lock_for_retry(self):
        """The lock is always released on context exit, allowing retries."""
        tracker = PipelineInFlightTracker()
        with tracker.try_start("42"):
            pass  # simulates a failed or errored analysis

        assert "42" not in tracker.in_flight
        # Retry is allowed
        retry = tracker.try_start("42")
        assert retry.ok


# ===========================================================================
# feat2 — HTTP 409 for two runs of the same pipeline
# ===========================================================================

class TestPipelineTrackerHTTP:
    def test_in_flight_pipeline_returns_409(self):
        """Pre-fill the tracker then fire the same pipeline — expect 409."""
        app = create_app(_make_settings(), http_transport=httpx.MockTransport(_make_combined_handler()))
        with TestClient(app) as client:
            tracker = client.app.state.pipeline_tracker

            async def _prefill():
                return tracker.try_start("42")  # pipeline definition id = 42

            held = asyncio.run(_prefill())
            assert held.ok

            response = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
            assert response.status_code == 409
            assert response.json()["code"] == "pipeline_already_processing"
            assert "request_id" in response.json()

            held.__exit__(None, None, None)

    def test_different_pipeline_not_blocked(self):
        """A second pipeline definition is never blocked by the first."""
        tracker = PipelineInFlightTracker()
        with tracker.try_start("42"):
            slot2 = tracker.try_start("99")
            assert slot2.ok

    def test_failed_analyze_releases_pipeline_lock(self):
        """Azure 401 → 502, pipeline lock released so retry is not 409."""
        def bad_azure(request: httpx.Request) -> httpx.Response:
            # Always return 401 for Azure calls
            if "stageazure" in request.url.host:
                return httpx.Response(401)
            return httpx.Response(200, json={"answer": "ok"})

        app = create_app(_make_settings(), http_transport=httpx.MockTransport(bad_azure))
        with TestClient(app) as client:
            r1 = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
            assert r1.status_code == 502

            # Lock should be released — retry must not get 409
            r2 = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
            assert r2.status_code == 502  # still fails Azure, but NOT 409


# ===========================================================================
# feat3 — Tag-based idempotency (server-side, durable)
# ===========================================================================

class TestTagBasedIdempotency:
    def test_already_analyzed_tag_returns_409(self):
        """If the build already has 'ai-analyzed' tag → 409 build_already_analyzed."""
        app = create_app(
            _make_settings(),
            http_transport=httpx.MockTransport(
                _make_combined_handler(tags=[AI_ANALYZED_TAG])
            ),
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
        assert response.status_code == 409
        assert response.json()["code"] == "build_already_analyzed"

    def test_no_tag_allows_analysis(self):
        """Build with no tags proceeds to full analysis."""
        app = create_app(
            _make_settings(),
            http_transport=httpx.MockTransport(
                _make_combined_handler(tags=[])
            ),
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
        assert response.status_code == 200

    def test_tag_is_written_after_successful_analysis(self):
        """PUT tag call must be made when AI analysis succeeds."""
        put_tag_calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "stageazure" in request.url.host:
                path = request.url.path
                if f"/tags/{AI_ANALYZED_TAG}" in path and request.method == "PUT":
                    put_tag_calls.append(path)
                    return httpx.Response(200, json=[AI_ANALYZED_TAG])
                return _azure_handler(request, tags=[])
            return httpx.Response(200, json={"answer": "ai result"})

        app = create_app(_make_settings(), http_transport=httpx.MockTransport(handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
        assert response.status_code == 200
        assert put_tag_calls, "Expected PUT tag call was never made"

    def test_tag_not_written_when_analysis_fails(self):
        """If the AI backend fails, the tag must NOT be written."""
        put_tag_calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "stageazure" in request.url.host:
                path = request.url.path
                if f"/tags/{AI_ANALYZED_TAG}" in path and request.method == "PUT":
                    put_tag_calls.append(path)
                    return httpx.Response(200, json=[AI_ANALYZED_TAG])
                return _azure_handler(request, tags=[])
            # AI backend: always fail
            return httpx.Response(503)

        app = create_app(_make_settings(), http_transport=httpx.MockTransport(handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
        assert response.status_code != 200
        assert not put_tag_calls, "Tag must not be written when AI backend fails"

    def test_tag_read_failure_does_not_block_analysis(self):
        """If the tags endpoint fails (503), we fail open and allow the analysis."""
        def handler(request: httpx.Request) -> httpx.Response:
            if "stageazure" in request.url.host:
                path = request.url.path
                # Tags GET returns 503
                if path.endswith("/tags") and request.method == "GET":
                    return httpx.Response(503)
                return _azure_handler(request, tags=[])
            return httpx.Response(200, json={"answer": "ai result"})

        app = create_app(_make_settings(), http_transport=httpx.MockTransport(handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
        # Should proceed normally despite tag read failure
        assert response.status_code == 200


# ===========================================================================
# feat4 — compact_log unit tests
# ===========================================================================

class TestCompactLog:
    def test_strips_ansi_codes(self):
        log = "\x1b[31mERROR: something failed\x1b[0m"
        result = compact_log(log)
        assert "\x1b" not in result
        assert "ERROR: something failed" in result

    def test_strips_azure_pipelines_timestamps(self):
        log = "2024-05-10T12:34:56.1234567Z Error: cannot find package"
        result = compact_log(log)
        assert "2024-05-10T" not in result
        assert "Error: cannot find package" in result

    def test_drops_boilerplate_lines(self):
        log = "\n".join([
            "##[section]Starting: Build",
            "##[command]dotnet build",
            "Starting: NuGet",
            "Error: package not found",
            "Finishing: NuGet",
        ])
        result = compact_log(log)
        assert "##[section]" not in result
        assert "##[command]" not in result
        assert "Starting: NuGet" not in result
        assert "Finishing: NuGet" not in result
        assert "Error: package not found" in result

    def test_collapses_duplicate_consecutive_lines(self):
        log = "\n".join(["Warning: retrying"] * 10 + ["Error: gave up"])
        result = compact_log(log)
        assert "[×10]" in result
        assert result.count("Warning: retrying") == 1
        assert "Error: gave up" in result

    def test_keeps_context_around_error_lines(self):
        lines = [f"line {i}" for i in range(20)]
        lines[10] = "Error: build failed here"
        log = "\n".join(lines)
        result = compact_log(log, context_lines=2)
        result_lines = result.splitlines()
        assert "Error: build failed here" in result_lines
        assert "line 8" in result_lines
        assert "line 12" in result_lines
        assert "line 0" not in result_lines

    def test_no_signal_lines_returns_cleaned_full_log(self):
        log = "all good\nnothing wrong here\nstep completed"
        result = compact_log(log)
        assert "all good" in result
        assert "nothing wrong here" in result

    def test_char_budget_truncates_with_note(self):
        log = "\n".join([f"Error: line {i} has a problem here" for i in range(100)])
        result = compact_log(log, char_budget=500)
        assert "truncated" in result
        assert len(result) <= 500 + 60

    def test_budget_not_exceeded_on_small_log(self):
        log = "Error: something small"
        result = compact_log(log, char_budget=10_000)
        assert "truncated" not in result

    def test_real_pipeline_log_shape(self):
        log = "\n".join([
            "2024-05-10T10:00:00.000Z ##[section]Starting: Build .NET",
            "2024-05-10T10:00:01.000Z ##[command]dotnet build /project.sln",
            "2024-05-10T10:00:02.000Z   Determining projects to restore...",
            "2024-05-10T10:00:03.000Z   Restored MyApp.csproj",
            "2024-05-10T10:00:04.000Z   Restored MyApp.csproj",
            "2024-05-10T10:00:05.000Z   Restored MyApp.csproj",
            "2024-05-10T10:00:06.000Z MyApp/Program.cs(42,10): error CS0246: "
            "The type or namespace name 'Foo' could not be found",
            "2024-05-10T10:00:07.000Z Build FAILED.",
            "2024-05-10T10:00:08.000Z ##[section]Finishing: Build .NET",
        ])
        result = compact_log(log)
        assert "CS0246" in result
        assert "Build FAILED" in result
        assert "2024-05-10T" not in result
        assert "##[section]" not in result
        assert "[×3]" in result


# ===========================================================================
# feat4 — compacted log is what reaches the AI backend
# ===========================================================================

class TestCompactionInAnalyzePipeline:
    def test_ai_receives_compacted_not_raw_log(self):
        raw_log = "\n".join([
            "2024-05-10T10:00:00.000Z ##[section]Starting: Compile",
            "2024-05-10T10:00:01.000Z ##[command]dotnet build",
            "2024-05-10T10:00:02.000Z error CS0246: type not found",
            "2024-05-10T10:00:03.000Z ##[section]Finishing: Compile",
        ])
        ai_messages: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "stageazure" in request.url.host:
                path = request.url.path
                if f"/tags/{AI_ANALYZED_TAG}" in path and request.method == "PUT":
                    return httpx.Response(200, json=[AI_ANALYZED_TAG])
                if path.endswith("/builds/1116401") and "timeline" not in path and "logs" not in path and "tags" not in path:
                    return httpx.Response(200, json=BUILD_DETAIL)
                if path.endswith("/tags") and request.method == "GET":
                    return httpx.Response(200, json=[])
                if "timeline" in path:
                    return httpx.Response(200, json=TIMELINE)
                if path.endswith("/logs/7"):
                    return httpx.Response(200, text=raw_log)
                return httpx.Response(404)
            body = json.loads(request.content)
            ai_messages.append(body.get("message", ""))
            return httpx.Response(200, json={"answer": "compacted result"})

        app = create_app(_make_settings(), http_transport=httpx.MockTransport(handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )

        assert response.status_code == 200
        assert ai_messages, "AI backend was never called"
        sent = ai_messages[0]
        assert "##[section]" not in sent
        assert "2024-05-10T" not in sent
        assert "CS0246" in sent or "error" in sent.lower()

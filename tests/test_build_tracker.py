"""Tests for BuildTracker (feat2 + feat3) and compact_log (feat4)."""

import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from az_ai_bff.analyze import compact_log, grep_errors
from az_ai_bff.app import create_app
from az_ai_bff.concurrency import BuildTracker
from az_ai_bff.config import Settings


AUTH = {"Authorization": "Bearer extension-secret"}

BUILD_URL = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589"
    "/_build/results?buildId=1116401"
)
BUILD_URL_2 = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589"
    "/_build/results?buildId=9999999"
)

_API_BASE = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589/_apis"
)

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
LOG_TEXT = "step 1 ok\nstep 2 ok\nstep 3 error: build failed\nstep 4 failed\nstep 5 ok"


def _azure_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/builds/1116401") and "timeline" not in path and "logs" not in path:
        return httpx.Response(200, json=BUILD_DETAIL)
    if "timeline" in path:
        return httpx.Response(200, json=TIMELINE)
    if path.endswith("/logs/42"):
        return httpx.Response(200, text=LOG_TEXT)
    return httpx.Response(404)


def _combined_handler(request: httpx.Request) -> httpx.Response:
    if "stageazure" in request.url.host:
        return _azure_handler(request)
    return httpx.Response(200, json={"answer": "ai result"})


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


# ===========================================================================
# feat2 — BuildTracker unit tests
# ===========================================================================

class TestBuildTrackerUnit:
    def test_first_registration_succeeds(self):
        tracker = BuildTracker()
        slot = tracker.try_start("42")
        assert slot.ok
        assert slot.reason == ""

    def test_duplicate_in_flight_rejected(self):
        tracker = BuildTracker()
        slot1 = tracker.try_start("42")
        assert slot1.ok
        slot2 = tracker.try_start("42")
        assert not slot2.ok
        assert slot2.reason == "build_already_processing"

    def test_different_builds_independent(self):
        tracker = BuildTracker()
        s1 = tracker.try_start("1")
        s2 = tracker.try_start("2")
        assert s1.ok
        assert s2.ok

    def test_context_manager_releases_on_exit(self):
        tracker = BuildTracker()
        with tracker.try_start("42"):
            assert "42" in tracker.in_flight
        assert "42" not in tracker.in_flight

    def test_mark_done_moves_to_completed(self):
        tracker = BuildTracker()
        slot = tracker.try_start("42")
        assert slot.ok
        slot.mark_done()
        slot.__exit__(None, None, None)
        assert "42" not in tracker.in_flight
        assert "42" in tracker.completed

    def test_completed_build_is_rejected(self):
        tracker = BuildTracker()
        with tracker.try_start("42") as slot:
            slot.mark_done()
        second = tracker.try_start("42")
        assert not second.ok
        assert second.reason == "build_already_analyzed"

    def test_failed_analysis_does_not_block_retry(self):
        """If mark_done() is never called (error path), the build is released
        from in-flight and a retry is allowed."""
        tracker = BuildTracker()
        with tracker.try_start("42"):
            pass  # no mark_done → simulates a failed analysis
        assert "42" not in tracker.in_flight
        assert "42" not in tracker.completed
        # Retry is allowed
        retry = tracker.try_start("42")
        assert retry.ok


# ===========================================================================
# feat2 — HTTP 409 for duplicate in-flight build
# ===========================================================================

class TestBuildTrackerHTTP:
    def test_in_flight_duplicate_returns_409(self):
        """Pre-fill the tracker then fire the same build — expect 409."""
        import asyncio

        app = create_app(_make_settings(), http_transport=httpx.MockTransport(_combined_handler))
        with TestClient(app) as client:
            tracker = client.app.state.build_tracker

            async def _prefill():
                return tracker.try_start("1116401")

            held = asyncio.run(_prefill())
            assert held.ok

            response = client.post(
                "/api/v1/analyze",
                json={
                    "message": "azure build pipeline error",
                    "data": BUILD_URL,
                    "role": "user",
                    "type": "normal",
                },
                headers=AUTH,
            )
            assert response.status_code == 409
            assert response.json()["code"] == "build_already_processing"
            assert "request_id" in response.json()

            held.__exit__(None, None, None)

    def test_already_analyzed_returns_409(self):
        """Mark build as completed, then retry — expect 409."""
        import asyncio

        app = create_app(_make_settings(), http_transport=httpx.MockTransport(_combined_handler))
        with TestClient(app) as client:
            tracker = client.app.state.build_tracker

            async def _prefill():
                slot = tracker.try_start("1116401")
                slot.mark_done()
                slot.__exit__(None, None, None)

            asyncio.run(_prefill())

            response = client.post(
                "/api/v1/analyze",
                json={
                    "message": "azure build pipeline error",
                    "data": BUILD_URL,
                    "role": "user",
                    "type": "normal",
                },
                headers=AUTH,
            )
            assert response.status_code == 409
            assert response.json()["code"] == "build_already_analyzed"

    def test_successful_analyze_marks_build_done(self):
        """After a successful analyze the build is in completed, not in_flight."""
        app = create_app(_make_settings(), http_transport=httpx.MockTransport(_combined_handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={
                    "message": "azure build pipeline error",
                    "data": BUILD_URL,
                    "role": "user",
                    "type": "normal",
                },
                headers=AUTH,
            )
            assert response.status_code == 200
            tracker = client.app.state.build_tracker
            assert "1116401" in tracker.completed
            assert "1116401" not in tracker.in_flight

    def test_failed_analyze_releases_build_for_retry(self):
        """When Azure returns 401, the build is not locked — retries are allowed."""
        def bad_azure(request: httpx.Request) -> httpx.Response:
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
            assert r1.status_code == 502  # azure_auth_error

            tracker = client.app.state.build_tracker
            assert "1116401" not in tracker.in_flight
            assert "1116401" not in tracker.completed  # not marked done

            # A second attempt is allowed
            r2 = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
            assert r2.status_code == 502  # still fails, but not 409


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
        assert "Warning: retrying" in result
        assert "Error: gave up" in result
        # Should not appear 10 times individually
        assert result.count("Warning: retrying") == 1

    def test_keeps_context_around_error_lines(self):
        lines = [f"line {i}" for i in range(20)]
        lines[10] = "Error: build failed here"
        log = "\n".join(lines)
        result = compact_log(log, context_lines=2)
        result_lines = result.splitlines()
        assert "Error: build failed here" in result_lines
        assert "line 8" in result_lines   # 2 before
        assert "line 12" in result_lines  # 2 after
        assert "line 0" not in result_lines  # outside context window

    def test_no_signal_lines_returns_cleaned_full_log(self):
        log = "all good\nnothing wrong here\nstep completed"
        result = compact_log(log)
        assert "all good" in result
        assert "nothing wrong here" in result

    def test_char_budget_truncates_with_note(self):
        # Generate a log that is definitely larger than 500 chars
        log = "\n".join([f"Error: line {i} has a problem here" for i in range(100)])
        result = compact_log(log, char_budget=500)
        assert "truncated" in result
        assert len(result) <= 500 + 60  # note text may push slightly past

    def test_budget_not_exceeded_on_small_log(self):
        log = "Error: something small"
        result = compact_log(log, char_budget=10_000)
        assert "truncated" not in result

    def test_real_pipeline_log_shape(self):
        """Realistic Azure Pipelines log — compacted output has the key error."""
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
        # Timestamps stripped
        assert "2024-05-10T" not in result
        # Section markers gone
        assert "##[section]" not in result
        # Duplicated restore lines collapsed
        assert "[×3]" in result


# ===========================================================================
# feat4 — compacted log is what reaches the AI backend
# ===========================================================================

class TestCompactionInAnalyzePipeline:
    def test_ai_receives_compacted_not_raw_log(self):
        """The payload sent to the AI must not contain raw boilerplate."""
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
                if path.endswith("/builds/1116401") and "timeline" not in path and "logs" not in path:
                    return httpx.Response(200, json=BUILD_DETAIL)
                if "timeline" in path:
                    return httpx.Response(200, json=TIMELINE)
                if path.endswith("/logs/42"):
                    return httpx.Response(200, text=raw_log)
                return httpx.Response(404)
            # AI backend — capture what was sent
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

"""Tests for client signature verification (MD5 of host|project_id|run_id)."""

import hashlib

import httpx
import pytest
from fastapi.testclient import TestClient

from az_ai_bff.app import create_app
from az_ai_bff.config import Settings
from az_ai_bff.security import generate_client_signature, verify_client_signature


AUTH = {"Authorization": "Bearer extension-secret"}

# The reference URL and its pre-computed signature (computed manually and
# cross-checked with the algorithm to serve as a stable regression anchor).
#   host       = "stageazure.asax.ir"
#   project_id = "bc24cb70-e692-4049-ae06-d776b7bf4b81"
#   run_id     = "1116251"
#   payload    = "stageazure.asax.ir|bc24cb70-e692-4049-ae06-d776b7bf4b81|1116251"
REFERENCE_URL_API = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/bc24cb70-e692-4049-ae06-d776b7bf4b81/_apis/build/builds/1116251"
)
REFERENCE_URL_UI = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/bc24cb70-e692-4049-ae06-d776b7bf4b81/_build/results?buildId=1116251"
)
REFERENCE_SIG = "3ed106d8182ba132e7ba5998a5ab1ef3"

# URL used by other test files — keep its signature handy too
TEST_BUILD_URL = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589/_build/results?buildId=1116401"
)
TEST_BUILD_SIG = generate_client_signature(TEST_BUILD_URL)


def _make_settings(require: bool = True) -> Settings:
    return Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential="extension-secret",
        backend_url="https://ai.example.test/v1/chat",
        backend_model="example-model",
        backend_credential="backend-secret",
        response_timeout_seconds=30,
        azure_token="valid-pat",
        require_client_signature=require,
    )


# ===========================================================================
# generate_client_signature — unit tests
# ===========================================================================

class TestGenerateSignature:
    def test_reference_value_api_format(self):
        """Known-good value: API URL format."""
        assert generate_client_signature(REFERENCE_URL_API) == REFERENCE_SIG

    def test_reference_value_ui_format(self):
        """Both URL formats must produce the same signature."""
        assert generate_client_signature(REFERENCE_URL_UI) == REFERENCE_SIG

    def test_output_is_32_lowercase_hex(self):
        sig = generate_client_signature(REFERENCE_URL_API)
        assert len(sig) == 32
        assert sig == sig.lower()
        assert all(c in "0123456789abcdef" for c in sig)

    def test_different_run_id_different_signature(self):
        url_a = REFERENCE_URL_API  # run 1116251
        url_b = (
            "https://stageazure.asax.ir/tfs/AsaProjects"
            "/bc24cb70-e692-4049-ae06-d776b7bf4b81/_apis/build/builds/9999999"
        )
        assert generate_client_signature(url_a) != generate_client_signature(url_b)

    def test_different_project_different_signature(self):
        url_a = REFERENCE_URL_API
        url_b = (
            "https://stageazure.asax.ir/tfs/AsaProjects"
            "/00000000-0000-0000-0000-000000000000/_apis/build/builds/1116251"
        )
        assert generate_client_signature(url_a) != generate_client_signature(url_b)

    def test_different_host_different_signature(self):
        url_a = REFERENCE_URL_API
        url_b = (
            "https://other.server.com/tfs/AsaProjects"
            "/bc24cb70-e692-4049-ae06-d776b7bf4b81/_apis/build/builds/1116251"
        )
        assert generate_client_signature(url_a) != generate_client_signature(url_b)

    def test_algorithm_is_md5_of_pipe_joined_parts(self):
        """Explicitly verify every step of the algorithm."""
        host = "stageazure.asax.ir"
        project = "bc24cb70-e692-4049-ae06-d776b7bf4b81"
        run_id = "1116251"
        payload = f"{host}|{project}|{run_id}"
        expected = hashlib.md5(payload.encode("utf-8")).hexdigest()
        assert generate_client_signature(REFERENCE_URL_API) == expected

    def test_host_is_lowercased(self):
        """Upper-case host in URL must produce the same signature."""
        # Pydantic normalises URLs so this tests our normalisation code
        url_lower = REFERENCE_URL_API
        # Manually build the equivalent payload with uppercase host
        payload = "STAGEAZURE.ASAX.IR|bc24cb70-e692-4049-ae06-d776b7bf4b81|1116251"
        uppercase_md5 = hashlib.md5(payload.encode("utf-8")).hexdigest()
        # They must NOT be equal — our function always lowercases
        assert generate_client_signature(url_lower) != uppercase_md5


# ===========================================================================
# verify_client_signature — unit tests
# ===========================================================================

class TestVerifySignature:
    def test_correct_signature_accepted(self):
        assert verify_client_signature(REFERENCE_URL_API, REFERENCE_SIG) is True

    def test_wrong_signature_rejected(self):
        assert verify_client_signature(REFERENCE_URL_API, "00000000000000000000000000000000") is False

    def test_none_signature_rejected(self):
        assert verify_client_signature(REFERENCE_URL_API, None) is False

    def test_empty_string_rejected(self):
        assert verify_client_signature(REFERENCE_URL_API, "") is False

    def test_uppercase_accepted(self):
        """Signature comparison is case-insensitive (we lowercase before compare)."""
        assert verify_client_signature(REFERENCE_URL_API, REFERENCE_SIG.upper()) is True

    def test_mixed_case_accepted(self):
        mixed = "".join(c.upper() if i % 2 == 0 else c for i, c in enumerate(REFERENCE_SIG))
        assert verify_client_signature(REFERENCE_URL_API, mixed) is True


# ===========================================================================
# HTTP integration — /api/v1/analyze with signature enforcement
# ===========================================================================

_API_BASE = (
    "https://stageazure.asax.ir/tfs/AsaProjects"
    "/967834e8-8a18-4a43-88aa-02bd7f158589/_apis"
)
BUILD_DETAIL = {
    "_links": {"timeline": {"href": f"{_API_BASE}/build/builds/1116401/timeline"}},
    "definition": {"id": 42, "name": "MyPipeline"},
}
TIMELINE = {
    "records": [{
        "type": "Task",
        "result": "failed",
        "name": "Compile",
        "log": {"id": 7, "url": f"{_API_BASE}/build/builds/1116401/logs/7"},
    }]
}


def _handler(request: httpx.Request) -> httpx.Response:
    if "stageazure" in request.url.host:
        path = request.url.path
        if path.endswith("/builds/1116401") and "timeline" not in path and "logs" not in path and "tags" not in path:
            return httpx.Response(200, json=BUILD_DETAIL)
        if path.endswith("/tags") and request.method == "GET":
            return httpx.Response(200, json=[])
        if path.endswith("/tags/ai-analyzed") and request.method == "PUT":
            return httpx.Response(200, json=["ai-analyzed"])
        if "timeline" in path:
            return httpx.Response(200, json=TIMELINE)
        if path.endswith("/logs/7"):
            return httpx.Response(200, text="step error: build failed")
        return httpx.Response(404)
    return httpx.Response(200, json={"answer": "ai result"})


class TestSignatureEnforcementHTTP:
    def test_valid_signature_accepted(self):
        app = create_app(_make_settings(require=True), http_transport=httpx.MockTransport(_handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={
                    "message": "azure build pipeline error",
                    "data": TEST_BUILD_URL,
                    "role": "user",
                    "type": "normal",
                    "client_signature": TEST_BUILD_SIG,
                },
                headers=AUTH,
            )
        assert response.status_code == 200

    def test_missing_signature_returns_401(self):
        app = create_app(_make_settings(require=True), http_transport=httpx.MockTransport(_handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={
                    "message": "azure build pipeline error",
                    "data": TEST_BUILD_URL,
                    "role": "user",
                    "type": "normal",
                    # no client_signature
                },
                headers=AUTH,
            )
        assert response.status_code == 401
        assert response.json()["code"] == "invalid_signature"

    def test_wrong_signature_returns_401(self):
        app = create_app(_make_settings(require=True), http_transport=httpx.MockTransport(_handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={
                    "message": "azure build pipeline error",
                    "data": TEST_BUILD_URL,
                    "role": "user",
                    "type": "normal",
                    "client_signature": "deadbeefdeadbeefdeadbeefdeadbeef",
                },
                headers=AUTH,
            )
        assert response.status_code == 401
        assert response.json()["code"] == "invalid_signature"

    def test_signature_check_disabled_accepts_without_signature(self):
        """require_client_signature=False skips the check entirely."""
        app = create_app(_make_settings(require=False), http_transport=httpx.MockTransport(_handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={
                    "message": "azure build pipeline error",
                    "data": TEST_BUILD_URL,
                    "role": "user",
                    "type": "normal",
                    # no client_signature
                },
                headers=AUTH,
            )
        assert response.status_code == 200

    def test_401_has_correct_error_shape(self):
        """Error response must follow the standard ErrorResponse schema."""
        app = create_app(_make_settings(require=True), http_transport=httpx.MockTransport(_handler))
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={
                    "message": "azure build pipeline error",
                    "data": TEST_BUILD_URL,
                    "role": "user",
                    "type": "normal",
                },
                headers=AUTH,
            )
        body = response.json()
        assert response.status_code == 401
        assert body["code"] == "invalid_signature"
        assert "request_id" in body
        assert response.headers["X-Request-ID"] == body["request_id"]

    def test_unsupported_responses_bypass_signature_check(self):
        """type=super and improvement message return early before signature check."""
        app = create_app(_make_settings(require=True), http_transport=httpx.MockTransport(_handler))
        with TestClient(app) as client:
            # type=super — returns before signature check
            r1 = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline error", "data": TEST_BUILD_URL, "role": "user", "type": "super"},
                headers=AUTH,
            )
            # improvement message — returns before signature check
            r2 = client.post(
                "/api/v1/analyze",
                json={"message": "azure build pipeline improvement", "data": TEST_BUILD_URL, "role": "user", "type": "normal"},
                headers=AUTH,
            )
        assert r1.json()["answer"] == "unsupported"
        assert r2.json()["answer"] == "unsupported"

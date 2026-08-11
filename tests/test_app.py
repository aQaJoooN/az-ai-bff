import logging
import re

import httpx
from fastapi.testclient import TestClient

from az_ai_bff.app import create_app
from az_ai_bff.config import Settings
from az_ai_bff.models import AIAnswer, ChatRequest, ChatResponse, ChatTurn, ErrorResponse


ACCESS_CREDENTIAL = "extension-secret"
MESSAGE = "private user message"


def make_app(handler=None):
    settings = Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential=ACCESS_CREDENTIAL,
        backend_url="https://ai.example.test/v1/chat",
        backend_model="example-model",
        backend_credential="backend-secret",
        response_timeout_seconds=30,
        azure_token="test-azure-pat",
    )
    if handler is None:
        handler = lambda _: httpx.Response(503)
    return create_app(settings, http_transport=httpx.MockTransport(handler))


def test_models_trim_messages_and_serialize_stable_fields() -> None:
    assert ChatRequest(message="  hello  ", conversation_id=" conversation ").model_dump() == {
        "message": "hello",
        "conversation_id": "conversation",
    }
    assert list(ChatResponse(answer="answer", conversation_id=None, backend_id="primary", request_id="id").model_dump()) == [
        "answer", "conversation_id", "backend_id", "request_id"
    ]
    assert list(ErrorResponse(code="error", message="message", request_id="id").model_dump()) == [
        "code", "message", "request_id"
    ]
    assert AIAnswer(text="answer", backend_id="primary").model_dump()["input_tokens"] is None
    assert ChatTurn(role="user", content="hello").role == "user"


def test_health_preserves_or_generates_request_id() -> None:
    with TestClient(make_app()) as client:
        preserved = client.get("/health", headers={"X-Request-ID": "request-123"})
        generated = client.get("/health", headers={"X-Request-ID": "invalid request id"})

    assert preserved.json() == {"status": "available"}
    assert preserved.headers["X-Request-ID"] == "request-123"
    assert re.fullmatch(r"[0-9a-f-]{36}", generated.headers["X-Request-ID"])

def test_chat_requires_valid_bearer_credential() -> None:
    with TestClient(make_app()) as client:
        missing = client.post("/api/v1/chat", json={"message": "hello"})
        invalid = client.post(
            "/api/v1/chat",
            json={"message": "hello"},
            headers={"Authorization": "Bearer wrong"},
        )
        authorized = client.post(
            "/api/v1/chat",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {ACCESS_CREDENTIAL}"},
        )

    for response in (missing, invalid):
        assert response.status_code == 401
        assert response.json()["code"] == "unauthorized"
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert authorized.status_code == 502
    assert authorized.json()["code"] == "ai_backend_error"


def test_invalid_chat_bodies_return_stable_400_errors() -> None:
    headers = {"Authorization": f"Bearer {ACCESS_CREDENTIAL}"}
    with TestClient(make_app()) as client:
        blank = client.post("/api/v1/chat", json={"message": "   "}, headers=headers)
        malformed = client.post(
            "/api/v1/chat",
            content="{",
            headers={**headers, "Content-Type": "application/json"},
        )

    assert blank.status_code == 400
    assert blank.json()["code"] == "invalid_message"
    assert malformed.status_code == 400
    assert malformed.json()["code"] == "invalid_json"
    assert set(malformed.json()) == {"code", "message", "request_id"}


def test_request_log_contains_safe_metadata(caplog) -> None:
    caplog.set_level(logging.INFO, logger="az_ai_bff.requests")
    with TestClient(make_app()) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": MESSAGE},
            headers={"Authorization": f"Bearer {ACCESS_CREDENTIAL}", "X-Request-ID": "logged-id"},
        )

    record = next(record for record in caplog.records if record.msg == "request_complete")
    assert (record.request_id, record.route, record.status) == ("logged-id", "/api/v1/chat", response.status_code)
    assert record.elapsed_ms >= 0
    rendered_logs = caplog.text
    assert MESSAGE not in rendered_logs
    assert ACCESS_CREDENTIAL not in rendered_logs
    assert "backend-secret" not in rendered_logs


def test_chat_maps_provider_answer_to_stable_response() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://ai.example.test/v1/chat"
        payload = request.read()
        if b'"message":"hello"' in payload:
            assert b'{"model":"example-model","message":"hello","conversation_id":"conversation"}' in payload
        else:
            assert b'"messages":[{"role":"user","content":"hello"}]' in payload
        return httpx.Response(200, json={"answer": "provider reply"})

    with TestClient(make_app(answer)) as client:
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello", "conversation_id": "conversation"},
            headers={"Authorization": f"Bearer {ACCESS_CREDENTIAL}", "X-Request-ID": "chat-id"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "provider reply",
        "conversation_id": "conversation",
        "backend_id": "primary",
        "request_id": "chat-id",
    }

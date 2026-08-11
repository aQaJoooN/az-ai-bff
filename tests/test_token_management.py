import json
import logging

import httpx
from fastapi.testclient import TestClient

from az_ai_bff.app import create_app
from az_ai_bff.config import Settings
from az_ai_bff.models import ChatTurn
from az_ai_bff.token_management import TokenManager


AUTH = {"Authorization": "Bearer extension-secret"}


def settings(**overrides) -> Settings:
    values = {
        "trusted_proxy_addresses": ["127.0.0.1"],
        "access_credential": "extension-secret",
        "backend_url": "https://primary.example.test/chat",
        "backend_model": "primary-model",
        "backend_credential": "primary-secret",
        "response_timeout_seconds": 30,
        "token_budget_enabled": True,
        "input_token_budget": 2,
        "output_token_limit": 7,
        "token_characters_per_token": 4,
        "azure_token": "test-azure-pat",
    }
    values.update(overrides)
    return Settings(**values)


def test_token_manager_fits_oldest_context_and_preserves_disabled_mode() -> None:
    context = [
        ChatTurn(role="user", content="12345678"),
        ChatTurn(role="assistant", content="1234"),
    ]
    enabled = TokenManager(True, 2, 7, 4).prepare("1234", context)
    disabled = TokenManager(False, 2, 7, 4).prepare("1234", context)

    assert enabled.context == (context[1],)
    assert (enabled.input_tokens, enabled.output_token_limit) == (2, 7)
    assert disabled.context == tuple(context)
    assert (disabled.input_tokens, disabled.output_token_limit) == (None, None)
def test_enabled_budget_applies_output_limit_and_logs_numeric_usage(caplog) -> None:
    payloads = []

    def answer(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "reply", "input_tokens": 1, "output_tokens": 2})

    caplog.set_level(logging.INFO, logger="az_ai_bff.tokens")
    with TestClient(create_app(settings(), http_transport=httpx.MockTransport(answer))) as client:
        response = client.post("/api/v1/chat", json={"message": "secret"}, headers=AUTH)

    assert response.status_code == 200
    assert payloads == [{"model": "primary-model", "message": "secret", "max_output_tokens": 7}]
    record = next(record for record in caplog.records if record.msg == "token_usage")
    assert (record.input_tokens, record.output_tokens, record.backend_id) == (1, 2, "primary")
    assert "secret" not in caplog.text


def test_budget_applies_to_every_multi_ai_backend() -> None:
    payloads = []

    def answer(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": request.url.host})

    configured = settings(
        multi_ai_enabled=True,
        additional_backends=[{
            "backend_url": "https://secondary.example.test/chat",
            "backend_model": "secondary-model",
            "backend_credential": "secondary-secret",
            "backend_id": "secondary",
        }],
    )
    with TestClient(create_app(configured, http_transport=httpx.MockTransport(answer))) as client:
        response = client.post("/api/v1/chat", json={"message": "hello"}, headers=AUTH)

    assert response.status_code == 200
    assert len(payloads) == 2
    assert {payload["max_output_tokens"] for payload in payloads} == {7}


def test_disabled_budget_keeps_provider_payload_unchanged() -> None:
    payloads = []

    def answer(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "reply"})

    with TestClient(create_app(settings(token_budget_enabled=False), http_transport=httpx.MockTransport(answer))) as client:
        response = client.post("/api/v1/chat", json={"message": "hello"}, headers=AUTH)

    assert response.status_code == 200
    assert payloads == [{"model": "primary-model", "message": "hello"}]

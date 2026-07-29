import httpx
from fastapi.testclient import TestClient

from az_ai_bff.app import create_app
from az_ai_bff.config import Settings


AUTH = {"Authorization": "Bearer extension-secret"}


def settings() -> Settings:
    return Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential="extension-secret",
        backend_url="https://primary.example.test/chat",
        backend_model="primary-model",
        backend_credential="primary-secret",
        backend_id="primary",
        response_timeout_seconds=30,
        multi_ai_enabled=True,
        additional_backends=[{
            "backend_url": "https://secondary.example.test/chat",
            "backend_model": "secondary-model",
            "backend_credential": "secondary-secret",
            "backend_id": "secondary",
        }],
    )


def post_chat(handler) -> httpx.Response:
    with TestClient(create_app(settings(), http_transport=httpx.MockTransport(handler))) as client:
        return client.post("/api/v1/chat", json={"message": "hello"}, headers=AUTH)


def test_multi_ai_synthesizes_all_successful_answers() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        backend = request.url.host.split(".")[0]
        return httpx.Response(200, json={"answer": f"{backend} answer"})

    response = post_chat(answer)

    assert response.status_code == 200
    assert response.json()["answer"] == "primary answer\n\nsecondary answer"
    assert response.json()["backend_id"] == "multi"


def test_multi_ai_uses_available_answer_after_partial_failure() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.host.startswith("primary"):
            return httpx.Response(503)
        return httpx.Response(200, json={"answer": "available answer"})

    response = post_chat(answer)

    assert response.status_code == 200
    assert response.json()["answer"] == "available answer"
    assert response.json()["backend_id"] == "multi"


def test_multi_ai_returns_stable_error_when_all_backends_fail() -> None:
    response = post_chat(lambda _: httpx.Response(503))

    assert response.status_code == 502
    assert response.json()["code"] == "all_ai_backends_failed"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]

import httpx
import pytest

from az_ai_bff.backend import (
    AIBackendRejectedError,
    AIBackendTimeoutError,
    AIBackendUnavailableError,
    HttpAIBackendClient,
)
from az_ai_bff.config import Settings


def settings() -> Settings:
    return Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential="extension-secret",
        backend_url="https://ai.example.test/v1/chat",
        backend_model="example-model",
        backend_credential="backend-secret",
        response_timeout_seconds=30,
        azure_token="test-azure-pat",
    )


@pytest.mark.parametrize(
    ("transport_error", "expected_error"),
    [
        (httpx.ConnectError("offline"), AIBackendUnavailableError),
        (httpx.ReadTimeout("slow"), AIBackendTimeoutError),
    ],
)
def test_provider_translates_transport_failures(transport_error, expected_error) -> None:
    async def fail(_: httpx.Request) -> httpx.Response:
        raise transport_error

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
            with pytest.raises(expected_error):
                await HttpAIBackendClient(client, settings()).generate("hello", None)

    import asyncio
    asyncio.run(exercise())


def test_provider_maps_success_and_rejection() -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, json={"answer": "reply", "input_tokens": 2, "output_tokens": 3})
        return httpx.Response(429, json={"error": "rejected"})

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            backend = HttpAIBackendClient(client, settings())
            answer = await backend.generate("hello", "conversation")
            assert answer.model_dump() == {
                "text": "reply", "backend_id": "primary", "input_tokens": 2, "output_tokens": 3
            }
            with pytest.raises(AIBackendRejectedError):
                await backend.generate("hello", None)

    import asyncio
    asyncio.run(exercise())
    assert requests[0].headers["Authorization"] == "Bearer backend-secret"

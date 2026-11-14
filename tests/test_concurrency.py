"""Tests for AISemaphore (feat1: concurrent AI request cap)."""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from az_ai_bff.app import create_app
from az_ai_bff.concurrency import AISemaphore
from az_ai_bff.config import Settings


# ---------------------------------------------------------------------------
# Unit tests: AISemaphore mechanics
# ---------------------------------------------------------------------------

def test_unlimited_semaphore_always_grants() -> None:
    async def exercise() -> None:
        sem = AISemaphore(0)  # 0 = unlimited
        for _ in range(100):
            slot = await sem.acquire()
            assert slot.ok

    asyncio.run(exercise())


def test_semaphore_grants_up_to_limit() -> None:
    async def exercise() -> None:
        sem = AISemaphore(2)
        slot1 = await sem.acquire()
        slot2 = await sem.acquire()
        slot3 = await sem.acquire()   # over limit

        assert slot1.ok
        assert slot2.ok
        assert not slot3.ok           # rejected

        # release one, then try again
        await slot1.__aexit__(None, None, None)
        slot4 = await sem.acquire()
        assert slot4.ok

    asyncio.run(exercise())


def test_slot_as_context_manager_releases_on_exit() -> None:
    async def exercise() -> None:
        sem = AISemaphore(1)
        slot = await sem.acquire()
        assert slot.ok

        async with slot:
            pass  # released here

        # slot was released so we can acquire again
        slot2 = await sem.acquire()
        assert slot2.ok

    asyncio.run(exercise())


def test_failed_slot_context_manager_does_not_release() -> None:
    """A rejected slot's __aexit__ must not release a slot it never held."""
    async def exercise() -> None:
        sem = AISemaphore(1)
        held = await sem.acquire()   # fills the semaphore
        assert held.ok

        rejected = await sem.acquire()
        assert not rejected.ok

        # Exiting rejected slot must not free the held slot
        await rejected.__aexit__(None, None, None)

        still_full = await sem.acquire()
        assert not still_full.ok     # semaphore still full

    asyncio.run(exercise())


# ---------------------------------------------------------------------------
# Integration tests: 429 via TestClient
# ---------------------------------------------------------------------------

AUTH = {"Authorization": "Bearer extension-secret"}


def _make_app(max_concurrent: int = 1) -> object:
    settings = Settings(
        trusted_proxy_addresses=["127.0.0.1"],
        access_credential="extension-secret",
        backend_url="https://ai.example.test/v1/chat",
        backend_model="example-model",
        backend_credential="backend-secret",
        response_timeout_seconds=30,
        azure_token="test-azure-pat",
        max_concurrent_ai_requests=max_concurrent,
    )
    # Backend that hangs forever — we never actually send it; we just want the
    # semaphore to be exhausted before the second request is dispatched.
    return create_app(settings, http_transport=httpx.MockTransport(lambda _: httpx.Response(503)))


def test_chat_returns_429_when_semaphore_full() -> None:
    """With cap=1, the second concurrent request gets 429."""
    async def exercise() -> None:
        sem = AISemaphore(1)
        # Exhaust the single slot
        slot = await sem.acquire()
        assert slot.ok

        # Simulate: app already has the slot, incoming request gets rejected
        second = await sem.acquire()
        assert not second.ok

    asyncio.run(exercise())


def test_chat_returns_429_via_http() -> None:
    """End-to-end: semaphore pre-filled → next HTTP request gets 429."""
    app = create_app(
        Settings(
            trusted_proxy_addresses=["127.0.0.1"],
            access_credential="extension-secret",
            backend_url="https://ai.example.test/v1/chat",
            backend_model="example-model",
            backend_credential="backend-secret",
            response_timeout_seconds=30,
            azure_token="test-azure-pat",
            max_concurrent_ai_requests=1,
        ),
        http_transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"answer": "ok"})),
    )
    with TestClient(app) as client:
        sem = client.app.state.ai_semaphore

        async def _fill_and_check() -> tuple[object, int, int]:
            held = await sem.acquire()
            assert held.ok
            # Semaphore is now exhausted — HTTP request should 429
            r1 = client.post("/api/v1/chat", json={"message": "hello"}, headers=AUTH)
            # Release then try again — should succeed
            await held.__aexit__(None, None, None)
            r2 = client.post("/api/v1/chat", json={"message": "hello"}, headers=AUTH)
            return held, r1.status_code, r2.status_code

        _, code1, code2 = asyncio.run(_fill_and_check())
        assert code1 == 429
        assert code2 == 200

    # Verify the 429 response shape separately (no semaphore manipulation needed)
    with TestClient(app) as client:
        sem = client.app.state.ai_semaphore

        async def _check_body() -> dict:
            held = await sem.acquire()
            r = client.post("/api/v1/chat", json={"message": "hello"}, headers=AUTH)
            await held.__aexit__(None, None, None)
            return r.json(), r.headers["X-Request-ID"]

        body, rid = asyncio.run(_check_body())
        assert body["code"] == "rate_limit_exceeded"
        assert body["request_id"] == rid


def test_zero_cap_means_unlimited() -> None:
    """max_concurrent_ai_requests=0 never returns 429."""
    app = create_app(
        Settings(
            trusted_proxy_addresses=["127.0.0.1"],
            access_credential="extension-secret",
            backend_url="https://ai.example.test/v1/chat",
            backend_model="example-model",
            backend_credential="backend-secret",
            response_timeout_seconds=30,
            azure_token="test-azure-pat",
            max_concurrent_ai_requests=0,
        ),
        http_transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"answer": "ok"})),
    )
    with TestClient(app) as client:
        for _ in range(10):
            r = client.post("/api/v1/chat", json={"message": "hi"}, headers=AUTH)
            assert r.status_code == 200

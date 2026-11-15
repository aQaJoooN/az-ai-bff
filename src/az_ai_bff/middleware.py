import logging
import re
from time import perf_counter
from typing import Any
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


REQUEST_ID_HEADER = "X-Request-ID"

# Allowed characters for a request ID: alphanumerics plus . _ : -
# This covers UUIDs, human-readable IDs, and common correlation token formats.
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")

logger = logging.getLogger("az_ai_bff.requests")


def _request_id(scope: Scope) -> str:
    """Return the X-Request-ID from request headers, or a fresh UUID."""
    for name, value in scope.get("headers", []):
        if name.lower() == b"x-request-id":
            candidate = value.decode("latin-1")
            if _REQUEST_ID_PATTERN.fullmatch(candidate):
                return candidate
            break
    return str(uuid4())


class RequestContextMiddleware:
    """Raw ASGI middleware that injects and echoes a request ID.

    Intentionally bypasses Starlette's BaseHTTPMiddleware to avoid the
    double-send issue that BaseHTTPMiddleware has with streaming responses.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id(scope)

        # Starlette initialises scope["state"] as a State object; we set an
        # attribute on it rather than treating it as a plain dict.
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id
        else:
            state.request_id = request_id  # type: ignore[union-attr]

        started = perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            # scope["route"] is set by Starlette routing after dispatch resolves,
            # so it is reliably populated here in the finally block.
            route: Any = scope.get("route")
            logger.info(
                "request_complete",
                extra={
                    "request_id": request_id,
                    "route": getattr(route, "path", scope["path"]),
                    "method": scope["method"],
                    "status": status_code,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 3),
                    "backend_id": scope["state"].get("backend_id")
                    if isinstance(scope["state"], dict)
                    else getattr(scope["state"], "backend_id", None),
                },
            )

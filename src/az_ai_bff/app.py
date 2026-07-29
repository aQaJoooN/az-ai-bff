from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from az_ai_bff.backend import AIBackendRegistry
from az_ai_bff.chat import ChatService
from az_ai_bff.config import Settings
from az_ai_bff.orchestration import AIOrchestrator
from az_ai_bff.errors import install_exception_handlers
from az_ai_bff.middleware import RequestContextMiddleware
from az_ai_bff.models import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from az_ai_bff.security import require_access_credential
from az_ai_bff.token_management import TokenManager


def create_app(
    settings: Settings | None = None,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Create an application after validating runtime settings."""
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            connect=resolved_settings.connect_timeout_seconds,
            read=resolved_settings.response_timeout_seconds,
            write=resolved_settings.response_timeout_seconds,
            pool=resolved_settings.connect_timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=timeout, transport=http_transport) as client:
            registry = AIBackendRegistry.from_settings(client, resolved_settings)
            orchestrator = AIOrchestrator(registry, resolved_settings.multi_ai_enabled)
            token_manager = TokenManager.from_settings(resolved_settings)
            app.state.http_client = client
            app.state.chat_service = ChatService(orchestrator, token_manager)
            yield

    app = FastAPI(title="Azure DevOps AI BFF", version="1.0.0", lifespan=lifespan)
    app.state.settings = resolved_settings
    install_exception_handlers(app)

    if resolved_settings.allowed_extension_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin).rstrip("/") for origin in resolved_settings.allowed_extension_origins],
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )
    app.add_middleware(RequestContextMiddleware)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
        tags=["chat"],
    )
    async def chat(
        chat_request: ChatRequest,
        request: Request,
        _: Annotated[None, Depends(require_access_credential)],
    ) -> ChatResponse:
        response = await request.app.state.chat_service.create_response(chat_request, request.state.request_id)
        request.state.backend_id = response.backend_id
        return response

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
            for path in schema.get("paths", {}).values():
                for operation in path.values():
                    if isinstance(operation, dict):
                        operation.get("responses", {}).pop("422", None)
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi
    return app

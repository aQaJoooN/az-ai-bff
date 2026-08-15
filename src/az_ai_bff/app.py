from contextlib import asynccontextmanager
import logging
from typing import Annotated, Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

from az_ai_bff.analyze import AzureAuthError, AzureDevOpsLogFetcher, BuildLogFetchError, BuildURLParseError
from az_ai_bff.backend import AIBackendRegistry
from az_ai_bff.chat import ChatService
from az_ai_bff.config import Settings
from az_ai_bff.orchestration import AIOrchestrator
from az_ai_bff.errors import APIError, install_exception_handlers
from az_ai_bff.middleware import RequestContextMiddleware
from az_ai_bff.models import AnalyzeRequest, AnalyzeResponse, ChatRequest, ChatResponse, ChatTurn, ErrorResponse, HealthResponse, UnsupportedResponse
from az_ai_bff.security import require_access_credential
from az_ai_bff.token_management import TokenManager

_analyze_logger = logging.getLogger("az_ai_bff.analyze")


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

    @app.post(
        "/api/v1/analyze",
        response_model=AnalyzeResponse | UnsupportedResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
        tags=["analyze"],
    )
    async def analyze(
        analyze_request: AnalyzeRequest,
        request: Request,
        _: Annotated[None, Depends(require_access_credential)],
    ) -> AnalyzeResponse | UnsupportedResponse:
        request_id = request.state.request_id

        if analyze_request.role != "user":
            raise APIError(400, "unsupported_role", "Only 'user' role is supported")

        if analyze_request.type == "super":
            return UnsupportedResponse(request_id=request_id)

        if analyze_request.message == "azure build pipeline improvement":
            return UnsupportedResponse(request_id=request_id)

        fetcher = AzureDevOpsLogFetcher(request.app.state.http_client, resolved_settings.azure_token.get_secret_value())
        try:
            error_logs = await fetcher.fetch_error_logs(str(analyze_request.data))
        except BuildURLParseError as exc:
            raise APIError(400, "invalid_build_url", str(exc)) from exc
        except AzureAuthError as exc:
            raise APIError(502, "azure_auth_error", str(exc)) from exc
        except BuildLogFetchError as exc:
            raise APIError(502, "azure_log_fetch_error", str(exc)) from exc

        line_count = error_logs.count("\n") + 1
        char_count = len(error_logs)

        _analyze_logger.info(
            "analyze_fetched_log_content",
            extra={
                "request_id": request_id,
                "build_id": str(analyze_request.data).split("buildId=")[-1],
                "log_content": error_logs,
            },
        )

        _analyze_logger.info(
            "analyze_fetched_log_stats",
            extra={
                "request_id": request_id,
                "build_id": str(analyze_request.data).split("buildId=")[-1],
                "log_line_count": line_count,
                "log_char_count": char_count,
            },
        )

        log_chat_request = ChatRequest(
            message=error_logs[:32_768],
            conversation_id=None,
            context=[
                ChatTurn(
                    role="system",
                    content=(
                        "You are an expert Azure DevOps and CI/CD build engineer. "
                        "Analyze the provided build pipeline error log and respond in exactly three clearly labeled sections. "
                        "Allocate your response as follows: ~35% explaining the problem, ~60% explaining the solution, ~5% for the summary.\n\n"
                        "## Problem\n"
                        "Explain what went wrong and why. Identify the root cause from the log.\n\n"
                        "## Solution\n"
                        "Provide concrete, step-by-step actions to fix the problem. Be specific — include commands, config changes, or settings where relevant.\n\n"
                        "## Summary\n"
                        "One or two sentences summarizing the issue and the fix."
                    ),
                ),
            ],
        )
        ai_response = await request.app.state.chat_service.create_response(log_chat_request, request_id)
        request.state.backend_id = ai_response.backend_id
        return AnalyzeResponse(answer=ai_response.answer, request_id=request_id)

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

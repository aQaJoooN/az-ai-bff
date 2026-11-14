from contextlib import asynccontextmanager
import logging
import time
from typing import Annotated, Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Query, Request
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from az_ai_bff.analyze import (
    AzureAuthError,
    AzureDevOpsLogFetcher,
    BuildLogFetchError,
    BuildURLParseError,
    parse_build_url,
)
from az_ai_bff.backend import AIBackendRegistry
from az_ai_bff.chat import ChatService
from az_ai_bff.concurrency import AISemaphore, PipelineInFlightTracker, _Slot
from az_ai_bff.config import Settings
from az_ai_bff.orchestration import AIOrchestrator
from az_ai_bff.errors import APIError, install_exception_handlers
from az_ai_bff.middleware import RequestContextMiddleware
from az_ai_bff.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    ChatTurn,
    ErrorResponse,
    HealthResponse,
    UnsupportedResponse,
)
from az_ai_bff.security import require_access_credential
from az_ai_bff.token_management import TokenManager

_analyze_logger = logging.getLogger("az_ai_bff.analyze")


async def _guarded_stream(slot: _Slot, inner: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield from *inner* and release *slot* when the stream finishes or errors."""
    try:
        async for chunk in inner:
            yield chunk
    finally:
        await slot.__aexit__(None, None, None)


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
            app.state.ai_semaphore = AISemaphore(resolved_settings.max_concurrent_ai_requests)
            app.state.pipeline_tracker = PipelineInFlightTracker()
            app.state.started_at = time.monotonic()
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
    async def health(request: Request) -> HealthResponse:
        uptime = time.monotonic() - getattr(request.app.state, "started_at", time.monotonic())
        return HealthResponse(uptime_seconds=round(uptime, 3))

    @app.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
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
        stream: bool = Query(default=False, description="Stream the response as SSE"),
    ) -> ChatResponse | StreamingResponse:
        slot = await request.app.state.ai_semaphore.acquire()
        if not slot.ok:
            raise APIError(429, "rate_limit_exceeded", "Too many concurrent AI requests — try again later")
        if stream:
            return StreamingResponse(
                _guarded_stream(
                    slot,
                    request.app.state.chat_service.create_stream(chat_request, request.state.request_id),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": request.state.request_id,
                },
            )
        async with slot:
            response = await request.app.state.chat_service.create_response(chat_request, request.state.request_id)
            request.state.backend_id = response.backend_id
            return response

    @app.post(
        "/api/v1/analyze",
        response_model=AnalyzeResponse | UnsupportedResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
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
        stream: bool = Query(default=False, description="Stream the AI analysis as SSE"),
    ) -> AnalyzeResponse | UnsupportedResponse | StreamingResponse:
        request_id = request.state.request_id

        if analyze_request.type == "super":
            return UnsupportedResponse(request_id=request_id)

        if analyze_request.message == "azure build pipeline improvement":
            return UnsupportedResponse(request_id=request_id)

        # --- Parse URL first so we have build_id and can call the AzDO API ---
        try:
            parsed_url = parse_build_url(str(analyze_request.data))
        except BuildURLParseError as exc:
            raise APIError(400, "invalid_build_url", str(exc)) from exc

        fetcher = AzureDevOpsLogFetcher(
            request.app.state.http_client,
            resolved_settings.azure_token.get_secret_value(),
        )

        # --- Fetch build object to get the pipeline definition ID (feat2) ---
        # We need the definition ID to key the in-flight lock by pipeline, not
        # by individual run.  This call also validates the token implicitly.
        try:
            _, build_obj = await fetcher.fetch_build_info(str(analyze_request.data))
        except BuildURLParseError as exc:
            raise APIError(400, "invalid_build_url", str(exc)) from exc
        except AzureAuthError as exc:
            raise APIError(502, "azure_auth_error", str(exc)) from exc
        except BuildLogFetchError as exc:
            raise APIError(502, "azure_log_fetch_error", str(exc)) from exc

        pipeline_id = str(build_obj.get("definition", {}).get("id", parsed_url.build_id))

        # --- feat2: pipeline-level in-flight lock ---
        # Prevents two runs of the same pipeline definition from being analyzed
        # at the same time.  The lock is in-memory and resets on restart.
        pipeline_slot = request.app.state.pipeline_tracker.try_start(pipeline_id)
        if not pipeline_slot.ok:
            raise APIError(
                409,
                "pipeline_already_processing",
                "Another run of this pipeline is already being analyzed — wait for it to finish",
            )

        with pipeline_slot:
            # --- feat3: run-level idempotency via Azure DevOps tags ---
            # Check whether the build already carries the "ai-analyzed" tag.
            # This is persistent across restarts and visible from any client.
            already_analyzed = await fetcher.check_already_analyzed(parsed_url)
            if already_analyzed:
                raise APIError(
                    409,
                    "build_already_analyzed",
                    "This build run has already been analyzed — the result is available in the AI analyze page",
                )

            # --- Fetch and compact the actual error logs ---
            try:
                error_logs = await fetcher.fetch_error_logs(str(analyze_request.data))
            except BuildURLParseError as exc:
                raise APIError(400, "invalid_build_url", str(exc)) from exc
            except AzureAuthError as exc:
                raise APIError(502, "azure_auth_error", str(exc)) from exc
            except BuildLogFetchError as exc:
                raise APIError(502, "azure_log_fetch_error", str(exc)) from exc

            _analyze_logger.info(
                "analyze_fetched_log_stats",
                extra={
                    "request_id": request_id,
                    "build_id": parsed_url.build_id,
                    "log_line_count": error_logs.count("\n") + 1,
                    "log_char_count": len(error_logs),
                },
            )

            log_chat_request = ChatRequest(
                message=error_logs,
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

            # --- feat1: AI concurrency cap ---
            slot = await request.app.state.ai_semaphore.acquire()
            if not slot.ok:
                raise APIError(429, "rate_limit_exceeded", "Too many concurrent AI requests — try again later")

            if stream:
                # For streaming: write the tag after the stream completes.
                # Both the AI semaphore slot and the pipeline lock must be held
                # for the full duration of the stream.
                async def _stream_mark_and_release() -> AsyncIterator[str]:
                    try:
                        async for chunk in _guarded_stream(
                            slot,
                            request.app.state.chat_service.create_stream(log_chat_request, request_id),
                        ):
                            yield chunk
                        # Stream finished cleanly — write the tag
                        await fetcher.mark_analyzed(parsed_url)
                    finally:
                        # Always release the pipeline lock
                        pipeline_slot.__exit__(None, None, None)

                # Transfer ownership of pipeline_slot to the generator.
                # Prevent the outer `with pipeline_slot:` from double-releasing.
                pipeline_slot.ok = False  # disarm the outer context manager
                return StreamingResponse(
                    _stream_mark_and_release(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "X-Request-ID": request_id,
                    },
                )

            # Non-streaming path
            async with slot:
                ai_response = await request.app.state.chat_service.create_response(log_chat_request, request_id)
                request.state.backend_id = ai_response.backend_id
                # Write tag on success; failure is logged but not propagated
                await fetcher.mark_analyzed(parsed_url)
                return AnalyzeResponse(
                    answer=ai_response.answer,
                    backend_id=ai_response.backend_id,
                    request_id=request_id,
                )

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

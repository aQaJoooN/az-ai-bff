from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
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
from az_ai_bff.mcp_client import AgenticLoopError, MCPClient, run_agentic_analysis
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
)
from az_ai_bff.security import require_access_credential, verify_client_signature
from az_ai_bff.token_management import TokenManager

_analyze_logger = logging.getLogger("az_ai_bff.analyze")


def _build_metadata_block(build_obj: dict, project: str) -> str:
    """Format key build fields into a human-readable block for the AI prompt.

    Uses only data already present in the build object returned by
    ``fetch_build_info`` — no extra API calls required.
    """
    def _get(path: list[str], fallback: str = "unknown") -> str:
        node = build_obj
        for key in path:
            if not isinstance(node, dict):
                return fallback
            node = node.get(key, {})
        return str(node) if node not in (None, {}, "") else fallback

    lines = [
        f"Project:       {project}",
        f"Pipeline:      {_get(['definition', 'name'])}",
        f"Build number:  {_get(['buildNumber'])}",
        f"Build result:  {_get(['result'])}",
        f"Branch:        {_get(['sourceBranch'])}",
        f"Commit:        {_get(['sourceVersion'])}",
        f"Repository:    {_get(['repository', 'name'])}",
        f"Triggered by:  {_get(['requestedFor', 'displayName'])}",
        f"Reason:        {_get(['reason'])}",
        f"Queue time:    {_get(['queueTime'])}",
        f"Start time:    {_get(['startTime'])}",
        f"Finish time:   {_get(['finishTime'])}",
    ]
    return "## Build Context\n" + "\n".join(lines)


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
            if resolved_settings.mcp_enabled:
                mcp_token = (
                    resolved_settings.mcp_token.get_secret_value()
                    if resolved_settings.mcp_token
                    else None
                )
                app.state.mcp_client = MCPClient(
                    http_client=client,
                    base_url=str(resolved_settings.mcp_url),
                    token=mcp_token,
                )
            else:
                app.state.mcp_client = None
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
        response_model=AnalyzeResponse,
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
    ) -> AnalyzeResponse | StreamingResponse:
        request_id = request.state.request_id

        if analyze_request.type == "super":
            raise APIError(400, "unsupported", "This analysis type is not supported")

        if analyze_request.message == "azure build pipeline improvement":
            raise APIError(400, "unsupported", "This analysis type is not supported")

        # --- Client signature check ---
        # Verify before any heavy work (Azure API calls, AI calls).
        if resolved_settings.require_client_signature:
            if not verify_client_signature(str(analyze_request.data), analyze_request.client_signature):
                raise APIError(
                    401,
                    "invalid_signature",
                    "Request signature is missing or does not match",
                )

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

        # --- Pipeline completion check ---
        # Azure DevOps sets status="completed" only when the run has fully
        # finished (succeeded, failed, canceled, or partiallySucceeded).
        # Any other status means the run is still in progress — there are no
        # final logs to analyze yet, so we return early with a clear message.
        build_status = build_obj.get("status", "")
        if build_status != "completed":
            raise APIError(
                409,
                "pipeline_not_done",
                "The pipeline run has not finished yet — please wait until it completes before requesting analysis",
            )

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
            already_analyzed = await fetcher.check_already_analyzed(parsed_url)
            if already_analyzed:
                if resolved_settings.enforce_analyzed_tag:
                    # Strict mode: block and return 409.
                    raise APIError(
                        409,
                        "build_already_analyzed",
                        "This build run has already been analyzed — the result is available in the AI analyze page",
                    )
                # Warning mode: continue but flag the response.
                _analyze_logger.warning(
                    "build_already_analyzed_warning",
                    extra={"request_id": request_id, "build_id": parsed_url.build_id},
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

            # --- feat1: AI concurrency cap ---
            slot = await request.app.state.ai_semaphore.acquire()
            if not slot.ok:
                raise APIError(429, "rate_limit_exceeded", "Too many concurrent AI requests — try again later")

            # ================================================================
            # AGENTIC PATH  (mcp_enabled = true)
            # The AI drives MCP tool calls itself in a loop until it has
            # enough context to produce the final answer.
            # Streaming is not supported in agentic mode — the loop requires
            # complete messages to parse tool_calls.
            # ================================================================
            if resolved_settings.mcp_enabled:
                async with slot:
                    try:
                        build_meta = _build_metadata_block(build_obj, parsed_url.project)
                        enriched_logs = f"{build_meta}\n\n## Failed Task Logs\n{error_logs}"
                        answer_text = await run_agentic_analysis(
                            error_logs=enriched_logs,
                            mcp_client=request.app.state.mcp_client,
                            ai_base_url=str(resolved_settings.backend_url),
                            ai_model=resolved_settings.backend_model,
                            ai_credential=(
                                resolved_settings.backend_credential.get_secret_value()
                                if resolved_settings.backend_credential
                                else ""
                            ),
                            http_client=request.app.state.http_client,
                            system_prompt=resolved_settings.analyze_system_prompt,
                            max_tokens=resolved_settings.max_tokens,
                        )
                    except AgenticLoopError as exc:
                        raise APIError(502, "ai_backend_error", str(exc)) from exc

                    analyzed_at = datetime.now(timezone.utc).isoformat()
                    request.state.backend_id = resolved_settings.backend_id
                    await fetcher.mark_analyzed(parsed_url)
                    return AnalyzeResponse(
                        answer=answer_text,
                        backend_id=resolved_settings.backend_id,
                        request_id=request_id,
                        already_analyzed=already_analyzed,
                        analyzed_at=analyzed_at,
                    )

            # ================================================================
            # STANDARD PATH  (mcp_enabled = false)
            # Unchanged from original — single AI call with compacted logs.
            # ================================================================
            build_meta = _build_metadata_block(build_obj, parsed_url.project)
            user_message = f"{build_meta}\n\n## Failed Task Logs\n{error_logs}"

            log_chat_request = ChatRequest(
                message=user_message,
                conversation_id=None,
                context=[
                    ChatTurn(
                        role="system",
                        content=resolved_settings.analyze_system_prompt,
                    ),
                ],
            )

            if stream:
                # For streaming: inject already_analyzed and analyzed_at into the
                # StreamDone event, write the tag, and release the pipeline lock
                # when finished.
                #
                # Capture tracker + pipeline_id so the generator can release
                # the lock in its own finally block.  Disarm the outer
                # `with pipeline_slot:` to prevent a double-release.
                _tracker = request.app.state.pipeline_tracker
                _pipeline_id = pipeline_slot.pipeline_id
                pipeline_slot.disarm()

                async def _stream_mark_and_release() -> AsyncIterator[str]:
                    try:
                        async for chunk in _guarded_stream(
                            slot,
                            request.app.state.chat_service.create_stream(log_chat_request, request_id),
                        ):
                            # Intercept the StreamDone event to inject already_analyzed
                            # and analyzed_at.  All other events (chunk, error) forwarded unchanged.
                            try:
                                event_data = json.loads(chunk.removeprefix("data:").strip().rstrip("\n"))
                                if event_data.get("type") == "done":
                                    event_data["already_analyzed"] = already_analyzed
                                    event_data["analyzed_at"] = datetime.now(timezone.utc).isoformat()
                                    yield f"data: {json.dumps(event_data)}\n\n"
                                    continue
                            except (json.JSONDecodeError, AttributeError):
                                pass
                            yield chunk
                        # Stream finished cleanly — write the tag
                        await fetcher.mark_analyzed(parsed_url)
                    finally:
                        # Always release the pipeline lock — covers clean finish,
                        # errors mid-stream, and generator garbage collection.
                        _tracker._release(_pipeline_id)

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
                analyzed_at = datetime.now(timezone.utc).isoformat()
                request.state.backend_id = ai_response.backend_id
                await fetcher.mark_analyzed(parsed_url)
                return AnalyzeResponse(
                    answer=ai_response.answer,
                    backend_id=ai_response.backend_id,
                    request_id=request_id,
                    already_analyzed=already_analyzed,
                    analyzed_at=analyzed_at,
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

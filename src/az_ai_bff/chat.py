from collections.abc import AsyncIterator

from az_ai_bff.backend import (
    AIBackendClient,
    AIBackendRejectedError,
    AllAIBackendsFailedError,
    AIBackendTimeoutError,
    AIBackendUnavailableError,
)
from az_ai_bff.errors import APIError
from az_ai_bff.models import ChatRequest, ChatResponse, StreamChunk, StreamDone, StreamError
from az_ai_bff.token_management import TokenBudgetExceededError, TokenManager


class ChatService:
    def __init__(self, backend: AIBackendClient, token_manager: TokenManager | None = None) -> None:
        self._backend = backend
        self._token_manager = token_manager

    async def create_response(self, request: ChatRequest, request_id: str) -> ChatResponse:
        try:
            prepared = self._token_manager.prepare(request.message, request.context) if self._token_manager else None
        except TokenBudgetExceededError as error:
            raise APIError(400, "token_budget_exceeded", "The message exceeds the input token budget") from error

        try:
            if prepared is None:
                answer = await self._backend.generate(request.message, request.conversation_id)
            else:
                answer = await self._backend.generate(
                    request.message,
                    request.conversation_id,
                    context=prepared.context,
                    output_token_limit=prepared.output_token_limit,
                )
        except AllAIBackendsFailedError as error:
            raise APIError(502, "all_ai_backends_failed", "All AI backends failed") from error
        except AIBackendTimeoutError as error:
            raise APIError(504, "ai_backend_timeout", "The AI backend response timed out") from error
        except AIBackendUnavailableError as error:
            raise APIError(503, "ai_backend_unavailable", "The AI backend is unavailable") from error
        except AIBackendRejectedError as error:
            raise APIError(502, "ai_backend_error", "The AI backend returned an unusable response") from error

        if self._token_manager and prepared:
            self._token_manager.record_usage(request_id, answer, prepared.input_tokens)

        return ChatResponse(
            answer=answer.text,
            conversation_id=request.conversation_id,
            backend_id=answer.backend_id,
            request_id=request_id,
        )

    async def create_stream(
        self,
        request: ChatRequest,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Yield SSE-formatted strings for a streaming chat response.

        Always yields at least a ``StreamDone`` or ``StreamError`` event so the
        client can detect completion even if no text chunks are produced.
        """
        try:
            prepared = self._token_manager.prepare(request.message, request.context) if self._token_manager else None
        except TokenBudgetExceededError:
            yield StreamError(
                code="token_budget_exceeded",
                message="The message exceeds the input token budget",
                request_id=request_id,
            ).to_sse()
            return

        backend_id = self._backend.backend_id if hasattr(self._backend, "backend_id") else "primary"

        try:
            stream = self._backend.stream_generate(
                request.message,
                request.conversation_id,
                context=prepared.context if prepared else (),
                output_token_limit=prepared.output_token_limit if prepared else None,
            )
            async for text in stream:
                yield StreamChunk(text=text, backend_id=backend_id).to_sse()
        except AllAIBackendsFailedError:
            yield StreamError(code="all_ai_backends_failed", message="All AI backends failed", request_id=request_id).to_sse()
            return
        except AIBackendTimeoutError:
            yield StreamError(code="ai_backend_timeout", message="The AI backend response timed out", request_id=request_id).to_sse()
            return
        except AIBackendUnavailableError:
            yield StreamError(code="ai_backend_unavailable", message="The AI backend is unavailable", request_id=request_id).to_sse()
            return
        except AIBackendRejectedError:
            yield StreamError(code="ai_backend_error", message="The AI backend returned an unusable response", request_id=request_id).to_sse()
            return

        yield StreamDone(
            conversation_id=request.conversation_id,
            backend_id=backend_id,
            request_id=request_id,
        ).to_sse()

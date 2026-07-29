from az_ai_bff.backend import (
    AIBackendClient,
    AIBackendRejectedError,
    AllAIBackendsFailedError,
    AIBackendTimeoutError,
    AIBackendUnavailableError,
)
from az_ai_bff.errors import APIError
from az_ai_bff.models import ChatRequest, ChatResponse
from az_ai_bff.token_management import TokenManager


class ChatService:
    def __init__(self, backend: AIBackendClient, token_manager: TokenManager | None = None) -> None:
        self._backend = backend
        self._token_manager = token_manager

    async def create_response(self, request: ChatRequest, request_id: str) -> ChatResponse:
        prepared = self._token_manager.prepare(request.message) if self._token_manager else None
        try:
            if prepared is None or not self._token_manager.enabled:
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

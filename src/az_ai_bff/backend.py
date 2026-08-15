import logging
from collections.abc import Sequence
from typing import Annotated, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from az_ai_bff.config import AIBackendSettings, Settings
from az_ai_bff.models import AIAnswer, ChatTurn

logger = logging.getLogger("az_ai_bff.backend")


class AIBackendError(Exception):
    """Base failure returned by an AI backend adapter."""


class AIBackendRejectedError(AIBackendError):
    pass


class AIBackendUnavailableError(AIBackendError):
    pass


class AIBackendTimeoutError(AIBackendError):
    pass


class AllAIBackendsFailedError(AIBackendError):
    pass


class AIBackendClient(Protocol):
    @property
    def backend_id(self) -> str: ...

    async def generate(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn] = (),
        output_token_limit: int | None = None,
    ) -> AIAnswer: ...


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: NonEmptyText
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class _OpenAIMessage(BaseModel):
    content: str | None = None  # nullable when model is cut off at max_tokens


class _OpenAIChoice(BaseModel):
    message: _OpenAIMessage
    finish_reason: str | None = None


class _OpenAIUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class _OpenAIResponse(BaseModel):
    choices: list[_OpenAIChoice] = Field(min_length=1)
    usage: _OpenAIUsage | None = None


class HttpAIBackendClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings | AIBackendSettings) -> None:
        self._client = client
        self._url = str(settings.backend_url)
        self._model = settings.backend_model
        self._credential = settings.backend_credential.get_secret_value() if settings.backend_credential else ""
        self._backend_id = settings.backend_id
        self._format = settings.backend_format
        # Global max_tokens cap — only available on the top-level Settings, not per-backend
        self._max_tokens: int | None = getattr(settings, "max_tokens", None)

    @property
    def backend_id(self) -> str:
        return self._backend_id

    async def generate(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn] = (),
        output_token_limit: int | None = None,
    ) -> AIAnswer:
        payload = self._build_payload(message, conversation_id, context, output_token_limit)
        headers = {"Authorization": f"Bearer {self._credential}"} if self._credential else None

        try:
            response = await self._client.post(self._url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise AIBackendTimeoutError from error
        except httpx.RequestError as error:
            raise AIBackendUnavailableError from error
        except httpx.HTTPStatusError as error:
            raise AIBackendRejectedError from error

        try:
            data = response.json()
            if self._format == "openai":
                provider_response = _OpenAIResponse.model_validate(data)
                usage = provider_response.usage
                content = provider_response.choices[0].message.content
                if not content or not content.strip():
                    logger.warning(
                        "backend_empty_content",
                        extra={
                            "backend_id": self._backend_id,
                            "finish_reason": provider_response.choices[0].finish_reason,
                            "raw_response": str(data)[:500],
                        },
                    )
                    raise AIBackendRejectedError("AI backend returned empty content")
                return AIAnswer(
                    text=content,
                    backend_id=self.backend_id,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                )
            provider_response = _ProviderResponse.model_validate(data)
        except AIBackendRejectedError:
            raise
        except (ValueError, ValidationError) as error:
            logger.warning(
                "backend_parse_error",
                extra={
                    "backend_id": self._backend_id,
                    "format": self._format,
                    "raw_response": str(data)[:500] if "data" in dir() else "could not decode JSON",
                    "error": str(error),
                },
            )
            raise AIBackendRejectedError from error

        return AIAnswer(
            text=provider_response.answer,
            backend_id=self.backend_id,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
        )

    def _build_payload(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn],
        output_token_limit: int | None,
    ) -> dict:
        # Resolve the token limit: explicit caller value takes priority, then global cap
        effective_limit = output_token_limit if output_token_limit is not None else self._max_tokens

        if self._format == "openai":
            payload = {
                "model": self._model,
                "messages": [
                    *(turn.model_dump() for turn in context),
                    {"role": "user", "content": message},
                ],
            }
            if effective_limit is not None:
                payload["max_tokens"] = effective_limit
            return payload

        payload = {"model": self._model, "message": message}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        if context:
            payload["context"] = [turn.model_dump() for turn in context]
        if effective_limit is not None:
            payload["max_output_tokens"] = effective_limit
        return payload


class AIBackendRegistry:
    def __init__(self, primary: AIBackendClient, additional: list[AIBackendClient] | None = None) -> None:
        self.primary = primary
        self.enabled_backends = (primary, *(additional or []))

    @classmethod
    def from_settings(cls, client: httpx.AsyncClient, settings: Settings) -> "AIBackendRegistry":
        primary = HttpAIBackendClient(client, settings)
        additional = [
            HttpAIBackendClient(client, backend)
            for backend in settings.additional_backends
            if backend.enabled
        ]
        return cls(primary, additional)

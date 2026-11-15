import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Annotated, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from az_ai_bff.config import AIBackendSettings, Settings
from az_ai_bff.models import AIAnswer, ChatTurn

logger = logging.getLogger("az_ai_bff.backend")


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class AIBackendError(Exception):
    """Base class for all AI backend failures."""


class AIBackendRejectedError(AIBackendError):
    """Backend returned an HTTP error or an unusable response body."""


class AIBackendUnavailableError(AIBackendError):
    """Backend could not be reached (network error)."""


class AIBackendTimeoutError(AIBackendError):
    """Backend did not respond within the configured timeout."""


class AllAIBackendsFailedError(AIBackendError):
    """Every configured backend failed during a multi-AI fan-out."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class AIBackendClient(Protocol):
    """Structural interface that every AI backend adapter must satisfy."""

    @property
    def backend_id(self) -> str: ...

    async def generate(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn] = (),
        output_token_limit: int | None = None,
    ) -> AIAnswer: ...

    # Note: implementations are async generators (not coroutines), so callers
    # must use `async for` directly — do NOT await the return value.
    async def stream_generate(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn] = (),
        output_token_limit: int | None = None,
    ) -> AsyncIterator[str]: ...


# ---------------------------------------------------------------------------
# Internal response models (not part of the public API)
# ---------------------------------------------------------------------------

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _ProviderResponse(BaseModel):
    """Expected shape of a 'simple' format backend response."""
    model_config = ConfigDict(extra="ignore")

    answer: NonEmptyText
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class _OpenAIMessage(BaseModel):
    content: str | None = None           # final answer; empty on reasoning models if max_tokens too low
    reasoning_content: str | None = None  # chain-of-thought on reasoning models (DeepSeek, Qwen3, etc.)


class _OpenAIChoice(BaseModel):
    message: _OpenAIMessage
    finish_reason: str | None = None


class _OpenAIUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class _OpenAIResponse(BaseModel):
    choices: list[_OpenAIChoice] = Field(min_length=1)
    usage: _OpenAIUsage | None = None


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class HttpAIBackendClient:
    """Sends requests to a single AI backend over HTTP.

    Supports two wire formats:
    - ``"simple"`` — custom JSON schema with ``answer`` / ``input_tokens`` / ``output_tokens``
    - ``"openai"`` — OpenAI Chat Completions API, including SSE streaming and reasoning models
    """

    def __init__(self, client: httpx.AsyncClient, settings: Settings | AIBackendSettings) -> None:
        self._client = client
        self._url = str(settings.backend_url)
        self._model = settings.backend_model
        self._credential = settings.backend_credential.get_secret_value() if settings.backend_credential else ""
        self._backend_id = settings.backend_id
        self._format = settings.backend_format
        self._reasoning_mode: bool = settings.backend_reasoning_mode
        # max_tokens is a global cap that only exists on the top-level Settings,
        # not on AIBackendSettings, so getattr is used to handle both types.
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
        """Send a single request and return the full AI answer."""
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
                choice = provider_response.choices[0]

                # In reasoning_mode the model always puts its answer in reasoning_content
                # (content is empty by design). Outside reasoning_mode, reasoning_content
                # is used as a silent fallback only when content is unexpectedly empty.
                if self._reasoning_mode:
                    content = choice.message.reasoning_content or choice.message.content
                else:
                    content = choice.message.content
                    if not content or not content.strip():
                        reasoning = choice.message.reasoning_content
                        if reasoning and reasoning.strip():
                            logger.warning(
                                "backend_used_reasoning_content",
                                extra={
                                    "backend_id": self._backend_id,
                                    "finish_reason": choice.finish_reason,
                                    "hint": "content empty — set AZ_AI_BFF_BACKEND_REASONING_MODE=true if this model always uses reasoning_content",
                                },
                            )
                            content = reasoning

                if not content or not content.strip():
                    logger.warning(
                        "backend_empty_content",
                        extra={
                            "backend_id": self._backend_id,
                            "finish_reason": choice.finish_reason,
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

    async def stream_generate(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn] = (),
        output_token_limit: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks from the backend.

        For ``openai`` format: opens a streaming HTTP connection and parses SSE lines.
        For ``simple`` format: falls back to a single buffered request and yields the
        full answer as one chunk (simple backends rarely support SSE).
        """
        payload = self._build_payload(message, conversation_id, context, output_token_limit)
        headers = {"Authorization": f"Bearer {self._credential}"} if self._credential else {}

        if self._format != "openai":
            answer = await self.generate(message, conversation_id, context, output_token_limit)
            yield answer.text
            return

        payload["stream"] = True
        try:
            async with self._client.stream("POST", self._url, json=payload, headers=headers) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as error:
                    raise AIBackendRejectedError from error

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # Respect reasoning_mode: prefer reasoning_content when set
                    if self._reasoning_mode:
                        text = delta.get("reasoning_content") or delta.get("content") or ""
                    else:
                        text = delta.get("content") or ""

                    if text:
                        yield text

        except httpx.TimeoutException as error:
            raise AIBackendTimeoutError from error
        except httpx.RequestError as error:
            raise AIBackendUnavailableError from error

    def _build_payload(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn],
        output_token_limit: int | None,
    ) -> dict:
        """Construct the request body for the configured backend format."""
        # Explicit caller limit takes priority; fall back to the global cap.
        effective_limit = output_token_limit if output_token_limit is not None else self._max_tokens

        if self._format == "openai":
            payload: dict = {
                "model": self._model,
                "messages": [
                    *(turn.model_dump() for turn in context),
                    {"role": "user", "content": message},
                ],
            }
            if effective_limit is not None:
                payload["max_tokens"] = effective_limit
            return payload

        # simple format — conversation_id and context are optional fields
        payload = {"model": self._model, "message": message}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        if context:
            payload["context"] = [turn.model_dump() for turn in context]
        if effective_limit is not None:
            payload["max_output_tokens"] = effective_limit
        return payload


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AIBackendRegistry:
    """Holds the primary backend and any additional enabled backends."""

    def __init__(self, primary: AIBackendClient, additional: list[AIBackendClient] | None = None) -> None:
        self.primary = primary
        self.enabled_backends = (primary, *(additional or []))

    @classmethod
    def from_settings(cls, client: httpx.AsyncClient, settings: Settings) -> "AIBackendRegistry":
        """Build the registry from validated settings."""
        primary = HttpAIBackendClient(client, settings)
        additional = [
            HttpAIBackendClient(client, backend)
            for backend in settings.additional_backends
            if backend.enabled
        ]
        return cls(primary, additional)

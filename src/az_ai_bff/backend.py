from collections.abc import Sequence
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from az_ai_bff.config import AIBackendSettings, Settings
from az_ai_bff.models import AIAnswer, ChatTurn


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


class _ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    answer: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class HttpAIBackendClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings | AIBackendSettings) -> None:
        self._client = client
        self._url = str(settings.backend_url)
        self._model = settings.backend_model
        self._credential = settings.backend_credential.get_secret_value()
        self._backend_id = settings.backend_id

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
        payload = {"model": self._model, "message": message}
        if conversation_id is not None:
            payload["conversation_id"] = conversation_id
        if context:
            payload["context"] = [turn.model_dump() for turn in context]
        if output_token_limit is not None:
            payload["max_output_tokens"] = output_token_limit

        try:
            response = await self._client.post(
                self._url,
                json=payload,
                headers={"Authorization": f"Bearer {self._credential}"},
            )
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise AIBackendTimeoutError from error
        except httpx.RequestError as error:
            raise AIBackendUnavailableError from error
        except httpx.HTTPStatusError as error:
            raise AIBackendRejectedError from error

        try:
            provider_response = _ProviderResponse.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise AIBackendRejectedError from error

        return AIAnswer(
            text=provider_response.answer,
            backend_id=self.backend_id,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
        )


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

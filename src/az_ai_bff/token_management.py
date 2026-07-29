import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from az_ai_bff.config import Settings
from az_ai_bff.models import AIAnswer, ChatTurn


logger = logging.getLogger("az_ai_bff.tokens")


class TokenBudgetExceededError(ValueError):
    """The user message cannot fit within the configured input budget."""


@dataclass(frozen=True)
class PreparedPrompt:
    context: tuple[ChatTurn, ...]
    input_tokens: int | None
    output_token_limit: int | None


class TokenManager:
    """Estimates usage and fits optional conversation context to a token budget."""

    def __init__(
        self,
        enabled: bool,
        input_token_budget: int,
        output_token_limit: int,
        characters_per_token: float,
    ) -> None:
        self.enabled = enabled
        self.input_token_budget = input_token_budget
        self.output_token_limit = output_token_limit
        self.characters_per_token = characters_per_token

    @classmethod
    def from_settings(cls, settings: Settings) -> "TokenManager":
        return cls(
            enabled=settings.token_budget_enabled,
            input_token_budget=settings.input_token_budget,
            output_token_limit=settings.output_token_limit,
            characters_per_token=settings.token_characters_per_token,
        )

    def estimate_text(self, text: str) -> int:
        return math.ceil(len(text) / self.characters_per_token)

    def estimate_input(self, message: str, context: Sequence[ChatTurn]) -> int:
        return self.estimate_text(message) + sum(self.estimate_text(turn.content) for turn in context)

    def fit_context(self, message: str, context: Sequence[ChatTurn]) -> tuple[ChatTurn, ...]:
        fitted = list(context)
        while fitted and self.estimate_input(message, fitted) > self.input_token_budget:
            fitted.pop(0)
        return tuple(fitted)

    def prepare(self, message: str, context: Sequence[ChatTurn] = ()) -> PreparedPrompt:
        if not self.enabled:
            return PreparedPrompt(tuple(context), None, None)
        message_tokens = self.estimate_text(message)
        if message_tokens > self.input_token_budget:
            raise TokenBudgetExceededError("message exceeds the configured input token budget")
        fitted = self.fit_context(message, context)
        return PreparedPrompt(
            context=fitted,
            input_tokens=self.estimate_input(message, fitted),
            output_token_limit=self.output_token_limit,
        )

    def record_usage(self, request_id: str, answer: AIAnswer, estimated_input: int | None) -> None:
        if not self.enabled:
            return
        logger.info(
            "token_usage",
            extra={
                "request_id": request_id,
                "backend_id": answer.backend_id,
                "input_tokens": (
                    answer.input_tokens
                    if answer.input_tokens is not None
                    else estimated_input * answer._backend_count
                    if estimated_input is not None
                    else None
                ),
                "output_tokens": (
                    answer.output_tokens
                    if answer.output_tokens is not None
                    else self.estimate_text(answer.text)
                ),
            },
        )

import asyncio
from collections.abc import AsyncIterator, Sequence

from az_ai_bff.backend import AIBackendClient, AIBackendError, AIBackendRegistry, AllAIBackendsFailedError
from az_ai_bff.models import AIAnswer, ChatTurn


class ResponseSynthesizer:
    """Combines available provider answers into one provider-neutral answer."""

    def synthesize(self, answers: Sequence[AIAnswer]) -> AIAnswer:
        answer = AIAnswer(
            text="\n\n".join(answer.text for answer in answers),
            backend_id="multi",
            input_tokens=self._sum_tokens(answers, "input_tokens"),
            output_tokens=self._sum_tokens(answers, "output_tokens"),
        )
        answer._backend_count = len(answers)
        return answer

    @staticmethod
    def _sum_tokens(answers: Sequence[AIAnswer], field: str) -> int | None:
        values = [getattr(answer, field) for answer in answers]
        return sum(values) if all(value is not None for value in values) else None


class AIOrchestrator:
    def __init__(
        self,
        registry: AIBackendRegistry,
        multi_ai_enabled: bool,
        synthesizer: ResponseSynthesizer | None = None,
    ) -> None:
        self._registry = registry
        self._multi_ai_enabled = multi_ai_enabled
        self._synthesizer = synthesizer or ResponseSynthesizer()

    async def generate(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn] = (),
        output_token_limit: int | None = None,
    ) -> AIAnswer:
        if not self._multi_ai_enabled:
            return await self._registry.primary.generate(
                message,
                conversation_id,
                context=context,
                output_token_limit=output_token_limit,
            )

        results = await asyncio.gather(
            *(
                backend.generate(
                    message,
                    conversation_id,
                    context=context,
                    output_token_limit=output_token_limit,
                )
                for backend in self._registry.enabled_backends
            ),
            return_exceptions=True,
        )
        answers: list[AIAnswer] = []
        for result in results:
            if isinstance(result, AIAnswer):
                answers.append(result)
            elif not isinstance(result, AIBackendError):
                raise result

        if not answers:
            raise AllAIBackendsFailedError
        return self._synthesizer.synthesize(answers)

    async def stream_generate(
        self,
        message: str,
        conversation_id: str | None,
        context: Sequence[ChatTurn] = (),
        output_token_limit: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream from the primary backend only.

        Multi-AI fan-out is intentionally not supported for streaming — combining
        parallel streams with unknown lengths would require buffering everything,
        which defeats the purpose of streaming.
        """
        async for chunk in self._registry.primary.stream_generate(
            message,
            conversation_id,
            context=context,
            output_token_limit=output_token_limit,
        ):
            yield chunk

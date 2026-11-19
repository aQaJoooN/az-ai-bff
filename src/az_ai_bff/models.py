import json
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints, model_serializer


TrimmedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MessageText = Annotated[TrimmedText, StringConstraints(max_length=32_768)]
Identifier = Annotated[TrimmedText, StringConstraints(max_length=128)]


class APIModel(BaseModel):
    """Base model that rejects unknown fields on all public request/response types."""
    model_config = ConfigDict(extra="forbid")


class ChatTurn(APIModel):
    role: Literal["user", "assistant", "system"]
    content: MessageText


class ChatRequest(APIModel):
    message: MessageText
    conversation_id: Identifier | None = None
    context: list[ChatTurn] = Field(default_factory=list, max_length=100)

    @model_serializer
    def serialize(self) -> dict[str, str | None]:
        # Intentionally omits `context` — context is passed directly to the
        # backend at the call site and must not be round-tripped through
        # serialisation (which could accidentally leak it into logs).
        return {"message": self.message, "conversation_id": self.conversation_id}


class AIResponse(APIModel):
    """Shared base for all AI-generated response models."""
    answer: str
    backend_id: str
    request_id: str


class ChatResponse(AIResponse):
    conversation_id: str | None

    # ConfigDict is repeated here to satisfy pydantic's MRO for model_config.
    model_config = ConfigDict(extra="forbid")

    def model_dump(self, **kwargs):
        # Enforce a stable field order: answer → conversation_id → backend_id → request_id
        d = super().model_dump(**kwargs)
        return {k: d[k] for k in ("answer", "conversation_id", "backend_id", "request_id")}


class ErrorResponse(APIModel):
    code: str
    message: str
    request_id: str


class HealthResponse(APIModel):
    status: Literal["available"] = "available"
    version: str = "1.0.0"
    uptime_seconds: float = Field(default_factory=lambda: 0.0, ge=0)


class AIAnswer(APIModel):
    """Internal model carrying the AI backend's raw answer and token counts."""
    text: MessageText
    backend_id: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    # Set by ResponseSynthesizer in multi-AI mode; used by TokenManager to scale estimates.
    _backend_count: int = PrivateAttr(default=1)


class AnalyzeRequest(APIModel):
    message: Literal["azure build pipeline error", "azure build pipeline improvement"]
    data: AnyHttpUrl
    role: Literal["user"]
    type: Literal["normal", "super"]
    # 32-char lowercase hex MD5 of "azure-host|project-id|run-id".
    # Required when AZ_AI_BFF_REQUIRE_CLIENT_SIGNATURE=true (the default).
    client_signature: str | None = None


class AnalyzeResponse(AIResponse):
    # True when the build already carried the "ai-analyzed" tag before this
    # request was processed.  The analysis still ran and answer is valid — this
    # is a warning to the client that a previous result may already exist.
    already_analyzed: bool = False
    # ISO-8601 UTC timestamp of when the AI printed the analysis result.
    analyzed_at: str | None = None
    # Wall-clock seconds from the start of the analyze flow to the finished
    # answer.  Also rendered as a banner at the top of the analyze page.
    elapsed_seconds: float | None = None


# ---------------------------------------------------------------------------
# SSE streaming models
# ---------------------------------------------------------------------------

class StreamChunk(BaseModel):
    """A single SSE data chunk sent during streaming."""
    type: Literal["chunk"] = "chunk"
    text: str
    backend_id: str

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.model_dump())}\n\n"


class StreamDone(BaseModel):
    """Final SSE event sent when streaming completes successfully."""
    type: Literal["done"] = "done"
    conversation_id: str | None
    backend_id: str
    request_id: str
    # True when the build already carried the "ai-analyzed" tag before this request.
    already_analyzed: bool = False
    # ISO-8601 UTC timestamp of when the AI printed the analysis result.
    analyzed_at: str | None = None
    # Wall-clock seconds from the start of the analyze flow to the finished
    # answer.  Also rendered as a banner at the top of the analyze page.
    elapsed_seconds: float | None = None

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.model_dump())}\n\n"


class StreamError(BaseModel):
    """SSE event sent when streaming fails mid-flight."""
    type: Literal["error"] = "error"
    code: str
    message: str
    request_id: str

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.model_dump())}\n\n"

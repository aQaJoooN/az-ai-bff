import json
import time
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints, model_serializer


TrimmedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MessageText = Annotated[TrimmedText, StringConstraints(max_length=32_768)]
Identifier = Annotated[TrimmedText, StringConstraints(max_length=128)]


class APIModel(BaseModel):
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
        return {"message": self.message, "conversation_id": self.conversation_id}


class AIResponse(APIModel):
    """Shared base for all AI-generated response models."""
    answer: str
    backend_id: str
    request_id: str


class ChatResponse(AIResponse):
    conversation_id: str | None

    # Preserve stable field order: answer, conversation_id, backend_id, request_id
    model_config = ConfigDict(extra="forbid")

    def model_dump(self, **kwargs):
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
    text: MessageText
    backend_id: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    _backend_count: int = PrivateAttr(default=1)


class AnalyzeRequest(APIModel):
    message: Literal["azure build pipeline error", "azure build pipeline improvement"]
    data: AnyHttpUrl
    role: Literal["user"]
    type: Literal["normal", "super"]


class AnalyzeResponse(AIResponse):
    pass


class UnsupportedResponse(APIModel):
    answer: Literal["unsupported"] = "unsupported"
    request_id: str


# --- SSE streaming models ---

class StreamChunk(BaseModel):
    """A single SSE data chunk sent during streaming."""
    type: Literal["chunk"] = "chunk"
    text: str
    backend_id: str

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.model_dump())}\n\n"


class StreamDone(BaseModel):
    """Final SSE event sent when streaming is complete."""
    type: Literal["done"] = "done"
    conversation_id: str | None
    backend_id: str
    request_id: str

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.model_dump())}\n\n"


class StreamError(BaseModel):
    """SSE event sent when streaming fails mid-stream."""
    type: Literal["error"] = "error"
    code: str
    message: str
    request_id: str

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.model_dump())}\n\n"

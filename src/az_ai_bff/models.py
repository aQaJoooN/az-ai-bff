from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


TrimmedText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MessageText = Annotated[TrimmedText, StringConstraints(max_length=32_768)]
Identifier = Annotated[TrimmedText, StringConstraints(max_length=128)]


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatRequest(APIModel):
    message: MessageText
    conversation_id: Identifier | None = None


class ChatResponse(APIModel):
    answer: str
    conversation_id: str | None
    backend_id: str
    request_id: str


class ErrorResponse(APIModel):
    code: str
    message: str
    request_id: str


class HealthResponse(APIModel):
    status: Literal["available"] = "available"


class AIAnswer(APIModel):
    text: str
    backend_id: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ChatTurn(APIModel):
    role: Literal["user", "assistant", "system"]
    content: str

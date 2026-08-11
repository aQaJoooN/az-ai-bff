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


class AnalyzeResponse(APIModel):
    answer: str
    request_id: str


class UnsupportedResponse(APIModel):
    answer: Literal["unsupported"] = "unsupported"
    request_id: str

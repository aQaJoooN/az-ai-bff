from typing import Annotated, Literal, Optional

from pydantic import (
    AfterValidator,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


def _validate_secret(value: Optional[SecretStr]) -> Optional[SecretStr]:
    if value is None:
        return None
    if not value.get_secret_value().strip():
        return None
    return value


RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OptionalSecret = Annotated[Optional[SecretStr], AfterValidator(_validate_secret)]


class AIBackendSettings(BaseModel):
    """Configuration for an additional AI backend."""

    model_config = ConfigDict(extra="forbid")

    backend_url: AnyHttpUrl
    backend_model: RequiredText
    backend_credential: OptionalSecret = None
    backend_id: RequiredText
    backend_format: Literal["simple", "openai"] = "simple"
    enabled: bool = True


class Settings(BaseSettings):
    """Validated runtime configuration loaded from AZ_AI_BFF_* variables."""

    model_config = SettingsConfigDict(
        env_prefix="AZ_AI_BFF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    internal_host: RequiredText = "127.0.0.1"
    internal_port: int = Field(default=8000, ge=1, le=65535)
    trusted_proxy_addresses: list[RequiredText]
    allowed_extension_origins: list[AnyHttpUrl] = Field(default_factory=list)
    access_credential: OptionalSecret = None
    backend_url: AnyHttpUrl
    backend_model: RequiredText
    backend_credential: OptionalSecret = None
    backend_id: RequiredText = "primary"
    backend_format: Literal["simple", "openai"] = "simple"
    multi_ai_enabled: bool = False
    additional_backends: list[AIBackendSettings] = Field(default_factory=list)
    token_budget_enabled: bool = False
    input_token_budget: int = Field(default=4096, ge=1)
    output_token_limit: int = Field(default=1024, ge=1)
    token_characters_per_token: float = Field(default=4.0, gt=0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0, le=300)
    response_timeout_seconds: float = Field(gt=0, le=600)

    @field_validator("trusted_proxy_addresses")
    @classmethod
    def require_trusted_proxy(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one trusted proxy address is required")
        return value

    @model_validator(mode="after")
    def validate_backends(self) -> "Settings":
        backend_ids = [self.backend_id, *(backend.backend_id for backend in self.additional_backends)]
        if len(backend_ids) != len(set(backend_ids)):
            raise ValueError("backend identifiers must be unique")
        if self.multi_ai_enabled and not any(backend.enabled for backend in self.additional_backends):
            raise ValueError("multi-AI mode requires an enabled additional backend")
        return self

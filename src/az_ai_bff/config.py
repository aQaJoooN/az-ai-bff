from typing import Annotated, Literal

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


def _validate_secret(value: SecretStr | None) -> SecretStr | None:
    """Treat blank strings the same as a missing secret — return None."""
    if value is None:
        return None
    if not value.get_secret_value().strip():
        return None
    return value


RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OptionalSecret = Annotated[SecretStr | None, AfterValidator(_validate_secret)]


class AIBackendSettings(BaseModel):
    """Configuration for a single additional AI backend in multi-AI mode."""

    model_config = ConfigDict(extra="forbid")

    backend_url: AnyHttpUrl
    backend_model: RequiredText
    backend_credential: OptionalSecret = None
    backend_id: RequiredText
    backend_format: Literal["simple", "openai"] = "simple"
    backend_reasoning_mode: bool = False
    enabled: bool = True


class Settings(BaseSettings):
    """Validated runtime configuration loaded from ``AZ_AI_BFF_*`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="AZ_AI_BFF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server binding
    internal_host: RequiredText = "127.0.0.1"
    internal_port: int = Field(default=8000, ge=1, le=65535)

    # Proxy trust and CORS
    trusted_proxy_addresses: list[RequiredText]
    allowed_extension_origins: list[AnyHttpUrl] = Field(default_factory=list)

    # API access credential (optional — leave empty to disable auth)
    access_credential: OptionalSecret = None

    # Primary AI backend
    backend_url: AnyHttpUrl
    backend_model: RequiredText
    backend_credential: OptionalSecret = None
    backend_id: RequiredText = "primary"
    backend_format: Literal["simple", "openai"] = "simple"
    backend_reasoning_mode: bool = False  # set True for models that answer in reasoning_content

    # Multi-AI fan-out (optional)
    multi_ai_enabled: bool = False
    additional_backends: list[AIBackendSettings] = Field(default_factory=list)

    # Token budget (optional)
    token_budget_enabled: bool = False
    input_token_budget: int = Field(default=4096, ge=1)
    output_token_limit: int = Field(default=1024, ge=1)
    token_characters_per_token: float = Field(default=4.0, gt=0)

    # Global max_tokens cap — if set, appended to every AI request
    max_tokens: int | None = Field(default=None, ge=1)

    # Azure DevOps PAT for fetching build logs (required)
    azure_token: SecretStr

    # MCP agentic tool-use (optional)
    # mcp_enabled=True  — the AI drives MCP tool calls itself in an agentic
    #                     loop for every /api/v1/analyze request.
    #                     mcp_url is required when mcp_enabled is true.
    # mcp_enabled=False — no MCP involvement; requests handled exactly as
    #                     before with no behaviour change.
    # mcp_token         — if set, sent as a Bearer token to the MCP server.
    #                     Leave blank when the MCP server needs no auth.
    mcp_enabled: bool = False
    mcp_url: AnyHttpUrl | None = None
    mcp_token: OptionalSecret = None

    # HTTP timeouts
    connect_timeout_seconds: float = Field(default=5.0, gt=0, le=300)
    response_timeout_seconds: float = Field(gt=0, le=600)

    # Maximum simultaneous AI backend calls across all endpoints; 0 = unlimited
    max_concurrent_ai_requests: int = Field(default=5, ge=0)

    # When True, every /api/v1/analyze request must include a valid client_signature
    # (MD5 of "host|project_id|run_id"). Set False only during development.
    require_client_signature: bool = True

    # Controls behaviour when the build already has the "ai-analyzed" tag:
    #   True  — block with 409 "build_already_analyzed"
    #   False — run the analysis and return already_analyzed=true as a warning
    enforce_analyzed_tag: bool = False

    # System prompt sent to the AI for every /api/v1/analyze request.
    # Override via AZ_AI_BFF_ANALYZE_SYSTEM_PROMPT to customise the AI's
    # behaviour without changing code.  The default instructs the model to
    # reply in Problem / Solution / Summary sections.
    analyze_system_prompt: str = (
        "You are a senior Azure DevOps and CI/CD engineer.\n"
        "You have access to tools that can query the Azure DevOps server "
        "for additional context (pipeline YAML, run history, variable groups, "
        "commits, etc.).\n\n"
        "The user will give you build metadata followed by compacted logs from "
        "one or more failed pipeline tasks. Each task section is prefixed with "
        "its name (=== Task Name ===).\n\n"
        "Use the available tools to gather whatever context you need to "
        "diagnose the failure precisely. When you have enough information, "
        "reply in exactly three sections — no other text outside them:\n\n"
        "## Problem\n"
        "State the root cause precisely. Quote the exact error line(s) from "
        "the log. Explain why that error occurs.\n\n"
        "## Solution\n"
        "Numbered, concrete steps to fix it. Include exact commands, file "
        "paths, config keys, or YAML snippets where applicable. Cover any "
        "prerequisite changes first.\n\n"
        "## Summary\n"
        "One sentence: what broke and how to fix it."
    )

    @field_validator("trusted_proxy_addresses")
    @classmethod
    def require_trusted_proxy(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one trusted proxy address is required")
        return value

    @model_validator(mode="after")
    def validate_backends(self) -> "Settings":
        """Ensure backend IDs are unique, multi-AI mode has an enabled backend,
        and MCP mode has a URL when enabled."""
        backend_ids = [self.backend_id, *(backend.backend_id for backend in self.additional_backends)]
        if len(backend_ids) != len(set(backend_ids)):
            raise ValueError("backend identifiers must be unique")
        if self.multi_ai_enabled and not any(backend.enabled for backend in self.additional_backends):
            raise ValueError("multi-AI mode requires an enabled additional backend")
        if self.mcp_enabled and not self.mcp_url:
            raise ValueError("mcp_url is required when mcp_enabled is true")
        return self

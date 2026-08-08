import pytest
from pydantic import ValidationError

from az_ai_bff.config import Settings


VALID_SETTINGS = {
    "trusted_proxy_addresses": ["127.0.0.1"],
    "access_credential": "extension-secret",
    "backend_url": "https://ai.example.test/v1/chat",
    "backend_model": "example-model",
    "backend_credential": "backend-secret",
    "response_timeout_seconds": 30,
    "internal_port": 8000,
}


def test_valid_settings_and_secret_serialization() -> None:
    settings = Settings(**VALID_SETTINGS)

    assert settings.internal_host == "127.0.0.1"
    assert settings.internal_port == 8000
    serialized = str(settings.model_dump())
    assert "extension-secret" not in serialized
    assert "backend-secret" not in serialized


def test_invalid_settings_report_all_fields() -> None:
    with pytest.raises(ValidationError) as error:
        Settings(
            trusted_proxy_addresses=[],
            access_credential="extension-secret",
            backend_url="invalid",
            backend_model="",
            backend_credential="backend-secret",
            response_timeout_seconds=0,
            internal_port=70000,
        )

    invalid_fields = {item["loc"][0] for item in error.value.errors()}
    assert invalid_fields == {
        "trusted_proxy_addresses",
        "backend_url",
        "backend_model",
        "response_timeout_seconds",
        "internal_port",
    }
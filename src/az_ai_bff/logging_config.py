import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Serialize application records without messages or credentials."""

    _fields = (
        "request_id",
        "route",
        "method",
        "status",
        "elapsed_ms",
        "backend_id",
        "input_tokens",
        "output_tokens",
        "build_url",
        "build_id",
        "failed_tasks",
        "log_line_count",
        "log_char_count",
        "finish_reason",
        "raw_response",
        "error",
        "hint",
        "max_concurrent",
        "remaining",
        "pipeline_id",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "severity": record.levelname,
            "event": record.getMessage(),
        }
        payload.update(
            (field, getattr(record, field))
            for field in self._fields
            if getattr(record, field, None) is not None
        )
        return json.dumps(payload, separators=(",", ":"))


def configure_logging() -> None:
    """Emit application logs as one JSON object per line."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("az_ai_bff")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from az_ai_bff.models import ErrorResponse


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    body = ErrorResponse(code=code, message=message, request_id=request_id)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers={"X-Request-ID": request_id},
    )


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, error: APIError) -> JSONResponse:
        return _error_response(request, error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
        code = "invalid_json" if any(item["type"] == "json_invalid" for item in error.errors()) else "invalid_message"
        message = "Request body is not valid JSON" if code == "invalid_json" else "A non-empty message is required"
        return _error_response(request, 400, code, message)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, _: Exception) -> JSONResponse:
        return _error_response(request, 500, "internal_error", "An unexpected error occurred")

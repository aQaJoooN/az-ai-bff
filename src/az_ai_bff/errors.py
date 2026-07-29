from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, error: StarletteHTTPException) -> JSONResponse:
        messages = {404: "The requested route was not found", 405: "The method is not allowed"}
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error_response(
            request,
            error.status_code,
            codes.get(error.status_code, "http_error"),
            messages.get(error.status_code, "The request could not be completed"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError) -> JSONResponse:
        errors = error.errors()
        if any(item["type"] == "json_invalid" for item in errors):
            return _error_response(request, 400, "invalid_json", "Request body is not valid JSON")
        if any("message" in item["loc"] for item in errors):
            return _error_response(request, 400, "invalid_message", "A non-empty message is required")
        return _error_response(request, 400, "invalid_request", "The request is invalid")

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, _: Exception) -> JSONResponse:
        return _error_response(request, 500, "internal_error", "An unexpected error occurred")

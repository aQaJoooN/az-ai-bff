from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from az_ai_bff.errors import APIError


_bearer = HTTPBearer(auto_error=False)


async def require_access_credential(
    request: Request,
    credential: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    access_credential = request.app.state.settings.access_credential
    
    if not access_credential or not access_credential.get_secret_value():
        return
    
    expected = access_credential.get_secret_value().encode("utf-8")
    supplied = credential.credentials.encode("utf-8") if credential and credential.scheme.lower() == "bearer" else b""
    if not supplied or not compare_digest(supplied, expected):
        raise APIError(401, "unauthorized", "A valid bearer credential is required")

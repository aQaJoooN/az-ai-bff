import hashlib
from hmac import compare_digest
from typing import Annotated
from urllib.parse import urlparse

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


# ---------------------------------------------------------------------------
# Client signature  (MD5 of "azure-host|project-id|run-id")
# ---------------------------------------------------------------------------

def _extract_signature_parts(url: str) -> tuple[str, str, str]:
    """Return (azure_host, project_id, run_id) extracted from an AzDO build URL.

    Both accepted URL formats are handled:

    UI  format: https://host/tfs/Collection/{project}/_build/results?buildId={id}
    API format: https://host/tfs/Collection/{project}/_apis/build/builds/{id}

    The host is normalised: scheme stripped, lowercase, no trailing slash.
    """
    parsed = urlparse(url)

    # Normalize host: lowercase, strip port if default (optional robustness)
    host = parsed.netloc.lower().split(":")[0]  # drop port if present

    path_parts = [p for p in parsed.path.split("/") if p]

    # API format: …/{project}/_apis/build/builds/{run_id}
    if "_apis" in path_parts:
        apis_idx = path_parts.index("_apis")
        project_id = path_parts[apis_idx - 1]
        # run_id is after "builds"
        builds_idx = path_parts.index("builds", apis_idx)
        run_id = path_parts[builds_idx + 1]
        return host, project_id, run_id

    # UI format: …/{project}/_build/results?buildId={run_id}
    if "_build" in path_parts:
        build_idx = path_parts.index("_build")
        project_id = path_parts[build_idx - 1]
        from urllib.parse import parse_qs
        qs = parse_qs(parsed.query)
        run_id = qs.get("buildId", [""])[0]
        return host, project_id, run_id

    raise ValueError(f"Cannot extract signature parts from URL: {url}")


def generate_client_signature(url: str) -> str:
    """Return the 32-character MD5 signature for a build URL.

    Algorithm (must be reproduced exactly in the extension):

    1. Extract the three components from the URL:
       - azure_host   : hostname only, lowercase, no protocol, no port, no slash
                        e.g. "stageazure.asax.ir"
       - project_id   : the project GUID or name segment from the URL path
                        e.g. "bc24cb70-e692-4049-ae06-d776b7bf4b81"
       - run_id       : the numeric build/run ID, treated as a plain string
                        e.g. "1116251"

    2. Concatenate with the pipe delimiter:
       payload = f"{azure_host}|{project_id}|{run_id}"

    3. Compute MD5 of the UTF-8–encoded payload:
       signature = md5(payload.encode("utf-8")).hexdigest()

    The result is exactly 32 lowercase hexadecimal characters.

    Example:
       URL       → "https://stageazure.asax.ir/tfs/AsaProjects/
                    bc24cb70-e692-4049-ae06-d776b7bf4b81/_apis/build/builds/1116251"
       payload   → "stageazure.asax.ir|bc24cb70-e692-4049-ae06-d776b7bf4b81|1116251"
       signature → "3ed106d8182ba132e7ba5998a5ab1ef3"
    """
    host, project_id, run_id = _extract_signature_parts(url)
    payload = f"{host}|{project_id}|{run_id}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def verify_client_signature(url: str, supplied: str | None) -> bool:
    """Return True when *supplied* matches the computed signature for *url*.

    Uses ``hmac.compare_digest`` to avoid timing-based side-channel leaks.
    Returns False (never raises) so the caller decides the HTTP response.
    """
    if not supplied:
        return False
    try:
        expected = generate_client_signature(url)
    except (ValueError, IndexError):
        return False
    try:
        return compare_digest(supplied.lower(), expected)
    except (TypeError, AttributeError):
        return False

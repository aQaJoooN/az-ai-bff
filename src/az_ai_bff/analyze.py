"""Azure DevOps on-premise build log fetcher and analyzer."""

import logging
import re
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger("az_ai_bff.analyze")

# Matches on-prem TFS/AzureDevOps build result URLs — two accepted formats:
#   1. UI URL:  https://host/tfs/<collection>/<project>/_build/results?buildId=<id>
#   2. API URL: https://host/tfs/<collection>/<project>/_apis/build/builds/<id>
_BUILD_UI_RE = re.compile(
    r"^https?://[^/]+"
    r"(?:/tfs/[^/]+)?"
    r"/(?P<project>[^/]+)"
    r"/_build/results",
    re.IGNORECASE,
)
_BUILD_API_RE = re.compile(
    r"^https?://[^/]+"
    r"(?:/tfs/[^/]+)?"
    r"/(?P<project>[^/]+)"
    r"/_apis/build/builds/(?P<build_id>\d+)",
    re.IGNORECASE,
)

CONTEXT_LINES = 5


class BuildURLParseError(ValueError):
    """Raised when the data URL is not a recognisable Azure DevOps build URL."""


class AzureAuthError(Exception):
    """Raised when the Azure DevOps token is rejected or missing."""


class BuildLogFetchError(Exception):
    """Raised when log retrieval fails for any other reason."""


class ParsedBuildURL(NamedTuple):
    base_url: str       # e.g. https://host/tfs/Collection
    project: str        # project GUID or name
    build_id: str


def parse_build_url(url: str) -> ParsedBuildURL:
    """Extract base_url, project and build_id from a TFS/AzureDevOps build URL.

    Accepts two formats:
    - UI:  https://host/tfs/Collection/project/_build/results?buildId=123
    - API: https://host/tfs/Collection/project/_apis/build/builds/123
    """
    parsed = urlparse(url)
    path_parts = parsed.path.split("/")

    # Try API URL format first: /_apis/build/builds/<id>
    api_match = _BUILD_API_RE.match(url)
    if api_match:
        project = api_match.group("project")
        build_id = api_match.group("build_id")
        try:
            proj_idx = path_parts.index(project)
        except ValueError:
            raise BuildURLParseError(f"Cannot locate project segment in path: {parsed.path}")
        base_path = "/".join(path_parts[:proj_idx])
        return ParsedBuildURL(
            base_url=f"{parsed.scheme}://{parsed.netloc}{base_path}",
            project=project,
            build_id=build_id,
        )

    # Try UI URL format: /_build/results?buildId=<id>
    ui_match = _BUILD_UI_RE.match(url)
    qs = parse_qs(parsed.query)
    build_ids = qs.get("buildId", [])
    if ui_match and build_ids:
        project = ui_match.group("project")
        build_id = build_ids[0]
        try:
            proj_idx = path_parts.index(project)
        except ValueError:
            raise BuildURLParseError(f"Cannot locate project segment in path: {parsed.path}")
        base_path = "/".join(path_parts[:proj_idx])
        return ParsedBuildURL(
            base_url=f"{parsed.scheme}://{parsed.netloc}{base_path}",
            project=project,
            build_id=build_id,
        )

    raise BuildURLParseError(f"Not a valid Azure DevOps build results URL: {url}")


def _auth_headers(token: str) -> dict[str, str]:
    import base64
    encoded = base64.b64encode(f":{token}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def grep_errors(log_text: str, context: int = CONTEXT_LINES) -> str:
    """Return lines containing 'error' with `context` lines before and after each match."""
    lines = log_text.splitlines()
    selected: set[int] = set()
    for i, line in enumerate(lines):
        if "error" in line.lower():
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                selected.add(j)
    if not selected:
        return log_text  # return full log if no error keyword found
    return "\n".join(lines[i] for i in sorted(selected))


class AzureDevOpsLogFetcher:
    def __init__(self, client: httpx.AsyncClient, token: str) -> None:
        self._client = client
        self._token = token
        self._headers = _auth_headers(token)

    async def fetch_error_logs(self, url: str) -> str:
        """
        Validate URL, find the failed task in the build, fetch its log,
        and return grepped error context.
        """
        parsed = parse_build_url(url)
        api_base = f"{parsed.base_url}/{parsed.project}/_apis"

        # 1. Validate token / fetch build details
        build = await self._get_json(f"{api_base}/build/builds/{parsed.build_id}?api-version=6.0")

        # 2. Fetch timeline (tasks + their results)
        timeline_url = (
            build.get("_links", {}).get("timeline", {}).get("href")
            or f"{api_base}/build/builds/{parsed.build_id}/timeline?api-version=6.0"
        )
        timeline = await self._get_json(timeline_url)

        # 3. Find failed tasks (result == "failed")
        records = timeline.get("records", [])
        failed_tasks = [
            r for r in records
            if r.get("type") == "Task" and r.get("result") == "failed"
        ]
        if not failed_tasks:
            # Fall back to any failed record
            failed_tasks = [r for r in records if r.get("result") == "failed"]
        if not failed_tasks:
            raise BuildLogFetchError("No failed tasks found in the build timeline")

        # 4. Fetch and aggregate error logs from all failed tasks
        all_grepped: list[str] = []
        for task in failed_tasks:
            log_ref = task.get("log")
            if not log_ref:
                continue
            log_url = log_ref.get("url")
            if not log_url:
                log_id = log_ref.get("id")
                if log_id is None:
                    continue
                log_url = f"{api_base}/build/builds/{parsed.build_id}/logs/{log_id}?api-version=6.0"

            log_text = await self._get_text(log_url)
            grepped = grep_errors(log_text)
            task_name = task.get("name", "unknown task")
            all_grepped.append(f"=== {task_name} ===\n{grepped}")

        if not all_grepped:
            raise BuildLogFetchError("Failed tasks found but no logs could be retrieved")

        result = "\n\n".join(all_grepped)
        logger.info(
            "analyze_logs_fetched",
            extra={
                "build_url": url,
                "build_id": parsed.build_id,
                "failed_tasks": len(all_grepped),
            },
        )
        return result

    async def _get_json(self, url: str) -> dict:
        try:
            response = await self._client.get(url, headers=self._headers)
        except httpx.RequestError as exc:
            raise BuildLogFetchError(f"Network error reaching Azure DevOps: {exc}") from exc

        if response.status_code == 401:
            raise AzureAuthError("Azure DevOps token is invalid or lacks permissions")
        if response.status_code == 403:
            raise AzureAuthError("Azure DevOps token does not have access to this project")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BuildLogFetchError(f"Azure DevOps API returned {response.status_code}") from exc

        return response.json()

    async def _get_text(self, url: str) -> str:
        try:
            response = await self._client.get(url, headers=self._headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BuildLogFetchError(f"Failed to fetch log: {exc}") from exc
        except httpx.RequestError as exc:
            raise BuildLogFetchError(f"Network error fetching log: {exc}") from exc
        return response.text

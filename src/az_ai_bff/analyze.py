"""Azure DevOps on-premise build log fetcher, compactor, and analyzer."""

import logging
import re
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger("az_ai_bff.analyze")

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Log compaction
# ---------------------------------------------------------------------------

# ANSI escape sequences (colours, cursor movement, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?(?:\x07|\x1b\\)")

# Azure Pipelines / TFS prepend ISO-8601 timestamps to every line, e.g.:
#   2024-05-10T12:34:56.1234567Z   or   2024-05-10T12:34:56Z
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\s*"
)

# Lines that are pure pipeline scaffolding and carry zero diagnostic value.
# Matched case-insensitively against the cleaned line content.
_BOILERPLATE_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^##\[section\]",                          # ##[section]Starting: …
        r"^##\[command\]",                          # ##[command]…
        r"^##\[debug\]",                            # ##[debug]…
        r"^##\[group\]",                            # ##[group]…
        r"^##\[endgroup\]",                         # ##[endgroup]
        r"^task\s+:.*version\s+\d",                 # Task : NuGet version 5.x
        r"^description\s+:",                        # Description : …
        r"^set agent variable",
        r"^preparing job",
        r"^initializing agent",
        r"^starting: ",
        r"^finishing: ",
        r"^agent version",
        r"^current agent version",
        r"^agent machine name",
        r"^environment:",
        r"^\s*$",                                   # blank lines
    ]
]

# Keywords that mark a line as diagnostically important.
_SIGNAL_RE = re.compile(
    r"error|warning|exception|traceback|fatal|fail|critical"
    r"|cannot|could not|unable to|not found|denied|refused"
    r"|stack trace|at \w+\.\w+|^\s+at ",
    re.IGNORECASE,
)

# A line is a stack-frame continuation if it starts with whitespace followed
# by "at " (Python/Java/C# tracebacks) or looks like a file reference.
_FRAME_RE = re.compile(r"^\s+at\s+\S|^\s+File\s+\"", re.IGNORECASE)


def compact_log(
    log_text: str,
    char_budget: int = 24_000,
    context_lines: int = 3,
) -> str:
    """Return a compacted, AI-friendly version of *log_text*.

    The compaction pipeline:

    1. Strip ANSI escape codes.
    2. Strip Azure Pipelines timestamp prefixes.
    3. Drop pure-boilerplate lines (agent setup, section markers, blank lines).
    4. Collapse runs of identical consecutive lines into one with a repeat
       count, e.g. ``[×12] The same warning...``.
    5. Keep every line that matches a diagnostic keyword, plus *context_lines*
       lines before and after each match.
    6. Hard-truncate to *char_budget* characters, cutting at a line boundary
       and appending a ``[... N lines truncated]`` note.

    If none of the above keywords are found (clean log), the full cleaned
    text is returned (still subject to the budget).
    """
    # --- 1 & 2: strip ANSI and timestamps ---
    cleaned: list[str] = []
    for raw_line in log_text.splitlines():
        line = _ANSI_RE.sub("", raw_line)
        line = _TIMESTAMP_RE.sub("", line)
        cleaned.append(line)

    # --- 3: drop boilerplate ---
    filtered: list[str] = []
    for line in cleaned:
        if any(pat.match(line) for pat in _BOILERPLATE_RES):
            continue
        filtered.append(line)

    # --- 4: collapse consecutive duplicate lines ---
    deduped: list[str] = []
    i = 0
    while i < len(filtered):
        line = filtered[i]
        j = i + 1
        while j < len(filtered) and filtered[j] == line:
            j += 1
        count = j - i
        if count > 1:
            deduped.append(f"[×{count}] {line}")
        else:
            deduped.append(line)
        i = j

    # --- 5: keep signal lines + context ---
    selected: set[int] = set()
    for idx, line in enumerate(deduped):
        if _SIGNAL_RE.search(line) or _FRAME_RE.match(line):
            lo = max(0, idx - context_lines)
            hi = min(len(deduped), idx + context_lines + 1)
            selected.update(range(lo, hi))

    if selected:
        kept = [deduped[idx] for idx in sorted(selected)]
    else:
        # No signal lines found — keep everything (still cleaned/deduped).
        kept = deduped

    # --- 6: hard-truncate to budget ---
    output_lines: list[str] = []
    used = 0
    truncated_at = -1
    for idx, line in enumerate(kept):
        cost = len(line) + 1  # +1 for newline
        if used + cost > char_budget:
            truncated_at = idx
            break
        output_lines.append(line)
        used += cost

    if truncated_at >= 0:
        remaining = len(kept) - truncated_at
        output_lines.append(f"[... {remaining} lines truncated to fit token budget]")

    return "\n".join(output_lines)


def grep_errors(log_text: str, context: int = CONTEXT_LINES) -> str:
    """Return lines containing 'error' with `context` lines before and after each match.

    This is the legacy extraction used before compaction was added.
    ``compact_log`` supersedes it for the analyze endpoint; this function is
    retained for tests and any callers that need the simpler behaviour.
    """
    lines = log_text.splitlines()
    selected: set[int] = set()
    for i, line in enumerate(lines):
        if "error" in line.lower():
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                selected.add(j)
    if not selected:
        return log_text  # return full log if no error keyword found
    return "\n".join(lines[i] for i in sorted(selected))


# ---------------------------------------------------------------------------
# Azure DevOps log fetcher
# ---------------------------------------------------------------------------

class AzureDevOpsLogFetcher:
    def __init__(self, client: httpx.AsyncClient, token: str) -> None:
        self._client = client
        self._token = token
        self._headers = _auth_headers(token)

    async def fetch_error_logs(self, url: str) -> str:
        """
        Validate URL, find the failed tasks in the build, fetch their logs,
        and return a compacted error log ready for the AI.
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

        # 4. Fetch and aggregate logs from all failed tasks, then compact each
        all_compacted: list[str] = []
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
            compacted = compact_log(log_text)
            task_name = task.get("name", "unknown task")
            all_compacted.append(f"=== {task_name} ===\n{compacted}")

        if not all_compacted:
            raise BuildLogFetchError("Failed tasks found but no logs could be retrieved")

        result = "\n\n".join(all_compacted)
        logger.info(
            "analyze_logs_fetched",
            extra={
                "build_url": url,
                "build_id": parsed.build_id,
                "failed_tasks": len(all_compacted),
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

"""MCP client and agentic tool-use loop for the Azure DevOps MCP server.

When ``mcp_enabled = true`` the analyze route delegates to
``run_agentic_analysis`` instead of calling the AI directly.  That function:

1. Builds a tool catalogue from the MCP server's ``tools/list`` response.
2. Sends the error logs, the system prompt, and the full tool catalogue to
   the AI backend in a single OpenAI-format chat-completions request.
3. If the model replies with ``finish_reason: tool_calls``:
   - Executes every requested tool against the MCP server.
   - Appends both the assistant turn (with tool_calls) and the tool result
     turns to the conversation.
   - Sends the updated conversation back to the AI.
4. Repeats until ``finish_reason: stop`` or the loop guard is reached.
5. Returns the final answer text.

The function never raises for MCP failures — a tool that errors returns a
plain error string as its result so the AI can reason about it and still
produce an answer.

Authentication is optional.  When ``mcp_token`` is set it is forwarded as a
``Bearer`` token on every request.  When it is absent, requests are sent
without an ``Authorization`` header.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger("az_ai_bff.mcp")

_MCP_PATH = "/mcp"
_ACCEPT = "application/json, text/event-stream"

# Maximum tool-call / AI-response rounds before giving up.
# Prevents an infinite loop if the model keeps calling tools without stopping.
_MAX_ITERATIONS = 10

# Hard ceiling on the characters of a single MCP tool result that are sent to
# the AI.  Keeps individual pipeline data responses from flooding the context
# window.  Only the tool result text is truncated — log compaction (compact_log
# in analyze.py) is intentionally NOT applied here because MCP responses may be
# structured JSON rather than log text.
_MCP_TOOL_RESULT_CHAR_LIMIT = 20_000


# Regex to extract the declared MCP IDs from the metadata header injected by
# _build_metadata_block.  Example header line:
#   MCP:{"projectId":"abc","runId":"1238384","pipelineId":"7426"}
# The IDs are always quoted strings in the prompt (to prevent LLM truncation),
# but the MCP tool schemas declare runId / pipelineId as integers, so we
# extract them as strings and coerce to int when correcting tool arguments.
_MCP_HEADER_RE = re.compile(
    r'MCP:\{"projectId":"(?P<projectId>[^"]+)"'
    r',"runId":"(?P<runId>\d+)"'
    r',"pipelineId":"(?P<pipelineId>\d+)"\}',
)


def _extract_declared_ids(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the {projectId, runId, pipelineId} declared in the first user message.

    Returns an empty dict if the header is absent (e.g. in tests that don't
    inject the metadata block).
    """
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content") or ""
            m = _MCP_HEADER_RE.search(str(content))
            if m:
                return {
                    "projectId": m.group("projectId"),
                    "runId": int(m.group("runId")),
                    "pipelineId": int(m.group("pipelineId")),
                }
            break
    return {}


def _fix_tool_args(
    tool_name: str,
    tool_args: dict[str, Any],
    declared: dict[str, Any],
) -> dict[str, Any]:
    """Replace any truncated or wrong runId / pipelineId in *tool_args*.

    The LLM sometimes truncates large integer IDs (e.g. 1238384 → 1238) even
    when the prompt says not to.  This guard compares the tool's runId and
    pipelineId against the values declared in the metadata header and
    overwrites them if they differ.  It is a no-op when *declared* is empty
    (no header present) or when the tool does not use those fields.
    """
    if not declared:
        return tool_args

    fixed = dict(tool_args)

    for key in ("runId", "pipelineId"):
        if key not in fixed:
            continue
        expected = declared.get(key)
        if expected is None:
            continue
        if fixed[key] != expected:
            logger.warning(
                "mcp_id_corrected",
                extra={
                    "tool": tool_name,
                    "field": key,
                    "llm_value": fixed[key],
                    "correct_value": expected,
                },
            )
            fixed[key] = expected

    return fixed


class MCPClient:
    """JSON-RPC client for the Azure DevOps MCP server.

    Parameters
    ----------
    http_client:
        Shared ``httpx.AsyncClient`` from ``app.state.http_client``.
    base_url:
        Root URL of the MCP server, e.g. ``http://fa-cicd-auto:8888``.
    token:
        Optional bearer token.  ``None`` means unauthenticated requests.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        token: str | None = None,
    ) -> None:
        self._client = http_client
        self._url = base_url.rstrip("/") + _MCP_PATH
        self._headers: dict[str, str] = {
            "Accept": _ACCEPT,
            "Content-Type": "application/json",
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # Tool catalogue
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool catalogue in OpenAI ``tools`` array format.

        Each entry looks like::

            {
                "type": "function",
                "function": {
                    "name": "get_pipeline_run_yaml",
                    "description": "...",
                    "parameters": { ... }   # JSON Schema
                }
            }

        Returns an empty list if the MCP server is unreachable or returns
        an error so the caller can decide how to proceed.
        """
        result = await self._rpc("tools/list", {})
        if result is None:
            return []
        tools_raw: list[dict] = result.get("tools", [])
        openai_tools: list[dict[str, Any]] = []
        for t in tools_raw:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                    },
                }
            )
        return openai_tools

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute one MCP tool and return its result as a plain string.

        Never raises — errors are returned as a descriptive string so the
        AI can reason about the failure and still produce an answer.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        try:
            response = await self._client.post(
                self._url, json=payload, headers=self._headers
            )
            response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            logger.warning("mcp_tool_call_failed", extra={"tool": name, "error": str(exc)})
            return f"[tool error: could not reach MCP server — {exc}]"

        try:
            body = response.json()
        except Exception as exc:
            logger.warning("mcp_tool_response_parse_failed", extra={"tool": name, "error": str(exc)})
            return "[tool error: MCP server returned non-JSON response]"

        # JSON-RPC protocol-level error
        if "error" in body:
            err = body["error"]
            logger.warning("mcp_protocol_error", extra={"tool": name, "error": err})
            return f"[tool error: {err.get('message', str(err))}]"

        mcp_result = body.get("result", {})

        # MCP application-level error (isError=true)
        if mcp_result.get("isError"):
            content = mcp_result.get("content", [{}])
            msg = content[0].get("text", "unknown error") if content else "unknown error"
            logger.warning("mcp_tool_error", extra={"tool": name, "mcp_message": msg})
            return f"[tool error: {msg}]"

        # Success — return the text content
        content = mcp_result.get("content", [])
        if not content:
            return "[tool returned no content]"
        return content[0].get("text", "[tool returned empty text]")

    # ------------------------------------------------------------------
    # Internal JSON-RPC helper
    # ------------------------------------------------------------------

    async def _rpc(self, method: str, params: dict) -> Any | None:
        """Send one raw JSON-RPC request; return the ``result`` field or ``None``."""
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        try:
            response = await self._client.post(
                self._url, json=payload, headers=self._headers
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            logger.warning("mcp_rpc_failed", extra={"method": method, "error": str(exc)})
            return None
        if "error" in body:
            logger.warning("mcp_rpc_error", extra={"method": method, "error": body["error"]})
            return None
        return body.get("result")


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

async def run_agentic_analysis(
    *,
    error_logs: str,
    mcp_client: MCPClient,
    ai_base_url: str,
    ai_model: str,
    ai_credential: str,
    http_client: httpx.AsyncClient,
    system_prompt: str,
    max_tokens: int | None = None,
) -> str:
    """Run the full agentic tool-use loop and return the final answer text.

    Flow
    ----
    1. Fetch the MCP tool catalogue.
    2. Send system prompt + error logs + tool catalogue to the AI.
    3. If the AI calls tools, execute them via MCP and loop.
    4. Return the final answer when ``finish_reason`` is ``stop`` or the
       iteration cap is reached.

    Raises ``AgenticLoopError`` only for unrecoverable AI backend failures
    (network errors, bad responses).  MCP tool failures are silently
    embedded in the conversation as error strings.
    """
    # 1. Build tool catalogue
    tools = await mcp_client.list_tools()

    if not tools:
        logger.warning("mcp_no_tools_available")

    # 2. Initial conversation
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": error_logs},
    ]

    # Extract the declared MCP IDs from the user message once so we can
    # correct any truncated values the model passes to tool calls.
    declared_ids = _extract_declared_ids(messages)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if ai_credential:
        headers["Authorization"] = f"Bearer {ai_credential}"

    for iteration in range(_MAX_ITERATIONS):
        payload: dict[str, Any] = {
            "model": ai_model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.info(
            "mcp_agentic_iteration",
            extra={"iteration": iteration, "messages": len(messages)},
        )

        # Debug: log the full request payload sent to the AI backend.
        # Each message is logged individually so very long content is still
        # readable in structured logs.  Credentials are never included —
        # only the payload body (messages, tools, model, token limits).
        logger.debug(
            "mcp_agentic_request",
            extra={
                "iteration": iteration,
                "ai_base_url": ai_base_url,
                "model": ai_model,
                "message_count": len(messages),
                "total_chars": sum(len(str(m.get("content") or "")) for m in messages),
                "messages": [
                    {
                        "role": m.get("role"),
                        "content_chars": len(str(m.get("content") or "")),
                        "content_preview": str(m.get("content") or "")[:500],
                        "tool_calls": m.get("tool_calls"),
                        "tool_call_id": m.get("tool_call_id"),
                    }
                    for m in messages
                ],
            },
        )

        # Call AI backend
        try:
            response = await http_client.post(ai_base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise AgenticLoopError("AI backend timed out during agentic loop") from exc
        except httpx.RequestError as exc:
            raise AgenticLoopError(f"AI backend unreachable: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise AgenticLoopError(f"AI backend returned {exc.response.status_code}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise AgenticLoopError("AI backend returned no choices")

        choice = choices[0]
        finish_reason = choice.get("finish_reason", "stop")
        assistant_message = choice.get("message", {})

        # Append the full assistant message (including tool_calls if any)
        messages.append(assistant_message)

        # --- Done ---
        if finish_reason == "stop":
            # Extract the final answer from content or reasoning_content
            content = assistant_message.get("content") or ""
            if not content.strip():
                content = assistant_message.get("reasoning_content") or ""
            if not content.strip():
                raise AgenticLoopError("AI returned stop but with empty content")
            logger.info(
                "mcp_agentic_complete",
                extra={"iterations": iteration + 1, "answer_chars": len(content)},
            )
            return content.strip()

        # --- Tool calls ---
        if finish_reason == "tool_calls":
            tool_calls = assistant_message.get("tool_calls", [])
            if not tool_calls:
                raise AgenticLoopError("finish_reason was tool_calls but no tool_calls in message")

            # Execute every requested tool and append results
            for tc in tool_calls:
                tool_id = tc.get("id", "unknown")
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                try:
                    tool_args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tool_args = {}

                logger.info(
                    "mcp_tool_call",
                    extra={"tool": tool_name, "tool_args": str(tool_args)[:200]},
                )

                tool_args = _fix_tool_args(tool_name, tool_args, declared_ids)

                tool_result = await mcp_client.call_tool(tool_name, tool_args)

                # Hard-truncate tool results to prevent pipeline data from
                # flooding the AI context window.  compact_log is NOT applied
                # here because MCP responses are often structured JSON (not log
                # text).  A plain character-based cap with a clear note is
                # sufficient and safe for all MCP response types.
                if len(tool_result) > _MCP_TOOL_RESULT_CHAR_LIMIT:
                    truncated_chars = len(tool_result) - _MCP_TOOL_RESULT_CHAR_LIMIT
                    tool_result = (
                        tool_result[:_MCP_TOOL_RESULT_CHAR_LIMIT]
                        + f"\n[... {truncated_chars} characters truncated — only logs are compressed, full pipeline data was truncated to fit context window]"
                    )
                    logger.info(
                        "mcp_tool_result_truncated",
                        extra={
                            "tool": tool_name,
                            "original_chars": len(tool_result),
                            "limit": _MCP_TOOL_RESULT_CHAR_LIMIT,
                            "truncated_chars": truncated_chars,
                        },
                    )

                logger.debug(
                    "mcp_tool_result",
                    extra={
                        "tool": tool_name,
                        "result_chars": len(tool_result),
                        "result_preview": tool_result[:500],
                    },
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result,
                    }
                )
            # Loop back to AI with updated messages
            continue

        # Unexpected finish_reason — treat whatever content is there as the answer
        content = assistant_message.get("content") or assistant_message.get("reasoning_content") or ""
        if content.strip():
            logger.warning(
                "mcp_agentic_unexpected_finish",
                extra={"finish_reason": finish_reason, "iteration": iteration},
            )
            return content.strip()

        raise AgenticLoopError(f"Unexpected finish_reason '{finish_reason}' with no content")

    # Iteration cap reached — use whatever the last message said
    last = messages[-1]
    fallback = last.get("content") or last.get("reasoning_content") or ""
    if fallback.strip():
        logger.warning("mcp_agentic_iteration_cap_reached", extra={"max": _MAX_ITERATIONS})
        return fallback.strip()

    raise AgenticLoopError(f"Agentic loop reached {_MAX_ITERATIONS} iterations without a final answer")


class AgenticLoopError(Exception):
    """Raised when the agentic loop cannot produce an answer."""

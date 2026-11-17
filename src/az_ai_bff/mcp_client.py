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
from typing import Any

import httpx

logger = logging.getLogger("az_ai_bff.mcp")

_MCP_PATH = "/mcp"
_ACCEPT = "application/json, text/event-stream"

# Maximum tool-call / AI-response rounds before giving up.
# Prevents an infinite loop if the model keeps calling tools without stopping.
_MAX_ITERATIONS = 10

# System prompt injected as the first message in every agentic analyze call.
_SYSTEM_PROMPT = (
    "You are a senior Azure DevOps and CI/CD engineer.\n"
    "You have access to tools that can query the Azure DevOps server "
    "for additional context (pipeline YAML, run history, variable groups, "
    "commits, etc.).\n\n"
    "The user will give you compacted build logs from one or more failed "
    "pipeline tasks. Each task section is prefixed with its name "
    "(=== Task Name ===).\n\n"
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
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": error_logs},
    ]

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
                    extra={"tool": tool_name, "args": str(tool_args)[:200]},
                )

                tool_result = await mcp_client.call_tool(tool_name, tool_args)

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

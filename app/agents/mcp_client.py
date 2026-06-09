"""MCP (Model Context Protocol) client wiring.

This module turns external MCP servers into LangChain `BaseTool`s that the
ReAct agent can call alongside its native `retrieve_context` / `memo` tools.

Design notes:
  * Tools are loaded ONCE at app startup (see `app/main.py` lifespan) and
    attached to the long-lived agent singleton — we never spawn a connection
    per request.
  * MCP is strictly opt-in (`MCP_ENABLED=true` + a configured server). When
    disabled or misconfigured the agent keeps working with just its native
    tools — MCP failures must never take down the API.
  * Tavily is exposed over two transports:
      - `http`  (default): the hosted streamable-HTTP endpoint. No Node needed,
        which keeps the python-only Docker image slim.
      - `stdio`: the `tavily-mcp` npm package via `npx` (requires Node).
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


def _tavily_connection(settings: Settings) -> dict[str, Any]:
    """Build the langchain-mcp-adapters connection dict for the Tavily server."""
    key = settings.tavily_api_key
    if settings.mcp_tavily_transport == "stdio":
        return {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {"TAVILY_API_KEY": key},
        }
    # Default: hosted streamable-HTTP endpoint — the API key travels in the URL.
    return {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={key}",
    }


def build_mcp_connections(settings: Settings) -> dict[str, dict[str, Any]]:
    """Map of {server_name: connection} for every configured MCP server."""
    connections: dict[str, dict[str, Any]] = {}
    if settings.tavily_api_key:
        connections["tavily"] = _tavily_connection(settings)
    return connections


async def load_mcp_tools(settings: Settings) -> list[BaseTool]:
    """Connect to the configured MCP servers and return their tools.

    Returns an empty list (never raises) when MCP is disabled, unconfigured, or
    if the connection fails — the agent then runs with only its native tools.
    """
    if not settings.mcp_enabled:
        log.info("mcp.disabled")
        return []

    connections = build_mcp_connections(settings)
    if not connections:
        log.warning(
            "mcp.enabled_but_no_servers",
            hint="set TAVILY_API_KEY to activate the Tavily web-search MCP",
        )
        return []

    # Imported lazily so the app still boots if the adapter isn't installed.
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections)
    try:
        tools = await client.get_tools()
    except Exception as exc:  # noqa: BLE001 — MCP must never break startup
        log.error(
            "mcp.load_failed",
            error=str(exc),
            servers=list(connections),
            transport=settings.mcp_tavily_transport,
        )
        return []

    log.info(
        "mcp.tools_loaded",
        count=len(tools),
        names=[t.name for t in tools],
        servers=list(connections),
        transport=settings.mcp_tavily_transport,
    )
    return tools


__all__ = ["build_mcp_connections", "load_mcp_tools"]

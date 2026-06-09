from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool

from app.config import Settings


def _fake_tool(name: str = "tavily-search") -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda query: "ok", name=name, description="fake web search tool"
    )


class TestBuildConnections:
    def test_http_is_default_transport(self) -> None:
        from app.agents.mcp_client import build_mcp_connections

        conns = build_mcp_connections(Settings(MCP_ENABLED=True, TAVILY_API_KEY="tvly-x"))
        assert set(conns) == {"tavily"}
        assert conns["tavily"]["transport"] == "streamable_http"
        assert "tvly-x" in conns["tavily"]["url"]

    def test_stdio_transport(self) -> None:
        from app.agents.mcp_client import build_mcp_connections

        conns = build_mcp_connections(
            Settings(MCP_ENABLED=True, TAVILY_API_KEY="tvly-x", MCP_TAVILY_TRANSPORT="stdio")
        )
        tavily = conns["tavily"]
        assert tavily["transport"] == "stdio"
        assert tavily["command"] == "npx"
        assert "tavily-mcp@latest" in tavily["args"]
        assert tavily["env"]["TAVILY_API_KEY"] == "tvly-x"

    def test_no_key_yields_no_servers(self) -> None:
        from app.agents.mcp_client import build_mcp_connections

        assert build_mcp_connections(Settings(TAVILY_API_KEY="")) == {}


class TestLoadMcpTools:
    async def test_disabled_returns_empty(self) -> None:
        from app.agents.mcp_client import load_mcp_tools

        tools = await load_mcp_tools(Settings(MCP_ENABLED=False, TAVILY_API_KEY="tvly-x"))
        assert tools == []

    async def test_enabled_without_server_returns_empty(self) -> None:
        from app.agents.mcp_client import load_mcp_tools

        tools = await load_mcp_tools(Settings(MCP_ENABLED=True, TAVILY_API_KEY=""))
        assert tools == []

    async def test_loads_tools_from_client(self) -> None:
        from app.agents.mcp_client import load_mcp_tools

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[_fake_tool()])
        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient", return_value=mock_client
        ):
            tools = await load_mcp_tools(Settings(MCP_ENABLED=True, TAVILY_API_KEY="tvly-x"))
        assert [t.name for t in tools] == ["tavily-search"]

    async def test_connection_failure_degrades_gracefully(self) -> None:
        from app.agents.mcp_client import load_mcp_tools

        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(side_effect=RuntimeError("connection refused"))
        with patch(
            "langchain_mcp_adapters.client.MultiServerMCPClient", return_value=mock_client
        ):
            tools = await load_mcp_tools(Settings(MCP_ENABLED=True, TAVILY_API_KEY="tvly-x"))
        assert tools == []  # never raises — agent keeps its native tools


class TestAgentMergesMcpTools:
    def _make_agent(self, fake_embeddings, tmp_persist_dir):
        from app.agents.react_agent import ReActAgent
        from app.rag.retriever import Retriever
        from app.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            embeddings=fake_embeddings,
            persist_dir=tmp_persist_dir,
            collection_name="test_mcp_merge",
        )
        retriever = Retriever(store=store, default_k=2)
        return ReActAgent(settings=Settings(OPENAI_API_KEY="sk-test"), retriever=retriever)

    def test_graph_includes_mcp_tools_and_prompt_addendum(
        self, fake_embeddings, tmp_persist_dir
    ) -> None:
        from app.agents.prompts import WEB_SEARCH_ADDENDUM

        agent = self._make_agent(fake_embeddings, tmp_persist_dir)
        agent.set_mcp_tools([_fake_tool("tavily-search")])

        with patch("app.agents.react_agent.create_react_agent") as mock_cra:
            agent._build_graph("sess")

        kwargs = mock_cra.call_args.kwargs
        tool_names = {t.name for t in kwargs["tools"]}
        assert {"retrieve_context", "memo", "tavily-search"} <= tool_names
        prompt = kwargs.get("prompt") or kwargs.get("state_modifier")
        assert WEB_SEARCH_ADDENDUM in prompt

    def test_graph_without_mcp_omits_addendum(
        self, fake_embeddings, tmp_persist_dir
    ) -> None:
        from app.agents.prompts import WEB_SEARCH_ADDENDUM

        agent = self._make_agent(fake_embeddings, tmp_persist_dir)  # no MCP tools

        with patch("app.agents.react_agent.create_react_agent") as mock_cra:
            agent._build_graph("sess")

        kwargs = mock_cra.call_args.kwargs
        tool_names = {t.name for t in kwargs["tools"]}
        assert tool_names == {"retrieve_context", "memo"}
        prompt = kwargs.get("prompt") or kwargs.get("state_modifier")
        assert WEB_SEARCH_ADDENDUM not in prompt

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.react_agent import _extract_answer_and_trace
from app.agents.tools import MemoStore, make_memo_tool


class TestTraceExtraction:
    def test_extracts_final_answer_only(self) -> None:
        state = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="Hello there."),
            ]
        }
        answer, trace = _extract_answer_and_trace(state)
        assert answer == "Hello there."
        assert len(trace) == 1
        assert trace[0].thought == "Hello there."

    def test_extracts_tool_calls_and_observations(self) -> None:
        ai_with_call = AIMessage(
            content="I need to search the YC knowledge base.",
            tool_calls=[{"name": "retrieve_context", "args": {"query": "YC deal"}, "id": "1"}],
        )
        state = {
            "messages": [
                HumanMessage(content="How much does YC invest?"),
                ai_with_call,
                ToolMessage(
                    content="[1] source=01_about_yc.md\nYC invests $500k per company.",
                    tool_call_id="1",
                ),
                AIMessage(content="YC invests $500k per company."),
            ]
        }
        answer, trace = _extract_answer_and_trace(state)
        assert answer == "YC invests $500k per company."
        assert any(s.action == "retrieve_context" for s in trace)
        assert any(s.observation and "$500k" in s.observation for s in trace)


class TestMemoTool:
    def test_memo_writes_and_clears(self) -> None:
        store = MemoStore()
        tool = make_memo_tool(store, session_id="s1")
        result = tool.invoke({"note": "first finding"})
        assert "MEMO_SAVED" in result
        assert "first finding" in result

        store.clear("s1")
        assert store.read("s1") == []

    def test_memo_isolates_sessions(self) -> None:
        store = MemoStore()
        make_memo_tool(store, "a").invoke({"note": "from-a"})
        make_memo_tool(store, "b").invoke({"note": "from-b"})
        assert store.read("a") == ["from-a"]
        assert store.read("b") == ["from-b"]


class TestAgentRun:
    """End-to-end agent run with the LLM and graph mocked out."""

    @pytest.mark.asyncio
    async def test_arun_returns_answer_and_contexts(
        self, fake_embeddings, tmp_persist_dir, sample_markdown_file, monkeypatch
    ) -> None:
        from app.agents.react_agent import ReActAgent
        from app.config import Settings
        from app.rag.chunker import Chunker
        from app.rag.loaders import load_document
        from app.rag.retriever import Retriever
        from app.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            embeddings=fake_embeddings,
            persist_dir=tmp_persist_dir,
            collection_name="test_agent",
        )
        chunks = Chunker(chunk_size=200, chunk_overlap=20).split(
            load_document(sample_markdown_file)
        )
        store.add_documents(chunks)
        retriever = Retriever(store=store, default_k=2)

        settings = Settings(OPENAI_API_KEY="sk-test", LLM_TIMEOUT_SECONDS=10)
        agent = ReActAgent(settings=settings, retriever=retriever)

        fake_state = {
            "messages": [
                HumanMessage(content="When was YC founded?"),
                AIMessage(content="Y Combinator was founded in March 2005."),
            ]
        }

        with patch.object(agent, "_build_graph") as mock_build:
            mock_build.return_value.ainvoke = AsyncMock(return_value=fake_state)
            response = await agent.arun("When was YC founded?")

        assert "2005" in response.answer
        assert response.contexts  # pre-retrieval populated
        assert response.run_id

    @pytest.mark.asyncio
    async def test_arun_timeout_raises(
        self, fake_embeddings, tmp_persist_dir
    ) -> None:
        import asyncio

        from app.agents.react_agent import ReActAgent
        from app.config import Settings
        from app.core.exceptions import LLMTimeoutError
        from app.rag.retriever import Retriever
        from app.vectorstore.chroma_store import ChromaVectorStore

        store = ChromaVectorStore(
            embeddings=fake_embeddings,
            persist_dir=tmp_persist_dir,
            collection_name="test_timeout",
        )
        retriever = Retriever(store=store, default_k=2)
        settings = Settings(OPENAI_API_KEY="sk-test", LLM_TIMEOUT_SECONDS=1)
        agent = ReActAgent(settings=settings, retriever=retriever)

        async def _hang(*_a, **_kw):
            await asyncio.sleep(10)

        with patch.object(agent, "_build_graph") as mock_build:
            mock_build.return_value.ainvoke = _hang
            with pytest.raises(LLMTimeoutError):
                await agent.arun("anything")

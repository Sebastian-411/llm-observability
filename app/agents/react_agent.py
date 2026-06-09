from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from app.agents.prompts import REACT_SYSTEM_PROMPT, WEB_SEARCH_ADDENDUM
from app.agents.tools import MemoStore, make_memo_tool, make_retrieve_tool
from app.config import Settings
from app.core.exceptions import LLMTimeoutError
from app.core.logging import get_logger, log_latency
from app.models import QueryResponse, ReActStep, RetrievedChunk
from app.rag.retriever import Retriever

log = get_logger(__name__)


class ReActAgent:
    """LangGraph ReAct agent over a RAG retriever.

    Notes:
      * The graph is built once per `ReActAgent` instance — cheap to reuse.
      * Each `arun` call is independent: it builds a fresh memo store keyed by
        a session id, so notes never leak between requests.
      * Traces are emitted as `ReActStep` records derived from the message
        stream — that's what the API surfaces to clients and what LangSmith
        records under the hood.
    """

    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        memo_store: MemoStore | None = None,
        mcp_tools: list[BaseTool] | None = None,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._memo_store = memo_store or MemoStore()
        self._mcp_tools: list[BaseTool] = list(mcp_tools or [])
        self._llm = self._build_llm()

    def set_mcp_tools(self, tools: list[BaseTool] | None) -> None:
        """Attach MCP-provided tools after construction (called at app startup).

        The graph is rebuilt per request, so newly attached tools take effect on
        the next `arun`/`astream` without recreating the agent.
        """
        self._mcp_tools = list(tools or [])

    # -------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------
    def _build_llm(self) -> ChatOpenAI:
        if not self._settings.openai_api_key:
            log.warning("agent.llm.no_api_key — calls will fail until OPENAI_API_KEY is set")
        return ChatOpenAI(
            api_key=self._settings.openai_api_key or "missing",
            model=self._settings.llm_model,
            temperature=self._settings.llm_temperature,
            timeout=self._settings.llm_timeout_seconds,
            max_retries=self._settings.llm_max_retries,
        )

    def _build_graph(self, session_id: str) -> Any:
        tools: list[BaseTool] = [
            make_retrieve_tool(self._retriever),
            make_memo_tool(self._memo_store, session_id),
            *self._mcp_tools,
        ]
        # Only advertise the web-search tools in the prompt when they're present.
        prompt = REACT_SYSTEM_PROMPT
        if self._mcp_tools:
            prompt += WEB_SEARCH_ADDENDUM
        # langgraph 0.2.x uses `state_modifier`; 0.3.x renamed it to `prompt`.
        # Try the new kwarg first and fall back so the code works on both.
        try:
            return create_react_agent(
                model=self._llm,
                tools=tools,
                prompt=prompt,
            )
        except TypeError:
            return create_react_agent(
                model=self._llm,
                tools=tools,
                state_modifier=prompt,
            )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    async def arun(
        self,
        question: str,
        *,
        top_k: int | None = None,
        session_id: str | None = None,
    ) -> QueryResponse:
        session_id = session_id or str(uuid.uuid4())
        start = time.perf_counter()
        run_id = str(uuid.uuid4())

        # Pre-retrieve to also return contexts to the client, decoupled from
        # whatever the agent decides to do internally.
        contexts = self._retriever.retrieve(question, k=top_k)

        graph = self._build_graph(session_id)
        inputs = {"messages": [HumanMessage(content=question)]}
        config = {
            "configurable": {"thread_id": session_id},
            "metadata": {"run_id": run_id, "session_id": session_id},
            "run_name": "react-rag",
        }

        try:
            with log_latency(log, "agent.run.completed", session_id=session_id):
                result = await asyncio.wait_for(
                    graph.ainvoke(inputs, config=config),
                    timeout=self._settings.llm_timeout_seconds * 2,
                )
        except asyncio.TimeoutError as exc:
            raise LLMTimeoutError(
                f"Agent execution exceeded {self._settings.llm_timeout_seconds * 2}s"
            ) from exc
        finally:
            self._memo_store.clear(session_id)

        answer, trace = _extract_answer_and_trace(result)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return QueryResponse(
            question=question,
            answer=answer,
            contexts=contexts,
            trace=trace,
            elapsed_ms=elapsed_ms,
            run_id=run_id,
        )

    async def astream(
        self,
        question: str,
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Token-level streaming of the final answer.

        We stream messages from the graph and yield each chunk as it arrives.
        """
        session_id = session_id or str(uuid.uuid4())
        graph = self._build_graph(session_id)
        inputs = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for event in graph.astream_events(inputs, config=config, version="v2"):
                if event.get("event") == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and getattr(chunk, "content", None):
                        yield str(chunk.content)
        finally:
            self._memo_store.clear(session_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_answer_and_trace(state: dict[str, Any]) -> tuple[str, list[ReActStep]]:
    """Turn the LangGraph message stream into a final answer + ReAct trace."""
    messages: list[Any] = state.get("messages", [])
    trace: list[ReActStep] = []
    step_idx = 0
    pending_thought: str | None = None
    pending_action: str | None = None
    pending_action_input: str | None = None
    final_answer = ""

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            content = (msg.content or "").strip() if isinstance(msg.content, str) else ""
            if tool_calls:
                for call in tool_calls:
                    step_idx += 1
                    pending_thought = content or pending_thought
                    pending_action = call.get("name")
                    pending_action_input = _coerce_input(call.get("args", {}))
                    trace.append(
                        ReActStep(
                            step=step_idx,
                            thought=pending_thought,
                            action=pending_action,
                            action_input=pending_action_input,
                        )
                    )
                    pending_thought = None
            else:
                final_answer = content or final_answer
                if content:
                    step_idx += 1
                    trace.append(
                        ReActStep(
                            step=step_idx,
                            thought=content,
                            action=None,
                            action_input=None,
                            observation=None,
                        )
                    )
        elif isinstance(msg, ToolMessage):
            if trace and trace[-1].observation is None:
                trace[-1].observation = _truncate(str(msg.content))
            else:
                step_idx += 1
                trace.append(
                    ReActStep(
                        step=step_idx,
                        observation=_truncate(str(msg.content)),
                    )
                )

    return final_answer, trace


def _coerce_input(args: dict[str, Any]) -> str:
    if not args:
        return ""
    if len(args) == 1:
        return str(next(iter(args.values())))
    return ", ".join(f"{k}={v}" for k, v in args.items())


def _truncate(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text) - limit} chars]"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_react_agent(settings: Settings, retriever: Retriever) -> ReActAgent:
    return ReActAgent(settings=settings, retriever=retriever)


__all__ = ["ReActAgent", "build_react_agent", "RetrievedChunk"]

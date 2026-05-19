from __future__ import annotations

from typing import AsyncIterator

from app.agents import ReActAgent
from app.core.logging import get_logger
from app.models import QueryResponse

log = get_logger(__name__)


class QueryService:
    """Thin orchestration layer in front of the ReAct agent.

    Kept separate from the agent so the API doesn't depend directly on the
    LangGraph object, and so we can compose pre/post-processing here later
    (rate limiting, request validation, caching) without touching the agent.
    """

    def __init__(self, agent: ReActAgent) -> None:
        self._agent = agent

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        include_trace: bool = True,
        session_id: str | None = None,
    ) -> QueryResponse:
        response = await self._agent.arun(
            question=question,
            top_k=top_k,
            session_id=session_id,
        )
        if not include_trace:
            response.trace = []
        return response

    async def stream_answer(
        self,
        question: str,
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        async for token in self._agent.astream(question, session_id=session_id):
            yield token

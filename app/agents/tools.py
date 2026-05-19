from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.rag.retriever import Retriever

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# retrieve_context tool — main RAG entrypoint for the agent
# ---------------------------------------------------------------------------
class RetrieveContextInput(BaseModel):
    query: str = Field(..., description="Question or keyword phrase to search the knowledge base.")
    k: int | None = Field(default=None, description="Optional number of chunks to retrieve.")


def _format_chunks(chunks: list[Any]) -> str:
    if not chunks:
        return "NO_RESULTS: the knowledge base did not return any matching chunks."
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        src = c.source or c.metadata.get("source", "unknown")
        page = c.metadata.get("page")
        header = f"[{i}] source={src}"
        if page is not None:
            header += f" page={page}"
        if c.score is not None:
            header += f" score={c.score:.3f}"
        parts.append(f"{header}\n{c.content}")
    return "\n\n".join(parts)


def make_retrieve_tool(retriever: Retriever) -> StructuredTool:
    """Bind a `Retriever` instance into a LangChain tool."""

    def _retrieve(query: str, k: int | None = None) -> str:
        log.info("agent.tool.retrieve_context", query=query, k=k)
        chunks = retriever.retrieve(query=query, k=k)
        return _format_chunks(chunks)

    return StructuredTool.from_function(
        func=_retrieve,
        name="retrieve_context",
        description=(
            "Semantic search over the indexed knowledge base. "
            "Use this whenever the question requires factual grounding. "
            "Returns the most relevant chunks with their source metadata."
        ),
        args_schema=RetrieveContextInput,
    )


# ---------------------------------------------------------------------------
# memo tool — scratchpad shared across ReAct steps within one run
# ---------------------------------------------------------------------------
class MemoInput(BaseModel):
    note: str = Field(..., description="A short note to remember across steps in this run.")


class MemoStore:
    """In-process notes keyed by session id.

    Cleared per-request from the service layer to keep memory bounded.
    """

    def __init__(self) -> None:
        self._notes: dict[str, list[str]] = {}

    def write(self, session_id: str, note: str) -> None:
        self._notes.setdefault(session_id, []).append(note)

    def read(self, session_id: str) -> list[str]:
        return list(self._notes.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._notes.pop(session_id, None)


def make_memo_tool(store: MemoStore, session_id: str) -> StructuredTool:
    def _memo(note: str) -> str:
        store.write(session_id, note)
        all_notes = store.read(session_id)
        return f"MEMO_SAVED ({len(all_notes)} total). Current notes:\n- " + "\n- ".join(all_notes)

    return StructuredTool.from_function(
        func=_memo,
        name="memo",
        description=(
            "Record a short note for yourself across ReAct steps in this run. "
            "Useful for tracking partial findings before producing the final answer."
        ),
        args_schema=MemoInput,
    )

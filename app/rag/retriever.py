from __future__ import annotations

from app.core.logging import get_logger, log_latency
from app.models import RetrievedChunk
from app.vectorstore import ChromaVectorStore

log = get_logger(__name__)


class Retriever:
    """Wraps the vector store with score-threshold filtering + nice DTOs."""

    def __init__(
        self,
        store: ChromaVectorStore,
        default_k: int = 4,
        score_threshold: float = 0.0,
    ) -> None:
        self._store = store
        self._default_k = default_k
        self._score_threshold = score_threshold

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        top_k = k or self._default_k
        with log_latency(log, "retrieval.completed", query_len=len(query), k=top_k):
            scored = self._store.similarity_search(
                query=query,
                k=top_k,
                score_threshold=self._score_threshold,
            )
        return [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source"),
                score=float(score) if score is not None else None,
                metadata=doc.metadata,
            )
            for doc, score in scored
        ]

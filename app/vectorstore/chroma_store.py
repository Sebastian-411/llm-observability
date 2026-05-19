from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import Settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.embeddings.provider import EmbeddingProvider

log = get_logger(__name__)


class ChromaVectorStore:
    """Thin wrapper around langchain_chroma.Chroma.

    Owns persistence, exposes a small surface area to the rest of the app,
    and centralises error handling so callers never deal with Chroma directly.
    """

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        persist_dir: str,
        collection_name: str,
    ) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        log.info(
            "vectorstore.initialized",
            backend="chroma",
            collection=collection_name,
            persist_dir=persist_dir,
        )

    # -------------------------------------------------------------------
    # Writes
    # -------------------------------------------------------------------
    def add_documents(self, docs: list[Document]) -> list[str]:
        if not docs:
            return []
        try:
            return self._store.add_documents(docs)
        except Exception as exc:  # noqa: BLE001
            log.error("vectorstore.add_documents.error", error=str(exc))
            raise RetrievalError(f"Failed to persist documents: {exc}") from exc

    # -------------------------------------------------------------------
    # Reads
    # -------------------------------------------------------------------
    def similarity_search(
        self,
        query: str,
        k: int,
        score_threshold: float = 0.0,
    ) -> list[tuple[Document, float]]:
        try:
            raw = self._store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as exc:  # noqa: BLE001
            log.error("vectorstore.similarity_search.error", error=str(exc))
            raise RetrievalError(f"Similarity search failed: {exc}") from exc

        if score_threshold <= 0:
            return raw
        return [(doc, score) for doc, score in raw if score >= score_threshold]

    def as_retriever(self, k: int) -> Any:
        return self._store.as_retriever(search_kwargs={"k": k})

    # -------------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        try:
            count = self._store._collection.count()  # noqa: SLF001 — official API path
        except Exception:  # noqa: BLE001
            count = -1
        return {
            "backend": "chroma",
            "collection": self._collection_name,
            "persist_dir": self._persist_dir,
            "document_count": count,
        }

    def reset(self) -> None:
        """Drop the collection — intended for tests and admin tooling."""
        try:
            self._store.delete_collection()
        except Exception as exc:  # noqa: BLE001
            log.warning("vectorstore.reset.error", error=str(exc))


def get_vectorstore(
    settings: Settings,
    embeddings: EmbeddingProvider,
) -> ChromaVectorStore:
    return ChromaVectorStore(
        embeddings=embeddings,
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )

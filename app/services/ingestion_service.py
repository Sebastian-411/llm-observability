from __future__ import annotations

import time
from pathlib import Path

from app.core.logging import get_logger
from app.models import IngestionResponse
from app.rag.chunker import Chunker
from app.rag.loaders import DocumentLoader
from app.vectorstore import ChromaVectorStore

log = get_logger(__name__)


class IngestionService:
    """Orchestrates: load → chunk → embed → persist.

    Embedding happens inside the vector store (it owns the embeddings
    callable) so this service stays small and easy to reason about.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        store: ChromaVectorStore,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._store = store

    def ingest_file(self, path: str | Path) -> IngestionResponse:
        start = time.perf_counter()
        docs = self._loader.load(path)
        chunks = self._chunker.split(docs)
        self._store.add_documents(chunks)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        stats = self._store.stats()
        log.info(
            "ingestion.completed",
            source=str(path),
            chunks=len(chunks),
            elapsed_ms=elapsed_ms,
        )
        return IngestionResponse(
            source=str(path),
            chunks_ingested=len(chunks),
            collection=stats.get("collection", "unknown"),
            elapsed_ms=elapsed_ms,
        )

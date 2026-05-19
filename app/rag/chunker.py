from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger

log = get_logger(__name__)


class Chunker:
    """Configurable recursive-character chunker preserving document metadata.

    Why recursive: it tries semantic boundaries first (\\n\\n, \\n, sentence,
    word) and only falls back to hard cuts if needed. That keeps chunks
    coherent for retrieval, which is the whole point of chunking for RAG.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, docs: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata.setdefault("chunk_index", i)
        log.info(
            "ingestion.chunked",
            input_docs=len(docs),
            output_chunks=len(chunks),
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        return chunks

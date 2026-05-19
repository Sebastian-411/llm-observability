from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import IngestionError, UnsupportedFormatError
from app.rag.chunker import Chunker
from app.rag.loaders import DocumentLoader, load_document
from app.rag.retriever import Retriever
from app.vectorstore.chroma_store import ChromaVectorStore


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
class TestLoaders:
    def test_loads_txt(self, sample_text_file: Path) -> None:
        docs = load_document(sample_text_file)
        assert len(docs) == 1
        assert "Y Combinator" in docs[0].page_content
        assert docs[0].metadata["format"] == "txt"

    def test_loads_markdown(self, sample_markdown_file: Path) -> None:
        docs = load_document(sample_markdown_file)
        assert len(docs) == 1
        assert "About YC" in docs[0].page_content
        assert docs[0].metadata["format"] == "md"

    def test_loads_json_list(self, sample_json_file: Path) -> None:
        docs = load_document(sample_json_file)
        assert len(docs) == 2
        assert all(d.metadata["format"] == "json" for d in docs)

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        weird = tmp_path / "image.png"
        weird.write_bytes(b"\x89PNG\r\n")
        with pytest.raises(UnsupportedFormatError):
            load_document(weird)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(IngestionError):
            load_document("/no/such/file.txt")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid", encoding="utf-8")
        with pytest.raises(IngestionError):
            load_document(bad)

    def test_loader_class_delegates(self, sample_text_file: Path) -> None:
        loader = DocumentLoader()
        assert loader.load(sample_text_file)[0].metadata["format"] == "txt"


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
class TestChunker:
    def test_rejects_bad_config(self) -> None:
        with pytest.raises(ValueError):
            Chunker(chunk_size=0)
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, chunk_overlap=200)

    def test_splits_into_multiple_chunks(self) -> None:
        from langchain_core.documents import Document

        long_text = ("Paragraph A. " * 200) + "\n\n" + ("Paragraph B. " * 200)
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = Chunker(chunk_size=200, chunk_overlap=20).split(docs)
        assert len(chunks) > 1
        assert all(c.metadata["source"] == "test" for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)


# ---------------------------------------------------------------------------
# Retriever (against a real Chroma instance with fake embeddings)
# ---------------------------------------------------------------------------
class TestRetriever:
    def test_retrieves_relevant_chunks(
        self, fake_embeddings, tmp_persist_dir, sample_markdown_file
    ) -> None:
        store = ChromaVectorStore(
            embeddings=fake_embeddings,
            persist_dir=tmp_persist_dir,
            collection_name="test_retriever",
        )
        chunks = Chunker(chunk_size=200, chunk_overlap=20).split(
            load_document(sample_markdown_file)
        )
        store.add_documents(chunks)

        retriever = Retriever(store=store, default_k=2)
        results = retriever.retrieve("When was Y Combinator founded?")
        assert len(results) >= 1
        assert all(r.source for r in results)

    def test_empty_retrieval_returns_empty_list(
        self, fake_embeddings, tmp_persist_dir
    ) -> None:
        store = ChromaVectorStore(
            embeddings=fake_embeddings,
            persist_dir=tmp_persist_dir,
            collection_name="test_empty",
        )
        retriever = Retriever(store=store, default_k=4)
        results = retriever.retrieve("anything")
        assert results == []

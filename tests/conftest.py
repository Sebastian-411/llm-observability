from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Force a test-only configuration BEFORE any app modules read settings.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp(prefix="chroma_test_"))
os.environ.setdefault("CHROMA_COLLECTION", "test_collection")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("LOG_LEVEL", "WARNING")


class FakeEmbeddings:
    """Deterministic, hashing-based fake embeddings.

    Avoids any network call while still producing distinct vectors for
    distinct strings, which is enough for similarity search to behave.
    """

    dimension = 64

    def _vec(self, text: str) -> list[float]:
        rng = sum(ord(c) for c in text) or 1
        return [((rng * (i + 1)) % 1000) / 1000.0 for i in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def tmp_persist_dir(tmp_path: Path) -> str:
    persist = tmp_path / "chroma"
    persist.mkdir(parents=True, exist_ok=True)
    return str(persist)


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text(
        "Y Combinator invests $500,000 in each startup: $125,000 for 7% equity "
        "on a post-money SAFE, plus $375,000 on an uncapped MFN SAFE.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_markdown_file(tmp_path: Path) -> Path:
    p = tmp_path / "guide.md"
    p.write_text(
        "# About YC\n\nY Combinator was founded in March 2005 by Paul Graham, "
        "Jessica Livingston, Robert Tapan Morris, and Trevor Blackwell.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.json"
    p.write_text(
        '[{"topic": "deal", "detail": "$125k for 7%"},'
        ' {"topic": "duration", "detail": "3-month batch ending with Demo Day"}]',
        encoding="utf-8",
    )
    return p


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

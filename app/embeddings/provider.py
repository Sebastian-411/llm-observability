from __future__ import annotations

from typing import Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

log = get_logger(__name__)


class EmbeddingProvider(Protocol):
    """Minimal contract used by the rest of the app."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class _RetryingEmbeddings:
    """Wrap any embeddings backend with exponential-backoff retries."""

    def __init__(self, inner: EmbeddingProvider, max_attempts: int = 3) -> None:
        self._inner = inner
        self._max_attempts = max_attempts

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._inner.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001
            log.warning("embedding.embed_documents.error", error=str(exc))
            raise

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
    )
    def embed_query(self, text: str) -> list[float]:
        try:
            return self._inner.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("embedding.embed_query.error", error=str(exc))
            raise


def get_embeddings(settings: Settings) -> EmbeddingProvider:
    """Build an embedding provider according to settings."""
    provider = settings.embedding_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise EmbeddingError("OPENAI_API_KEY is required for the openai provider")
        from langchain_openai import OpenAIEmbeddings

        backend: EmbeddingProvider = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
    elif provider == "sentence-transformers":
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        backend = SentenceTransformerEmbeddings(model_name=settings.embedding_model)
    else:  # pragma: no cover — guarded by pydantic Literal
        raise EmbeddingError(f"Unknown embedding provider: {provider}")

    log.info(
        "embeddings.initialized",
        provider=provider,
        model=settings.embedding_model,
    )
    return _RetryingEmbeddings(backend, max_attempts=3)

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.agents import ReActAgent, build_react_agent
from app.config import Settings, get_settings
from app.embeddings import EmbeddingProvider, get_embeddings
from app.evaluation import RagasEvaluator
from app.rag.chunker import Chunker
from app.rag.loaders import DocumentLoader
from app.rag.retriever import Retriever
from app.services import IngestionService, QueryService
from app.vectorstore import ChromaVectorStore, get_vectorstore


# ---------------------------------------------------------------------------
# Composition root (cached singletons) — kept tiny and explicit on purpose.
# Each provider here pulls from settings and stitches the next layer.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _embeddings_singleton() -> EmbeddingProvider:
    return get_embeddings(get_settings())


@lru_cache(maxsize=1)
def _vectorstore_singleton() -> ChromaVectorStore:
    return get_vectorstore(get_settings(), _embeddings_singleton())


@lru_cache(maxsize=1)
def _retriever_singleton() -> Retriever:
    settings = get_settings()
    return Retriever(
        store=_vectorstore_singleton(),
        default_k=settings.top_k,
        score_threshold=settings.score_threshold,
    )


@lru_cache(maxsize=1)
def _agent_singleton() -> ReActAgent:
    return build_react_agent(get_settings(), _retriever_singleton())


# ---------------------------------------------------------------------------
# FastAPI dependencies — the public surface of this module.
# ---------------------------------------------------------------------------
def provide_settings() -> Settings:
    return get_settings()


def provide_vectorstore() -> ChromaVectorStore:
    return _vectorstore_singleton()


def provide_ingestion_service(
    settings: Settings = Depends(provide_settings),
) -> IngestionService:
    return IngestionService(
        loader=DocumentLoader(),
        chunker=Chunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        store=_vectorstore_singleton(),
    )


def provide_agent() -> ReActAgent:
    return _agent_singleton()


def provide_query_service() -> QueryService:
    return QueryService(agent=_agent_singleton())


def provide_evaluator(
    query_service: QueryService = Depends(provide_query_service),
) -> RagasEvaluator:
    return RagasEvaluator(query_service=query_service)


def reset_singletons() -> None:
    """Test-only helper — clears every lru_cache in this module."""
    _embeddings_singleton.cache_clear()
    _vectorstore_singleton.cache_clear()
    _retriever_singleton.cache_clear()
    _agent_singleton.cache_clear()
    get_settings.cache_clear()

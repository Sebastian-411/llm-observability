from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    # --- Embeddings ---
    embedding_provider: Literal["openai", "sentence-transformers"] = Field(
        default="openai", alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=1536, alias="EMBEDDING_DIMENSION")

    # --- Vector Store ---
    vectorstore_provider: Literal["chroma"] = Field(
        default="chroma", alias="VECTORSTORE_PROVIDER"
    )
    chroma_persist_dir: str = Field(default="./.chroma", alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field(default="rag_documents", alias="CHROMA_COLLECTION")

    # --- RAG ---
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, alias="TOP_K")
    score_threshold: float = Field(default=0.0, alias="SCORE_THRESHOLD")

    # --- LangSmith ---
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT"
    )
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="reto3-react-rag", alias="LANGCHAIN_PROJECT"
    )

    # --- API ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=False, alias="API_RELOAD")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langchain_tracing_v2 and self.langchain_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (used as FastAPI dependency)."""
    return Settings()

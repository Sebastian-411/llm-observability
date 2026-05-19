from __future__ import annotations


class AppError(Exception):
    """Base error for the application."""

    status_code: int = 500

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class UnsupportedFormatError(AppError):
    status_code = 415


class IngestionError(AppError):
    status_code = 422


class EmbeddingError(AppError):
    status_code = 502


class RetrievalError(AppError):
    status_code = 502


class LLMTimeoutError(AppError):
    status_code = 504


class EvaluationError(AppError):
    status_code = 500

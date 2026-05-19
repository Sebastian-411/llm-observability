from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import __version__
from app.api.dependencies import (
    provide_evaluator,
    provide_ingestion_service,
    provide_query_service,
    provide_settings,
    provide_vectorstore,
)
from app.config import Settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.evaluation import RagasEvaluator
from app.models import (
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)
from app.services import IngestionService, QueryService
from app.vectorstore import ChromaVectorStore

log = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(
    settings: Settings = Depends(provide_settings),
    store: ChromaVectorStore = Depends(provide_vectorstore),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        vectorstore=store.stats(),
        langsmith_enabled=settings.langsmith_enabled,
    )


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@router.post("/ingest", response_model=IngestionResponse, tags=["ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    service: IngestionService = Depends(provide_ingestion_service),
) -> IngestionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = Path(file.filename).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        try:
            return service.ingest_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/ingest/path", response_model=IngestionResponse, tags=["ingestion"])
def ingest_local_path(
    path: str,
    service: IngestionService = Depends(provide_ingestion_service),
) -> IngestionResponse:
    """Ingest a file already on the server filesystem (useful for batch jobs)."""
    try:
        return service.ingest_file(path)
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query(
    payload: QueryRequest,
    service: QueryService = Depends(provide_query_service),
) -> QueryResponse:
    try:
        return await service.answer(
            question=payload.question,
            top_k=payload.top_k,
            include_trace=payload.include_trace,
            session_id=payload.session_id,
        )
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/query/stream", tags=["query"])
async def query_stream(
    payload: QueryRequest,
    service: QueryService = Depends(provide_query_service),
) -> StreamingResponse:
    async def _gen():
        try:
            async for chunk in service.stream_answer(
                payload.question, session_id=payload.session_id
            ):
                yield chunk
        except AppError as exc:
            yield f"\n[error] {exc.message}"

    return StreamingResponse(_gen(), media_type="text/plain")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@router.post("/evaluate", response_model=EvaluationResponse, tags=["evaluation"])
async def evaluate(
    payload: EvaluationRequest,
    evaluator: RagasEvaluator = Depends(provide_evaluator),
) -> EvaluationResponse:
    try:
        return await evaluator.evaluate(
            samples=payload.samples,
            metric_names=payload.metrics,
        )
    except AppError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

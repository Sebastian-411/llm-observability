from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    environment: str
    vectorstore: dict[str, Any] = Field(default_factory=dict)
    langsmith_enabled: bool = False


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestionResponse(BaseModel):
    source: str
    chunks_ingested: int
    collection: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Retrieval / Query
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    content: str
    source: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReActStep(BaseModel):
    """A single Thought/Action/Observation cycle of the ReAct loop."""

    step: int
    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    include_trace: bool = True
    session_id: str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    trace: list[ReActStep] = Field(default_factory=list)
    elapsed_ms: float
    run_id: str | None = None  # LangSmith run id when tracing is on


# ---------------------------------------------------------------------------
# Evaluation (RAGAS)
# ---------------------------------------------------------------------------
class EvaluationSample(BaseModel):
    question: str
    ground_truth: str


class EvaluationRequest(BaseModel):
    samples: list[EvaluationSample] = Field(..., min_length=1)
    metrics: list[str] | None = Field(
        default=None,
        description="Subset of metrics to compute. Default = all four core metrics.",
    )


class EvaluationResponse(BaseModel):
    n_samples: int
    metrics: dict[str, float]
    per_sample: list[dict[str, Any]] = Field(default_factory=list)

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import EvaluationError
from app.evaluation import RagasEvaluator
from app.models import EvaluationSample, QueryResponse, RetrievedChunk


@pytest.mark.asyncio
async def test_evaluate_rejects_empty_samples() -> None:
    evaluator = RagasEvaluator(query_service=AsyncMock())
    with pytest.raises(EvaluationError):
        await evaluator.evaluate(samples=[])


@pytest.mark.asyncio
async def test_evaluate_aggregates_metrics(monkeypatch) -> None:
    qs = AsyncMock()
    qs.answer = AsyncMock(
        return_value=QueryResponse(
            question="When was YC founded?",
            answer="March 2005",
            contexts=[RetrievedChunk(content="YC was founded in March 2005.", source="01_about_yc.md")],
            trace=[],
            elapsed_ms=1.0,
        )
    )
    evaluator = RagasEvaluator(query_service=qs)

    fake_metrics = {
        "faithfulness": 0.9,
        "answer_relevancy": 0.85,
        "context_precision": 0.8,
        "context_recall": 0.75,
    }
    monkeypatch.setattr(
        evaluator,
        "_run_ragas",
        lambda **_kwargs: fake_metrics,
    )

    response = await evaluator.evaluate(
        samples=[EvaluationSample(question="When was YC founded?", ground_truth="March 2005.")]
    )
    assert response.n_samples == 1
    assert response.metrics == fake_metrics


@pytest.mark.asyncio
async def test_evaluate_propagates_ragas_errors(monkeypatch) -> None:
    qs = AsyncMock()
    qs.answer = AsyncMock(
        return_value=QueryResponse(
            question="q?",
            answer="x",
            contexts=[],
            trace=[],
            elapsed_ms=1.0,
        )
    )
    evaluator = RagasEvaluator(query_service=qs)

    def _boom(**_kwargs):
        raise EvaluationError("ragas blew up")

    monkeypatch.setattr(evaluator, "_run_ragas", _boom)

    with pytest.raises(EvaluationError):
        await evaluator.evaluate(
            samples=[EvaluationSample(question="q?", ground_truth="g")],
        )

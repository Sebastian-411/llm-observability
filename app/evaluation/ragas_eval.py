from __future__ import annotations

from typing import Any

from app.core.exceptions import EvaluationError
from app.core.logging import get_logger
from app.models import EvaluationResponse, EvaluationSample
from app.services.query_service import QueryService

log = get_logger(__name__)


_DEFAULT_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


class RagasEvaluator:
    """RAGAS-based evaluation harness.

    For each sample we run the agent end-to-end to capture (answer, contexts),
    then hand the resulting dataset to RAGAS. We import RAGAS lazily so the
    rest of the app doesn't pay its import cost on cold start.
    """

    def __init__(self, query_service: QueryService) -> None:
        self._query_service = query_service

    async def evaluate(
        self,
        samples: list[EvaluationSample],
        metric_names: list[str] | None = None,
    ) -> EvaluationResponse:
        if not samples:
            raise EvaluationError("At least one evaluation sample is required")

        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        ground_truths: list[str] = []
        per_sample: list[dict[str, Any]] = []

        for sample in samples:
            response = await self._query_service.answer(
                sample.question,
                include_trace=False,
            )
            ctx_texts = [c.content for c in response.contexts]
            questions.append(sample.question)
            answers.append(response.answer)
            contexts.append(ctx_texts)
            ground_truths.append(sample.ground_truth)
            per_sample.append(
                {
                    "question": sample.question,
                    "answer": response.answer,
                    "ground_truth": sample.ground_truth,
                    "num_contexts": len(ctx_texts),
                }
            )

        metrics_summary = self._run_ragas(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths,
            metric_names=metric_names,
        )

        return EvaluationResponse(
            n_samples=len(samples),
            metrics=metrics_summary,
            per_sample=per_sample,
        )

    # -------------------------------------------------------------------
    # RAGAS plumbing
    # -------------------------------------------------------------------
    def _run_ragas(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
        metric_names: list[str] | None,
    ) -> dict[str, float]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:
            raise EvaluationError(
                "ragas / datasets are required for evaluation but are not installed"
            ) from exc

        metric_registry = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

        names = metric_names or list(_DEFAULT_METRIC_NAMES)
        unknown = [m for m in names if m not in metric_registry]
        if unknown:
            raise EvaluationError(f"Unknown RAGAS metrics: {unknown}")

        selected = [metric_registry[name] for name in names]

        ds = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        try:
            result = evaluate(ds, metrics=selected)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

        return {name: float(result[name]) for name in names if name in result}

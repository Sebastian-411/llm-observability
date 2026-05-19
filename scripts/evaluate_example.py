"""Run RAGAS evaluation over a small built-in test set.

Usage:
    python -m scripts.evaluate_example
"""
from __future__ import annotations

import asyncio
import json

from app.api.dependencies import provide_evaluator, provide_query_service
from app.core.logging import configure_logging
from app.models import EvaluationSample


SAMPLES = [
    EvaluationSample(
        question="How much does Y Combinator invest in each startup?",
        ground_truth=(
            "Since 2022 YC invests $500,000 per company: $125,000 for 7% on a "
            "post-money SAFE, plus $375,000 on an uncapped MFN SAFE."
        ),
    ),
    EvaluationSample(
        question="Who founded Y Combinator and when?",
        ground_truth=(
            "Y Combinator was founded in March 2005 by Paul Graham, Jessica "
            "Livingston, Robert Tapan Morris, and Trevor Blackwell."
        ),
    ),
    EvaluationSample(
        question="What is the main idea of Paul Graham's essay 'Do Things That Don't Scale'?",
        ground_truth=(
            "Successful startups typically begin with manual, unscalable work "
            "(recruiting users by hand, concierge onboarding) because the "
            "early-stage bottleneck is learning, not scale."
        ),
    ),
    EvaluationSample(
        question="Is OpenAI a YC company?",
        ground_truth=(
            "No. OpenAI never went through a YC batch. Sam Altman cofounded "
            "OpenAI in 2015 while he was president of YC, but OpenAI was "
            "founded independently as a non-profit research lab."
        ),
    ),
    EvaluationSample(
        question="Who currently runs Y Combinator?",
        ground_truth="Garry Tan has been President and CEO of YC since 2023.",
    ),
]


async def _main() -> int:
    configure_logging("INFO")
    evaluator = provide_evaluator(provide_query_service())
    result = await evaluator.evaluate(samples=SAMPLES)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

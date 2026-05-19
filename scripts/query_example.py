"""Run a single question against the agent from the command line.

Usage:
    python -m scripts.query_example "How does authentication work?"
"""
from __future__ import annotations

import asyncio
import json
import sys

from app.api.dependencies import provide_query_service
from app.core.logging import configure_logging, get_logger


async def _main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.query_example '<question>'", file=sys.stderr)
        return 2

    question = " ".join(sys.argv[1:])
    configure_logging("INFO")
    log = get_logger("scripts.query")

    service = provide_query_service()
    response = await service.answer(question)

    log.info("query.done", elapsed_ms=response.elapsed_ms, run_id=response.run_id)
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

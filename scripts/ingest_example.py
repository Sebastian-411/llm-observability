"""Ingest a folder of documents into the vector store.

Usage:
    python -m scripts.ingest_example data/sample
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.api.dependencies import provide_ingestion_service, provide_settings
from app.core.logging import configure_logging, get_logger

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".json"}


def main() -> int:
    settings = provide_settings()
    configure_logging(settings.log_level)
    log = get_logger("scripts.ingest")

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
    if not target.exists():
        log.error("ingest.target_missing", path=str(target))
        return 1

    service = provide_ingestion_service(settings)
    files = (
        [target] if target.is_file()
        else [p for p in target.rglob("*") if p.suffix.lower() in SUPPORTED]
    )
    if not files:
        log.warning("ingest.no_files", path=str(target))
        return 0

    total_chunks = 0
    for f in files:
        result = service.ingest_file(f)
        total_chunks += result.chunks_ingested
        log.info(
            "ingest.file_done",
            file=str(f),
            chunks=result.chunks_ingested,
            elapsed_ms=result.elapsed_ms,
        )

    log.info("ingest.all_done", files=len(files), total_chunks=total_chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

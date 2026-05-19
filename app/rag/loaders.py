from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from app.core.exceptions import IngestionError, UnsupportedFormatError
from app.core.logging import get_logger

log = get_logger(__name__)


class SupportedFormat(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    MARKDOWN = "md"
    JSON = "json"


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------
def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader  # local import keeps cold start fast

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "format": "pdf", "page": idx + 1},
            )
        )
    return docs


def _load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        Document(
            page_content=text,
            metadata={"source": str(path), "format": "txt"},
        )
    ]


def _load_markdown(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        Document(
            page_content=text,
            metadata={"source": str(path), "format": "md"},
        )
    ]


def _load_json(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IngestionError(f"Invalid JSON in {path}: {exc}") from exc

    docs: list[Document] = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            docs.append(
                Document(
                    page_content=_stringify(item),
                    metadata={"source": str(path), "format": "json", "index": i},
                )
            )
    elif isinstance(data, dict):
        docs.append(
            Document(
                page_content=_stringify(data),
                metadata={"source": str(path), "format": "json"},
            )
        )
    else:
        docs.append(
            Document(
                page_content=str(data),
                metadata={"source": str(path), "format": "json"},
            )
        )
    return docs


def _stringify(obj: object) -> str:
    if isinstance(obj, (dict, list)):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return str(obj)


_LOADERS: dict[SupportedFormat, Callable[[Path], list[Document]]] = {
    SupportedFormat.PDF: _load_pdf,
    SupportedFormat.TXT: _load_txt,
    SupportedFormat.MARKDOWN: _load_markdown,
    SupportedFormat.JSON: _load_json,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class DocumentLoader:
    """Selects the right loader based on file extension."""

    def load(self, path: str | Path) -> list[Document]:
        return load_document(path)


def load_document(path: str | Path) -> list[Document]:
    file_path = Path(path)
    if not file_path.exists():
        raise IngestionError(f"File not found: {file_path}")

    ext = file_path.suffix.lower().lstrip(".")
    if ext == "markdown":
        ext = "md"

    try:
        fmt = SupportedFormat(ext)
    except ValueError as exc:
        raise UnsupportedFormatError(
            f"Unsupported file extension: .{ext} (supported: pdf, txt, md, json)"
        ) from exc

    docs = _LOADERS[fmt](file_path)
    log.info(
        "ingestion.loaded",
        path=str(file_path),
        format=fmt.value,
        documents=len(docs),
    )
    return docs

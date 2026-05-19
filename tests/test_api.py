from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    provide_evaluator,
    provide_ingestion_service,
    provide_query_service,
    provide_vectorstore,
    reset_singletons,
)
from app.main import create_app
from app.models import (
    EvaluationResponse,
    IngestionResponse,
    QueryResponse,
    ReActStep,
    RetrievedChunk,
)


@pytest.fixture
def client() -> TestClient:
    reset_singletons()
    app = create_app()

    # ---- Fake services with no LLM / network --------------------------
    class _FakeIngestion:
        def ingest_file(self, path):  # noqa: ARG002
            return IngestionResponse(
                source=str(path),
                chunks_ingested=3,
                collection="test",
                elapsed_ms=5.0,
            )

    class _FakeQuery:
        async def answer(self, question, *, top_k=None, include_trace=True, session_id=None):  # noqa: ARG002
            return QueryResponse(
                question=question,
                answer="YC invests $500k per company.",
                contexts=[
                    RetrievedChunk(
                        content="YC standard deal: $500k per company.",
                        source="01_about_yc.md",
                        score=0.9,
                    )
                ],
                trace=[ReActStep(step=1, thought="search", action="retrieve_context")],
                elapsed_ms=12.3,
                run_id="run-1",
            )

        async def stream_answer(self, question, *, session_id=None):  # noqa: ARG002
            for tok in ["YC ", "invests ", "$500k ", "per ", "company."]:
                yield tok

    class _FakeEvaluator:
        async def evaluate(self, samples, metric_names=None):  # noqa: ARG002
            return EvaluationResponse(
                n_samples=len(samples),
                metrics={
                    "faithfulness": 0.92,
                    "answer_relevancy": 0.88,
                    "context_precision": 0.81,
                    "context_recall": 0.79,
                },
                per_sample=[],
            )

    class _FakeStore:
        def stats(self):
            return {"backend": "chroma", "collection": "test", "document_count": 3}

    app.dependency_overrides[provide_ingestion_service] = lambda: _FakeIngestion()
    app.dependency_overrides[provide_query_service] = lambda: _FakeQuery()
    app.dependency_overrides[provide_evaluator] = lambda: _FakeEvaluator()
    app.dependency_overrides[provide_vectorstore] = lambda: _FakeStore()

    return TestClient(app)


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body


class TestQuery:
    def test_query_returns_answer(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": "How much does YC invest?"})
        assert r.status_code == 200
        body = r.json()
        assert "$500k" in body["answer"]
        assert body["contexts"][0]["source"] == "01_about_yc.md"
        assert body["trace"]

    def test_query_validation(self, client: TestClient) -> None:
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 422

    def test_query_stream(self, client: TestClient) -> None:
        with client.stream("POST", "/query/stream", json={"question": "YC?"}) as r:
            assert r.status_code == 200
            body = b"".join(r.iter_bytes())
        assert b"$500k" in body


class TestIngest:
    def test_ingest_upload(self, client: TestClient, tmp_path: Path) -> None:
        p = tmp_path / "doc.txt"
        p.write_text("hello world", encoding="utf-8")
        with p.open("rb") as fh:
            r = client.post("/ingest", files={"file": ("doc.txt", fh, "text/plain")})
        assert r.status_code == 200
        assert r.json()["chunks_ingested"] == 3


class TestEvaluate:
    def test_evaluate(self, client: TestClient) -> None:
        r = client.post(
            "/evaluate",
            json={"samples": [{"question": "auth?", "ground_truth": "OAuth2."}]},
        )
        assert r.status_code == 200
        metrics = r.json()["metrics"]
        for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            assert key in metrics

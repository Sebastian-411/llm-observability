from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api import router
from app.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("app.lifespan")

    # Wire LangSmith env vars for the LangChain runtime to pick up.
    if settings.langsmith_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        log.info("langsmith.enabled", project=settings.langchain_project)
    else:
        log.info("langsmith.disabled")

    log.info(
        "app.starting",
        version=__version__,
        environment=settings.environment,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        vectorstore=settings.vectorstore_provider,
    )

    # Load external MCP tools (e.g. Tavily web search) once and attach them to
    # the long-lived agent. Opt-in via MCP_ENABLED; failures degrade gracefully.
    if settings.mcp_active:
        from app.agents.mcp_client import load_mcp_tools
        from app.api.dependencies import provide_agent

        mcp_tools = await load_mcp_tools(settings)
        if mcp_tools:
            provide_agent().set_mcp_tools(mcp_tools)
            log.info("agent.mcp_attached", count=len(mcp_tools))
    else:
        log.info("mcp.inactive")

    yield
    log.info("app.stopping")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Reto3 — ReAct + RAG Agent",
        version=__version__,
        description=(
            "ReAct agent orchestrated with LangGraph over a Chroma-backed RAG "
            "pipeline, with LangSmith tracing and RAGAS evaluation."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "detail": exc.detail},
        )

    app.include_router(router)
    return app


app = create_app()

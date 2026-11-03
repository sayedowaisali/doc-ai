"""FastAPI entrypoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app.api import chat, documents, health, metrics
from app.config import get_settings
from app.db import init_db
from app.services.llm import LLMError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.requests")


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_settings().storage_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="Docu AI",
    description="Upload documents, get AI insights, and chat with them using RAG.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %d (%.0f ms)",
        request.method,
        request.url.path,
        response.status_code,
        (perf_counter() - start) * 1000,
    )
    return response


@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


for router in (documents.router, chat.router, metrics.router, health.router):
    app.include_router(router)


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    """Serve the single-page UI (plain HTML/JS, no build step)."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")

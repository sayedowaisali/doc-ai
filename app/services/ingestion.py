"""Document processing pipeline executed by the Celery worker.

extract -> chunk -> embed -> insights. Each stage is timed and stored on a ProcessingJob row,
and failures are recorded on the document so clients can poll `GET /documents/{id}`.
"""

import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from sqlalchemy import delete

from app.config import get_settings
from app.db import SessionLocal
from app.models import (
    STATUS_FAILED,
    STATUS_PROCESSING,
    STATUS_READY,
    Document,
    DocumentChunk,
    ProcessingJob,
)
from app.services import llm
from app.services.chunking import select_text_for_insights, split_pages
from app.services.extraction import ExtractionError, extract_pages
from app.services.insights import generate_insights

logger = logging.getLogger(__name__)


@contextmanager
def _stage(timings: dict[str, float], name: str) -> Iterator[None]:
    start = perf_counter()
    try:
        yield
    finally:
        timings[name] = round(perf_counter() - start, 3)


def _fail(db, document: Document, job: ProcessingJob, message: str, started: float) -> None:
    db.rollback()
    document.status = STATUS_FAILED
    document.error = message[:500]
    job.status = "failed"
    job.error = message[:500]
    job.duration_seconds = round(perf_counter() - started, 3)
    job.finished_at = datetime.now(timezone.utc)
    db.commit()


def process_document(document_id: uuid.UUID) -> None:
    settings = get_settings()
    with SessionLocal() as db:
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("Document %s no longer exists; skipping", document_id)
            return

        job = ProcessingJob(document_id=document.id)
        document.status = STATUS_PROCESSING
        document.error = None
        db.add(job)
        db.commit()

        started = perf_counter()
        timings: dict[str, float] = {}
        usage = llm.Usage()
        try:
            with _stage(timings, "extract"):
                pages = extract_pages(Path(document.storage_path))

            with _stage(timings, "chunk"):
                chunks = split_pages(pages)

            with _stage(timings, "embed"):
                vectors, embed_usage = llm.embed_texts([c.content for c in chunks])
                usage.add(embed_usage)

            with _stage(timings, "insights"):
                text = select_text_for_insights(chunks, settings.insight_token_budget)
                insights, insights_usage = generate_insights(text)
                usage.add(insights_usage)

            # Replace chunks so reprocessing never leaves duplicates behind.
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
            db.add_all(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=c.index,
                    page_number=c.page_number,
                    content=c.content,
                    token_count=c.token_count,
                    embedding=vector,
                )
                for c, vector in zip(chunks, vectors, strict=True)
            )
            document.page_count = len(pages)
            document.summary = insights.summary
            document.insights = insights.model_dump(exclude={"summary"})
            document.status = STATUS_READY

            job.status = "succeeded"
            job.stage_timings = timings
            job.duration_seconds = round(perf_counter() - started, 3)
            job.prompt_tokens = usage.prompt_tokens
            job.completion_tokens = usage.completion_tokens
            job.cost_usd = usage.cost_usd
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
            logger.info("Processed %s in %.1fs (%d chunks)", document.id, job.duration_seconds, len(chunks))
        except (ExtractionError, llm.LLMError) as exc:
            logger.warning("Processing %s failed: %s", document.id, exc)
            _fail(db, document, job, str(exc), started)
        except Exception:
            logger.exception("Unexpected error processing %s", document.id)
            _fail(db, document, job, "Unexpected error while processing the document.", started)

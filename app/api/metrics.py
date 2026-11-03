"""Monitoring endpoints backed by aggregate queries over documents, jobs and chat messages."""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import Float, case, cast, func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ChatMessage, ChatSession, Document, DocumentChunk, ProcessingJob

router = APIRouter(prefix="/metrics", tags=["metrics"])

STAGES = ("extract", "chunk", "embed", "insights")


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


@router.get("/documents")
def document_statistics(db: Session = Depends(get_db)) -> dict:
    by_status = dict(db.execute(select(Document.status, func.count()).group_by(Document.status)).all())
    by_category = dict(
        db.execute(
            select(Document.insights["category"].astext, func.count())
            .where(Document.insights.is_not(None))
            .group_by(Document.insights["category"].astext)
        ).all()
    )
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    return {
        "total_documents": sum(by_status.values()),
        "by_status": by_status,
        "by_category": by_category,
        "total_size_bytes": db.scalar(select(func.coalesce(func.sum(Document.size_bytes), 0))),
        "total_pages": db.scalar(select(func.coalesce(func.sum(Document.page_count), 0))),
        "total_chunks": db.scalar(select(func.count()).select_from(DocumentChunk)),
        "uploaded_last_24h": db.scalar(
            select(func.count()).select_from(Document).where(Document.created_at >= since)
        ),
    }


@router.get("/processing")
def processing_metrics(db: Session = Depends(get_db)) -> dict:
    succeeded = ProcessingJob.status == "succeeded"
    stage_averages = [
        func.avg(cast(ProcessingJob.stage_timings[stage].astext, Float)).filter(succeeded)
        for stage in STAGES
    ]
    row = db.execute(
        select(
            func.count(),
            func.count().filter(succeeded),
            func.count().filter(ProcessingJob.status == "failed"),
            func.count().filter(ProcessingJob.status == "running"),
            func.avg(ProcessingJob.duration_seconds).filter(succeeded),
            func.percentile_cont(0.95).within_group(ProcessingJob.duration_seconds),
            func.coalesce(func.sum(ProcessingJob.prompt_tokens + ProcessingJob.completion_tokens), 0),
            func.coalesce(func.sum(ProcessingJob.cost_usd), 0.0),
            *stage_averages,
        )
    ).one()
    total, ok, failed, running, avg_duration, p95, tokens, cost, *stage_values = row
    finished = ok + failed
    recent_failures = db.execute(
        select(ProcessingJob.document_id, ProcessingJob.error, ProcessingJob.finished_at)
        .where(ProcessingJob.status == "failed")
        .order_by(ProcessingJob.id.desc())
        .limit(5)
    ).all()
    return {
        "total_jobs": total,
        "succeeded": ok,
        "failed": failed,
        "running": running,
        "success_rate": _round(ok / finished) if finished else None,
        "avg_duration_seconds": _round(avg_duration),
        "p95_duration_seconds": _round(p95),
        "avg_stage_seconds": {s: _round(v) for s, v in zip(STAGES, stage_values)},
        "total_tokens": tokens,
        "total_cost_usd": _round(cost, 6),
        "recent_failures": [
            {"document_id": str(d), "error": e, "finished_at": f} for d, e, f in recent_failures
        ],
    }


@router.get("/chat")
def chat_metrics(db: Session = Depends(get_db)) -> dict:
    assistant = ChatMessage.role == "assistant"
    row = db.execute(
        select(
            func.count().filter(assistant),
            func.avg(ChatMessage.latency_ms).filter(assistant),
            func.percentile_cont(0.95).within_group(ChatMessage.latency_ms).filter(assistant),
            func.coalesce(func.sum(ChatMessage.prompt_tokens + ChatMessage.completion_tokens), 0),
            func.coalesce(func.sum(ChatMessage.cost_usd), 0.0),
            func.count().filter(assistant & (func.jsonb_array_length(ChatMessage.citations) == 0)),
        )
    ).one()
    answers, avg_latency, p95_latency, tokens, cost, uncited = row
    return {
        "total_sessions": db.scalar(select(func.count()).select_from(ChatSession)),
        "total_answers": answers,
        "avg_latency_ms": _round(avg_latency, 0),
        "p95_latency_ms": _round(p95_latency, 0),
        "total_tokens": tokens,
        "total_cost_usd": _round(cost, 6),
        # Answers without citations are mostly "not found" replies: a rough retrieval-quality signal.
        "answers_without_citations": uncited,
    }

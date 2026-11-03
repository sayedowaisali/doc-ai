"""Vector retrieval over pgvector, scoped to a set of documents."""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentChunk


@dataclass
class RetrievedChunk:
    document_id: uuid.UUID
    filename: str
    page_number: int
    content: str
    score: float  # cosine similarity, 1.0 = identical direction


def retrieve_chunks(
    db: Session, query_vector: list[float], document_ids: list[uuid.UUID], top_k: int
) -> list[RetrievedChunk]:
    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    rows = db.execute(
        select(DocumentChunk, Document.filename, distance)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.document_id.in_(document_ids))
        .order_by(distance)
        .limit(top_k)
    ).all()
    return [
        RetrievedChunk(c.document_id, filename, c.page_number, c.content, round(1 - dist, 4))
        for c, filename, dist in rows
    ]

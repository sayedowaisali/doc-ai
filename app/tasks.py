"""Celery tasks. Kept thin: all logic lives in app.services.ingestion."""

import uuid

from app.services.ingestion import process_document
from app.worker import celery_app


@celery_app.task(name="process_document")
def process_document_task(document_id: str) -> None:
    process_document(uuid.UUID(document_id))

"""Document upload, retrieval and insights endpoints."""

import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_READY,
    Document,
)
from app.schemas import DocumentList, DocumentOut, InsightsOut
from app.services.extraction import SUPPORTED_EXTENSIONS
from app.tasks import process_document_task

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


def _get_or_404(db: Session, document_id: uuid.UUID) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found.")
    return document


def _enqueue(db: Session, document: Document) -> None:
    """Queue processing; if the broker is down, fail loudly instead of leaving a stuck 'pending' doc."""
    try:
        process_document_task.delay(str(document.id))
    except Exception as exc:
        logger.exception("Could not enqueue document %s", document.id)
        document.status = STATUS_FAILED
        document.error = "Processing queue is unavailable. Try reprocessing later."
        db.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Processing queue is unavailable."
        ) from exc


def _read_validated_upload(file: UploadFile) -> tuple[str, bytes]:
    settings = get_settings()
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type '{extension}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)  # read one extra byte to detect oversize without loading it all
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB."
        )
    if not data.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    return filename, data


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def upload_document(
    response: Response, file: UploadFile = File(...), db: Session = Depends(get_db)
) -> Document:
    """Store the file and queue processing. Poll `GET /documents/{id}` for status.

    Re-uploading an identical file returns the existing document (HTTP 200) without reprocessing.
    """
    filename, data = _read_validated_upload(file)
    content_hash = hashlib.sha256(data).hexdigest()

    existing = db.scalar(select(Document).where(Document.content_hash == content_hash))
    if existing:
        response.status_code = status.HTTP_200_OK
        return existing

    storage_dir = get_settings().storage_dir
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / f"{uuid.uuid4()}{Path(filename).suffix.lower()}"
    path.write_bytes(data)

    document = Document(
        filename=filename,
        content_type=file.content_type or "application/octet-stream",
        size_bytes=len(data),
        content_hash=content_hash,
        storage_path=str(path),
        status=STATUS_PENDING,
    )
    db.add(document)
    try:
        db.commit()
    except IntegrityError:  # concurrent upload of the same file won the race
        db.rollback()
        path.unlink(missing_ok=True)
        response.status_code = status.HTTP_200_OK
        return db.scalar(select(Document).where(Document.content_hash == content_hash))

    _enqueue(db, document)
    return document


@router.get("", response_model=DocumentList)
def list_documents(
    status_filter: str | None = Query(None, alias="status"),
    category: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> DocumentList:
    query = select(Document)
    if status_filter:
        query = query.where(Document.status == status_filter)
    if category:
        query = query.where(Document.insights["category"].astext == category)
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    items = db.scalars(query.order_by(Document.created_at.desc()).limit(limit).offset(offset)).all()
    return DocumentList(total=total, items=items)


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> Document:
    return _get_or_404(db, document_id)


@router.get("/{document_id}/insights", response_model=InsightsOut)
def get_insights(document_id: uuid.UUID, db: Session = Depends(get_db)) -> InsightsOut:
    document = _get_or_404(db, document_id)
    if document.status != STATUS_READY:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Insights are not available: document status is '{document.status}'.",
        )
    return InsightsOut(
        document_id=document.id,
        filename=document.filename,
        summary=document.summary or "",
        **(document.insights or {}),
    )


@router.post("/{document_id}/reprocess", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def reprocess_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> Document:
    document = _get_or_404(db, document_id)
    if document.status in (STATUS_PENDING, STATUS_PROCESSING):
        raise HTTPException(status.HTTP_409_CONFLICT, "Document is already being processed.")
    document.status = STATUS_PENDING
    document.error = None
    db.commit()
    _enqueue(db, document)
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)) -> None:
    document = _get_or_404(db, document_id)
    if document.status == STATUS_PROCESSING:
        raise HTTPException(status.HTTP_409_CONFLICT, "Cannot delete a document while it is processing.")
    Path(document.storage_path).unlink(missing_ok=True)
    db.delete(document)  # chunks, jobs and chat links are removed by ON DELETE CASCADE
    db.commit()

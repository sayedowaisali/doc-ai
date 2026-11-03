"""Chat session endpoints (RAG over one or more documents)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import STATUS_READY, ChatMessage, ChatSession, Document
from app.schemas import MessageOut, QuestionIn, SessionCreate, SessionOut
from app.services.chat import answer_question

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_session_or_404(db: Session, session_id: uuid.UUID) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat session not found.")
    return session


@router.post("/sessions", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def start_session(body: SessionCreate, db: Session = Depends(get_db)) -> ChatSession:
    """Start a chat over one or more *ready* documents (multi-document chat)."""
    ids = list(dict.fromkeys(body.document_ids))
    documents = db.scalars(select(Document).where(Document.id.in_(ids))).all()
    missing = set(ids) - {d.id for d in documents}
    if missing:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Documents not found: {sorted(str(i) for i in missing)}"
        )
    not_ready = [d.filename for d in documents if d.status != STATUS_READY]
    if not_ready:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Documents not ready for chat yet: {', '.join(not_ready)}"
        )
    title = body.title or ", ".join(d.filename for d in documents)[:255]
    session = ChatSession(title=title, documents=documents)
    db.add(session)
    db.commit()
    return session


@router.post("/sessions/{session_id}/messages", response_model=MessageOut)
def ask_question(
    session_id: uuid.UUID, body: QuestionIn, db: Session = Depends(get_db)
) -> ChatMessage:
    """Ask a question; earlier messages in the session provide multi-turn context."""
    session = _get_session_or_404(db, session_id)
    if not session.documents:
        raise HTTPException(status.HTTP_409_CONFLICT, "All documents in this session were deleted.")
    return answer_question(db, session, body.question)


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def get_history(
    session_id: uuid.UUID,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[ChatMessage]:
    _get_session_or_404(db, session_id)
    return list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id)
            .limit(limit)
            .offset(offset)
        )
    )

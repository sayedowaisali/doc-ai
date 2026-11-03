"""RAG chat: condense follow-up -> embed -> retrieve -> grounded answer with citations."""

import re
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import ChatMessage, ChatSession, Document
from app.services import llm
from app.services.prompts import ANSWER_SYSTEM_PROMPT, CONDENSE_SYSTEM_PROMPT, NOT_FOUND_ANSWER
from app.services.retrieval import RetrievedChunk, retrieve_chunks

SNIPPET_CHARS = 240
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def recent_history(db: Session, session_id, limit: int) -> list[ChatMessage]:
    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    ).all()
    return list(reversed(rows))


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Number passages from 1 so the model's [n] citations map back to chunks."""
    return "\n\n".join(
        f"[{i}] (document: {c.filename}, page {c.page_number})\n{c.content}"
        for i, c in enumerate(chunks, 1)
    )


def build_overview(documents: list[Document]) -> str:
    """Summaries generated at ingestion. Top-k passages cannot answer broad questions about a
    whole document, so these give the model that big picture at a small, fixed token cost."""
    blocks = []
    for doc in documents:
        key_points = "\n".join(f"- {p}" for p in (doc.insights or {}).get("key_points", []))
        blocks.append(f"Document: {doc.filename}\nSummary: {doc.summary or 'n/a'}\nKey points:\n{key_points}")
    return "\n\n".join(blocks)


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Return only the passages the model actually cited, ignoring invalid numbers."""
    cited = sorted({int(n) for n in _CITATION_PATTERN.findall(answer) if 1 <= int(n) <= len(chunks)})
    return [
        {
            "index": i,
            "document_id": str(chunks[i - 1].document_id),
            "filename": chunks[i - 1].filename,
            "page_number": chunks[i - 1].page_number,
            "snippet": chunks[i - 1].content[:SNIPPET_CHARS],
            "score": chunks[i - 1].score,
        }
        for i in cited
    ]


def _to_openai_messages(history: list[ChatMessage]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in history]


def _standalone_question(question: str, history: list[ChatMessage], usage: llm.Usage) -> str:
    if not history:
        return question
    transcript = "\n".join(f"{m.role}: {m.content}" for m in history)
    messages = [
        {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"},
    ]
    text, condense_usage = llm.complete(messages, temperature=0)
    usage.add(condense_usage)
    return text.strip() or question


def answer_question(db: Session, session: ChatSession, question: str) -> ChatMessage:
    settings = get_settings()
    started = perf_counter()
    usage = llm.Usage()
    history = recent_history(db, session.id, settings.history_messages)

    standalone = _standalone_question(question, history, usage)
    vectors, embed_usage = llm.embed_texts([standalone])
    usage.add(embed_usage)
    chunks = retrieve_chunks(
        db, vectors[0], [d.id for d in session.documents], settings.retrieval_top_k
    )

    if chunks:
        messages = [
            {
                "role": "system",
                "content": (
                    f"{ANSWER_SYSTEM_PROMPT}\n\nDocument overviews:\n{build_overview(session.documents)}"
                    f"\n\nContext passages:\n{build_context(chunks)}"
                ),
            },
            *_to_openai_messages(history),
            {"role": "user", "content": question},
        ]
        answer, answer_usage = llm.complete(messages, temperature=0.2)
        usage.add(answer_usage)
    else:
        answer = NOT_FOUND_ANSWER

    # Persist both turns only after the model succeeded, so a failed call leaves no orphan question.
    db.add(ChatMessage(session_id=session.id, role="user", content=question))
    assistant = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=answer,
        citations=extract_citations(answer, chunks),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_usd=usage.cost_usd,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    db.add(assistant)
    db.commit()
    return assistant

"""Pydantic request/response models."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---- Documents -------------------------------------------------------------------------------
class DocumentOut(ORMModel):
    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: str
    error: str | None
    page_count: int | None
    category: str | None
    tags: list[str]
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    total: int
    items: list[DocumentOut]


class DocumentInsights(BaseModel):
    """Structured output requested from the LLM at ingestion time."""

    summary: str
    key_points: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    category: str = "other"
    tags: list[str] = Field(default_factory=list)
    sentiment: Literal["positive", "neutral", "negative", "mixed"] = "neutral"


class InsightsOut(DocumentInsights):
    document_id: uuid.UUID
    filename: str


# ---- Chat ------------------------------------------------------------------------------------
class SessionCreate(BaseModel):
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=10)
    title: str | None = Field(default=None, max_length=255)


class SessionDocument(ORMModel):
    id: uuid.UUID
    filename: str


class SessionOut(ORMModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    documents: list[SessionDocument]


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    index: int
    document_id: uuid.UUID
    filename: str
    page_number: int
    snippet: str
    score: float


class MessageOut(ORMModel):
    id: int
    role: str
    content: str
    citations: list[Citation] | None
    latency_ms: int | None
    created_at: datetime

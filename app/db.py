"""Database engine, session factory and schema bootstrap.

A single synchronous SQLAlchemy engine is shared by the API (FastAPI runs sync
endpoints in a threadpool) and the Celery worker (which is sync by nature).
"""

from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    with SessionLocal() as session:
        yield session


def init_db() -> None:
    """Enable pgvector and create tables. Alembic migrations are out of scope for this challenge."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)

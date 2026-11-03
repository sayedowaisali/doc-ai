"""Application settings loaded from environment variables (or a local .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://docuai:docuai@localhost:5432/docuai"
    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # Must match the pgvector column size; changing it requires re-embedding all chunks.
    embedding_dimensions: int = 1536

    storage_dir: Path = Path("storage")
    max_upload_mb: int = 20

    # ~800 tokens keeps a chunk focused enough to retrieve precisely but large enough to be
    # self-explanatory; the overlap avoids cutting an answer in half at a boundary.
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100
    retrieval_top_k: int = 6
    # Recent messages fed to the model for multi-turn context (older turns are dropped to cap cost).
    history_messages: int = 6
    # Max tokens of document text sent to the model when generating summary/insights.
    insight_token_budget: int = 24000


@lru_cache
def get_settings() -> Settings:
    return Settings()

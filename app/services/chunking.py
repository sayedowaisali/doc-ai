"""Token-aware chunking. LangChain is used only for the splitter; everything else is hand-written."""

from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.extraction import Page

ENCODING_NAME = "cl100k_base"


@dataclass
class Chunk:
    index: int
    page_number: int
    content: str
    token_count: int


@lru_cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


@lru_cache
def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    # Recursive splitting prefers paragraph, then sentence, then word boundaries.
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )


def split_pages(pages: list[Page]) -> list[Chunk]:
    """Split each page separately so every chunk maps to exactly one page for citations."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in _splitter().split_text(page.text):
            chunks.append(Chunk(len(chunks), page.number, text, count_tokens(text)))
    return chunks


def select_text_for_insights(chunks: list[Chunk], token_budget: int) -> str:
    """Build the text sent for summarisation, staying within a token budget.

    Short documents are sent whole. Long ones are sampled at evenly spaced chunks so the
    summary covers the beginning, middle and end rather than only the first pages.
    """
    total = sum(c.token_count for c in chunks)
    if total <= token_budget:
        return "\n\n".join(c.content for c in chunks)
    avg = max(total // len(chunks), 1)
    keep = max(token_budget // avg, 1)
    step = len(chunks) / keep
    sampled = [chunks[int(i * step)] for i in range(keep)]
    return "\n\n[...]\n\n".join(c.content for c in sampled)

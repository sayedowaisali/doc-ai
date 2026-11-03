"""Thin wrapper around the OpenAI SDK.

Centralising calls here gives one place for error translation, retries and cost tracking,
and lets tests replace the provider by patching two functions (`embed_texts`, `complete`).
"""

from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI, OpenAIError

from app.config import get_settings

# USD per 1M tokens as (input, output). Update when OpenAI pricing changes; unknown models cost 0.
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}
EMBEDDING_BATCH_SIZE = 100


class LLMError(Exception):
    """The AI provider is misconfigured or failed; the message is safe to show to users."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


@lru_cache
def _client() -> OpenAI:
    api_key = get_settings().openai_api_key
    if not api_key or api_key.startswith("sk-your-key"):
        raise LLMError("OPENAI_API_KEY is not configured.")
    # The SDK retries 429/5xx/connection errors with exponential backoff.
    return OpenAI(api_key=api_key, max_retries=3, timeout=60)


def embed_texts(texts: list[str]) -> tuple[list[list[float]], Usage]:
    """Embed texts in batches; returns vectors in input order."""
    settings = get_settings()
    vectors: list[list[float]] = []
    usage = Usage()
    try:
        client = _client()
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=texts[start : start + EMBEDDING_BATCH_SIZE],
                dimensions=settings.embedding_dimensions,
            )
            vectors.extend(item.embedding for item in response.data)
            tokens = response.usage.prompt_tokens
            usage.add(Usage(tokens, 0, _cost(settings.embedding_model, tokens, 0)))
    except OpenAIError as exc:
        raise LLMError(f"Embedding request failed: {exc.__class__.__name__}") from exc
    return vectors, usage


def complete(
    messages: list[dict], *, json_mode: bool = False, temperature: float = 0.2
) -> tuple[str, Usage]:
    """Run one chat completion; returns the text and token/cost usage."""
    model = get_settings().chat_model
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    try:
        response = _client().chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
    except OpenAIError as exc:
        raise LLMError(f"Chat completion failed: {exc.__class__.__name__}") from exc
    used = response.usage
    usage = Usage(
        used.prompt_tokens,
        used.completion_tokens,
        _cost(model, used.prompt_tokens, used.completion_tokens),
    )
    return response.choices[0].message.content or "", usage

"""LLM-generated summary and structured insights for a document."""

from pydantic import ValidationError

from app.schemas import DocumentInsights
from app.services import llm
from app.services.prompts import INSIGHTS_SYSTEM_PROMPT


def generate_insights(document_text: str) -> tuple[DocumentInsights, llm.Usage]:
    messages = [
        {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Document text:\n\n{document_text}"},
    ]
    raw, usage = llm.complete(messages, json_mode=True, temperature=0.2)
    try:
        return DocumentInsights.model_validate_json(raw), usage
    except ValidationError as exc:
        raise llm.LLMError("The AI returned malformed insights; try reprocessing.") from exc

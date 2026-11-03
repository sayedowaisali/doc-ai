"""Text extraction from uploaded files, returned page by page so citations can point to pages."""

import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import PyPdfError

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


class ExtractionError(Exception):
    """The file could not be read or contains no extractable text (message is user-safe)."""


@dataclass
class Page:
    number: int
    text: str


def _clean(text: str) -> str:
    # Postgres TEXT columns reject NUL bytes, which some PDFs contain.
    return text.replace("\x00", "").strip()


# Some PDF producers position every word separately, so pypdf returns "word\n \nword\n \n...".
# Blank-but-not-empty lines never occur in normal paragraph breaks ("\n\n"), so collapsing
# only those restores sentences without merging real paragraphs.
_WORD_PER_LINE_BREAKS = re.compile(r"\n(?:[ \t]+\n)+")


def clean_pdf_text(text: str) -> str:
    return _clean(_WORD_PER_LINE_BREAKS.sub(" ", text))


def _extract_pdf(path: Path) -> list[Page]:
    try:
        reader = PdfReader(path)
        return [Page(i, clean_pdf_text(p.extract_text() or "")) for i, p in enumerate(reader.pages, 1)]
    except (PyPdfError, ValueError, OSError) as exc:
        raise ExtractionError("The PDF is corrupted, encrypted, or unreadable.") from exc


def _extract_docx(path: Path) -> list[Page]:
    try:
        doc = DocxDocument(str(path))
    except Exception as exc:  # python-docx raises assorted zip/xml errors for bad files
        raise ExtractionError("The DOCX file is corrupted or unreadable.") from exc
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        parts.extend(" | ".join(cell.text for cell in row.cells) for row in table.rows)
    # DOCX has no fixed pagination, so the whole file is treated as one page.
    return [Page(1, _clean("\n".join(parts)))]


def _extract_text(path: Path) -> list[Page]:
    return [Page(1, _clean(path.read_text(encoding="utf-8", errors="replace")))]


_EXTRACTORS = {".pdf": _extract_pdf, ".docx": _extract_docx, ".txt": _extract_text, ".md": _extract_text}


def extract_pages(path: Path) -> list[Page]:
    """Return non-empty pages; raise ExtractionError if nothing can be extracted."""
    extractor = _EXTRACTORS.get(path.suffix.lower())
    if extractor is None:
        raise ExtractionError(f"Unsupported file type: {path.suffix}")
    pages = [page for page in extractor(path) if page.text]
    if not pages:
        raise ExtractionError(
            "No text could be extracted. The file may be empty or a scanned image (OCR is not supported)."
        )
    return pages

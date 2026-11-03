"""Upload validation happens before any DB or queue access, so no services are needed here."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)  # not used as a context manager: skips the DB-initialising lifespan


def _upload(name: str, content: bytes):
    return client.post("/documents", files={"file": (name, content, "application/octet-stream")})


def test_rejects_unsupported_extension():
    response = _upload("malware.exe", b"data")
    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]


def test_rejects_empty_file():
    assert _upload("empty.txt", b"  \n").status_code == 400


def test_rejects_oversized_file(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)
    assert _upload("big.txt", b"x").status_code == 413


def test_chat_session_requires_at_least_one_document():
    response = client.post("/chat/sessions", json={"document_ids": []})
    assert response.status_code == 422


@pytest.mark.parametrize("question", ["", "x" * 2001])
def test_question_length_is_validated(question):
    response = client.post(
        "/chat/sessions/00000000-0000-0000-0000-000000000000/messages", json={"question": question}
    )
    assert response.status_code == 422

"""End-to-end demo against a running server: upload -> wait -> insights -> multi-turn chat -> metrics.

Usage: python scripts/demo.py [path ...]   (defaults to the files in samples/)
"""

import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
SAMPLES = sorted(Path(__file__).resolve().parent.parent.joinpath("samples").glob("*.md"))
QUESTIONS = [
    "How much is the home office stipend?",
    "And the internet allowance?",  # follow-up: relies on multi-turn context
    "Compare the vacation days with the remote work eligibility rules.",
    "What is the CEO's name?",  # not in the documents: should be refused
]


def show(title: str, payload) -> None:
    print(f"\n=== {title} ===")
    print(payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str))


def wait_until_processed(client: httpx.Client, document_id: str, timeout: int = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        document = client.get(f"/documents/{document_id}").json()
        if document["status"] in ("ready", "failed"):
            return document
        time.sleep(2)
    raise TimeoutError(f"Document {document_id} was not processed within {timeout}s")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or SAMPLES
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        document_ids = []
        for path in paths:
            response = client.post("/documents", files={"file": (path.name, path.read_bytes())})
            response.raise_for_status()
            document = wait_until_processed(client, response.json()["id"])
            show(f"Processed {path.name}: {document['status']}", document.get("error") or "")
            if document["status"] != "ready":
                sys.exit(1)
            document_ids.append(document["id"])
            show(f"Insights for {path.name}", client.get(f"/documents/{document['id']}/insights").json())

        session = client.post("/chat/sessions", json={"document_ids": document_ids}).json()
        show("Chat session", session)
        for question in QUESTIONS:
            answer = client.post(f"/chat/sessions/{session['id']}/messages", json={"question": question})
            answer.raise_for_status()
            body = answer.json()
            show(f"Q: {question}", body["content"])
            for c in body["citations"]:
                print(f"   [{c['index']}] {c['filename']} p.{c['page_number']} (score {c['score']})")

        show("Processing metrics", client.get("/metrics/processing").json())
        show("Chat metrics", client.get("/metrics/chat").json())


if __name__ == "__main__":
    main()

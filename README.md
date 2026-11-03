# Docu AI: chat with your documents

Upload PDF, DOCX, TXT or Markdown files, get an AI summary and insights, and ask questions in a
multi-turn chat that answers **only from your documents, with citations** (RAG).

## Quick start

Requires Docker (Compose v2) and an OpenAI API key. Setting up a fresh Mac? Follow
[SETUP.md](SETUP.md) for the full step-by-step guide and troubleshooting.

```bash
cp .env.example .env          # then set OPENAI_API_KEY
docker compose up --build     # Postgres+pgvector, Redis, API, Celery worker
```

- **Web UI: http://localhost:8000** (upload, status, insights, multi-document chat with citations)
- API docs (Swagger): http://localhost:8000/docs
- Demo (second terminal): `pip install httpx && python scripts/demo.py`

Local development without Docker for the API: `python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`,
run Postgres/Redis via `docker compose up db redis`, then `uvicorn app.main:app --reload` and
`celery -A app.worker worker --loglevel=info`. Tests: `.venv/bin/pytest`.

## Architecture

```
            POST /documents                         Celery worker
 client ───────────────────► API ──queue (Redis)──► extract → chunk → embed → insights
   ▲                          │                          │
   │ poll status / chat       ▼                          ▼
   └───────────────────── PostgreSQL (+ pgvector): documents, chunks+embeddings,
                           processing_jobs, chat_sessions, chat_messages
```

- **FastAPI** API; **Celery + Redis** for async processing; **PostgreSQL + pgvector** for metadata,
  chats, metrics *and* vectors (one datastore to run, chunks join directly to documents).
- **Hybrid RAG:** LangChain is used only for token-aware text splitting. Retrieval is a SQL
  cosine-similarity query and prompts are hand-written (`app/services/prompts.py`), so each
  step is explicit and testable.
- Files are stored on the local filesystem (`storage/`), shared by API and worker via a volume.

### Processing pipeline (`app/services/ingestion.py`)

1. **Upload** validates type/size/emptiness, de-duplicates by SHA-256, stores the file, queues a
   task and returns `202` immediately.
2. **Extract** text page by page (pypdf, python-docx) so citations carry page numbers.
3. **Chunk** ~800 tokens with 100 overlap, per page.
4. **Embed** in batches (`text-embedding-3-small`, 1536 dims) into pgvector with an HNSW index.
5. **Insights:** one JSON-mode GPT call returns summary, key points, topics, category, tags and
   sentiment. Long documents are sampled evenly so the summary covers the whole file.
6. Status goes `pending → processing → ready | failed`. Each attempt is a `processing_jobs` row
   with per-stage timings, tokens and cost.

### Chat flow (`app/services/chat.py`)

1. Rewrite a follow-up ("and the internet allowance?") into a standalone question using recent history.
2. Embed it and fetch the top 6 chunks across the session's documents.
3. Answer with a grounded prompt: use only the numbered passages (plus each document's stored
   summary/key points as an overview for broad questions), cite `[n]`, refuse if not found.
4. Return the answer plus citations (document, page, snippet, similarity) for the passages cited.

## Web UI

A single static page (`app/static/index.html`, plain HTML/CSS/JS, no build step) served by FastAPI
at `/`, so there is no extra service and no CORS. It only calls the public API described below:
drag-and-drop upload, live status polling, per-document insights, selecting several documents to
start one chat, citations as clickable `[n]` chips that open the source passage, and a restored
session after a page refresh. Model output is HTML-escaped before rendering.

## API summary

| Endpoint | Purpose |
|---|---|
| `POST /documents` | Upload (202; 200 with the existing record if the file is a duplicate) |
| `GET /documents`, `GET /documents/{id}` | List (filter by `status`, `category`) / status of one document |
| `GET /documents/{id}/insights` | Summary, key points, topics, category, tags, sentiment |
| `POST /documents/{id}/reprocess`, `DELETE /documents/{id}` | Retry a failed/ready doc / remove doc, chunks and file |
| `POST /chat/sessions` | Start a chat over one **or several** ready documents (`document_ids`) |
| `POST /chat/sessions/{id}/messages` | Ask a question |
| `GET /chat/sessions/{id}/messages` | Chat history with citations |
| `GET /metrics/documents` | Counts by status/category, size, pages, chunks, uploads in 24h |
| `GET /metrics/processing` | Success rate, avg/p95 duration, per-stage averages, tokens, cost, recent failures |
| `GET /metrics/chat` | Answers, avg/p95 latency, tokens, cost, answers without citations |
| `GET /health` | Postgres and Redis checks (503 when degraded) |

Errors use standard codes: 404 unknown id, 409 document not ready / already processing,
413 too large, 415 unsupported type, 422 invalid body, 503 queue or AI provider unavailable.

## Design decisions and assumptions

- **Status in Postgres, not Celery results:** clients poll the document; Celery is only a job runner.
  `acks_late` re-delivers a document if a worker dies mid-job.
- **Failures are first class:** scanned/empty/corrupt files and AI errors end as `failed` with a
  readable `error`; `reprocess` retries. If Redis is down at upload, the document is marked failed
  and the API returns 503 instead of leaving it `pending` forever.
- **No auth, single tenant, no OCR**, tables auto-created at startup (Alembic would replace this
  in production). Models, chunking and retrieval sizes are configurable in `app/config.py`.
- **Multi-document chat** is a session linked to N documents; retrieval is a single top-k across all
  of them. Trade-off: one very relevant document can crowd out another in comparison questions.
- **Prompt-injection:** document text is marked untrusted in the prompts.
- **Cost tracking:** token usage and USD cost (price table in `app/services/llm.py`) are stored
  per processing job and per answer, and exposed via the metrics endpoints.

## Scaling to thousands of documents

- API and workers scale horizontally; processing is queued, so uploads stay fast. Add worker
  replicas or split queues (extract vs embed) as load grows.
- HNSW indexing keeps vector search fast; the similarity query filters by document, and
  chunks are indexed by `document_id`.
- Embeddings are batched (100 per request); the SDK retries 429/5xx with backoff.
- Next steps: S3 storage instead of a shared volume, Redis caching of query embeddings and
  answers, per-user quotas/rate limits, Alembic migrations, SSE streaming for answers.

## Tests

`pytest` covers extraction (txt/docx/corrupt/empty), chunking and sampling, citation parsing,
cost maths and upload/input validation. Tests that touch OpenAI or Postgres are not included;
`scripts/demo.py` exercises the full stack.

See [AI_USAGE.md](AI_USAGE.md) for how AI tools and prompts were used, and
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) for a file-by-file explanation of the backend.

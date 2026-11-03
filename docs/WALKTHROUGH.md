# Backend walkthrough

How the backend works, file by file, plus the trade-offs and weak spots worth knowing before
discussing the implementation. For setup see [SETUP.md](../SETUP.md); for the short design
summary see [README.md](../README.md).

## 1. The big picture

```
Browser/curl ──► FastAPI (api container) ──► Postgres + pgvector
                      │  .delay(doc_id)              ▲
                      ▼                              │
                 Redis (queue) ──► Celery worker ────┘
                                      │
                                      └──► OpenAI (embeddings + GPT)
```

- **FastAPI** handles HTTP requests quickly and never does slow work itself.
- **Redis** is only a message queue. It holds "process document X" messages.
- **Celery worker** does the slow work: reading the file, chunking, calling OpenAI.
- **Postgres** is the single datastore: metadata, chat history, metrics, and vectors (pgvector).
- **OpenAI** is called by the worker at indexing time and by the API at chat time. Only the
  similarity search itself is local.

### Flow A: Upload (`POST /documents`)
1. Validate the file: extension (415), size (413), non-empty (400).
2. Compute a SHA-256 hash of the bytes. If a document with the same hash exists, return it (200) and stop.
3. Save the file as `storage/<uuid>.<ext>` and insert a `Document` row with status `pending`.
4. `process_document_task.delay(id)` puts a JSON message on Redis.
5. Return **202 Accepted** immediately. The client polls `GET /documents/{id}`.

### Flow B: Processing (worker)
1. Pick up the message, open its own DB session, create a `ProcessingJob` row, set the document to `processing`.
2. **extract** (text per page) → **chunk** (about 800 tokens) → **embed** (OpenAI) → **insights** (GPT, JSON output).
3. In one transaction: delete old chunks, insert new chunks with embeddings, fill in the document
   summary and insights, set status `ready`, record timings, tokens and cost on the job.
4. On any failure: roll back, set the document to `failed` with a readable error, mark the job failed.

### Flow C: Chat (`POST /chat/sessions/{id}/messages`)
1. Load the last 6 messages (the history).
2. If there is history, ask GPT to rewrite the question as a standalone question.
3. Embed that standalone question (OpenAI).
4. Vector similarity search in Postgres over only this session's documents; take the top 6 chunks.
5. Build a prompt: system rules, numbered chunks, the history, and the real question. GPT answers.
6. Parse `[n]` markers from the answer to produce citations.
7. Save the user and assistant messages together, with tokens, cost and latency.

| Step | Where it runs |
|---|---|
| Index: split into chunks | Local |
| Index: embed chunks | OpenAI (embeddings) |
| Index: summary and insights | OpenAI (GPT) |
| Chat: rewrite a follow-up as a standalone question | OpenAI (GPT, only from the second question on) |
| Chat: embed the question | OpenAI (embeddings) |
| Chat: find the closest chunks | Local (pgvector in Postgres) |
| Chat: write the answer from those chunks | OpenAI (GPT) |

## 2. File by file

### `app/config.py`
`Settings` extends pydantic's `BaseSettings`, so each field is read from an environment variable
of the same name, case-insensitive (`OPENAI_API_KEY` → `openai_api_key`), with defaults otherwise.
`get_settings()` uses `@lru_cache`, so it is built once per process.

Tunable numbers live here so they can change without touching code: `chunk_size_tokens=800`,
`chunk_overlap_tokens=100`, `retrieval_top_k=6`, `history_messages=6`, `insight_token_budget=24000`,
`embedding_dimensions=1536` (must match the pgvector column).

### `app/db.py`
- `create_engine(..., pool_pre_ping=True)` checks a pooled connection is alive before using it,
  avoiding errors after Postgres restarts.
- `sessionmaker(expire_on_commit=False)` keeps objects readable after commit. Without it,
  touching `document.id` after `commit()` would trigger a reload, and returning ORM objects from
  endpoints would be awkward.
- `get_db()` is a FastAPI dependency: one session per request, closed afterwards.
- `init_db()` runs `CREATE EXTENSION IF NOT EXISTS vector` and then `create_all`. The extension must
  exist before a table with a `vector` column is created. There are no migrations (Alembic would
  replace this in production).
- The API and the worker both use **synchronous** SQLAlchemy. FastAPI runs plain `def` endpoints
  in a threadpool, so they do not block the event loop. Celery is synchronous anyway.

### `app/models.py`
- **`documents`**: filename, size, `content_hash` (unique, used for de-duplication), `storage_path`,
  `status` (`pending`/`processing`/`ready`/`failed`, indexed), `error`, `page_count`, `summary`,
  `insights` (JSONB). `category` and `tags` are plain Python `@property`s reading from the
  `insights` JSON, which keeps the schema small.
- **`document_chunks`**: `document_id`, `chunk_index`, `page_number`, `content`, `token_count`,
  `embedding Vector(1536)`. The foreign key has `ON DELETE CASCADE`, so deleting a document removes
  its chunks in the database without loading them. The **HNSW index** with `vector_cosine_ops`
  builds an approximate nearest-neighbour graph so search stays fast with many chunks
  (`m=16`, `ef_construction=64` are standard build parameters).
- **`processing_jobs`**: one row per processing attempt: status, error, `stage_timings`
  (JSON such as `{"extract":0.001,"embed":2.3}`), total duration, tokens, cost. The source for
  processing metrics.
- **`chat_sessions`** + **`chat_session_documents`** (link table): many-to-many, which is what makes
  multi-document chat possible. `ChatSession.documents` is a `relationship(secondary=...)`.
- **`chat_messages`**: `role`, `content`, `citations` (JSON list), tokens, cost, `latency_ms`.

**`JSONB(none_as_null=True)`:** by default SQLAlchemy stores Python `None` in a JSON column as the
JSON value `null`, which is not SQL `NULL`; that breaks queries such as `insights IS NOT NULL`.
The flag makes `None` become real SQL NULL.

### `app/schemas.py`
Pydantic models define what the API accepts and returns (Swagger is generated from them).
`from_attributes=True` lets them read directly from ORM objects. `DocumentInsights` is used twice:
it is the shape requested from GPT *and* it validates GPT's output; an invalid `sentiment` fails
validation and the job fails cleanly. `Field(min_length=..., max_length=...)` on `document_ids` and
`question` gives automatic 422 responses.

### `app/worker.py` and `app/tasks.py`
- `Celery("docu_ai", broker=redis_url, include=["app.tasks"])`: `include` makes the worker import
  the tasks module so `process_document` is registered.
- **No result backend**: task state lives in Postgres (document status).
- `task_acks_late=True` + `task_reject_on_worker_lost=True`: the message is acknowledged only after
  the task finishes, so if the worker crashes mid-task the message is re-delivered, not lost.
- `worker_prefetch_multiplier=1`: each worker takes one job at a time instead of hoarding a batch;
  matters because jobs last seconds.
- `tasks.py` is deliberately thin (convert id to UUID, call `process_document`); the logic lives in
  `services/ingestion.py`, so it is testable without Celery.

### `app/services/extraction.py`
Returns a list of `Page(number, text)` so citations can name pages.
- **PDF:** `pypdf`, page by page. `extract_text()` can return `None`, hence `or ""`. Corrupt or
  encrypted files raise library errors that are converted to `ExtractionError` with a friendly message.
- **DOCX:** `python-docx`. No real pages, so the whole file is "page 1". Paragraphs first, then table
  rows as `cell | cell` (original paragraph/table order is not preserved).
  `clean_pdf_text` also repairs PDFs that return one word per line (`word\n \nword\n \n...`) by
  collapsing blank-but-not-empty lines; real paragraph breaks (`\n\n`) are untouched. Without it a
  4-page PDF became 7 noisy chunks with weak embeddings; now it is 4 clean chunks.
- **TXT/MD:** UTF-8 with `errors="replace"`, so odd bytes never crash it.
- `_clean` removes NUL bytes (`\x00`), which Postgres TEXT columns reject.
- Empty pages are filtered out. If nothing is left (typical of **scanned PDFs**, since there is no
  OCR) it raises "No text could be extracted...".
- A dispatch dict `_EXTRACTORS` keyed by extension replaces an if/elif chain.

### `app/services/chunking.py`
- `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` is the only LangChain piece. It measures
  size in **tokens** (`cl100k_base`), and tries paragraphs first, then lines, spaces, characters, so
  chunks end at natural boundaries.
- **Each page is split separately.** Benefit: every chunk belongs to one page, so a citation can say
  "page 3". Cost: a short page gives a small chunk, and ideas that span a page break are cut.
- Overlap (100 tokens) repeats the end of one chunk at the start of the next, so an answer sitting
  on a boundary is not lost.
- `select_text_for_insights`: if the document fits the 24,000-token budget, send it all. Otherwise
  compute how many chunks fit (`keep = budget // average chunk size`) and pick evenly spaced chunks
  (`chunks[int(i*step)]`) joined with a `[...]` marker, so the summary reflects the beginning,
  middle and end rather than only the first pages.

### `app/services/llm.py`
All OpenAI access goes through here: one place for errors, retries and cost.
- `PRICING`: USD per million tokens (unknown model costs 0). `Usage` is a small dataclass with
  `add()` so costs accumulate across several calls in one operation.
- `_client()` is cached. It raises `LLMError("OPENAI_API_KEY is not configured.")` if the key is
  empty or still the placeholder (fail fast with a clear message). `max_retries=3, timeout=60`:
  the SDK retries rate limits, 5xx and connection errors with exponential backoff.
- `embed_texts` sends texts in batches of 100 (fewer HTTP calls) and passes `dimensions=` so the
  vector size always matches the DB column. Cost comes from the token usage in the response.
- `complete` runs one chat completion. With `json_mode=True` it sets
  `response_format={"type": "json_object"}`, which forces valid JSON (OpenAI requires the prompt to
  contain the word "JSON"; the insights prompt does).
- All `OpenAIError`s become `LLMError` carrying only the exception class name (for example
  `AuthenticationError`), so nothing sensitive leaks. `main.py` maps `LLMError` to HTTP 503.
- Settings and the client are cached for the life of the process, which is why changing `.env`
  needs a container recreate.

### `app/services/prompts.py`
1. **Answer prompt:** answer only from the document overviews and numbered passages, cite `[n]`, use an exact refusal
   sentence when the answer is absent, flag conflicts between documents, treat passages as untrusted
   (prompt-injection defence). The exact refusal sentence makes "not found" easy to detect.
2. **Condense prompt:** rewrites "and the internet allowance?" into a standalone question. Retrieval
   works on embeddings and a pronoun-heavy question embeds poorly, so the query is fixed before
   searching. The answer step still sees the real history.
3. **Insights prompt:** JSON mode with an explicit key list and a closed category list, so categories
   are consistent and filterable.

### `app/services/insights.py`
Calls `complete(json_mode=True)` and validates with `DocumentInsights.model_validate_json`. A
`ValidationError` becomes `LLMError("The AI returned malformed insights; try reprocessing.")`.

### `app/services/ingestion.py` (the heart of the worker)
- `_stage` is a context manager that times a block into a `timings` dict, even if the block raises
  (`finally`).
- `process_document`:
  1. Load the document; if it was deleted meanwhile, log and return.
  2. Create the job row, set `processing`, and **commit first** so the status is visible immediately.
  3. Run extract, chunk, embed, insights.
  4. Delete existing chunks and bulk-insert new ones. `zip(..., strict=True)` raises if chunk and
     vector counts ever differ. Deleting first makes reprocessing **idempotent**: no duplicate chunks,
     even if Celery re-delivers the task.
  5. Update document and job and **commit once**, so a document never shows `ready` without its chunks.
- Two error levels: `ExtractionError`/`LLMError` have user-safe messages and are saved as the
  document's `error`. Any other exception is logged with a stack trace, and the user only sees
  "Unexpected error while processing the document." (no internals leaked).
- `_fail` calls `rollback()` first (the session may be in a broken transaction), then records the
  failure and commits.

### `app/services/retrieval.py`
```python
distance = DocumentChunk.embedding.cosine_distance(query_vector)   # pgvector <=> operator
select(DocumentChunk, Document.filename, distance)
  .join(Document, ...).where(document_id IN session_docs)
  .order_by(distance).limit(top_k)
```
Score is `1 - distance` (1.0 = identical direction). `WHERE document_id IN (...)` scopes retrieval to
the session's documents. Joining `Document` gets the filename for citations in one query.

### `app/services/chat.py`
- `recent_history` fetches the newest N messages descending, then reverses them so the prompt reads
  chronologically.
- `build_overview` adds each session document's stored summary and key points to the prompt.
  Top-6 passages cannot answer broad questions ("what is this document about?"), so the overview
  gives the model the big picture at a small fixed cost. Answers drawn only from an overview carry
  no `[n]` citation, by design; specific facts must still come from numbered passages.
- `build_context` numbers passages from 1: `[1] (document: x.pdf, page 2)\n<text>`. The numbering
  is what lets `[n]` map back to chunks.
- `extract_citations` finds `[n]` with a regex, keeps valid numbers in range, de-duplicates and sorts.
  It returns only passages the model **actually cited**, with filename, page, a 240-character snippet
  and score. Bogus numbers such as `[9]` are ignored (covered in `tests/test_chat_logic.py`).
- `answer_question` order: history → condense (skipped without history, saving a call) → embed →
  retrieve → answer (temperature 0.2) → save. If retrieval finds nothing, it returns the "couldn't
  find" answer without calling GPT.
- Messages are saved **only after the model succeeds**, so a failed OpenAI call leaves no orphan
  question.
- Context is rebuilt every turn and is **not** stored in the history, so old passages do not pile up
  in the prompt.
- Usage from every call in the turn (condense, embed, answer) is summed into one token and cost
  figure on the assistant message.

### `app/api/documents.py`
- `_read_validated_upload` reads `limit + 1` bytes, so oversize is detected without loading a huge
  file. `Path(filename).name` strips directory parts (path-traversal defence), and the stored
  filename is a fresh UUID, so the user's filename never touches the filesystem path.
- The upload endpoint is a plain `def`, not `async def`: it does blocking DB and Redis work that
  would stall the event loop if it were `async`.
- **De-dup race:** two simultaneous identical uploads can both pass the "exists?" check; the `unique`
  constraint makes one insert fail with `IntegrityError`, which is caught: the extra file is deleted
  and the winner's record is returned.
- `_enqueue`: if Redis is down, the document is marked `failed` and the API returns 503, instead of
  leaving it `pending` forever.
- **List** filters by `status` and `category` (`insights["category"].astext` queries inside JSONB),
  with `limit`/`offset` and a total count.
- **Insights:** 409 unless the document is `ready`. **Reprocess:** 409 if already pending/processing.
- **Delete:** 409 while processing (the worker would otherwise hit foreign-key errors). Removes the
  file and the row; `ON DELETE CASCADE` handles chunks, jobs and chat links.

### `app/api/chat.py`
- **Start session:** `dict.fromkeys(ids)` removes duplicate ids while keeping order. 404 lists any
  missing ids; 409 if any document is not `ready`. The default title is the filenames.
- **Ask:** 404 for an unknown session; 409 if every document was deleted since.
- **History:** ordered by id, with limit and offset.

### `app/api/metrics.py`
All computed with SQL aggregates, not Python loops:
- **Documents:** counts by status and category (`GROUP BY`), total size/pages/chunks, uploads in 24h.
- **Processing:** `count(*) FILTER (WHERE status = ...)` for succeeded/failed/running in one query;
  `avg(...)` for average duration; `percentile_cont(0.95) WITHIN GROUP (ORDER BY duration)` for p95;
  `avg((stage_timings->>'embed')::float)` per stage; plus tokens, cost, success rate and the 5 most
  recent failures.
- **Chat:** answers, average and p95 latency, tokens, cost, and "answers without citations" (mostly
  refusals, a rough retrieval-quality signal).

### `app/api/health.py`
`SELECT 1` against Postgres and `PING` against Redis. 200 if both pass, **503** if either fails, so
load balancers and Docker can use it.

### `app/main.py`
`lifespan` creates the storage folder and runs `init_db()` at startup. A middleware logs
`METHOD path -> status (ms)` for every request. An exception handler maps `LLMError` to a 503 JSON
message. It includes the four routers and serves the UI at `/`.

### `app/static/index.html`
A single static page (plain HTML/CSS/JS, no build step) served by FastAPI, so there is no extra
service and no CORS. It only calls the public API. Model output is HTML-escaped before the app adds
its own tags (bold, bullets, citation chips), so answers cannot inject HTML.

### Docker setup
- `docker-compose.yml` runs `db` (pgvector image), `redis`, `api`, `worker`. Health checks plus
  `depends_on: condition: service_healthy` make the app wait for Postgres and Redis.
- `storage/` is shared by the API and the worker: the API saves the upload, the worker reads it.
- `DATABASE_URL`/`REDIS_URL` are overridden inside containers to use the service names `db` and
  `redis` instead of `localhost`.
- The API runs with `--reload` and `app/` is mounted. **The worker does not reload**; use
  `docker compose restart worker` after editing worker-side code.

## 3. Likely questions

**Why RAG instead of sending the whole document to the model?** It scales to many documents, cuts
cost and latency, and gives page-level citations. Whole documents stop fitting in context windows
once there are thousands.

**Why pgvector and not Pinecone?** One datastore to run, vectors sit next to metadata so filtering
by document is plain SQL, and it is enough at this scale. At very large scale a dedicated vector
database may be better.

**Why Celery and not FastAPI `BackgroundTasks`?** `BackgroundTasks` runs inside the API process: no
retries, no independent scaling, and tasks are lost if the API restarts. Celery with Redis gives
durability, `acks_late` redelivery, and workers that scale separately.

**How do you handle a document that fails halfway?** The transaction is rolled back, so no partial
chunks are saved. The document is marked `failed` with a readable error and `/reprocess` retries.
Reprocessing is idempotent because old chunks are deleted first.

**How does multi-turn context work?** In two places: the condense step rewrites the follow-up so
retrieval works, and the answer step also sees the last 6 messages as real chat turns.

**How do you reduce hallucinations?** Grounded prompt, exact refusal sentence, low temperature,
numbered sources with citations, and an untrusted-text instruction. If retrieval returns nothing,
GPT is not called at all.

**How would it scale to thousands of documents?** More worker replicas, split queues, S3 instead of
a shared volume, Redis caching of query embeddings and answers, rate limits and quotas, and Alembic
migrations.

## 4. Known weaknesses

- **Multi-document crowding:** retrieval takes the overall top 6, so one very relevant document can
  push out another in comparison questions. A fix is taking the top few per document.
- **pgvector filtering:** with an HNSW index the `document_id` filter is applied after the index
  search, so a restrictive filter over a large table can return fewer than 6 results. Newer pgvector
  versions support iterative scans, or the table can be partitioned.
- Chunks never cross pages, and DOCX has no page numbers.
- No OCR, so scanned PDFs fail (with a clear message).
- No authentication or multi-tenancy: anyone with a session id can read it.
- `create_all` instead of migrations.
- No task-level retry: OpenAI errors are retried inside the SDK, but a failed document needs a manual
  reprocess.
- Chunk token counts use `cl100k_base`, which matches the embedding model but only approximates
  GPT-4o-mini.
- Category is not enforced by the schema: the prompt asks for a closed list, but the Pydantic field
  is a free string with default `other`.

Bugs found and fixed during development (useful honest stories about AI-assisted work): the upload
endpoint was `async` but did blocking calls; JSONB stored `None` as JSON `null`; and a UI bug
rendered the document chips as `[object HTMLSpanElement]` because `replaceChildren` was given an
array instead of spread arguments.

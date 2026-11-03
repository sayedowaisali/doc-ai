# AI Usage

## AI coding tools

Built with **Cursor** (AI agent). How it was used:

- **Requirements review first:** the agent read the case study, listed ambiguities (infra, RAG
  framework, queue, scope) and I decided each before any code was written.
- **Planning:** the schema, endpoint list and pipeline stages were agreed up front, then
  implemented in one pass.
- **Implementation and iteration:** the agent wrote the FastAPI/Celery/pgvector code and tests,
  and checked library APIs against the installed packages instead of guessing.
- **Review pass:** the agent caught two issues before release: a blocking `async` upload
  handler (switched to sync so it runs in the threadpool) and JSONB `None` being stored as JSON
  `null` instead of SQL `NULL` (broke aggregate queries).
- **What I did myself:** scope and architecture decisions, choosing a simple hybrid RAG over a
  full framework so it can be explained end to end, and reviewing/running the result.

## Prompt engineering (`app/services/prompts.py`)

| Prompt | Design choices and why |
|---|---|
| **Answer** | Answer only from numbered passages; cite `[n]`; exact refusal sentence when the answer is absent (prevents hallucination and makes "not found" detectable); flag conflicts between documents; treat passages as untrusted (prompt-injection defence). Temperature 0.2. |
| **Condense** | Rewrites follow-ups into standalone questions so retrieval works in multi-turn chat. The answer step still receives the real history. Temperature 0. |
| **Insights** | JSON mode with an explicit key list and a *closed* category set so categories are filterable; asks for specific key points rather than generic ones; long documents are sampled across their whole length and marked `[...]`. Output is validated with Pydantic and a malformed response fails the job with a retryable error. |

## Other AI-driven features

- Automatic summary, key points, topics, category, tags and sentiment at ingestion.
- Citation mapping: only passages the model actually cited are returned, with page and snippet.
- Token and cost tracking per processing job and per answer.

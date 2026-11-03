# Setup Guide (fresh macOS)

Step-by-step instructions to run Docu AI on a new Mac. Written so a human **or a Cursor agent**
can follow it. Everything runs in Docker; the only secret needed is an OpenAI API key.

> **For Cursor agents:** run the commands in order, check each "Verify" line before moving on,
> and never print or commit the contents of `.env`. Ask the user for the OpenAI key to be pasted
> into `.env` by them rather than asking them to paste it in chat.

## 1. Prerequisites

| Tool | Why | Install |
|---|---|---|
| Homebrew | Package manager | https://brew.sh |
| Git | Clone the repo | `brew install git` |
| GitHub CLI | Authenticate to the (private) repo | `brew install gh` |
| OrbStack (or Docker Desktop) | Runs Postgres, Redis, API, worker | `brew install --cask orbstack` |
| Python 3.11+ (optional) | Only for running the demo script and tests outside Docker | `brew install python@3.12` |
| OpenAI API key | Embeddings + chat | https://platform.openai.com/api-keys |

Verify: `brew --version && git --version`

## 2. Authenticate with GitHub

```bash
gh auth login
```

Choose: **GitHub.com → HTTPS → Yes (authenticate git) → Login with a web browser.** Enter the
one-time code at https://github.com/login/device while signed in to the account that owns the repo.

Then register `gh` as git's credential helper (login alone does not do this, and without it
`git clone`/`git push` fail with `could not read Username for 'https://github.com'`):

```bash
gh auth setup-git
```

Verify: `gh auth status` shows `Logged in to github.com account <name>`, and `git push`/`git clone`
over HTTPS now use this account.

## 3. Start OrbStack (Docker)

1. `open -a OrbStack` and complete the first-run screens (sign-in, approve the system prompt).
   Ignore the "New Container" dialog if it appears: click **Cancel**.
2. Make the `docker` CLI available. OrbStack keeps it in `~/.orbstack/bin`, which may not be on `PATH`:

```bash
echo 'export PATH="$HOME/.orbstack/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Verify: `docker --version && docker compose version` both print a version.
(Docker Desktop users: skip the PATH step; `docker` is already available once the app is running.)

## 4. Get the code and configure

```bash
git clone https://github.com/sayedowaisali/docu-ai.git
cd docu-ai
cp .env.example .env
```

Open `.env` in an editor and replace the placeholder:

```
OPENAI_API_KEY=sk-...your-real-key...
```

`.env` is gitignored: never commit it. Other values in it already match `docker-compose.yml`.
The compose file fails to start if `.env` is missing, so do not skip the `cp`.

Verify (without printing the key): `grep -q "^OPENAI_API_KEY=sk-" .env && ! grep -q "sk-your-key-here" .env && echo KEY_SET_OK`

## 5. Run the stack

```bash
docker compose up --build -d
```

The first run downloads images and builds (about 1-3 minutes). It starts four containers:
`db` (Postgres + pgvector), `redis`, `api` (port 8000) and `worker` (Celery).

Verify:

```bash
curl http://localhost:8000/health
# {"status":"ok","checks":{"database":true,"redis":true}}
```

Web UI: http://localhost:8000 (API docs with a clickable page per endpoint: http://localhost:8000/docs)

## 6. Try it

**Option A: web UI** at http://localhost:8000. Drop a file from `samples/` onto the page, wait for
the green `ready` badge, tick one or more documents, click **Chat with N documents**, and ask a
question. Click an inline `[n]` to see the source passage.

**Option B: demo script** (uploads both sample documents, shows insights, runs a multi-turn chat, prints metrics):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/demo.py
```

(The script only needs `httpx`, so `python3 -m pip install httpx` is enough if you do not want the full venv.)

**Option C: Swagger UI** at http://localhost:8000/docs:
`POST /documents` (upload a file from `samples/`) → poll `GET /documents/{id}` until `status` is
`ready` → `POST /chat/sessions` with `{"document_ids": ["<id>"]}` → `POST /chat/sessions/{id}/messages`
with `{"question": "..."}`.

Expected demo result: both documents `ready`, answers with `[n]` citations (document + page),
a correct follow-up ("And the internet allowance?"), and "I couldn't find this in the provided
documents." for a question the documents do not answer.

## 7. Run the tests

```bash
.venv/bin/pytest
```

17 unit tests; they need no Docker, database or OpenAI key.

## 8. Everyday commands

```bash
docker compose up -d                # start (add --build after code/dependency changes)
docker compose logs -f api worker   # follow logs
docker compose ps                   # container status
docker compose down                 # stop; data is kept in the pgdata volume
docker compose down -v              # stop AND delete the database (full reset)
```

Uploaded files live in `./storage/` (gitignored). The `app/` folder is mounted into the
containers: the API reloads on code changes, but **the worker does not**, so restart it with
`docker compose restart worker` after editing worker-side code (`app/services/ingestion.py`, etc.).

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker: command not found` | OrbStack's CLI is not on `PATH`: do the step 3 PATH commands, or open a new terminal. |
| `Cannot connect to the Docker daemon` | OrbStack/Docker Desktop app is not running: `open -a OrbStack`. |
| `env file .env not found` | Run `cp .env.example .env`. |
| `port is already allocated` (5432, 6379 or 8000) | Another Postgres/Redis/app is using the port: stop it (`brew services stop postgresql redis`) or change the left side of the port mapping in `docker-compose.yml`. |
| Document ends `failed` with "OPENAI_API_KEY is not configured." | Key missing or still the placeholder: fix `.env`, then `docker compose up -d --force-recreate api worker` (env is read at container start), then `POST /documents/{id}/reprocess`. |
| Chat shows "Embedding request failed: AuthenticationError" | The key in `.env` is wrong or truncated: fix it, then `docker compose up -d --force-recreate api worker`. |
| Document ends `failed` with "Embedding request failed: ..." | Invalid key, no credit, or rate limit: check the key and billing at platform.openai.com. |
| Document stuck in `pending` | Worker is not running or cannot reach Redis: `docker compose logs worker`. |
| `/health` returns 503 | `checks` shows which dependency is down: `docker compose ps` and check logs. |
| Changed `embedding_dimensions` or the embedding model | Existing vectors no longer match the column: `docker compose down -v` and re-upload documents. |
| `gh auth login` code rejected or wrong account | Open an incognito window, sign in to the correct GitHub account, then retry at https://github.com/login/device. Use `gh auth logout` to switch accounts. |

## 10. Project layout

```
app/static/     Single-page web UI (index.html, served at /)
app/api/        FastAPI routers (documents, chat, metrics, health)
app/services/   extraction, chunking, embeddings/LLM, retrieval, chat, ingestion, prompts
app/models.py   SQLAlchemy tables (incl. pgvector chunks)
app/worker.py   Celery app; app/tasks.py task entrypoint
samples/        two small documents for the demo
scripts/        demo.py
tests/          pytest unit tests
```

Architecture and design decisions are in [README.md](README.md); prompt design in [AI_USAGE.md](AI_USAGE.md).

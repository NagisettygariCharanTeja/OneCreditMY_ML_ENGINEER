# Similar companies

The service behind the "Similar companies" panel on the OneCredit portal. A consumer reads company events from the registry's RabbitMQ queue, cleans them, embeds each company's business description and keeps Postgres up to date. An API returns the most similar active companies for any company, ranked by pgvector inside Postgres.

- **Design, decisions and evaluation:** [DESIGN.md](DESIGN.md)
- The original brief is in the template repo: [OneCreditMY/mle-take-home](https://github.com/OneCreditMY/mle-take-home)

## Run it

Needs [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). uv installs Python 3.14 and all dependencies.

```bash
just setup      # install Python and dependencies
just up         # start Postgres (pgvector) and RabbitMQ
```

**Terminal 1: the consumer.** It creates the tables on first start and then waits for messages. The first start downloads the embedding model (~220 MB).
```bash
uv run python -m similar_companies.consumer
```

**Terminal 2: replay the registry feed.**
```bash
just publish
```
The consumer logs progress every 500 messages and finishes in a minute or two.

**Terminal 3: the API.**
```bash
uv run uvicorn similar_companies.api:app --port 8000
curl http://localhost:8000/companies/SYN-001550/similar
```

**Check the result:**
```bash
docker compose exec postgres psql -U postgres -d companies -c "SELECT status, count(*) FROM companies GROUP BY 1;"
# active 1992, struck_off 8
docker compose exec postgres psql -U postgres -d companies -c "SELECT reason, count(*) FROM rejected_events GROUP BY 1;"
# 32 quarantined messages
```

Stop everything with Ctrl+C in each terminal, then `just down`.

## API

`GET /companies/{registration_number}/similar?limit=5`

`limit` is 1 to 50 (default 5). The registration number is normalized, so ` syn-001550 ` works too.

```json
{
  "registrationNumber": "SYN-001550",
  "similar": [
    {"registrationNumber": "SYN-000539", "companyName": "Seri Sejahtera Food Sdn. Bhd.",
     "industry": "Food & Beverage", "state": "Johor", "similarity": 1.0},
    {"registrationNumber": "SYN-000263", "companyName": "Gemilang Delima Food Sdn. Bhd.",
     "industry": "Food & Beverage", "state": "Johor", "similarity": 0.95}
  ]
}
```
(abridged and rounded: the default response has 5 cards, most similar first)

| Status | When |
|---|---|
| 200 | The company is active: its most similar active companies, never itself |
| 404 | The company has never been in the feed |
| 410 | The company was struck off |
| 422 | Malformed registration number or `limit` |
| 503 | Postgres is unreachable |

`GET /health` returns `{"status": "ok"}` when the database is reachable. Interactive docs: http://localhost:8000/docs

## Tests and checks

```bash
uv run pytest -q                    # all 141 tests (~3 min: two replay the whole feed)
uv run pytest -q -m "not slow"      # quick loop (~15 s)
uv run ruff check .                 # lint
```
The tests use a separate `companies_test` database in the same Postgres, so `just up` must be running. They never touch the `companies` database.

| Command | What it does |
|---|---|
| `uv run python -m scripts.pipeline_check` | Full pipeline with a crash: producer → RabbitMQ → consumer killed with SIGKILL mid-feed → restart → whole feed published again → API checks. 13 checks; uses its own database. Takes a few minutes. |
| `uv run python -m scripts.evaluate_models` | Compares embedding set-ups by industry precision@5 (results in DESIGN.md) |
| `uv run python -m scripts.profile_feed` | The data-quality profile of the feed (EDA) |
| `uv run python -m scripts.peek_models` | Similarity scores of one café against five real descriptions, per model |
| `RUN_MODEL_TESTS=1 uv run pytest tests/test_embedding.py` | Smoke tests against the real models (downloads them) |

## Configuration

All settings are environment variables with defaults that match `docker-compose.yml` (see `similar_companies/config.py`).

| Variable | Default |
|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:${POSTGRES_PORT:-5432}/companies` |
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:${RABBITMQ_PORT:-5672}/` |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`hashing-bow-384` = offline placeholder, no download) |
| `PREFETCH_COUNT` | `20` |

If ports 5432 or 5672 are taken, set `POSTGRES_PORT` / `RABBITMQ_PORT` before `just up`; the consumer, API, producer and scripts all follow them.

## Layout

```
similar_companies/
  config.py       settings from environment variables
  schema.sql      tables: companies, rejected_events, company_neighbours
  normalize.py    pure functions: registration numbers, UTC timestamps, states, descriptions
  validation.py   raw message -> clean event, or a rejection with a reason code
  embedding.py    text to embed + embedders (fastembed model, hashing placeholder)
  repository.py   newer-wins writes, strike-offs, quarantine
  neighbours.py   similarity search: live exact scan + exact precomputed top 20
  consumer.py     RabbitMQ -> validate -> embed -> write -> commit -> ack
  api.py          FastAPI app
scripts/          pipeline check, model evaluation, EDA, model peek
tests/            unit, feed-contract, integration and end-to-end tests
```

I added files but didn't change the template's `producer/`, `data/`, `docker-compose.yml` or `Justfile` (the registry team's side of the scenario). Dependencies were added with `uv add`.

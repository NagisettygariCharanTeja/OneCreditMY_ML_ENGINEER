# Design

How I built the similar-companies service, the decisions behind it, and what I'd do next.

## 1. What I built

A consumer reads the registry's `company-events` queue, validates and normalizes every message, embeds each company's business description with an open-source multilingual model on CPU, and writes it to Postgres with a newer-wins rule, committing before it acks so nothing is lost. A precomputed, exactly maintained neighbour table sits next to the vectors in pgvector. A FastAPI endpoint returns, for any company, its most similar active companies, with the name, industry, state and registration number each card needs, most similar first.

After replaying the feed: 1,992 active and 8 struck-off companies (all 2,000 in the feed), 32 bad messages quarantined, every active company searchable.

## 2. Decisions

| Area | Choice | Why | Trade-off |
|---|---|---|---|
| Similarity signal | `natureOfBusiness` embedding | Brief says this is enough for v1; the goal is "do similar things" | Ignores location |
| Embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, CPU, via fastembed) | Half the descriptions are Malay. Measured on 1,890 companies: industry precision@5 of 0.862 vs 0.522 for the README's English-only `bge-small-en-v1.5` (see Deep dive) | ~220 MB model download on first start |
| Text sent to the model | The cleaned description; for the 39 companies without one, `industry. company name` | The brief's similarity signal; the fallback keeps every company recommendable. Stripping locations was measured and made no difference (0.861 vs 0.862), so the text stays as the registry sent it | Fallback companies match on industry and name only |
| Ordering rule | A write applies only if its `updated_at` (converted to UTC) is strictly newer than the stored one, for both upserts and strike-offs | 66 messages arrive after a newer one; one atomic `INSERT … ON CONFLICT … DO UPDATE … WHERE` handles it without locks. A strike-off also advances `updated_at`, so a late older upsert can't revive the company. | Relies on the registry's clocks |
| Duplicates | No separate dedup table: an equal timestamp is a no-op in the same conditional write | Every message carries full company state, so writes are idempotent by nature | Duplicates still cost a parse (cheap) |
| Upsert semantics | Each valid upsert replaces the whole record (full state), not a per-field merge | In this feed every valid upsert carries all four fields and no newer upsert ever nulls a field (checked on all 331 companies with several valid upserts), so a per-field merge would give the identical result with four extra timestamp columns. It would also have to guess whether `null` means "unknown" or "cleared" | If the registry starts sending partial updates, see What's left |
| Delivery guarantee | Commit the database transaction, then ack | A crash in between only causes a redelivery, which is a no-op; acking first could lose a company. Verified with `kill -9` mid-feed: nothing lost | Redeliveries cost a parse and a lookup |
| Database outage | `nack(requeue=True)`, reconnect with backoff (1–30 s), then continue | Transient failures must not quarantine good data | Messages wait in the queue meanwhile |
| Bad messages | Quarantine in `rejected_events` (stored once per message_id), then ack | The queue is owned by the registry team with no arguments, so no dead-letter queue is possible; a table keeps evidence and never blocks the queue | Needs manual review |
| Reactivation | A newer upsert after a strike-off reactivates the company (logged as a warning) | The registry is the source of truth and outside our control; struck-off companies can be reinstated. The consumer mirrors the registry instead of second-guessing it. | A registry mistake would show up on the portal until corrected upstream |
| Struck-off companies | Keep the row as a tombstone (`status = struck_off`); exclude from all results; its own page gets **410 Gone** (unknown company: 404) | Deleting would let a late, older upsert recreate it; the API enforces this rather than trusting the portal | Keeps a few never-shown rows |
| Schema | `companies` (one row per company, PK = registration number, `status`, UTC `updated_at`, card fields, `embedding vector(384)`, `embedding_model`) `rejected_events` (raw body as text + reason code) and `company_neighbours` (the cache, below) | Registration number is the natural key; raw text because bad messages may not be valid JSON; a CHECK constraint makes every active company displayable and searchable | Rejected rows aren't reprocessed automatically |
| Vector index | None in v1: exact scan with a deterministic tie-breaker | At 2,000 rows the exact top 5 takes ~1.4 ms (measured), always returns the true neighbours, and breaks ties reproducibly. Many companies share an identical description, and HNSW returned an arbitrary subset of tied companies and isn't used at all with a tie-breaker. | Gets slow at very large volumes, see 10x |
| API | `GET /companies/{reg_no}/similar?limit=5` → cards (reg no, name, industry, state, similarity), most similar first; 404 unknown, 410 struck off, 422 malformed, 503 database down | The registration number lets a click open the next page; distinct codes let the portal show "not found" vs "unavailable" | |
| Search query | One SQL statement: exact cosine scan over active companies, excluding the company itself, ties broken by registration number | Runs in Postgres as required; reproducible. Used directly as the fallback, and as the reference the cache is tested against | Linear in company count |
| Neighbour cache | `company_neighbours`: each active company's top 20 active neighbours, kept exact by the consumer in the same transaction as every write. The API reads these candidates, joins live company data, filters out struck-off companies, re-scores with live vectors, and falls back to the live search when needed | The production pattern of precomputed candidates plus hard filters on live data at request time. Query 1.30 → 0.15 ms; request 5.46 → 3.52 ms | Writes are slower: full-feed replay 1.4 s → 45 s (~19 ms per applied write) |
| Connections | One short connection per request, no pool | Simplest option; whole request measured at median 11.7 ms, p95 15.7 ms on my machine (Docker on a Mac) | Connection setup per request; a pool at higher traffic |

## 3. Assumptions

- **Latest = newest `updatedAt`**, not arrival order: the registry note says updates can arrive late.
- **Timestamps without a timezone are Malaysia time (UTC+8).** Evidence: read as Malaysia time, the 14 such timestamps sit a median 6.2 min from neighbouring lines; read as UTC, 474.7 min (~8 h) away.
- **An upsert is the company's full current record.** Evidence: no valid upsert in the feed omits a field that an earlier one had.
- A struck-off company has been removed from the register: it stays in the database as a record but is never shown on the portal or in recommendations.

## 4. How the consumer handles the feed

| Issue in the feed | Count | What the consumer does |
|---|---|---|
| Truncated JSON / literal `null` | 5 / 6 | Quarantine, ack |
| Exact duplicate messages | 68 | No-op (equal `updated_at`) |
| Older update arriving after a newer one | 66 | No-op (newer-wins) |
| Reg no padded / lowercase | 27 / 22 | Trim + uppercase |
| Reg no integer / missing | 3 / 7 | Quarantine (never guessed) |
| `updatedAt` as `Z` / no timezone / epoch ms | 28 / 14 / 30 | Converted to UTC; no timezone = Malaysia time |
| `updatedAt` missing | 7 | Quarantine |
| `data` missing / a string | 5 / 4 | Quarantine (an empty upsert must not wipe a company) |
| `state` in 46 spellings | — | Matched to the 16 states |
| Description repeated end to end | 5 distinct | Collapsed to one copy |
| Empty / null / missing description | 15 / 29 / 14 | Stored as null; embedding uses fallback text |

## 5. Testing

- **Unit** (normalize, validation, embedding): every message shape found in the feed, as parametrized cases.
- **Feed contract**: the normalizers and validator over the real `data/feed.jsonl` reproduce the EDA counts exactly.
- **Integration** (real Postgres): schema constraints, newer-wins writes for every ordering case, consumer outcomes, a simulated database outage, and the neighbour cache compared with the live search after every one of 250 random changes (inserts, moves, strike-offs, reactivations, exact ties).
- **End-to-end**: the whole feed through the consumer gives 1,992 active + 8 struck-off companies and 32 quarantined messages; delivered twice in shuffled order, the portal-visible state is identical.
- **Full pipeline check** (`scripts/pipeline_check.py`): real producer → RabbitMQ → consumer → Postgres with the real model; SIGKILL mid-feed, restart, re-publish the whole feed. Also checks the cache against the live search and the API (200/410/404/422, never itself or struck-off, response time). 13 checks, all passing on my machine (API median 11.7 ms).
- **Model evaluation** (`scripts/evaluate_models.py`): see Deep dive.

In total: 141 automated tests (2 marked slow because they replay the whole feed) plus 2 opt-in tests against the real models.

## 6. At 10x volume

| Change | When it's needed | Why not now |
|---|---|---|
| Connection pool (`psycopg_pool`) in the API | Many concurrent requests | Requests take 11.7 ms median end to end; a pool adds a dependency and failure modes for no visible gain |
| HNSW index (`vector_cosine_ops`) + pgvector iterative scans for the status filter | The exact scan gets slow (roughly 100k+ companies) | At 2,000 rows the exact scan is ~1.4 ms and exact; HNSW is approximate and can't keep the tie-break order |
| Several consumers in parallel | Feed volume outgrows one consumer | The company write is atomic per row, but cache maintenance rewrites other companies' neighbour lists, so parallel consumers would race on them. They'd need one cache-maintenance worker (or an advisory lock around it). One consumer handles this feed in a minute or two |
| Skip re-embedding when the description is unchanged (compare with the stored `embedded_text`) | Many updates only touch the name or state | Embedding a sentence takes milliseconds at this volume; the stored text makes it a small change later |
| Batch embeddings (embed many messages per model call) | Embedding becomes the bottleneck | One sentence per message takes milliseconds at this volume |
| Cache maintenance without full scans (ANN for the rebuild step, or batched rebuilds) | Large catalogues: each rebuilt list is a scan of every company | At 2,000 companies a rebuild is 2.7 ms |
| Re-embedding job when the model changes | Switching models | Today a feed replay does it; `embedding_model` per row shows what's stale |

## 7. What's left

| Improvement | Benefit | Why it isn't implemented now |
|---|---|---|
| Learning from user clicks (below) | Recommendations that improve with real usage | Needs click data the take-home doesn't have, plus A/B testing|
| Per-field merge for partial upserts (missing key = no change, explicit `null` = cleared) | Correct if the registry starts sending partial updates | It depends on the registry's contract, not on volume: build it when the registry sends partial updates. It isn't cheaper per write (both find the row through the primary key, and Postgres writes a new copy of the whole row on every UPDATE); it costs a timestamp per field, a `CASE` per column and more tests. No partial updates exist in this feed |
| Reprocessing tool for `rejected_events` | Replay parked messages after a fix (raw bodies are kept for this) | No rejected message in this feed is recoverable; a manual SQL query is enough for 32 rows |
| Metrics (counts of applied / skipped / rejected, queue lag) instead of log lines | Alerting in production | Logs show the same numbers at this scale; needs a metrics stack |
| `just` recipes for the consumer and API | Shorter commands | The Justfile belongs to the template; commands are in the README instead |

### Learning from usage (my main next step as an ML engineer)

Embeddings capture what descriptions *say*; clicks capture what users actually find similar. The plan:

1. **Log impressions and clicks server-side**: (viewed company, shown companies with positions, clicked company). The client sends only these IDs. It never sends vectors, which anyone could forge to poison results.
2. **Re-rank with co-clicks**: blend the cosine score with how often users who viewed A clicked B, only above a minimum count, and corrected for **position bias** (the first card gets clicked more just for being first, so naive learning would reinforce whatever is already ranked first).
3. **Fine-tune the embedding model** later on clicked pairs (positives) and shown-but-skipped pairs (hard negatives), and ship it only if it wins an A/B test on click-through.

An earlier idea was to store the mean of the two companies' vectors after each click. A mean vector lies between the two companies, but no query uses it, so rankings wouldn't change; the steps above use the same signal in a way that does.

## 8. Deep dive: recommendation quality

**Question:** which embedding set-up gives the best "similar companies" for this mixed English and Malay feed?

**Method** (`scripts/evaluate_models.py`): build the final state of the feed with the consumer's own validation and newer-wins rule; take the 1,890 active companies with a known industry; rank neighbours exactly as the API does (cosine, then registration number, never itself); score **industry precision@5**, the share of a company's top 5 in its industry. Industry is a coarse, weak label (11 values), a proxy for "does similar things", not ground truth.

Many companies share an identical description, which any method matches trivially, so the second column excludes candidates with the same original description. That column measures whether a model understands meaning across wording and language.

| Set-up | P@5 | P@5, identical text excluded |
|---|---|---|
| Random ranking | 0.091 | 0.091 |
| Word hashing (baseline) | 0.988 | 0.363 |
| `bge-small-en-v1.5` (README suggestion) | 0.986 | 0.522 |
| **`paraphrase-multilingual-MiniLM-L12-v2`** | **0.996** | **0.862** |
| Multilingual, trailing locations stripped | 0.996 | 0.861 |
| Multilingual + pseudo-relevance feedback (re-query with the mean of the first top 5) | 0.997 | 0.857 |

**Findings**
- The multilingual model is clearly best: 86% vs 52% same-industry neighbours when exact copies are excluded. bge-small treats Malay largely as unknown tokens; in the README's own mock it placed a Malay logistics company next to a café. With the multilingual model, that slot holds coffee companies.
- Stripping locations ("in the Klang Valley", "di Penang") doesn't help the multilingual model, which already weights place names lightly; the location problem was specific to bge-small.
- Pseudo-relevance feedback doesn't help: the mean is built from neighbours already found, so it adds no new information.
- The last three rows differ by at most 0.005, within noise (about ±0.004 over 9,450 neighbour slots), so the simplest set-up wins.
- The first column can't separate the models (0.986–0.997) because identical descriptions dominate it; an evaluation without the second column would have hidden the difference.

**Caveats:** synthetic data, a weak label, and one language pair. With real users the next step is click-through evaluation (see Learning from usage).

## 9. AI tools used

I used **Claude** (Anthropic, through claude.ai) throughout, to draft the code and as a tutor to learn new concepts. Claude has been used to refine the README.md and DESIGN.md.

**What I did myself:**
- Ran the EDA and reconciled its numbers, including fixing the missing-key vs null miscount.
- Made the design decisions in the decision log. I accepted, rejected or changed each proposal: for example the reactivation policy, the 410 response and the model order.
- Challenged Claude's claim that a neighbour cache would threaten freshness, using how food-delivery recommenders work. The research that followed changed its advice, and I chose to build the cache against its original recommendation.
- Ensured the unit tests ran on the real feed, which led to the feed contract tests.
- Reviewed the code. I found the missed case/punctuation repeats and the needless allocations in `normalize.py`.
- Proposed the two ideas tested or designed here: the mean-of-neighbours query (evaluated: no gain) and learning from clicks (section 7).
- Ran every test, the pipeline check and the model evaluation on my machine.

## 10. Time spent

| Session | Work | Time |
|---|---|---|
| 21 Nov | EDA script and findings | 46 min |
| 22 Nov | Producer review, struck-off and API status decisions | 20 min |
| 23 Nov | Schema, index experiment, newer-wins repository | 60 min |
| 23 Nov | Normalization review fixes, feed contract tests, lint, EDA cleanup | 25 min |
| 23 Nov | Validation, embedding | ~20 min |
| 23 Nov | Model peek, consumer, end-to-end replay tests | ~25 min |
| 23 Nov | Pipeline check, API | ~35 min |
| 23 Nov | Neighbour cache, profiling | ~55 min |
| 23 Nov | Model evaluation, final docs | ~20 min |
| **Total** | | **~5 h 06 min** |


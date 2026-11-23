"""Full pipeline check: real producer -> RabbitMQ -> consumer -> Postgres, with a crash.

1. Publishes the feed with the template's producer.
2. Starts the consumer, then kills it with SIGKILL part-way through (a hard crash).
3. Restarts it and waits until the queue is drained.
4. Checks the end state: 1,992 active + 8 struck-off companies, 32 quarantined messages,
   every active company embedded with the configured model.
5. Publishes the whole feed again (every message duplicated) and checks that nothing the
   portal can see has changed.
6. Checks the neighbour cache equals the live search, then starts the API on that
   database and checks 200 / 410 / 404 / 422, that no result is the company itself or
   struck off, and the response time; prints the panel for SYN-001550, the company in
   the README's mock.

It writes to its own database (companies_pipeline_check) so your normal `companies` data
is untouched. Needs `just up` first; uses the same settings as the consumer.

Usage (from the repo root): uv run python -m scripts.pipeline_check
"""

import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import aio_pika
import psycopg

from similar_companies import neighbours
from similar_companies.config import load_settings
from similar_companies.consumer import QUEUE

CHECK_DB = "companies_pipeline_check"
KILL_AFTER_COMPANIES = 300  # crash the first consumer once this many companies are stored
EXPECTED_STATUS = {"active": 1992, "struck_off": 8}
EXPECTED_REJECTED = 32
LOG_DIR = Path(tempfile.gettempdir())
API_PORT = 8765  # not 8000, so it never clashes with an API you are running yourself
API = f"http://127.0.0.1:{API_PORT}"
MOCK_COMPANY = "SYN-001550"  # Sinar Teguh Bakery, the company in the README's mock panel

FINGERPRINT_SQL = """
SELECT md5(string_agg(
    registration_number || status || updated_at::text ||
    CASE WHEN status = 'active'
         THEN coalesce(company_name, '') || coalesce(nature_of_business, '') ||
              coalesce(state, '') || coalesce(industry, '') || coalesce(embedded_text, '')
         ELSE '' END,
    ',' ORDER BY registration_number))
FROM companies
"""


def queue_depth(rabbitmq_url: str) -> int:
    """Messages ready in the queue (passive declare: reads the queue, never changes it)."""

    async def depth() -> int:
        connection = await aio_pika.connect(rabbitmq_url)
        async with connection:
            channel = await connection.channel()
            queue = await channel.declare_queue(QUEUE, passive=True)
            return queue.declaration_result.message_count

    return asyncio.run(depth())


def progress(conn: psycopg.Connection) -> tuple:
    return conn.execute(
        "SELECT (SELECT count(*) FROM companies), (SELECT count(*) FROM rejected_events),"
        " (SELECT max(updated_at) FROM companies)"
    ).fetchone()


def wait_until_drained(conn: psycopg.Connection, rabbitmq_url: str, timeout: int = 900) -> None:
    """Wait until the queue is empty and the database has stopped changing for 3 seconds."""
    deadline = time.monotonic() + timeout
    last, stable_since = None, time.monotonic()
    while time.monotonic() < deadline:
        state = progress(conn)
        if queue_depth(rabbitmq_url) == 0 and state == last:
            if time.monotonic() - stable_since >= 3:
                return
        else:
            last, stable_since = state, time.monotonic()
        time.sleep(1)
    raise TimeoutError("queue did not drain in time")


def start_consumer(env: dict, log_name: str) -> subprocess.Popen:
    log = open(LOG_DIR / log_name, "w")  # left open for the child process; closed on exit
    return subprocess.Popen(
        [sys.executable, "-m", "similar_companies.consumer"], env=env, stdout=log, stderr=log
    )


def get(path: str) -> tuple[int, dict]:
    """GET from the API; returns (status code, JSON body) for success and errors alike."""
    try:
        with urlopen(API + path, timeout=10) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def start_api(env: dict) -> subprocess.Popen:
    log = open(LOG_DIR / "pipeline_check_api.log", "w")  # left open for the child process
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "similar_companies.api:app", "--port", str(API_PORT)],
        env=env,
        stdout=log,
        stderr=log,
    )
    for _ in range(60):
        try:
            if get("/health")[0] == 200:
                return api
        except URLError:
            pass  # not listening yet
        time.sleep(0.5)
    api.kill()
    sys.exit("API did not start; see pipeline_check_api.log")


def check_api(conn: psycopg.Connection, env: dict) -> list[tuple[str, bool, str]]:
    struck_off = {
        r[0]
        for r in conn.execute(
            "SELECT registration_number FROM companies WHERE status = 'struck_off'"
        )
    }
    active = [
        r[0]
        for r in conn.execute("SELECT registration_number FROM companies WHERE status = 'active'")
    ]
    api = start_api(env)
    try:
        status, body = get(f"/companies/{MOCK_COMPANY}/similar")
        cards = body.get("similar", [])
        numbers = [card["registrationNumber"] for card in cards]
        print(f"\n   Similar companies for {MOCK_COMPANY}:")
        for card in cards:
            print(
                f"   {card['similarity']:.3f}  {card['companyName']:<38}"
                f" {card['industry'] or '-':<16} {card['state'] or '-'}"
            )
        sample = random.Random(7).sample(active, 50)
        timings, clean = [], True
        for number in sample:
            started = time.perf_counter()
            _, result = get(f"/companies/{number}/similar")
            timings.append((time.perf_counter() - started) * 1000)
            returned = {card["registrationNumber"] for card in result["similar"]}
            clean = clean and number not in returned and not returned & struck_off
        p95 = statistics.quantiles(timings, n=20)[-1]
        cache_exact = neighbours.is_complete(conn) and all(
            neighbours.similar(conn, number, 5) == neighbours.live_similar(conn, number, 5)
            for number in sample
        )
        return [
            (
                "neighbour cache complete and equal to the live search (50 companies)",
                cache_exact,
                "",
            ),
            ("API 200 with 5 cards for the mock company", status == 200 and len(cards) == 5, ""),
            (
                "API never returns itself or struck-off (51 companies)",
                clean and not set(numbers) & struck_off,
                "",
            ),
            (
                "API 410 for a struck-off company",
                get(f"/companies/{min(struck_off)}/similar")[0] == 410,
                "",
            ),
            ("API 404 for an unknown company", get("/companies/SYN-999999/similar")[0] == 404, ""),
            ("API 422 for a malformed number", get("/companies/hello/similar")[0] == 422, ""),
            (
                "API response time",
                p95 < 200,
                f"median {statistics.median(timings):.1f} ms, p95 {p95:.1f} ms",
            ),
        ]
    finally:
        api.terminate()
        api.wait()


def publish_feed() -> None:
    subprocess.run([sys.executable, "-m", "producer.publish"], check=True)


def main() -> None:
    settings = load_settings()
    check_url = psycopg.conninfo.make_conninfo(settings.database_url, dbname=CHECK_DB)
    admin_url = psycopg.conninfo.make_conninfo(settings.database_url, dbname="postgres")
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {CHECK_DB} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {CHECK_DB}")
    env = {**os.environ, "DATABASE_URL": check_url}
    results: list[tuple[str, bool, str]] = []

    print(f"1. queue has {queue_depth(settings.rabbitmq_url)} messages; publishing the feed")
    publish_feed()

    print(f"2. starting consumer; will kill -9 it after {KILL_AFTER_COMPANIES} companies")
    consumer = start_consumer(env, "pipeline_check_consumer_1.log")
    while True:
        time.sleep(0.2)
        try:
            with psycopg.connect(check_url) as conn:
                if progress(conn)[0] >= KILL_AFTER_COMPANIES:
                    break
        except psycopg.errors.UndefinedTable:
            pass  # the consumer hasn't created the tables yet
        if consumer.poll() is not None:
            sys.exit("consumer exited early; see pipeline_check_consumer_1.log")
    consumer.kill()  # SIGKILL: no clean shutdown, unacked messages go back to the queue
    consumer.wait()
    with psycopg.connect(check_url) as conn:
        stored = progress(conn)[0]
    print(f"   killed with {stored} companies stored, {queue_depth(settings.rabbitmq_url)} queued")

    print("3. restarting consumer and waiting for the queue to drain")
    consumer = start_consumer(env, "pipeline_check_consumer_2.log")
    try:
        with psycopg.connect(check_url, autocommit=True) as conn:
            wait_until_drained(conn, settings.rabbitmq_url)
            status = dict(
                conn.execute("SELECT status, count(*) FROM companies GROUP BY 1").fetchall()
            )
            rejected = conn.execute("SELECT count(*) FROM rejected_events").fetchone()[0]
            unembedded = conn.execute(
                "SELECT count(*) FROM companies WHERE status = 'active' AND embedding IS NULL"
            ).fetchone()[0]
            models = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT embedding_model FROM companies"
                    " WHERE embedding_model IS NOT NULL"
                )
            ]
            results += [
                ("companies by status", status == EXPECTED_STATUS, str(status)),
                ("quarantined messages", rejected == EXPECTED_REJECTED, str(rejected)),
                ("active without embedding", unembedded == 0, str(unembedded)),
                ("embedding model", models == [settings.embedding_model], str(models)),
            ]
            before = conn.execute(FINGERPRINT_SQL).fetchone()[0]

            print("4. publishing the whole feed again (every message is now a duplicate)")
            publish_feed()
            wait_until_drained(conn, settings.rabbitmq_url)
            after = conn.execute(FINGERPRINT_SQL).fetchone()[0]
            rejected_after = conn.execute("SELECT count(*) FROM rejected_events").fetchone()[0]
            results += [
                ("visible state unchanged after re-publish", before == after, ""),
                ("quarantine unchanged after re-publish", rejected_after == rejected, ""),
            ]

            print("5. starting the API on the same database")
            results += check_api(conn, env)
    finally:
        consumer.terminate()
        consumer.wait()

    print("\nResults")
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}  {detail}")
    print(f"\nConsumer logs: {LOG_DIR / 'pipeline_check_consumer_*.log'}")
    print(f"Inspect or drop the check database: {CHECK_DB}")
    sys.exit(0 if all(passed for _, passed, _ in results) else 1)


if __name__ == "__main__":
    main()

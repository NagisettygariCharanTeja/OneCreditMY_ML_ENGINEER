"""Integration tests: the neighbour cache always gives exactly the live search's answer.

The main test runs a long random sequence of inserts, description changes (vector moves),
strike-offs and reactivations, with duplicated vectors to create ties, and after every
single operation compares the cached top 20 of every active company with the live search.
"""

import random
from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from similar_companies import neighbours
from similar_companies.neighbours import CACHE_SIZE, live_similar, similar
from similar_companies.repository import CompanyUpsert, mark_struck_off, upsert_company

START = datetime(2026, 8, 14, tzinfo=UTC)


@pytest.fixture
def conn(test_db_url):
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        conn.execute("TRUNCATE companies, rejected_events, company_neighbours")
        yield conn
        conn.execute("TRUNCATE companies, rejected_events, company_neighbours")


class Registry:
    """Applies writes the way the consumer does: newer-wins write, then cache refresh."""

    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn
        self.clock = START

    def tick(self) -> datetime:
        self.clock += timedelta(seconds=1)
        return self.clock

    def upsert(self, number: str, vector: list[float]) -> None:
        company = CompanyUpsert(
            registration_number=number,
            updated_at=self.tick(),
            company_name=f"Company {number}",
            nature_of_business="x",
            state=None,
            industry=None,
            embedded_text="x",
            embedding=vector,
            embedding_model="test",
        )
        if upsert_company(self.conn, company):
            neighbours.refresh_after_change(self.conn, number)

    def strike_off(self, number: str) -> None:
        if mark_struck_off(self.conn, number, self.tick()):
            neighbours.refresh_after_change(self.conn, number)


def cached_only(conn: psycopg.Connection, number: str) -> list:
    rows = conn.execute(neighbours.CACHED, {"reg": number, "limit": CACHE_SIZE}).fetchall()
    return [neighbours.Neighbour(*row) for row in rows]


def active_numbers(conn: psycopg.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute("SELECT registration_number FROM companies WHERE status = 'active'")
    ]


def assert_cache_exact(conn: psycopg.Connection) -> None:
    for number in active_numbers(conn):
        assert cached_only(conn, number) == live_similar(conn, number, CACHE_SIZE), number


def random_vector(rng: random.Random) -> list[float]:
    values = [rng.gauss(0, 1) for _ in range(8)] + [0.0] * 376
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


def test_cache_stays_exact_through_random_changes(conn):
    rng = random.Random(1)
    registry = Registry(conn)
    shared = [random_vector(rng) for _ in range(5)]  # reused vectors create exact ties
    numbers = [f"SYN-{i:06d}" for i in range(1, 61)]
    struck: set[str] = set()
    for _ in range(250):
        number = rng.choice(numbers)
        action = rng.random()
        if action < 0.15 and number not in struck:
            registry.strike_off(number)
            struck.add(number)
        else:
            vector = rng.choice(shared) if rng.random() < 0.4 else random_vector(rng)
            registry.upsert(number, vector)  # insert, move, or reactivate
            struck.discard(number)
        assert_cache_exact(conn)


def test_rebuild_all_matches_incremental_maintenance(conn):
    rng = random.Random(2)
    registry = Registry(conn)
    for i in range(1, 41):
        registry.upsert(f"SYN-{i:06d}", random_vector(rng))
    registry.strike_off("SYN-000007")
    incremental = conn.execute("SELECT * FROM company_neighbours ORDER BY 1, 2").fetchall()
    neighbours.rebuild_all(conn)
    assert conn.execute("SELECT * FROM company_neighbours ORDER BY 1, 2").fetchall() == incremental


def test_struck_off_company_has_no_list_and_appears_in_none(conn):
    rng = random.Random(3)
    registry = Registry(conn)
    for i in range(1, 31):
        registry.upsert(f"SYN-{i:06d}", random_vector(rng))
    registry.strike_off("SYN-000005")
    count = conn.execute(
        "SELECT count(*) FROM company_neighbours"
        " WHERE registration_number = 'SYN-000005' OR neighbour = 'SYN-000005'"
    ).fetchone()[0]
    assert count == 0


def test_is_complete_detects_a_missing_cache(conn):
    rng = random.Random(4)
    registry = Registry(conn)
    for i in range(1, 11):
        registry.upsert(f"SYN-{i:06d}", random_vector(rng))
    assert neighbours.is_complete(conn)
    conn.execute("TRUNCATE company_neighbours")
    assert not neighbours.is_complete(conn)
    neighbours.rebuild_all(conn)
    assert neighbours.is_complete(conn)


def test_similar_falls_back_to_live_when_limit_exceeds_cache(conn):
    rng = random.Random(5)
    registry = Registry(conn)
    for i in range(1, 41):
        registry.upsert(f"SYN-{i:06d}", random_vector(rng))
    result = similar(conn, "SYN-000001", 30)
    assert len(result) == 30
    assert result == live_similar(conn, "SYN-000001", 30)


def test_similar_falls_back_to_live_when_cache_is_empty(conn):
    rng = random.Random(6)
    registry = Registry(conn)
    for i in range(1, 11):
        registry.upsert(f"SYN-{i:06d}", random_vector(rng))
    conn.execute("TRUNCATE company_neighbours")
    assert similar(conn, "SYN-000001", 5) == live_similar(conn, "SYN-000001", 5)

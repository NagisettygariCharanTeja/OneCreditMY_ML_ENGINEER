"""Integration tests for the consumer's message handling, against real Postgres.
"""

import hashlib
import json
import random

import psycopg
import pytest

from similar_companies.consumer import (
    APPLIED,
    PROCESSING_ERROR,
    REJECTED,
    SKIPPED,
    TRANSIENT_ERRORS,
    MessageHandler,
)
from similar_companies.embedding import HashingEmbedder
from tests.test_feed_contract import FEED_PATH


class CountingEmbedder(HashingEmbedder):
    """The hashing placeholder, counting how many texts it was asked to embed."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return super().embed(texts)


class BrokenEmbedder(HashingEmbedder):
    def embed(self, texts):
        raise RuntimeError("model crashed")


@pytest.fixture
def handler(test_db_url):
    handler = MessageHandler(test_db_url, CountingEmbedder())
    handler.conn.execute("TRUNCATE companies, rejected_events")
    yield handler
    handler.reconnect()  # the test may have closed it
    handler.conn.execute("TRUNCATE companies, rejected_events")  # leave nothing for other tests
    handler.conn.close()


def upsert(updated_at: str, name: str = "Sinar Teguh Bakery Sdn. Bhd.") -> bytes:
    return json.dumps(
        {
            "type": "company-upserted",
            "registrationNumber": "SYN-001550",
            "updatedAt": updated_at,
            "data": {"companyName": name, "natureOfBusiness": "Operating cafes"},
        }
    ).encode()


def struck_off(updated_at: str) -> bytes:
    return json.dumps(
        {"type": "company-struck-off", "registrationNumber": "SYN-001550", "updatedAt": updated_at}
    ).encode()


def row(handler) -> tuple:
    return handler.conn.execute(
        "SELECT status, company_name, embedding IS NOT NULL FROM companies"
    ).fetchone()


def test_valid_upsert_is_applied_with_an_embedding(handler):
    assert handler.handle(upsert("2026-08-14T10:00:00+08:00"), "m1") == APPLIED
    assert row(handler) == ("active", "Sinar Teguh Bakery Sdn. Bhd.", True)


def test_duplicate_is_skipped_without_embedding_again(handler):
    handler.handle(upsert("2026-08-14T10:00:00+08:00"), "m1")
    assert handler.handle(upsert("2026-08-14T10:00:00+08:00"), "m1") == SKIPPED
    assert handler.embedder.calls == 1


def test_stale_update_is_skipped(handler):
    handler.handle(upsert("2026-08-14T12:00:00+08:00", name="New"), "m1")
    assert handler.handle(upsert("2026-08-14T10:00:00+08:00", name="Old"), "m2") == SKIPPED
    assert row(handler)[1] == "New"


def test_struck_off_then_reactivated(handler):
    handler.handle(upsert("2026-08-14T10:00:00+08:00"), "m1")
    assert handler.handle(struck_off("2026-08-14T14:00:00+08:00"), "m2") == APPLIED
    assert row(handler)[0] == "struck_off"
    assert handler.handle(upsert("2026-08-14T15:00:00+08:00"), "m3") == APPLIED
    assert row(handler)[0] == "active"


def test_bad_message_is_quarantined_with_its_reason(handler):
    assert (
        handler.handle(b'{"type": "company-upserted", "data": {"companyName": "Bes', "m1")
        == REJECTED
    )
    stored = handler.conn.execute(
        "SELECT message_id, reason, raw_body FROM rejected_events"
    ).fetchone()
    assert stored == (
        "m1",
        "invalid_json",
        '{"type": "company-upserted", "data": {"companyName": "Bes',
    )


def test_unexpected_error_is_quarantined_not_raised(handler):
    handler.embedder = BrokenEmbedder()
    assert handler.handle(upsert("2026-08-14T10:00:00+08:00"), "m1") == REJECTED
    reason = handler.conn.execute("SELECT reason FROM rejected_events").fetchone()[0]
    assert reason == PROCESSING_ERROR
    assert handler.conn.execute("SELECT count(*) FROM companies").fetchone()[0] == 0  # rolled back


def test_database_outage_raises_transient_error_then_recovers(handler):
    handler.conn.close()  # simulate the database going away
    with pytest.raises(TRANSIENT_ERRORS):
        handler.handle(upsert("2026-08-14T10:00:00+08:00"), "m1")
    handler.reconnect()
    assert handler.handle(upsert("2026-08-14T10:00:00+08:00"), "m1") == APPLIED


# ---- Replaying the whole real feed ------------------------------------------------------


def feed_messages() -> list[tuple[bytes, str]]:
    """Each feed line as the producer publishes it: body + SHA-256 message_id."""
    lines = FEED_PATH.read_text(encoding="utf-8").splitlines()
    return [(line.encode(), hashlib.sha256(line.encode()).hexdigest()) for line in lines]


def visible_state(conn: psycopg.Connection) -> tuple[list[tuple], ...]:
    """Everything the portal can ever see, plus the quarantine.

    Active companies in full. Struck-off companies only as status and time: they are never
    shown, and the card data a tombstone keeps depends on whether the strike-off arrived
    before or after the company's data (see WORKLOG, struck-off arriving first).
    """
    active = conn.execute(
        "SELECT registration_number, updated_at, company_name, nature_of_business, state,"
        " industry, embedded_text FROM companies WHERE status = 'active' ORDER BY 1"
    ).fetchall()
    struck_off = conn.execute(
        "SELECT registration_number, updated_at FROM companies"
        " WHERE status = 'struck_off' ORDER BY 1"
    ).fetchall()
    rejected = conn.execute("SELECT message_id, reason FROM rejected_events ORDER BY 1").fetchall()
    return active, struck_off, rejected


def test_replaying_the_feed_gives_the_expected_end_state(handler):
    for body, message_id in feed_messages():
        handler.handle(body, message_id)
    counts = dict(
        handler.conn.execute("SELECT status, count(*) FROM companies GROUP BY status").fetchall()
    )
    assert counts == {"active": 1992, "struck_off": 8}  # all 2,000 companies, none lost
    assert handler.conn.execute("SELECT count(*) FROM rejected_events").fetchone()[0] == 32
    missing_embeddings = handler.conn.execute(
        "SELECT count(*) FROM companies WHERE status = 'active' AND embedding IS NULL"
    ).fetchone()[0]
    assert missing_embeddings == 0


def test_visible_state_does_not_depend_on_order_or_duplicates(handler):
    messages = feed_messages()
    for body, message_id in messages:
        handler.handle(body, message_id)
    in_order = visible_state(handler.conn)

    handler.conn.execute("TRUNCATE companies, rejected_events")
    scrambled = messages * 2  # every message delivered twice...
    random.Random(42).shuffle(scrambled)  # ...in a random (but repeatable) order
    for body, message_id in scrambled:
        handler.handle(body, message_id)

    assert visible_state(handler.conn) == in_order

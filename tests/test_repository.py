"""Integration tests: newer-wins writes behave correctly for every ordering case."""

from dataclasses import replace
from datetime import UTC, datetime

from similar_companies.repository import (
    CompanyUpsert,
    mark_struck_off,
    save_rejected,
    stored_updated_at,
    upsert_company,
)

REG_NO = "SYN-001550"
T10, T12, T14, T15 = (datetime(2026, 8, 14, h, tzinfo=UTC) for h in (10, 12, 14, 15))


def make_company(updated_at: datetime, name: str = "Sinar Teguh Bakery Sdn. Bhd.") -> CompanyUpsert:
    return CompanyUpsert(
        registration_number=REG_NO,
        updated_at=updated_at,
        company_name=name,
        nature_of_business="Operating cafes and coffee shops",
        state="Selangor",
        industry="Food & Beverage",
        embedded_text="Operating cafes and coffee shops",
        embedding=[0.1] * 384,
        embedding_model="test-model",
    )


def stored(db) -> tuple:
    return db.execute(
        "SELECT status, updated_at, company_name FROM companies WHERE registration_number = %s",
        (REG_NO,),
    ).fetchone()


def test_first_upsert_inserts(db):
    assert upsert_company(db, make_company(T10)) is True
    assert stored(db) == ("active", T10, "Sinar Teguh Bakery Sdn. Bhd.")


def test_newer_upsert_overwrites(db):
    upsert_company(db, make_company(T10, name="Old Name"))
    assert upsert_company(db, make_company(T12, name="New Name")) is True
    assert stored(db) == ("active", T12, "New Name")


def test_older_upsert_arriving_late_is_ignored(db):
    upsert_company(db, make_company(T12, name="New Name"))
    assert upsert_company(db, make_company(T10, name="Old Name")) is False
    assert stored(db) == ("active", T12, "New Name")


def test_duplicate_with_same_timestamp_is_ignored(db):
    upsert_company(db, make_company(T10))
    assert upsert_company(db, make_company(T10)) is False


def test_struck_off_keeps_card_fields_and_advances_time(db):
    upsert_company(db, make_company(T10))
    assert mark_struck_off(db, REG_NO, T14) is True
    assert stored(db) == ("struck_off", T14, "Sinar Teguh Bakery Sdn. Bhd.")


def test_stale_upsert_cannot_revive_struck_off_company(db):
    # upsert 10:00, struck off 14:00, then a late upsert from 12:00 arrives
    upsert_company(db, make_company(T10))
    mark_struck_off(db, REG_NO, T14)
    assert upsert_company(db, make_company(T12)) is False
    assert stored(db)[0] == "struck_off"


def test_newer_upsert_after_struck_off_reactivates(db):
    # policy: the registry is the source of truth; a newer upsert means the company is active again
    upsert_company(db, make_company(T10))
    mark_struck_off(db, REG_NO, T14)
    assert upsert_company(db, make_company(T15)) is True
    assert stored(db)[0] == "active"


def test_struck_off_before_any_upsert_creates_tombstone(db):
    assert mark_struck_off(db, REG_NO, T14) is True
    assert stored(db) == ("struck_off", T14, None)
    assert upsert_company(db, make_company(T12)) is False  # older data can't resurrect it


def test_stored_updated_at(db):
    assert stored_updated_at(db, REG_NO) is None
    upsert_company(db, make_company(T10))
    assert stored_updated_at(db, REG_NO) == T10


def test_rejected_message_saved_once(db):
    save_rejected(db, "m1", "null", "not_an_object")
    save_rejected(db, "m1", "null", "not_an_object")
    assert db.execute("SELECT count(*) FROM rejected_events").fetchone()[0] == 1


def test_company_upsert_is_immutable():
    company = make_company(T10)
    assert replace(company, company_name="X").company_name == "X"
    assert company.company_name == "Sinar Teguh Bakery Sdn. Bhd."

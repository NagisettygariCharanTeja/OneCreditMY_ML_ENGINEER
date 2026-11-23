"""Integration tests: the database schema enforces our data rules on its own."""

import psycopg
import pytest

from tests.conftest import SCHEMA_PATH

ZERO_VECTOR = "[" + ",".join(["0"] * 384) + "]"


def test_schema_can_be_applied_twice(db):
    db.execute(SCHEMA_PATH.read_text())  # every statement is IF NOT EXISTS


def test_active_company_with_name_and_embedding_is_accepted(db):
    db.execute(
        "INSERT INTO companies (registration_number, status, updated_at, company_name, embedding)"
        " VALUES ('SYN-000001', 'active', now(), 'A Sdn. Bhd.', %s::vector)",
        (ZERO_VECTOR,),
    )


def test_active_company_without_embedding_is_refused(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "INSERT INTO companies (registration_number, status, updated_at, company_name)"
            " VALUES ('SYN-000001', 'active', now(), 'A Sdn. Bhd.')"
        )


def test_struck_off_tombstone_needs_no_data(db):
    db.execute(
        "INSERT INTO companies (registration_number, status, updated_at)"
        " VALUES ('SYN-000002', 'struck_off', now())"
    )


def test_unknown_status_is_refused(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "INSERT INTO companies (registration_number, status, updated_at)"
            " VALUES ('SYN-000003', 'deleted', now())"
        )


def test_rejected_body_may_be_invalid_json(db):
    db.execute(
        "INSERT INTO rejected_events (message_id, raw_body, reason) VALUES (%s, %s, %s)",
        ("m1", '{"type": "company-upserted", "data": {"companyName": "Bes', "invalid_json"),
    )


def test_same_rejected_message_is_stored_once(db):
    insert = (
        "INSERT INTO rejected_events (message_id, raw_body, reason)"
        " VALUES ('m2', 'null', 'not_an_object')"
        " ON CONFLICT (message_id) DO NOTHING"
    )
    db.execute(insert)
    db.execute(insert)
    assert (
        db.execute("SELECT count(*) FROM rejected_events WHERE message_id = 'm2'").fetchone()[0]
        == 1
    )

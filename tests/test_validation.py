"""Unit tests for validation.py: every message shape found in the feed, plus edge cases."""

import json
from datetime import UTC, datetime

import pytest

from similar_companies.validation import (
    INVALID_DATA,
    INVALID_ENCODING,
    INVALID_JSON,
    INVALID_REGISTRATION_NUMBER,
    INVALID_UPDATED_AT,
    NOT_AN_OBJECT,
    UNKNOWN_TYPE,
    Rejection,
    StruckOffEvent,
    UpsertEvent,
    parse_event,
)

VALID_DATA = {
    "companyName": "Sinar Teguh Bakery Sdn. Bhd.",
    "natureOfBusiness": "Operating cafes and coffee shops in the Klang Valley",
    "state": "Selangor",
    "industry": "Food & Beverage",
}


def message(**overrides) -> str:
    """A valid upsert message as JSON text, with fields overridden (None removes a field)."""
    base = {
        "type": "company-upserted",
        "registrationNumber": "SYN-001550",
        "updatedAt": "2026-08-14T12:45:52+08:00",
        "data": VALID_DATA,
    }
    base.update(overrides)
    return json.dumps({key: value for key, value in base.items() if value is not None})


def test_valid_upsert_is_normalized():
    event = parse_event(
        message(
            registrationNumber=" syn-001550 ",
            updatedAt="2026-08-14T04:45:52Z",
            data={**VALID_DATA, "state": "SELANGOR", "natureOfBusiness": "Kedai Kopi Kedai Kopi"},
        )
    )
    assert event == UpsertEvent(
        registration_number="SYN-001550",
        updated_at=datetime(2026, 8, 14, 4, 45, 52, tzinfo=UTC),
        company_name="Sinar Teguh Bakery Sdn. Bhd.",
        nature_of_business="Kedai Kopi",
        state="Selangor",
        industry="Food & Beverage",
    )


def test_bytes_body_is_accepted():
    assert isinstance(parse_event(message().encode("utf-8")), UpsertEvent)


def test_extra_metadata_keys_are_ignored():
    assert isinstance(parse_event(message(source="partner-api", ingestedBy="x")), UpsertEvent)


def test_optional_fields_may_be_null_or_missing():
    event = parse_event(message(data={"companyName": "A Sdn. Bhd.", "natureOfBusiness": None}))
    assert event.nature_of_business is None
    assert event.state is None
    assert event.industry is None


def test_struck_off_needs_no_data():
    event = parse_event(
        json.dumps(
            {
                "type": "company-struck-off",
                "registrationNumber": "SYN-000186",
                "updatedAt": "2026-08-14T16:00:00+08:00",
            }
        )
    )
    assert event == StruckOffEvent("SYN-000186", datetime(2026, 8, 14, 8, 0, tzinfo=UTC))


def test_struck_off_ignores_any_data_it_carries():
    event = parse_event(message(type="company-struck-off", data="anything"))
    assert isinstance(event, StruckOffEvent)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (b"\xff\xfe not utf-8", INVALID_ENCODING),
        (
            '{"type": "company-upserted", "data": {"companyName": "Bes',  # truncated, like 5 lines
            INVALID_JSON,
        ),
        ("", INVALID_JSON),
        ("null", NOT_AN_OBJECT),  # 6 lines in the feed
        ("[1, 2]", NOT_AN_OBJECT),
        (message(type="company-deleted"), UNKNOWN_TYPE),  # never guessed as a strike-off
        (message(type=None), UNKNOWN_TYPE),
        (message(registrationNumber=119), INVALID_REGISTRATION_NUMBER),  # the "X" messages
        (message(registrationNumber=None), INVALID_REGISTRATION_NUMBER),  # "No Reg Sdn. Bhd."
        (message(registrationNumber="SYN-1550"), INVALID_REGISTRATION_NUMBER),
        (message(updatedAt=None), INVALID_UPDATED_AT),  # the "Unknown" messages
        (message(updatedAt="yesterday"), INVALID_UPDATED_AT),
        (message(updatedAt=True), INVALID_UPDATED_AT),
        (message(data=None), INVALID_DATA),  # upsert without data must not wipe a company
        (message(data="companyName=Foo; state=Perak"), INVALID_DATA),
        (message(data={"natureOfBusiness": "Cafes"}), INVALID_DATA),  # no company name
        (message(data={**VALID_DATA, "companyName": "   "}), INVALID_DATA),
    ],
)
def test_rejections(body, reason):
    result = parse_event(body)
    assert isinstance(result, Rejection)
    assert result.reason == reason
    assert result.detail  # every rejection explains itself for the logs


def test_first_failing_check_wins():
    # bad registration number AND missing updatedAt: reported as the earlier check
    result = parse_event(message(registrationNumber=None, updatedAt=None))
    assert result.reason == INVALID_REGISTRATION_NUMBER

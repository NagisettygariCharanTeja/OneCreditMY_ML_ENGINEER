"""Turn one raw queue message into a clean event, or a rejection with a reason code.

Checks run in a fixed order and stop at the first failure, so every bad message gets
exactly one reason: encoding -> JSON -> object -> type -> registration number ->
updatedAt -> data. Nothing is guessed: an unknown event type or an integer registration
number is rejected, not interpreted.
"""

import json
from dataclasses import dataclass
from datetime import datetime

from similar_companies.normalize import (
    clean_description,
    clean_text,
    normalize_registration_number,
    normalize_state,
    normalize_updated_at,
)

UPSERTED = "company-upserted"
STRUCK_OFF = "company-struck-off"

# Rejection reason codes, stored in rejected_events.reason.
INVALID_ENCODING = "invalid_encoding"
INVALID_JSON = "invalid_json"
NOT_AN_OBJECT = "not_an_object"
UNKNOWN_TYPE = "unknown_type"
INVALID_REGISTRATION_NUMBER = "invalid_registration_number"
INVALID_UPDATED_AT = "invalid_updated_at"
INVALID_DATA = "invalid_data"


@dataclass(frozen=True)
class UpsertEvent:
    """A valid company-upserted message, normalized. The full current state of a company."""

    registration_number: str
    updated_at: datetime  # UTC
    company_name: str
    nature_of_business: str | None
    state: str | None
    industry: str | None


@dataclass(frozen=True)
class StruckOffEvent:
    """A valid company-struck-off message. Any data it carries is ignored."""

    registration_number: str
    updated_at: datetime  # UTC


@dataclass(frozen=True)
class Rejection:
    """Why a message can't be used. `detail` is for logs; `reason` is the stored code."""

    reason: str
    detail: str


Event = UpsertEvent | StruckOffEvent


def parse_event(body: bytes | str) -> Event | Rejection:
    """Validate and normalize one raw message body."""
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8")
        except UnicodeDecodeError as error:
            return Rejection(INVALID_ENCODING, str(error))

    try:
        message = json.loads(body)
    except json.JSONDecodeError as error:
        return Rejection(INVALID_JSON, error.msg)
    if not isinstance(message, dict):
        return Rejection(NOT_AN_OBJECT, f"got {type(message).__name__}")

    event_type = message.get("type")
    if event_type not in (UPSERTED, STRUCK_OFF):
        return Rejection(UNKNOWN_TYPE, f"type={event_type!r}")

    registration_number = normalize_registration_number(message.get("registrationNumber"))
    if registration_number is None:
        return Rejection(
            INVALID_REGISTRATION_NUMBER, f"registrationNumber={message.get('registrationNumber')!r}"
        )

    updated_at = normalize_updated_at(message.get("updatedAt"))
    if updated_at is None:
        return Rejection(INVALID_UPDATED_AT, f"updatedAt={message.get('updatedAt')!r}")

    if event_type == STRUCK_OFF:
        return StruckOffEvent(registration_number, updated_at)

    return _parse_upsert(registration_number, updated_at, message.get("data"))


def _parse_upsert(
    registration_number: str, updated_at: datetime, data: object
) -> UpsertEvent | Rejection:
    """An upsert must carry a data object with a company name; everything else is optional."""
    if not isinstance(data, dict):
        return Rejection(INVALID_DATA, f"data is {type(data).__name__}")
    company_name = clean_text(data.get("companyName"))
    if company_name is None:
        return Rejection(INVALID_DATA, "companyName is missing or empty")
    return UpsertEvent(
        registration_number=registration_number,
        updated_at=updated_at,
        company_name=company_name,
        nature_of_business=clean_description(data.get("natureOfBusiness")),
        state=normalize_state(data.get("state")),
        industry=clean_text(data.get("industry")),
    )

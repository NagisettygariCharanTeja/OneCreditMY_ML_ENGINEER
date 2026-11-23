"""Contract tests: run the normalizers over the real data/feed.jsonl.

Unit tests (test_normalize.py) pin each rule with small hand-picked inputs. These tests
check the rules against the whole real feed and assert the counts found during EDA, so
if the feed or a rule changes, the difference shows up as a failing number.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from similar_companies.normalize import (
    clean_description,
    normalize_registration_number,
    normalize_state,
    normalize_updated_at,
)

FEED_PATH = Path(__file__).resolve().parent.parent / "data" / "feed.jsonl"


@pytest.fixture(scope="module")
def events() -> list[dict]:
    """Every line of the feed that parses to a JSON object."""
    parsed = []
    for line in FEED_PATH.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            parsed.append(event)
    return parsed


@pytest.fixture(scope="module")
def records(events) -> list[dict]:
    """The `data` objects of all events that have one."""
    return [e["data"] for e in events if isinstance(e.get("data"), dict)]


def test_feed_has_expected_number_of_objects(events):
    assert len(events) == 2507  # 2,518 lines - 5 truncated - 6 literal null


def test_registration_numbers(events):
    normalized = [normalize_registration_number(e.get("registrationNumber")) for e in events]
    assert sum(r is not None for r in normalized) == 2497  # 2,507 - 7 missing - 3 integers
    assert len(set(normalized) - {None}) == 2000


def test_every_present_timestamp_parses(events):
    present = [e["updatedAt"] for e in events if "updatedAt" in e]
    assert len(present) == 2500
    assert all(normalize_updated_at(value) is not None for value in present)


def test_states_map_to_the_16_known_states(records):
    raw_states = [r["state"] for r in records if isinstance(r.get("state"), str)]
    assert all(normalize_state(value) is not None for value in raw_states)
    assert len({normalize_state(value) for value in raw_states}) == 16


def test_descriptions(records):
    outcomes = Counter()
    for record in records:
        raw = record.get("natureOfBusiness")
        cleaned = clean_description(raw)
        if cleaned is None:
            outcomes["none"] += 1
        elif cleaned != raw:
            outcomes["changed"] += 1
    assert outcomes["none"] == 58  # 29 null + 14 missing key + 15 empty
    assert outcomes["changed"] == 7  # the messages carrying repeated-phrase descriptions
    assert max(len(clean_description(r.get("natureOfBusiness")) or "") for r in records) < 200

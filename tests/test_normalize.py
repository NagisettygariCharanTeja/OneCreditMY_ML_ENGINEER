"""Unit tests for normalize.py. Inputs are real shapes found in the feed during EDA."""

from datetime import UTC, datetime

import pytest

from similar_companies.normalize import (
    clean_description,
    clean_text,
    normalize_registration_number,
    normalize_state,
    normalize_updated_at,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SYN-001550", "SYN-001550"),
        (" SYN-001497 ", "SYN-001497"),  # padded (27 in the feed)
        ("syn-000587", "SYN-000587"),  # lowercase (22 in the feed)
        (119, None),  # integer: could be any company, so never guessed (3 in the feed)
        (None, None),  # missing (7 in the feed)
        ("", None),
        ("SYN-1550", None),  # wrong number of digits
        ("ABC-001550", None),
        ("SYN-001550-X", None),
    ],
)
def test_normalize_registration_number(raw, expected):
    assert normalize_registration_number(raw) == expected


SAME_MOMENT = datetime(2026, 8, 14, 1, 35, 39, tzinfo=UTC)  # 09:35:39 in Malaysia


@pytest.mark.parametrize(
    "raw",
    [
        "2026-08-14T09:35:39+08:00",  # offset (2,428 in the feed)
        "2026-08-14T01:35:39Z",  # UTC 'Z' (28)
        "2026-08-14T09:35:39",  # no timezone -> assumed Malaysia time (14)
        1786671339000,  # epoch milliseconds (30)
        " 2026-08-14T09:35:39+08:00 ",
    ],
)
def test_every_timestamp_format_gives_the_same_utc_instant(raw):
    assert normalize_updated_at(raw) == SAME_MOMENT


def test_normalized_timestamps_compare_correctly_across_formats():
    # As raw text, "...T09:34:39+08:00" sorts after "...T01:35:39Z", but it is one minute earlier.
    earlier = normalize_updated_at("2026-08-14T09:34:39+08:00")
    later = normalize_updated_at("2026-08-14T01:35:39Z")
    assert earlier < later


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "yesterday",
        "2026-13-45T00:00:00",
        True,
        3.5,
        [],
        {},
        10**20,
        0,
        "1999-12-31T23:59:59Z",
    ],
)
def test_unusable_timestamps_return_none(raw):
    assert normalize_updated_at(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Sinar   Teguh Bakery  ", "Sinar Teguh Bakery"),
        ("", None),
        ("   ", None),
        (None, None),
        (42, None),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Selangor", "Selangor"),
        ("SELANGOR", "Selangor"),
        ("kuala lumpur", "Kuala Lumpur"),
        ("  Pulau   Pinang ", "Pulau Pinang"),
        ("negeri sembilan", "Negeri Sembilan"),
        ("Atlantis", None),  # not a Malaysian state: left empty, not invented
        (None, None),
        ("", None),
    ],
)
def test_normalize_state(raw, expected):
    assert normalize_state(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Operating cafes and coffee shops", "Operating cafes and coffee shops"),
        (
            "Hotel butik di Seremban " * 3,
            "Hotel butik di Seremban",
        ),  # repeated phrase (5 in the feed)
        (
            "Perunding pengurusan untuk PKS Perunding pengurusan untuk PKS",
            "Perunding pengurusan untuk PKS",
        ),
        ("very very good service", "very very good service"),  # partial repeat is kept
        ("Bakery Bakery", "Bakery"),
        ("  Logistics   and   courier  ", "Logistics and courier"),
        ("Kedai Kopi kedai kopi", "Kedai Kopi"),  # repeat with different casing
        ("Kedai Kopi, Kedai Kopi", "Kedai Kopi"),  # repeat with a separating comma
        ("Kedai Kopi, kedai kopi.", "Kedai Kopi"),  # both, plus a full stop
        ("Kopi , Kopi ,", "Kopi"),  # punctuation as separate tokens
        (
            "Cafes, bakeries and catering.",
            "Cafes, bakeries and catering.",
        ),  # no repeat: punctuation kept
        ("", None),  # 15 empty in the feed
        (None, None),  # 29 null in the feed
    ],
)
def test_clean_description(raw, expected):
    assert clean_description(raw) == expected

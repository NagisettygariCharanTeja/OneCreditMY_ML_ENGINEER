"""Profile the registry feed (data/feed.jsonl) before building the consumer.

Prints the data-quality findings the consumer has to handle: parse failures,
schema variants, identifier and timestamp formats, duplicates, out-of-order
updates and field values. Uses the consumer's own normalizers, so the EDA and
the consumer can never disagree about a rule.

Usage (from the repo root): uv run python -m scripts.profile_feed [--file data/feed.jsonl]
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime, timezone
from pathlib import Path

from similar_companies.normalize import (
    MALAYSIA_TIME,
    normalize_registration_number,
    normalize_state,
    normalize_updated_at,
)

OFFSET_PATTERN = re.compile(r"[+-]\d\d:\d\d$")
MISSING = object()  # marks an absent key, so it is never confused with a null value


def section(title: str) -> None:
    print(f"\n== {title} " + "=" * (60 - len(title)))


def parse_lines(lines: list[str]) -> tuple[list[tuple[int, object]], list[tuple[int, str]]]:
    """Split lines into (line_no, parsed JSON) and (line_no, error) for unparseable ones."""
    parsed, failed = [], []
    for line_no, line in enumerate(lines, start=1):
        try:
            parsed.append((line_no, json.loads(line)))
        except json.JSONDecodeError as error:
            failed.append((line_no, error.msg))
    return parsed, failed


def timestamp_format(value: object) -> str:
    """Name the format of an updatedAt value, without converting it."""
    if value is None:
        return "missing"
    if isinstance(value, int):
        return "epoch-ms"
    if isinstance(value, str):
        if value.endswith("Z"):
            return "iso-Z (UTC)"
        if OFFSET_PATTERN.search(value):
            return "iso-offset"
        return "iso-naive"
    return f"other:{type(value).__name__}"


def reg_no_shape(value: object) -> str:
    """Show the shape of a registration number with digits masked, e.g. ' SYN-999999 '."""
    if value is None:
        return "missing"
    if not isinstance(value, str):
        return f"type:{type(value).__name__}"
    return re.sub(r"\d", "9", value)


def count_field(records: list[dict], field: str) -> tuple[Counter, int, int]:
    """Count a field's values; return (values, null_count, missing_key_count) separately."""
    values = Counter(record.get(field, MISSING) for record in records)
    return values, values.pop(None, 0), values.pop(MISSING, 0)


def naive_as(value: str, tz: timezone) -> datetime:
    """Read a timestamp without a timezone as if it were in `tz`, converted to UTC.

    Deliberately separate from normalize_updated_at: this tests the hypothesis
    (UTC vs Malaysia time) that normalize_updated_at then relies on.
    """
    return datetime.fromisoformat(value).replace(tzinfo=tz).astimezone(UTC)


def naive_timezone_evidence(events: list[tuple[int, dict]]) -> None:
    """Test the assumption that timestamps without a timezone are Malaysia time.

    The feed is roughly in time order, so a correctly read timestamp should sit close
    to the timestamps of nearby lines that carry an explicit offset. Compare the median
    gap under 'naive = UTC' vs 'naive = Malaysia time'.
    """
    anchors = [
        (line_no, normalize_updated_at(e["updatedAt"]))
        for line_no, e in events
        if timestamp_format(e.get("updatedAt")) == "iso-offset"
    ]
    naive = [
        (n, e["updatedAt"])
        for n, e in events
        if timestamp_format(e.get("updatedAt")) == "iso-naive"
    ]
    for label, tz in (("naive treated as UTC", UTC), ("naive treated as MYT", MALAYSIA_TIME)):
        gaps = []
        for line_no, value in naive:
            nearest = min(anchors, key=lambda anchor: abs(anchor[0] - line_no))
            gaps.append(abs((naive_as(value, tz) - nearest[1]).total_seconds()) / 60)
        print(f"  {label:<24} median gap to nearest neighbour: {statistics.median(gaps):7.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=Path("data/feed.jsonl"))
    lines = parser.parse_args().file.read_text(encoding="utf-8").splitlines()

    parsed, failed = parse_lines(lines)
    events = [(n, obj) for n, obj in parsed if isinstance(obj, dict)]

    section("Volume and parsing")
    print(f"  lines: {len(lines)}   blank: {sum(1 for line in lines if not line.strip())}")
    print(f"  unparseable JSON: {len(failed)}  (lines {[n for n, _ in failed]})")
    not_objects = Counter(type(o).__name__ for _, o in parsed if not isinstance(o, dict))
    print(f"  parsed but not an object: {dict(not_objects)}")
    print(f"  identical duplicate lines (extra copies): {len(lines) - len(set(lines))}")

    section("Event types and top-level keys")
    print(f"  types: {dict(Counter(e.get('type') for _, e in events))}")
    for keys, count in Counter(tuple(sorted(e)) for _, e in events).most_common():
        print(f"  {count:5}  {keys}")

    section("registrationNumber")
    shapes = Counter(reg_no_shape(e.get("registrationNumber")) for _, e in events)
    for shape, count in shapes.most_common():
        print(f"  {count:5}  {shape!r}")
    companies = {normalize_registration_number(e.get("registrationNumber")) for _, e in events}
    print(f"  distinct companies after trim + uppercase: {len(companies - {None})}")

    section("updatedAt")
    formats = Counter(timestamp_format(e.get("updatedAt")) for _, e in events)
    for fmt, count in formats.most_common():
        print(f"  {count:5}  {fmt}")
    naive_timezone_evidence(events)

    section("data field")
    data_types = Counter(type(e["data"]).__name__ if "data" in e else "missing" for _, e in events)
    print(f"  data type: {dict(data_types)}")
    records = [e["data"] for _, e in events if isinstance(e.get("data"), dict)]
    for keys, count in Counter(tuple(sorted(d)) for d in records).most_common():
        print(f"  {count:5}  {keys}")
    placeholders = Counter(
        d.get("companyName") for d in records if d.get("companyName") in {"Unknown", "X", "Foo"}
    )
    print(f"  companyName values that look like placeholders: {dict(placeholders)}")

    section("Field values")
    descriptions, null_desc, missing_desc = count_field(records, "natureOfBusiness")
    empty_desc = descriptions.pop("", 0)
    print(
        f"  natureOfBusiness: distinct {len(descriptions)}, null {null_desc},"
        f" missing key {missing_desc}, empty {empty_desc}"
    )
    long_ones = [text for text in descriptions if isinstance(text, str) and len(text) > 200]
    print(
        f"  descriptions longer than 200 chars: {len(long_ones)}  (e.g. {long_ones[0][:60]!r}...)"
    )

    states, null_state, missing_state = count_field(records, "state")
    known_states = {normalize_state(raw) for raw in states} - {None}
    print(
        f"  state: {len(states)} raw spellings -> {len(known_states)} known states,"
        f" null {null_state}, missing key {missing_state}"
    )

    industries, null_industry, missing_industry = count_field(records, "industry")
    print(
        f"  industry: {dict(industries.most_common())},"
        f" null {null_industry}, missing key {missing_industry}"
    )

    section("History per company (time-ordered, naive = MYT)")
    history = defaultdict(list)
    for line_no, e in events:
        reg_no = normalize_registration_number(e.get("registrationNumber"))
        if reg_no:
            history[reg_no].append((line_no, e))
    events_per_company = Counter(len(v) for v in history.values())
    print(f"  events per company: {dict(sorted(events_per_company.items()))}")

    stale = 0
    for company_events in history.values():
        newest = None
        for _, e in company_events:  # arrival order
            ts = normalize_updated_at(e.get("updatedAt"))
            if ts is None:
                continue
            if newest and ts < newest:
                stale += 1
            newest = max(newest, ts) if newest else ts
    print(f"  stale arrivals (older updatedAt than one already seen): {stale}")

    no_description = [
        reg_no
        for reg_no, company_events in history.items()
        if not any(
            isinstance(e.get("data"), dict) and e["data"].get("natureOfBusiness")
            for _, e in company_events
        )
    ]
    print(f"  companies with no non-empty description in any event: {len(no_description)}")
    struck = sorted(
        reg_no
        for reg_no, company_events in history.items()
        if any(e.get("type") == "company-struck-off" for _, e in company_events)
    )
    print(f"  struck-off companies: {len(struck)}  {struck}")


if __name__ == "__main__":
    main()

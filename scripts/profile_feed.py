"""Profile the registry feed (data/feed.jsonl) before building the consumer.

Prints the data-quality findings the consumer has to handle: parse failures,
schema variants, identifier and timestamp formats, duplicates, out-of-order
updates and field values.

"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

MYT = timezone(timedelta(hours=8))  # Malaysia time
REG_NO_PATTERN = re.compile(r"SYN-\d{6}")
OFFSET_PATTERN = re.compile(r"[+-]\d\d:\d\d$")


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


def to_utc(value: object, naive_tz: timezone) -> datetime | None:
    """Convert any known updatedAt format to UTC. naive_tz is the assumption for naive strings."""
    if isinstance(value, int):
        return datetime.fromtimestamp(value / 1000, UTC)
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=naive_tz)).astimezone(UTC)
    return None


def normalize_reg_no(value: object) -> str | None:
    """Trim and uppercase string IDs. Non-strings are not guessed at."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().upper()
    return candidate if REG_NO_PATTERN.fullmatch(candidate) else None


def reg_no_shape(value: object) -> str:
    if value is None:
        return "missing"
    if not isinstance(value, str):
        return f"type:{type(value).__name__}"
    return re.sub(r"\d", "9", value)  # e.g. " SYN-999999 "


def naive_timezone_evidence(events: list[tuple[int, dict]]) -> None:
    """Test the assumption that naive timestamps are Malaysia time.

    The feed is roughly in time order, so a correctly converted timestamp should sit
    close to the timestamps of nearby offset-bearing lines. Compare the median gap
    under 'naive = UTC' vs 'naive = MYT'.
    """
    anchors = [
        (line_no, to_utc(e["updatedAt"], MYT))
        for line_no, e in events
        if timestamp_format(e.get("updatedAt")) == "iso-offset"
    ]
    for label, tz in (("naive treated as UTC", UTC), ("naive treated as MYT", MYT)):
        gaps = []
        for line_no, e in events:
            if timestamp_format(e.get("updatedAt")) != "iso-naive":
                continue
            nearest = min(anchors, key=lambda anchor: abs(anchor[0] - line_no))
            gaps.append(abs((to_utc(e["updatedAt"], tz) - nearest[1]).total_seconds()) / 60)
        print(f"  {label:<24} median gap to nearest neighbour: {statistics.median(gaps):7.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=Path("data/feed.jsonl"))
    lines = parser.parse_args().file.read_text(encoding="utf-8").splitlines()

    parsed, failed = parse_lines(lines)
    events = [(n, obj) for n, obj in parsed if isinstance(obj, dict)]

    section("Volume and parsing")
    print(f"  lines: {len(lines)}   blank: {sum(1 for l in lines if not l.strip())}")
    print(f"  unparseable JSON: {len(failed)}  (lines {[n for n, _ in failed]})")
    print(f"  parsed but not an object: {Counter(type(o).__name__ for _, o in parsed if not isinstance(o, dict))}")
    print(f"  identical duplicate lines (extra copies): {len(lines) - len(set(lines))}")

    section("Event types and top-level keys")
    print(f"  types: {dict(Counter(e.get('type') for _, e in events))}")
    for keys, count in Counter(tuple(sorted(e)) for _, e in events).most_common():
        print(f"  {count:5}  {keys}")

    section("registrationNumber")
    for shape, count in Counter(reg_no_shape(e.get("registrationNumber")) for _, e in events).most_common():
        print(f"  {count:5}  {shape!r}")
    companies = {normalize_reg_no(e.get("registrationNumber")) for _, e in events} - {None}
    print(f"  distinct companies after trim + uppercase: {len(companies)}")

    section("updatedAt")
    for fmt, count in Counter(timestamp_format(e.get("updatedAt")) for _, e in events).most_common():
        print(f"  {count:5}  {fmt}")
    naive_timezone_evidence(events)

    section("data field")
    print(f"  data type: {dict(Counter(type(e.get('data')).__name__ if 'data' in e else 'missing' for _, e in events))}")
    records = [e["data"] for _, e in events if isinstance(e.get("data"), dict)]
    for keys, count in Counter(tuple(sorted(d)) for d in records).most_common():
        print(f"  {count:5}  {keys}")
    print(f"  companyName values that look like placeholders: "
          f"{dict(Counter(d.get('companyName') for d in records if d.get('companyName') in {'Unknown', 'X', 'Foo'}))}")

    section("Field values")
    MISSING = object()
    descriptions = Counter(d.get("natureOfBusiness", MISSING) for d in records)
    missing_key = descriptions.pop(MISSING, 0)
    print(f"  natureOfBusiness: distinct {len(descriptions)}, null {descriptions[None]}, missing key {missing_key}, empty {descriptions['']}")
    long_ones = [text for text in descriptions if isinstance(text, str) and len(text) > 200]
    print(f"  descriptions longer than 200 chars: {len(long_ones)}  (e.g. {long_ones[0][:60]!r}...)")

    states = Counter(d.get("state", MISSING) for d in records)
    missing_state = states.pop(MISSING, 0)
    null_state = states.pop(None, 0)
    canonical_states = {s.strip().title() for s in states if isinstance(s, str)}
    print(f"  state: {len(states)} raw spellings -> {len(canonical_states)} after title-case, null {null_state}, missing key {missing_state}")
    
    industry = Counter(d.get('industry', MISSING) for d in records)
    missing_industry = industry.pop(MISSING, 0)
    print(f"  industry: {dict(industry.most_common())}, missing {missing_industry}")

    section("History per company (time-ordered, naive = MYT)")
    history = defaultdict(list)
    for line_no, e in events:
        reg_no = normalize_reg_no(e.get("registrationNumber"))
        if reg_no:
            history[reg_no].append((line_no, e))
    print(f"  events per company: {dict(sorted(Counter(len(v) for v in history.values()).items()))}")

    stale = 0
    for company_events in history.values():
        newest = None
        for _, e in company_events:  # arrival order
            ts = to_utc(e.get("updatedAt"), MYT)
            if ts is None:
                continue
            if newest and ts < newest:
                stale += 1
            newest = max(newest, ts) if newest else ts
    print(f"  stale arrivals (older updatedAt than one already seen): {stale}")

    no_description = [
        reg_no for reg_no, company_events in history.items()
        if not any(isinstance(e.get("data"), dict) and e["data"].get("natureOfBusiness") for _, e in company_events)
    ]
    print(f"  companies with no non-empty description in any event: {len(no_description)}")
    struck = [reg_no for reg_no, ce in history.items() if any(e.get("type") == "company-struck-off" for _, e in ce)]
    print(f"  struck-off companies: {len(struck)}  {sorted(struck)}")


if __name__ == "__main__":
    main()
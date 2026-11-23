"""Functions that turn raw feed values into one canonical form.

Each function either returns a cleaned value or None. None means "unusable"; the caller
decides whether that rejects the whole message (registration number, updatedAt) or just
leaves an optional field empty (state, description). Nothing here guesses.
"""

import re
from datetime import UTC, datetime, timedelta, timezone

MALAYSIA_TIME = timezone(timedelta(hours=8))  # assumed for timestamps without a timezone (A-001)
REGISTRATION_NUMBER = re.compile(r"SYN-\d{6}")
PUNCTUATION = ".,;:!?"  # ignored at word edges when detecting repeated descriptions

# Sanity window for timestamps: anything outside it is a broken value, not a real update.
EARLIEST_VALID = datetime(2000, 1, 1, tzinfo=UTC)
LATEST_VALID = datetime(2100, 1, 1, tzinfo=UTC)

# The 16 Malaysian states and federal territories, keyed by lowercase spelling.
STATES = {
    name.lower(): name
    for name in (
        "Johor",
        "Kedah",
        "Kelantan",
        "Kuala Lumpur",
        "Labuan",
        "Melaka",
        "Negeri Sembilan",
        "Pahang",
        "Perak",
        "Perlis",
        "Pulau Pinang",
        "Putrajaya",
        "Sabah",
        "Sarawak",
        "Selangor",
        "Terengganu",
    )
}


def normalize_registration_number(value: object) -> str | None:
    """' syn-000587 ' -> 'SYN-000587'. Non-strings (e.g. the integer 119) are not guessed at."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().upper()
    return candidate if REGISTRATION_NUMBER.fullmatch(candidate) else None


def normalize_updated_at(value: object) -> datetime | None:
    """Convert any known updatedAt format to a UTC datetime.

    - ISO 8601 with an offset or 'Z'  -> converted to UTC
    - ISO 8601 without a timezone     -> assumed Malaysia time (UTC+8), then converted
    - integer                         -> epoch milliseconds
    """
    if isinstance(value, bool):  # bool is a subclass of int in Python; True is not a timestamp
        return None
    try:
        if isinstance(value, int):
            parsed = datetime.fromtimestamp(value / 1000, UTC)
        elif isinstance(value, str):
            parsed = datetime.fromisoformat(value.strip())
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=MALAYSIA_TIME)
        else:
            return None
    except ValueError, OverflowError, OSError:
        return None
    parsed = parsed.astimezone(UTC)
    return parsed if EARLIEST_VALID <= parsed < LATEST_VALID else None


def clean_text(value: object) -> str | None:
    """Trim and collapse internal whitespace. Non-strings and empty strings become None."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def normalize_state(value: object) -> str | None:
    """'KUALA LUMPUR' -> 'Kuala Lumpur'. Unknown spellings become None rather than a guess."""
    cleaned = clean_text(value)
    return STATES.get(cleaned.lower()) if cleaned else None


def repeat_unit_length(keys: list[str]) -> int:
    """Length of the shortest unit that, repeated end to end, makes up `keys`.

    Returns len(keys) when there is no repetition. Compares in place, without building
    candidate lists, and stops at the first mismatch for each candidate length.
    """
    total = len(keys)
    for unit in range(1, total // 2 + 1):
        if total % unit == 0 and all(keys[i] == keys[i % unit] for i in range(unit, total)):
            return unit
    return total


def clean_description(value: object) -> str | None:
    """Clean a natureOfBusiness value and collapse a phrase repeated end to end.

    Repeats are detected ignoring case and surrounding punctuation; the first copy is kept
    as written: 'Kedai Kopi, kedai kopi.' -> 'Kedai Kopi'. Text without a repeat is only
    whitespace-cleaned.
    """
    if not isinstance(value, str):
        return None
    words = value.split()  # splits on any whitespace run and drops the ends: one pass
    if not words:
        return None
    keys = [word.strip(PUNCTUATION).casefold() for word in words]
    unit = repeat_unit_length(keys)
    if unit == len(words):
        return " ".join(words)
    return " ".join(words[:unit]).rstrip(PUNCTUATION + " ")

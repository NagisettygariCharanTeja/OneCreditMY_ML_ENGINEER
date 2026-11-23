"""All SQL the consumer runs. Every write is idempotent and newer-wins.

Ordering rule: a write only changes a stored company if its updated_at is strictly
newer than the stored one. Duplicates (equal time) and stale messages are no-ops.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import psycopg


@dataclass(frozen=True)
class CompanyUpsert:
    """A validated, normalized company-upserted event, ready to store."""

    registration_number: str
    updated_at: datetime  # timezone-aware, UTC
    company_name: str
    nature_of_business: str | None
    state: str | None
    industry: str | None
    embedded_text: str
    embedding: Sequence[float]
    embedding_model: str


UPSERT_COMPANY = """
INSERT INTO companies (
    registration_number, status, updated_at, company_name, nature_of_business,
    state, industry, embedded_text, embedding, embedding_model
) VALUES (
    %(registration_number)s, 'active', %(updated_at)s, %(company_name)s, %(nature_of_business)s,
    %(state)s, %(industry)s, %(embedded_text)s, %(embedding)s::vector, %(embedding_model)s
)
ON CONFLICT (registration_number) DO UPDATE SET
    status             = 'active',
    updated_at         = EXCLUDED.updated_at,
    company_name       = EXCLUDED.company_name,
    nature_of_business = EXCLUDED.nature_of_business,
    state              = EXCLUDED.state,
    industry           = EXCLUDED.industry,
    embedded_text      = EXCLUDED.embedded_text,
    embedding          = EXCLUDED.embedding,
    embedding_model    = EXCLUDED.embedding_model
WHERE companies.updated_at < EXCLUDED.updated_at
RETURNING registration_number
"""

MARK_STRUCK_OFF = """
INSERT INTO companies (registration_number, status, updated_at)
VALUES (%(registration_number)s, 'struck_off', %(updated_at)s)
ON CONFLICT (registration_number) DO UPDATE SET
    status     = 'struck_off',
    updated_at = EXCLUDED.updated_at
WHERE companies.updated_at < EXCLUDED.updated_at
RETURNING registration_number
"""

SAVE_REJECTED = """
INSERT INTO rejected_events (message_id, raw_body, reason)
VALUES (%(message_id)s, %(raw_body)s, %(reason)s)
ON CONFLICT (message_id) DO NOTHING
"""


def to_pgvector(values: Sequence[float]) -> str:
    """Format a vector as pgvector's text input, e.g. '[0.1,0.2]'."""
    return "[" + ",".join(f"{v:.8g}" for v in values) + "]"


def upsert_company(conn: psycopg.Connection, company: CompanyUpsert) -> bool:
    """Insert or overwrite a company if this event is newer. Returns True if it was applied."""
    params = {**company.__dict__, "embedding": to_pgvector(company.embedding)}
    return conn.execute(UPSERT_COMPANY, params).fetchone() is not None


def mark_struck_off(conn: psycopg.Connection, registration_number: str, updated_at: datetime) -> bool:
    """Mark a company struck off (or create a tombstone) if this event is newer. Returns True if applied."""
    params = {"registration_number": registration_number, "updated_at": updated_at}
    return conn.execute(MARK_STRUCK_OFF, params).fetchone() is not None


def save_rejected(conn: psycopg.Connection, message_id: str | None, raw_body: str, reason: str) -> None:
    """Quarantine a message the consumer cannot use. A redelivered message is stored once."""
    conn.execute(SAVE_REJECTED, {"message_id": message_id, "raw_body": raw_body, "reason": reason})


def stored_updated_at(conn: psycopg.Connection, registration_number: str) -> datetime | None:
    """The updated_at currently stored for a company, or None if unknown. Lets the consumer skip
    embedding work for stale events; the conditional writes above stay the real safeguard."""
    row = conn.execute(
        "SELECT updated_at FROM companies WHERE registration_number = %s", (registration_number,)
    ).fetchone()
    return row[0] if row else None

"""Nearest-neighbour search, live and precomputed.

`company_neighbours` stores each active company's top CACHE_SIZE active neighbours. The
consumer keeps it exact: after every applied write to company X it recomputes, in the
same transaction,
  - X's own list,
  - every list that contains X (X moved, was renamed or struck off), and
  - every list X now belongs in (X is new or its vector moved closer).
So a cached answer always equals the live query. The API still joins the live `companies`
table and filters on status at read time, the way production recommenders apply hard
filters after retrieval, and falls back to the live query whenever the cache can't answer.
"""

from dataclasses import dataclass

import psycopg

CACHE_SIZE = 20  # larger than the default page of 5, so the read-time filter has headroom


@dataclass(frozen=True)
class Neighbour:
    registration_number: str
    company_name: str
    industry: str | None
    state: str | None
    similarity: float  # 1 - cosine distance


LIVE = """
WITH target AS (SELECT embedding FROM companies WHERE registration_number = %(reg)s)
SELECT c.registration_number, c.company_name, c.industry, c.state,
       1 - (c.embedding <=> target.embedding)
FROM companies AS c, target
WHERE c.status = 'active' AND c.registration_number <> %(reg)s
ORDER BY c.embedding <=> target.embedding, c.registration_number
LIMIT %(limit)s
"""

CACHED = """
WITH target AS (SELECT embedding FROM companies WHERE registration_number = %(reg)s)
SELECT c.registration_number, c.company_name, c.industry, c.state,
       1 - (c.embedding <=> target.embedding)
FROM company_neighbours AS n
JOIN companies AS c ON c.registration_number = n.neighbour, target
WHERE n.registration_number = %(reg)s AND c.status = 'active'
ORDER BY c.embedding <=> target.embedding, c.registration_number
LIMIT %(limit)s
"""

DELETE_LISTS = "DELETE FROM company_neighbours WHERE registration_number = ANY(%(regs)s)"

RECOMPUTE = """
INSERT INTO company_neighbours (registration_number, rank, neighbour, distance)
SELECT c.registration_number, n.rank, n.registration_number, n.distance
FROM companies AS c
CROSS JOIN LATERAL (
    SELECT o.registration_number,
           o.embedding <=> c.embedding AS distance,
           row_number() OVER (ORDER BY o.embedding <=> c.embedding, o.registration_number) AS rank
    FROM companies AS o
    WHERE o.status = 'active' AND o.registration_number <> c.registration_number
    ORDER BY o.embedding <=> c.embedding, o.registration_number
    LIMIT %(size)s
) AS n
WHERE c.registration_number = ANY(%(regs)s) AND c.status = 'active'
"""

# Lists X now qualifies for: X ranks ahead of the list's last entry (same order as the
# search: distance, then registration number), or the list isn't full (no last entry).
LISTS_X_JOINS = """
WITH x AS (
    SELECT embedding FROM companies WHERE registration_number = %(reg)s AND status = 'active'
)
SELECT c.registration_number, c.embedding <=> x.embedding
FROM companies AS c CROSS JOIN x
LEFT JOIN company_neighbours AS last
       ON last.registration_number = c.registration_number AND last.rank = %(size)s
WHERE c.status = 'active' AND c.registration_number <> %(reg)s
  AND (last.registration_number IS NULL
       OR (c.embedding <=> x.embedding, %(reg)s) < (last.distance, last.neighbour))
  AND NOT EXISTS (
      SELECT 1 FROM company_neighbours AS n
      WHERE n.registration_number = c.registration_number AND n.neighbour = %(reg)s
  )
"""

# Merge X into lists it newly joins, using only the stored entries (no scan of all companies).
MERGE_X = """
SELECT owner, row_number() OVER (PARTITION BY owner ORDER BY distance, neighbour) AS rank,
       neighbour, distance
FROM (
    SELECT registration_number AS owner, neighbour, distance
    FROM company_neighbours WHERE registration_number = ANY(%(owners)s)
    UNION ALL
    SELECT owner, %(reg)s, distance
    FROM unnest(%(owners)s::text[], %(distances)s::float8[]) AS joined(owner, distance)
) AS candidates
"""

INSERT_ROWS = """
INSERT INTO company_neighbours (registration_number, rank, neighbour, distance)
SELECT * FROM unnest(%(owners)s::text[], %(ranks)s::int[], %(neighbours)s::text[],
                     %(distances)s::float8[])
"""

LISTS_CONTAINING_X = "SELECT registration_number FROM company_neighbours WHERE neighbour = %(reg)s"


def similar(conn: psycopg.Connection, registration_number: str, limit: int) -> list[Neighbour]:
    """Top `limit` active companies most similar to this one: from the cache when it can
    answer in full, otherwise from the live exact search (identical results either way)."""
    params = {"reg": registration_number, "limit": limit}
    rows = conn.execute(CACHED, params).fetchall() if limit <= CACHE_SIZE else []
    if len(rows) < limit:
        rows = conn.execute(LIVE, params).fetchall()
    return [Neighbour(*row) for row in rows]


def live_similar(conn: psycopg.Connection, registration_number: str, limit: int) -> list[Neighbour]:
    """The exact search without the cache (the reference the cache is tested against)."""
    rows = conn.execute(LIVE, {"reg": registration_number, "limit": limit}).fetchall()
    return [Neighbour(*row) for row in rows]


def recompute(conn: psycopg.Connection, registration_numbers: list[str]) -> None:
    """Rebuild the lists of these companies (struck-off companies just lose theirs)."""
    if registration_numbers:
        params = {"regs": registration_numbers, "size": CACHE_SIZE}
        conn.execute(DELETE_LISTS, params)
        conn.execute(RECOMPUTE, params)


def refresh_after_change(conn: psycopg.Connection, registration_number: str) -> int:
    """Keep the cache exact after company X was inserted, updated, struck off or reactivated.

    Must run in the same transaction as the write. Lists that contained X are rebuilt with
    a full scan (X may have moved away, so the next-best company is unknown); lists X newly
    joins are updated by merging X into their stored entries, with no scan. Returns how
    many lists changed.
    """
    params = {"reg": registration_number, "size": CACHE_SIZE}
    contained = [row[0] for row in conn.execute(LISTS_CONTAINING_X, params)]
    recompute(conn, sorted({registration_number, *contained}))

    joins = conn.execute(LISTS_X_JOINS, params).fetchall()
    if joins:
        owners = [owner for owner, _ in joins]
        merged = conn.execute(
            MERGE_X, {**params, "owners": owners, "distances": [d for _, d in joins]}
        ).fetchall()
        kept = [row for row in merged if row[1] <= CACHE_SIZE]
        conn.execute(DELETE_LISTS, {"regs": owners})
        conn.execute(
            INSERT_ROWS,
            {
                "owners": [row[0] for row in kept],
                "ranks": [row[1] for row in kept],
                "neighbours": [row[2] for row in kept],
                "distances": [row[3] for row in kept],
            },
        )
    return 1 + len(contained) + len(joins)


def rebuild_all(conn: psycopg.Connection) -> None:
    """Rebuild every list, e.g. to backfill a database filled before the cache existed."""
    conn.execute("TRUNCATE company_neighbours")
    active = [
        r[0]
        for r in conn.execute("SELECT registration_number FROM companies WHERE status = 'active'")
    ]
    recompute(conn, active)


def is_complete(conn: psycopg.Connection) -> bool:
    """True if every active company has a list (the cache was built and maintained)."""
    missing = conn.execute(
        "SELECT count(*) FROM companies c WHERE c.status = 'active' AND NOT EXISTS"
        " (SELECT 1 FROM company_neighbours n WHERE n.registration_number = c.registration_number)"
        " AND (SELECT count(*) FROM companies WHERE status = 'active') > 1"
    ).fetchone()[0]
    return missing == 0

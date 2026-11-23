"""HTTP API the portal calls to draw the "Similar companies" panel.

GET /companies/{registration_number}/similar?limit=5
  200  the most similar active companies, most similar first, with what a card needs
  404  the company has never been in the feed
  410  the company was struck off (it existed but must not be shown)
  422  the registration number or limit is malformed
  503  the database is unreachable

The search runs inside Postgres with pgvector: an exact cosine scan over active companies,
excluding the company itself, with the registration number as a tie-breaker so results
are reproducible (many companies share an identical description).
"""

import psycopg
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from similar_companies.config import load_settings
from similar_companies.normalize import normalize_registration_number

MAX_LIMIT = 50

COMPANY_STATUS = "SELECT status FROM companies WHERE registration_number = %(reg)s"

SIMILAR_COMPANIES = """
WITH target AS (
    SELECT embedding FROM companies WHERE registration_number = %(reg)s
)
SELECT c.registration_number, c.company_name, c.industry, c.state,
       1 - (c.embedding <=> target.embedding) AS similarity
FROM companies AS c, target
WHERE c.status = 'active' AND c.registration_number <> %(reg)s
ORDER BY c.embedding <=> target.embedding, c.registration_number
LIMIT %(limit)s
"""


class CamelModel(BaseModel):
    """snake_case in Python, camelCase in JSON (matching the feed and the portal)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class CompanyCard(CamelModel):
    registration_number: str
    company_name: str
    industry: str | None
    state: str | None
    similarity: float  # cosine similarity, 1.0 = identical description


class SimilarCompanies(CamelModel):
    registration_number: str
    similar: list[CompanyCard]


def create_app(database_url: str | None = None) -> FastAPI:
    """Build the API. Tests pass their own database URL; production reads config."""
    database_url = database_url or load_settings().database_url
    app = FastAPI(title="Similar companies")

    @app.exception_handler(psycopg.OperationalError)
    async def database_unavailable(request: Request, error: psycopg.OperationalError):
        return JSONResponse(status_code=503, content={"detail": "database unavailable"})

    @app.get("/health")
    def health() -> dict:
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return {"status": "ok"}

    @app.get("/companies/{registration_number}/similar", response_model=SimilarCompanies)
    def similar_companies(
        registration_number: str, limit: int = Query(5, ge=1, le=MAX_LIMIT)
    ) -> SimilarCompanies:
        reg_no = normalize_registration_number(registration_number)
        if reg_no is None:
            raise HTTPException(422, f"{registration_number!r} is not a registration number")
        params = {"reg": reg_no, "limit": limit}
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            row = conn.execute(COMPANY_STATUS, params).fetchone()
            if row is None:
                raise HTTPException(404, f"company {reg_no} not found")
            if row[0] == "struck_off":
                raise HTTPException(410, f"company {reg_no} has been struck off")
            rows = conn.execute(SIMILAR_COMPANIES, params).fetchall()
        cards = [
            CompanyCard(
                registration_number=number,
                company_name=name,
                industry=industry,
                state=state,
                similarity=round(similarity, 4),
            )
            for number, name, industry, state, similarity in rows
        ]
        return SimilarCompanies(registration_number=reg_no, similar=cards)

    return app


app = create_app()

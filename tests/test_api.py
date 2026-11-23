"""Integration tests for the API against real Postgres with hand-made vectors.

Vectors are chosen so the expected order is known exactly:
  target  SYN-000001  [1, 0, ...]
  near    SYN-000002  mostly x, a little y      -> most similar
  tie     SYN-000003  identical vector to near  -> tie, ordered after it by reg no
  far     SYN-000004  mostly y
  gone    SYN-000005  identical to the target but struck off -> never returned
"""

import math
from datetime import UTC, datetime

import psycopg
import pytest
from fastapi.testclient import TestClient

from similar_companies.api import create_app
from similar_companies.repository import CompanyUpsert, mark_struck_off, upsert_company

T = datetime(2026, 8, 14, tzinfo=UTC)


def unit(x: float, y: float) -> list[float]:
    norm = math.hypot(x, y)
    return [x / norm, y / norm] + [0.0] * 382


COMPANIES = {
    "SYN-000001": ("Target Kopi Sdn. Bhd.", unit(1, 0)),
    "SYN-000002": ("Near Cafe Sdn. Bhd.", unit(1, 0.2)),
    "SYN-000003": ("Tie Cafe Sdn. Bhd.", unit(1, 0.2)),
    "SYN-000004": ("Far Logistics Sdn. Bhd.", unit(0.2, 1)),
    "SYN-000005": ("Gone Kopi Sdn. Bhd.", unit(1, 0)),
}


@pytest.fixture
def client(test_db_url):
    with psycopg.connect(test_db_url, autocommit=True) as conn:
        conn.execute("TRUNCATE companies, rejected_events")
        for number, (name, vector) in COMPANIES.items():
            upsert_company(
                conn,
                CompanyUpsert(
                    registration_number=number,
                    updated_at=T,
                    company_name=name,
                    nature_of_business="Cafes",
                    state="Selangor",
                    industry="Food & Beverage",
                    embedded_text="Cafes",
                    embedding=vector,
                    embedding_model="test",
                ),
            )
        mark_struck_off(conn, "SYN-000005", datetime(2026, 8, 15, tzinfo=UTC))
        yield TestClient(create_app(test_db_url))
        conn.execute("TRUNCATE companies, rejected_events")


def numbers(response) -> list[str]:
    return [card["registrationNumber"] for card in response.json()["similar"]]


def test_returns_most_similar_first_with_card_fields(client):
    response = client.get("/companies/SYN-000001/similar")
    assert response.status_code == 200
    body = response.json()
    assert body["registrationNumber"] == "SYN-000001"
    assert numbers(response) == ["SYN-000002", "SYN-000003", "SYN-000004"]
    first = body["similar"][0]
    assert set(first) == {"registrationNumber", "companyName", "industry", "state", "similarity"}
    assert first["companyName"] == "Near Cafe Sdn. Bhd."
    similarities = [card["similarity"] for card in body["similar"]]
    assert similarities == sorted(similarities, reverse=True)


def test_never_returns_the_company_itself_or_struck_off_companies(client):
    assert "SYN-000001" not in numbers(client.get("/companies/SYN-000001/similar"))
    assert "SYN-000005" not in numbers(client.get("/companies/SYN-000001/similar"))


def test_ties_are_broken_by_registration_number(client):
    assert numbers(client.get("/companies/SYN-000004/similar"))[:2] == ["SYN-000002", "SYN-000003"]


def test_limit(client):
    assert numbers(client.get("/companies/SYN-000001/similar?limit=1")) == ["SYN-000002"]


@pytest.mark.parametrize("limit", [0, -1, 51, "abc"])
def test_invalid_limit_is_422(client, limit):
    assert client.get(f"/companies/SYN-000001/similar?limit={limit}").status_code == 422


def test_registration_number_is_normalized(client):
    response = client.get("/companies/%20syn-000001%20/similar")
    assert response.status_code == 200
    assert response.json()["registrationNumber"] == "SYN-000001"


def test_malformed_registration_number_is_422(client):
    assert client.get("/companies/hello/similar").status_code == 422


def test_unknown_company_is_404(client):
    assert client.get("/companies/SYN-999999/similar").status_code == 404


def test_struck_off_company_is_410(client):
    response = client.get("/companies/SYN-000005/similar")
    assert response.status_code == 410
    assert "struck off" in response.json()["detail"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_database_down_is_503():
    client = TestClient(create_app("postgresql://postgres:postgres@127.0.0.1:1/companies"))
    assert client.get("/companies/SYN-000001/similar").status_code == 503
    assert client.get("/health").status_code == 503

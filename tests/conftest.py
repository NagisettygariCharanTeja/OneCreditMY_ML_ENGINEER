"""Shared pytest fixtures: a throwaway test database with the real schema applied."""

import os
from pathlib import Path

import psycopg
import pytest

# Admin connection to the docker-compose Postgres; override with TEST_ADMIN_DATABASE_URL.
ADMIN_URL = os.environ.get(
    "TEST_ADMIN_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)
TEST_DB_NAME = "companies_test"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "similar_companies" / "schema.sql"


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Create a fresh test database once per test run and apply schema.sql to it."""
    with psycopg.connect(ADMIN_URL, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME} WITH (FORCE)")
        admin.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    with psycopg.connect(ADMIN_URL, dbname=TEST_DB_NAME, autocommit=True) as conn:
        conn.execute(SCHEMA_PATH.read_text())
    return psycopg.conninfo.make_conninfo(ADMIN_URL, dbname=TEST_DB_NAME)


@pytest.fixture
def db(test_db_url: str):
    """A connection whose changes are rolled back after each test, so tests stay independent."""
    with psycopg.connect(test_db_url) as conn:
        yield conn
        conn.rollback()

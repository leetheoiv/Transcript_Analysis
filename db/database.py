"""
db/database.py

Database connection helper for the Transcript Analysis Automation pipeline.
Loads Postgres credentials from config/secrets.toml and returns a psycopg2
connection. All repository classes use get_connection() as their entry point.

Expected secrets.toml keys under [postgres]:
    host, port, dbname, user, password
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import psycopg2
from psycopg2.extensions import connection as PgConnection


def _load_postgres_config() -> dict:
    secrets_path = Path(__file__).parent.parent / "config" / "secrets.toml"
    if not secrets_path.exists():
        raise FileNotFoundError(f"secrets.toml not found at {secrets_path}")
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    return secrets["postgres"]


def get_connection() -> PgConnection:
    """Return a new psycopg2 connection using credentials from secrets.toml."""
    cfg = _load_postgres_config()
    return psycopg2.connect(
        host=cfg["host"],
        port=cfg.get("port", 5432),
        dbname=cfg["dbname"],
        user=cfg["user"],
        password=cfg["password"],
    )

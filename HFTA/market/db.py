# HFTA/market/db.py

from __future__ import annotations

import os
from peewee import PostgresqlDatabase

DB_NAME = os.getenv("HFTA_DB_NAME", "hfta")
DB_USER = os.getenv("HFTA_DB_USER", "postgres")
DB_PASSWORD = os.getenv("HFTA_DB_PASSWORD", "postgres")
DB_HOST = os.getenv("HFTA_DB_HOST", "localhost")
DB_PORT = int(os.getenv("HFTA_DB_PORT", "5432"))

db = PostgresqlDatabase(
    DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
)

"""Read-only database explorer used by the HR "Database" page.

Safety: table names are only accepted from the live catalog (no string-built SQL from user input),
and free-form queries must be a single SELECT-style statement executed inside a READ ONLY
transaction with a statement timeout and a row limit.
"""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

MAX_ROWS = 500
_ALLOWED_START = ("select", "with", "explain", "table", "values", "show")
# server-side functions that could read files / affect other sessions even from a SELECT
_BLOCKED = re.compile(r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file|lo_import|lo_export|"
                      r"pg_terminate_backend|pg_cancel_backend|pg_reload_conf|dblink|set_config|"
                      r"pg_sleep)\b", re.IGNORECASE)


class QueryRejected(ValueError):
    pass


def list_tables(engine: Engine) -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name"))]


def describe_table(engine: Engine, table: str) -> tuple[list[dict], list[dict], int]:
    """(columns, constraints/indexes, row_count) for one table."""
    if table not in list_tables(engine):
        raise QueryRejected(f"Unknown table '{table}'")
    with engine.connect() as conn:
        columns = [dict(r._mapping) for r in conn.execute(text(
            """SELECT column_name AS "Column", data_type AS "Type",
                      CASE WHEN is_nullable = 'YES' THEN 'yes' ELSE 'no' END AS "Nullable",
                      column_default AS "Default"
                 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = :t ORDER BY ordinal_position"""), {"t": table})]
        rules = [dict(r._mapping) for r in conn.execute(text(
            """SELECT conname AS "Name",
                      CASE contype WHEN 'p' THEN 'primary key' WHEN 'f' THEN 'foreign key' WHEN 'u' THEN 'unique'
                                   WHEN 'c' THEN 'check' WHEN 'x' THEN 'exclusion' END AS "Kind",
                      pg_get_constraintdef(oid) AS "Definition"
                 FROM pg_constraint WHERE conrelid = to_regclass(:t) ORDER BY contype, conname"""),
            {"t": f"public.{table}"})]
        rules += [{"Name": r[0], "Kind": "index", "Definition": r[1]} for r in conn.execute(text(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = :t "
            "AND indexname NOT IN (SELECT conname FROM pg_constraint) ORDER BY indexname"), {"t": table})]
        quoted = conn.dialect.identifier_preparer.quote(table)
        count = conn.execute(text(f"SELECT count(*) FROM public.{quoted}")).scalar() or 0
    return columns, rules, count


def _readable(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def fetch_rows(engine: Engine, table: str, limit: int = 200) -> list[dict]:
    if table not in list_tables(engine):
        raise QueryRejected(f"Unknown table '{table}'")
    limit = max(1, min(int(limit), MAX_ROWS))
    with engine.connect() as conn:
        quoted = conn.dialect.identifier_preparer.quote(table)
        result = conn.execute(text(f"SELECT * FROM public.{quoted} ORDER BY 1 DESC LIMIT :n"), {"n": limit})
        return [{k: _readable(v) for k, v in row._mapping.items()} for row in result]


def validate_query(sql: str) -> str:
    query = (sql or "").strip().rstrip(";").strip()
    if not query:
        raise QueryRejected("Enter a query.")
    if ";" in query:
        raise QueryRejected("Only one statement at a time (no ';' inside the query).")
    if query.split(None, 1)[0].lower() not in _ALLOWED_START:
        raise QueryRejected("Only read-only queries are allowed (start with SELECT or WITH).")
    if _BLOCKED.search(query):
        raise QueryRejected("That function is not allowed here.")
    return query


def run_readonly_query(engine: Engine, sql: str, timeout_ms: int = 5000) -> tuple[list[dict], bool]:
    """Run one read-only query. Returns (rows, truncated)."""
    query = validate_query(sql)
    with engine.connect() as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text("SELECT set_config('statement_timeout', :t, true)"), {"t": str(int(timeout_ms))})
            result = conn.execute(text(query))
            if not result.returns_rows:
                return [], False
            rows = result.fetchmany(MAX_ROWS + 1)
    truncated = len(rows) > MAX_ROWS
    return [{k: _readable(v) for k, v in r._mapping.items()} for r in rows[:MAX_ROWS]], truncated

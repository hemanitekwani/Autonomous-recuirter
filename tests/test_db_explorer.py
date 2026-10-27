"""Read-only database explorer (HR 'Database' page)."""
from __future__ import annotations

import pytest

from src.db_explorer import QueryRejected, validate_query


@pytest.mark.parametrize("sql", [
    "DELETE FROM jobs", "UPDATE jobs SET title='x'", "DROP TABLE jobs", "insert into jobs values (1)",
    "SELECT 1; DROP TABLE jobs", "SELECT pg_read_file('C:/Windows/win.ini')", "SELECT pg_sleep(60)", "",
])
def test_unsafe_queries_rejected(sql):
    with pytest.raises(QueryRejected):
        validate_query(sql)


def test_trailing_semicolon_allowed():
    assert validate_query("  SELECT * FROM jobs;  ") == "SELECT * FROM jobs"


@pytest.mark.integration
def test_schema_data_and_readonly_sql(db, ml_job):
    from sqlalchemy import text

    from src.db_explorer import describe_table, fetch_rows, list_tables, run_readonly_query

    assert {"jobs", "applications", "interviews", "audit_log"} <= set(list_tables(db))
    columns, rules, count = describe_table(db, "interviews")
    assert "starts_at" in [c["Column"] for c in columns] and count == 0
    assert any(r["Kind"] == "exclusion" for r in rules)

    rows = fetch_rows(db, "jobs")
    assert rows[0]["title"] == "Machine Learning Engineer" and rows[0]["skills"].startswith("{")
    with pytest.raises(QueryRejected):
        fetch_rows(db, "jobs; DROP TABLE jobs")

    result, truncated = run_readonly_query(db, "SELECT job_id, title FROM jobs")
    assert result == [{"job_id": ml_job, "title": "Machine Learning Engineer"}] and not truncated

    # a write hidden inside a WITH passes the text check but is stopped by the READ ONLY transaction
    with pytest.raises(Exception, match="read-only transaction"):
        run_readonly_query(db, "WITH d AS (DELETE FROM jobs RETURNING *) SELECT * FROM d")
    with db.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM jobs")).scalar() == 1

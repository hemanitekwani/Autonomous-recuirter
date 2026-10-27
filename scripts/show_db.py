"""Look inside the database without any extra tools.

  python -m scripts.show_db schema         # tables and columns (as created from sql/init.sql)
  python -m scripts.show_db jobs           # every job with its requirements (the "JD")
  python -m scripts.show_db apps           # every application with status
  python -m scripts.show_db match <id>     # how application <id> was matched against its job
  python -m scripts.show_db sql            # print the screening SQL that does the matching
  python -m scripts.show_db query "SELECT * FROM jobs"   # run any read-only SQL, print a table
  python -m scripts.show_db shell          # open the interactive PostgreSQL shell (psql)
  python -m scripts.show_db psql           # just print the psql command / connection details
"""
from __future__ import annotations

import json
import sys

from sqlalchemy import text

from src.database import get_engine


def schema(conn) -> None:
    rows = conn.execute(text(
        """SELECT table_name, column_name, data_type, is_nullable, column_default
             FROM information_schema.columns
            WHERE table_schema = 'public' ORDER BY table_name, ordinal_position""")).all()
    current = None
    for table, column, dtype, nullable, default in rows:
        if table != current:
            current = table
            print(f"\n=== {table} ===")
        print(f"  {column:<28} {dtype:<26} {'NULL' if nullable == 'YES' else 'NOT NULL':<9} "
              f"{('default ' + str(default))[:40] if default else ''}")
    print("\n--- constraints & indexes ---")
    for name, table, definition in conn.execute(text(
            """SELECT conname, conrelid::regclass::text, pg_get_constraintdef(oid) FROM pg_constraint
                WHERE connamespace = 'public'::regnamespace AND contype IN ('c','u','x','f') ORDER BY 2, 1""")):
        print(f"  [{table}] {name}: {definition[:110]}")
    for name, definition in conn.execute(text(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname='public' AND indexname LIKE 'ux_%'")):
        print(f"  {name}: {definition}")


def jobs(conn) -> None:
    for job_id, title, min_exp, skills, status in conn.execute(text(
            "SELECT job_id, title, min_experience_years, skills, status FROM jobs ORDER BY job_id")):
        print(f"\n#{job_id} {title}  [{status}]  minimum experience: {min_exp} years")
        print("  mandatory:", [s["canonical_name"] for s in skills.get("mandatory", [])])
        print("  preferred:", [s["canonical_name"] for s in skills.get("preferred", [])])
        print("  stored JSON:", json.dumps(skills))


def apps(conn) -> None:
    for row in conn.execute(text(
            """SELECT a.application_id, a.name, a.email, j.title, a.experience, a.screening_status, a.pipeline_status
                 FROM applications a JOIN jobs j ON j.job_id = a.job_id ORDER BY a.application_id""")):
        print(f"#{row[0]:<4} {row[1][:22]:<22} {row[2][:30]:<30} {row[3][:26]:<26} exp={row[4]!s:<5} "
              f"{row[5]:<16} {row[6]}")


def match(conn, app_id: int) -> None:
    row = conn.execute(text(
        """SELECT a.name, j.title, j.min_experience_years, j.skills, a.experience, a.skills, a.screening_status,
                  a.screening_result
             FROM applications a JOIN jobs j ON j.job_id = a.job_id WHERE a.application_id = :id"""),
        {"id": app_id}).first()
    if row is None:
        raise SystemExit(f"Application {app_id} not found")
    name, title, min_exp, job_skills, exp, app_skills, status, result = row
    print(f"Application #{app_id}: {name}  ->  job '{title}'   RESULT: {status}\n")
    print(f"Experience: candidate {exp} years  vs  required {min_exp} years")
    print("\nCandidate skills (from resume, after normalization + validation):")
    for s in app_skills:
        print(f"  {s['raw_name'][:24]:<24} -> {str(s['canonical_name']):<28} {s['match_type']:<10} "
              f"{s['validation_status']}")
    print("\nJob requirement  vs  candidate  =  result")
    for check in result.get("checks", []):
        print(f"  {check['kind']:<16} {check['requirement']:<26} {check['result']:<8} {check.get('reason') or ''}")
    if result.get("review_reasons"):
        print("\nReview reasons:", result["review_reasons"])


def print_table(rows: list[dict], width: int = 40) -> None:
    if not rows:
        return
    cols = list(rows[0])
    cell = lambda v: ("NULL" if v is None else str(v)).replace("\n", " ")[:width]  # noqa: E731
    widths = {c: max(len(c), *(len(cell(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(cell(r[c]).ljust(widths[c]) for c in cols))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "sql":
        from src.screening_service import REQUIREMENT_MATCH_SQL

        print(REQUIREMENT_MATCH_SQL.text)
        return
    engine = get_engine()
    if cmd in ("psql", "shell"):
        import shutil
        import subprocess
        from pathlib import Path

        url = engine.url
        exe = shutil.which("psql")
        try:
            import pgserver

            bundled = Path(pgserver.__file__).parent / "pginstall" / "bin" / ("psql.exe" if sys.platform == "win32"
                                                                               else "psql")
            exe = str(bundled) if bundled.exists() else exe
        except ImportError:
            pass
        if not exe:
            raise SystemExit("psql not found - use `python -m scripts.show_db query \"...\"` or a GUI tool instead")
        args = [exe, "-h", str(url.host), "-p", str(url.port), "-U", str(url.username), "-d", str(url.database)]
        print(f"Connection: host={url.host} port={url.port} user={url.username} database={url.database}")
        print("psql tips:  \\dt  list tables   \\d jobs  describe a table   SELECT * FROM jobs;   \\x  wide rows   "
              "\\q  quit")
        if cmd == "psql":
            print(" ".join(f'"{a}"' if " " in a else a for a in args))
            return
        env = {**__import__("os").environ, "PGPASSWORD": url.password or ""}
        raise SystemExit(subprocess.call(args, env=env))
    if cmd == "query":
        from src.db_explorer import QueryRejected, run_readonly_query

        if len(sys.argv) < 3:
            raise SystemExit('Usage: python -m scripts.show_db query "SELECT ..."')
        try:
            rows, truncated = run_readonly_query(engine, " ".join(sys.argv[2:]))
        except QueryRejected as exc:
            raise SystemExit(str(exc))
        print_table(rows)
        print(f"({len(rows)} row(s){', truncated at 500' if truncated else ''})")
        return
    with engine.connect() as conn:
        if cmd == "schema":
            schema(conn)
        elif cmd == "jobs":
            jobs(conn)
        elif cmd == "apps":
            apps(conn)
        elif cmd == "match" and len(sys.argv) > 2:
            match(conn, int(sys.argv[2]))
        else:
            print(__doc__)


if __name__ == "__main__":
    main()

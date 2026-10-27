"""Database engine, per-operation session scope and ORM mappings.

The schema itself is owned by sql/init.sql (constraints, exclusion
constraint, partial indexes). The ORM classes below only map columns;
`init_db()` runs the SQL file through a controlled setup command, never on
every Streamlit rerun.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.config import PROJECT_ROOT, get_settings

INIT_SQL_PATH = PROJECT_ROOT / "sql" / "init.sql"

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    min_experience_years: Mapped[int] = mapped_column(Integer)
    skills: Mapped[dict[str, Any]] = mapped_column(JSONB)
    interview_duration_minutes: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))


class Application(Base):
    __tablename__ = "applications"

    application_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id"))
    email: Mapped[str] = mapped_column(String(254))
    name: Mapped[str] = mapped_column(String(200))
    skills: Mapped[list[Any]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    experience: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resume_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    profile: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    screening_status: Mapped[str] = mapped_column(String(30), server_default=text("'PENDING'"))
    screening_result: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    pipeline_status: Mapped[str] = mapped_column(String(30), server_default=text("'SCREENING'"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))


class Interview(Base):
    __tablename__ = "interviews"

    interview_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.job_id"))
    application_id: Mapped[int] = mapped_column(Integer)
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    timezone: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(30), server_default=text("'PENDING_CONFIRMATION'"))
    google_event_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    meeting_link: Mapped[str | None] = mapped_column(Text, nullable=True)
    email_status: Mapped[str] = mapped_column(String(20), server_default=text("'NOT_SENT'"))
    email_attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    email_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))


class AuditLog(Base):
    __tablename__ = "audit_log"

    audit_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(30))
    entity_id: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(60))
    actor: Mapped[str] = mapped_column(String(100))
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("NOW()"))


def resolve_database_url(url: str) -> str:
    """Support `pgserver:<folder>` = embedded local PostgreSQL (no Docker / no install needed).

    The server is started on first use and left running (data persists in <folder>);
    later processes reconnect to the same server.
    """
    if not url.startswith("pgserver:"):
        return url
    try:
        import pgserver
    except ImportError as exc:
        raise RuntimeError("DATABASE_URL uses pgserver: but the 'pgserver' package is not installed "
                           "(pip install pgserver)") from exc
    import os

    import pgserver.postgres_server as pg_module

    folder = Path(url.split(":", 1)[1] or "private/pgdata")
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    folder.mkdir(parents=True, exist_ok=True)
    # pgserver picks a random free port by default; pin it (LOCAL_DB_PORT, default 5433) so that
    # GUI tools such as DBeaver / pgAdmin can keep one saved connection. Only used when the server starts.
    port = int(os.getenv("LOCAL_DB_PORT", "5433"))
    pg_module.find_suitable_port = lambda address=None: port
    server = pgserver.get_server(str(folder), cleanup_mode=None)
    return server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)


def get_engine(database_url: str | None = None) -> Engine:
    """Return the process-wide engine (connection pool is shared; sessions are not)."""
    global _engine, _session_factory
    if database_url is not None:
        engine = create_engine(resolve_database_url(database_url), pool_pre_ping=True, future=True)
        return engine
    if _engine is None:
        _engine = create_engine(resolve_database_url(get_settings().database_url), pool_pre_ping=True,
                                pool_size=5, max_overflow=5)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def configure_engine(engine: Engine) -> None:
    """Point the module at a specific engine (used by tests and scripts)."""
    global _engine, _session_factory
    _engine = engine
    _session_factory = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """One short-lived session per unit of work: commit on success, rollback on error."""
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Engine | None = None, sql_path: Path = INIT_SQL_PATH) -> None:
    """Execute sql/init.sql (idempotent). Called from scripts/init_db.py, not from the UI."""
    engine = engine or get_engine()
    sql = sql_path.read_text(encoding="utf-8")
    # psycopg3 accepts multiple statements in one execute when no parameters are bound.
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)

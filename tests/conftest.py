"""Shared fixtures.

Unit tests need nothing external. Integration tests (marked `integration`) need
PostgreSQL. They use, in order:
  1. TEST_DATABASE_URL  (an existing PostgreSQL test database, e.g.
     postgresql+psycopg://user:password@localhost:5432/recruiter_test)
  2. a throw-away local server via the optional `pgserver` package
  3. otherwise they are skipped.
WARNING: the test database is wiped. Never point TEST_DATABASE_URL at real data.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

os.environ.setdefault("MOCK_MODE", "true")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://unused:unused@localhost:1/unused")

from src.config import load_settings  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "sample_data" / "resumes"


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs a PostgreSQL database")


@pytest.fixture
def settings(tmp_path):
    base = load_settings(validate=False)
    return dataclasses.replace(
        base,
        mock_mode=True, mock_llm=True, mock_calendar=True, mock_email=True,
        resume_storage_dir=tmp_path / "resumes", mock_data_dir=tmp_path / "mock",
        default_timezone="Asia/Kolkata", working_hours_start=10, working_hours_end=18, slot_step_minutes=30,
        min_notice_hours=24, scheduling_horizon_days=10, max_proposed_slots=8, email_max_attempts=3,
        count_internships=False, auto_approve_eligible=False, google_calendar_ids=(), email_from="hr@example.com",
        recruiter_contact_email="hr@example.com", llm_max_repair_attempts=2,
    )


def sample_text(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------
_PG_SERVER = None


def _database_url(tmp_path_factory) -> str | None:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url
    global _PG_SERVER
    try:
        import pgserver  # optional dev dependency
    except ImportError:
        return None
    if _PG_SERVER is None:
        _PG_SERVER = pgserver.get_server(str(tmp_path_factory.mktemp("pgdata")), cleanup_mode="stop")
    uri = _PG_SERVER.get_uri()
    return uri.replace("postgresql://", "postgresql+psycopg://", 1)


@pytest.fixture(scope="session")
def pg_engine(tmp_path_factory):
    url = _database_url(tmp_path_factory)
    if not url:
        pytest.skip("No PostgreSQL available (set TEST_DATABASE_URL or `pip install pgserver`)")
    from sqlalchemy import create_engine, text

    from src.database import configure_engine, init_db

    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS interviews, applications, jobs, audit_log CASCADE"))
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"PostgreSQL not reachable: {type(exc).__name__}")
    init_db(engine)
    configure_engine(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db(pg_engine):
    """Clean tables before each integration test."""
    from sqlalchemy import text

    with pg_engine.begin() as conn:
        conn.execute(text("TRUNCATE interviews, applications, jobs, audit_log RESTART IDENTITY CASCADE"))
    return pg_engine


@pytest.fixture
def ml_job(db):
    from src import repository as repo
    from src.database import session_scope
    from src.schemas import JobCreate, JobSkills
    from src.skill_normalizer import normalize_job_skills

    skills, _ = normalize_job_skills(["Python", "PyTorch", "ML"], ["SQL", "Docker"])
    with session_scope() as s:
        return repo.create_job(s, JobCreate(title="Machine Learning Engineer", min_experience_years=3,
                                            skills=JobSkills.model_validate(skills)), "test")


def make_pdf(text: str, path: Path) -> Path:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 800), text, fontsize=10, fontname="helv")
    doc.save(str(path))
    doc.close()
    return path


def make_docx(text: str, path: Path) -> Path:
    import docx

    document = docx.Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    document.save(str(path))
    return path


@pytest.fixture
def submit(settings, ml_job):
    """Store a sample resume and create its application row. Returns application_id."""
    from src import repository as repo
    from src.database import session_scope
    from src.resume_extractor import save_resume
    from src.schemas import CandidateSubmission

    def _submit(sample: str, email: str | None = None, job_id: int | None = None, text: str | None = None) -> int:
        body = text if text is not None else sample_text(sample)
        tmp = settings.resume_storage_dir.parent / f"{sample}.pdf"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        data = make_pdf(body, tmp).read_bytes()
        stored = save_resume(data, ".pdf", settings.resume_storage_dir)
        with session_scope() as s:
            return repo.create_application(
                s, CandidateSubmission(job_id=job_id or ml_job, name=sample.replace("_", " ").title(),
                                       email=email or f"{sample}@example.com"), stored.name)

    return _submit

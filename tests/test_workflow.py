"""LangGraph workflows end-to-end in mock mode (tests 16, 18, 24, 25 + acceptance flow)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.llm_client import LLMError, MockLLMClient

pytestmark = pytest.mark.integration


def _app(app_id):
    from src import repository as repo
    from src.database import session_scope

    with session_scope() as s:
        a = repo.get_application(s, app_id)
        return {"screening": a.screening_status, "pipeline": a.pipeline_status, "experience": a.experience,
                "skills": a.skills, "profile": a.profile, "result": a.screening_result}


@pytest.fixture
def intake(settings):
    from src.recruitment_workflow import IntakeWorkflow

    return IntakeWorkflow(settings, MockLLMClient())


def test_eligible_candidate(db, intake, submit):
    app_id = submit("priya_sharma")
    state = intake.run(app_id)
    assert state["status"] == "ELIGIBLE", state.get("errors")
    app = _app(app_id)
    assert app["screening"] == "ELIGIBLE" and app["experience"] >= 3
    canon = {s["canonical_name"] for s in app["skills"] if s["validation_status"] == "VALID"}
    assert {"Python", "PyTorch", "Machine Learning"} <= canon
    torch = next(s for s in app["skills"] if s["raw_name"].lower() == "torch")
    assert torch["match_type"] == "ALIAS" and torch["grounded"]
    assert app["profile"]["parser"]["mode"] == "mock"


def test_not_eligible_candidate(db, intake, submit):
    app_id = submit("rahul_verma")
    assert intake.run(app_id)["status"] == "NOT_ELIGIBLE"
    failed = {c["requirement"] for c in _app(app_id)["result"]["checks"] if c["result"] == "FAIL"}
    assert {"minimum_experience", "Python", "PyTorch", "Machine Learning"} <= failed


def test_ambiguous_dates_route_to_review(db, intake, submit):
    app_id = submit("anita_rao")
    state = intake.run(app_id)
    assert state["status"] == "REVIEW_REQUIRED"
    app = _app(app_id)
    assert app["experience"] is None
    assert any("Experience could not be calculated" in r for r in app["profile"]["review_reasons"])


def test_prompt_injection_does_not_help(db, intake, submit):
    app_id = submit("injection_attempt")
    assert intake.run(app_id)["status"] == "NOT_ELIGIBLE"
    assert any("prompt injection" in r for r in _app(app_id)["profile"]["review_reasons"])


def test_upload_only_application_takes_name_from_resume(db, intake, settings, ml_job):
    from src import repository as repo
    from src.database import session_scope
    from src.resume_extractor import find_email, save_resume
    from src.schemas import CandidateSubmission
    from tests.conftest import make_pdf, sample_text

    text = sample_text("priya_sharma")
    data = make_pdf(text, settings.resume_storage_dir.parent / "upload.pdf").read_bytes()
    stored = save_resume(data, ".pdf", settings.resume_storage_dir)
    with session_scope() as s:
        app_id = repo.create_application(
            s, CandidateSubmission(job_id=ml_job, name=repo.NAME_FROM_RESUME, email=find_email(text)), stored.name)
    intake.run(app_id)
    with session_scope() as s:
        app = repo.get_application(s, app_id)
        assert (app.name, app.email) == ("Priya Sharma", "priya.sharma@example.com")


def test_typed_name_is_never_overwritten(db, intake, submit):
    from src import repository as repo
    from src.database import session_scope

    app_id = submit("priya_sharma")  # name typed as "Priya Sharma" by the fixture -> keep whatever was typed
    with session_scope() as s:
        repo.get_application(s, app_id).name = "P. Sharma (typed)"
    intake.run(app_id)
    with session_scope() as s:
        assert repo.get_application(s, app_id).name == "P. Sharma (typed)"


def _reapply(settings, ml_job, sample, email, cooldown=0, now=None):
    from src import repository as repo
    from src.database import session_scope
    from src.resume_extractor import save_resume
    from src.schemas import CandidateSubmission
    from tests.conftest import make_pdf, sample_text

    data = make_pdf(sample_text(sample), settings.resume_storage_dir.parent / f"{sample}-re.pdf").read_bytes()
    stored = save_resume(data, ".pdf", settings.resume_storage_dir)
    with session_scope() as s:
        return repo.reapply(s, CandidateSubmission(job_id=ml_job, name=repo.NAME_FROM_RESUME, email=email),
                            stored.name, cooldown_days=cooldown, now=now)


def test_each_attempt_gets_new_id_and_history_is_kept(db, intake, settings, ml_job):
    from src import repository as repo
    from src.database import session_scope

    first, prev = _reapply(settings, ml_job, "rahul_verma", "same@example.com")
    assert prev == [] and intake.run(first)["status"] == "NOT_ELIGIBLE"

    # re-upload while the first attempt is still in screening -> new ID, first one SUPERSEDED (kept)
    second, prev = _reapply(settings, ml_job, "priya_sharma", "SAME@example.com")
    assert second != first and prev == [first]
    assert intake.run(second)["status"] == "ELIGIBLE"
    with session_scope() as s:
        old, new = repo.get_application(s, first), repo.get_application(s, second)
        assert old.pipeline_status == "SUPERSEDED" and old.screening_status == "NOT_ELIGIBLE"  # history intact
        assert old.resume_path != new.resume_path and new.pipeline_status == "SCREENING"
        assert [a.application_id for a in repo.list_attempts(s, ml_job, "same@example.com")] == [first, second]
        assert [a.application_id for a in repo.list_applications(s, job_id=ml_job)] == [second]  # superseded hidden
        assert "REAPPLIED" in [a.action for a in repo.list_audit(s, "application", second)]


def test_rejected_candidate_can_apply_again(db, intake, settings, ml_job):
    from src import repository as repo
    from src.database import session_scope
    from src.schemas import PipelineStatus

    first, _ = _reapply(settings, ml_job, "rahul_verma", "again@example.com")
    with session_scope() as s:
        repo.set_pipeline_status(s, first, PipelineStatus.REJECTED, "hr")
    second, prev = _reapply(settings, ml_job, "priya_sharma", "again@example.com")
    assert prev == [first]
    with session_scope() as s:
        assert repo.get_application(s, first).pipeline_status == "REJECTED"  # rejection stays on record
        # HR cannot reopen the old attempt while the newer one is active
        with pytest.raises(repo.InvalidTransitionError, match="newer active application"):
            repo.set_pipeline_status(s, first, PipelineStatus.SCREENING, "hr")


def test_reapply_blocked_while_in_progress_and_by_cooldown(db, settings, ml_job):
    from datetime import datetime, timedelta, timezone

    from src import repository as repo
    from src.database import session_scope
    from src.schemas import PipelineStatus

    first, _ = _reapply(settings, ml_job, "priya_sharma", "busy@example.com")
    with session_scope() as s:
        repo.set_pipeline_status(s, first, PipelineStatus.APPROVED, "hr")
    with pytest.raises(repo.ReapplyNotAllowedError, match="approved"):
        _reapply(settings, ml_job, "priya_sharma", "busy@example.com")

    with session_scope() as s:
        repo.set_pipeline_status(s, first, PipelineStatus.REJECTED, "hr")
    with pytest.raises(repo.ReapplyNotAllowedError, match="apply for this job again from"):
        _reapply(settings, ml_job, "priya_sharma", "busy@example.com", cooldown=30)
    later = datetime.now(timezone.utc) + timedelta(days=31)
    new_id, prev = _reapply(settings, ml_job, "priya_sharma", "busy@example.com", cooldown=30, now=later)
    assert prev == [first] and new_id != first


def test_database_allows_only_one_active_application(db, ml_job):
    from sqlalchemy import text

    insert = text("INSERT INTO applications (job_id, email, name, pipeline_status) VALUES (:j, :e, 'X', :p)")
    with db.begin() as conn:
        conn.execute(insert, {"j": ml_job, "e": "a@x.com", "p": "REJECTED"})
        conn.execute(insert, {"j": ml_job, "e": "A@x.com", "p": "SUPERSEDED"})
        conn.execute(insert, {"j": ml_job, "e": "a@x.com", "p": "SCREENING"})  # closed ones don't block
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError, match="ux_applications_job_email_active"):
        with db.begin() as conn:
            conn.execute(insert, {"j": ml_job, "e": "A@X.COM", "p": "SCREENING"})


def test_duplicate_application_policy(db, submit):
    from src import repository as repo

    submit("priya_sharma", email="dup@example.com")
    with pytest.raises(repo.DuplicateApplicationError):
        submit("priya_sharma", email="DUP@example.com")  # case-insensitive


def test_workflow_rerun_is_idempotent(db, intake, submit):
    app_id = submit("priya_sharma")
    first = intake.run(app_id)
    snapshot = _app(app_id)
    second = intake.run(app_id)
    again = _app(app_id)
    assert first["status"] == second["status"]
    assert snapshot["skills"] == again["skills"] and snapshot["experience"] == again["experience"]


class _BrokenLLM(MockLLMClient):
    mode = "llm"

    def __init__(self, error: Exception | None = None):
        self.error = error

    def generate_json(self, **kwargs):
        if self.error:
            raise self.error
        return {"unexpected": True}


def test_invalid_parser_output_goes_to_manual_review(db, settings, submit):
    from src.recruitment_workflow import IntakeWorkflow

    app_id = submit("priya_sharma")
    state = IntakeWorkflow(settings, _BrokenLLM()).run(app_id)
    assert state["status"] == "REVIEW_REQUIRED"
    assert any("Automatic parsing failed" in r for r in _app(app_id)["profile"]["review_reasons"])


def test_llm_outage_marks_error(db, settings, submit):
    from src.recruitment_workflow import IntakeWorkflow

    app_id = submit("priya_sharma")
    state = IntakeWorkflow(settings, _BrokenLLM(LLMError("Could not reach the LLM API"))).run(app_id)
    assert state["status"] == "ERROR"
    app = _app(app_id)
    assert app["screening"] == "ERROR" and "Could not reach" in app["result"]["error"]


def test_hr_correction_rescreens_and_is_audited(db, intake, submit, settings):
    from src import repository as repo
    from src.database import session_scope
    from src.recruitment_workflow import apply_hr_corrections
    from src.schemas import EmploymentEntry, ParsedProfile

    app_id = submit("anita_rao")
    intake.run(app_id)
    profile = ParsedProfile.model_validate(_app(app_id)["profile"])
    fixed = [e if e.start_date else EmploymentEntry(**{**e.model_dump(), "start_date": "2021-01",
                                                       "end_date": "2024-12"})
             for e in profile.employment_history]
    result = apply_hr_corrections(app_id, profile.skills, fixed, "hr:tester", clear_review_reasons=True,
                                  settings=settings)
    assert result["status"] == "ELIGIBLE"
    app = _app(app_id)
    assert app["experience"] == 3 and app["profile"]["human_edited"] is True
    with session_scope() as s:
        actions = [a.action for a in repo.list_audit(s, "application", app_id)]
    assert "PROFILE_EDITED" in actions and "HR_RESCREEN" in actions


def test_end_to_end_schedule_after_approval(db, intake, submit, settings):
    from src import repository as repo
    from src.calendar_service import MockCalendarService
    from src.database import session_scope
    from src.email_service import MockEmailSender
    from src.recruitment_workflow import SchedulingWorkflow
    from src.scheduling_agent import SchedulingAgent
    from src.schemas import PipelineStatus

    app_id = submit("priya_sharma")
    assert intake.run(app_id)["status"] == "ELIGIBLE"
    with session_scope() as s:
        repo.set_pipeline_status(s, app_id, PipelineStatus.APPROVED, "hr:tester")

    now = datetime(2026, 11, 2, 3, 30, tzinfo=timezone.utc)
    agent = SchedulingAgent(settings, MockCalendarService(settings), MockEmailSender(settings),
                            now=lambda: now)
    wf = SchedulingWorkflow(settings, agent)
    proposed = wf.propose(app_id)
    assert proposed["available_slots"]
    from src.schemas import TimeSlot

    booked = wf.book(app_id, TimeSlot.model_validate(proposed["available_slots"][0]), "hr:tester")
    assert booked["booking"]["outcome"] == "CONFIRMED" and booked["email_status"] == "SENT"
    assert _app(app_id)["pipeline"] == "INTERVIEW_SCHEDULED"
    import email
    from email import policy

    raw = next((settings.mock_data_dir / "outbox").glob("*.eml")).read_bytes()
    msg = email.message_from_bytes(raw, policy=policy.default)
    body = msg.get_body(("plain",)).get_content()
    ics = next(msg.iter_attachments()).get_content()
    assert "Machine Learning Engineer" in body and "Asia/Kolkata" in body
    ics = ics.decode() if isinstance(ics, bytes) else ics
    assert "BEGIN:VCALENDAR" in ics

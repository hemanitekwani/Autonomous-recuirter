"""All database reads/writes. Every query is parameterized (ORM or bound `text()` params).

Functions take an explicit Session; transaction boundaries are owned by the
caller via `database.session_scope()`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database import Application, AuditLog, Interview, Job
from src.schemas import (
    CandidateSubmission,
    EmailStatus,
    InterviewStatus,
    JobCreate,
    JobSkills,
    ParsedProfile,
    PipelineStatus,
    ScreeningResult,
    ScreeningStatus,
)

# Placeholder name for upload-only applications; replaced by the name extracted from the resume.
NAME_FROM_RESUME = "(from resume)"

ACTIVE_INTERVIEW_STATUSES =(InterviewStatus.PENDING_CONFIRMATION.value, InterviewStatus.CONFIRMED.value)

# Allowed HR / interview pipeline transitions (enforced in one place).
PIPELINE_TRANSITIONS: dict[str, set[str]] = {
    "SCREENING": {"APPROVED", "REJECTED"},
    "APPROVED": {"INTERVIEW_SCHEDULED", "REJECTED", "SCREENING"},
    "INTERVIEW_SCHEDULED": {"INTERVIEWED", "APPROVED", "REJECTED"},
    "INTERVIEWED": {"SELECTED", "REJECTED"},
    "SELECTED": set(),
    "REJECTED": {"SCREENING"},  # HR may reopen a rejected application (if no newer attempt is active)
    "SUPERSEDED": set(),        # replaced by a newer application of the same candidate
}
# Closed attempts do not block a new application for the same job.
CLOSED_STAGES = ("REJECTED", "SUPERSEDED")
# While one attempt is in these stages, the candidate cannot apply again for the same job.
IN_PROGRESS_STAGES = ("APPROVED", "INTERVIEW_SCHEDULED", "INTERVIEWED", "SELECTED")


class RepositoryError(RuntimeError):
    pass


class NotFoundError(RepositoryError):
    pass


class DuplicateApplicationError(RepositoryError):
    def __init__(self, application_id: int | None):
        super().__init__("An application for this job with this email already exists.")
        self.application_id = application_id


class InvalidTransitionError(RepositoryError):
    pass


class SlotConflictError(RepositoryError):
    """The requested time overlaps another active interview (DB exclusion constraint)."""


class AlreadyScheduledError(RepositoryError):
    """The application already has an active interview."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
def audit(session: Session, entity_type: str, entity_id: int, action: str, actor: str,
          details: dict[str, Any] | None = None) -> None:
    session.add(AuditLog(entity_type=entity_type, entity_id=entity_id, action=action[:60], actor=actor[:100],
                         details=details or {}))


def list_audit(session: Session, entity_type: str, entity_id: int) -> list[AuditLog]:
    stmt = (select(AuditLog).where(AuditLog.entity_type == entity_type, AuditLog.entity_id == entity_id)
            .order_by(AuditLog.created_at.desc()))
    return list(session.scalars(stmt))


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
def create_job(session: Session, job: JobCreate, actor: str = "hr") -> int:
    validated = JobSkills.model_validate(job.skills.model_dump())  # re-validate before persisting JSONB
    row = Job(
        title=job.title,
        min_experience_years=job.min_experience_years,
        skills=validated.model_dump(mode="json"),
        interview_duration_minutes=job.interview_duration_minutes,
        status=job.status,
    )
    session.add(row)
    session.flush()
    audit(session, "job", row.job_id, "JOB_CREATED", actor, {"title": job.title})
    return row.job_id


def get_job(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise NotFoundError(f"Job {job_id} not found")
    return job


def list_jobs(session: Session, status: str | None = None) -> list[Job]:
    stmt = select(Job).order_by(Job.created_at.desc())
    if status:
        stmt = stmt.where(Job.status == status)
    return list(session.scalars(stmt))


def set_job_interview_duration(session: Session, job_id: int, minutes: int, actor: str = "hr") -> None:
    if not 15 <= minutes <= 240:
        raise ValueError("Interview duration must be between 15 and 240 minutes")
    job = get_job(session, job_id)
    old = job.interview_duration_minutes
    job.interview_duration_minutes = minutes
    audit(session, "job", job_id, "JOB_DURATION_CHANGED", actor, {"from": old, "to": minutes})


def set_job_status(session: Session, job_id: int, status: str, actor: str = "hr") -> None:
    if status not in ("OPEN", "CLOSED"):
        raise ValueError("status must be OPEN or CLOSED")
    job = get_job(session, job_id)
    job.status = status
    audit(session, "job", job_id, f"JOB_{status}", actor)


# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------
def _same_candidate(job_id: int, email: str):
    return (Application.job_id == job_id, func.lower(Application.email) == email.strip().lower())


def find_active_application(session: Session, job_id: int, email: str) -> Application | None:
    """The single non-closed application of this candidate for this job (closed = REJECTED / SUPERSEDED)."""
    stmt = select(Application).where(*_same_candidate(job_id, email),
                                     Application.pipeline_status.not_in(CLOSED_STAGES))
    return session.scalars(stmt).first()


def list_attempts(session: Session, job_id: int, email: str) -> list[Application]:
    """Every application (attempt) of this candidate for this job, oldest first."""
    stmt = select(Application).where(*_same_candidate(job_id, email)).order_by(Application.created_at,
                                                                                  Application.application_id)
    return list(session.scalars(stmt))


def create_application(session: Session, submission: CandidateSubmission, resume_path: str) -> int:
    """Insert a new application. Policy: only one ACTIVE application per (job, email) - see `reapply`."""
    job = get_job(session, submission.job_id)
    if job.status != "OPEN":
        raise RepositoryError("This job is not accepting applications.")
    existing = find_active_application(session, submission.job_id, submission.email)
    if existing is not None:
        raise DuplicateApplicationError(existing.application_id)
    row = Application(job_id=submission.job_id, email=submission.email.lower(), name=submission.name,
                      resume_path=resume_path, skills=[], profile={}, screening_result={})
    try:
        with session.begin_nested():  # savepoint so a race on the unique index leaves the session usable
            session.add(row)
            session.flush()
    except IntegrityError as exc:
        if "ux_applications_job_email_active" in str(exc.orig):
            raise DuplicateApplicationError(None) from exc
        raise
    audit(session, "application", row.application_id, "APPLICATION_SUBMITTED", "candidate",
          {"job_id": submission.job_id})
    return row.application_id


class ReapplyNotAllowedError(RepositoryError):
    """Candidate may not apply again right now (application in progress, selected, or cooldown)."""


def reapply(session: Session, submission: CandidateSubmission, resume_path: str,
            cooldown_days: int = 0, now: datetime | None = None) -> tuple[int, list[int]]:
    """Apply (again) for a job. Every attempt gets a NEW application ID; older attempts are kept as history.

    * no previous attempt              -> new application
    * previous attempt still SCREENING -> it becomes SUPERSEDED, new application created
    * previous attempt REJECTED        -> new application (after REAPPLY_COOLDOWN_DAYS, default 0)
    * previous attempt APPROVED / interview / SELECTED -> refused (process already under way)
    Returns (new_application_id, ids_of_previous_attempts).
    """
    now = now or _now()
    attempts = list_attempts(session, submission.job_id, submission.email)
    for previous in attempts:
        if previous.pipeline_status in IN_PROGRESS_STAGES:
            stage = previous.pipeline_status.replace("_", " ").lower()
            raise ReapplyNotAllowedError(
                f"You already have application #{previous.application_id} for this job at the '{stage}' stage. "
                "Please contact the recruiting team instead of applying again.")
    last_rejection = max((a.updated_at for a in attempts if a.pipeline_status == "REJECTED"), default=None)
    if cooldown_days and last_rejection is not None and now < last_rejection + timedelta(days=cooldown_days):
        allowed = (last_rejection + timedelta(days=cooldown_days)).date().isoformat()
        raise ReapplyNotAllowedError(f"You can apply for this job again from {allowed}.")

    for previous in attempts:
        if previous.pipeline_status == PipelineStatus.SCREENING.value:
            previous.pipeline_status = PipelineStatus.SUPERSEDED.value
            audit(session, "application", previous.application_id, "PIPELINE_SUPERSEDED", "candidate",
                  {"reason": "Candidate submitted a new resume for the same job"})
    session.flush()  # free the "one active application" slot before inserting the new attempt
    new_id = create_application(session, submission, resume_path)
    previous_ids = [a.application_id for a in attempts]
    if previous_ids:
        audit(session, "application", new_id, "REAPPLIED", "candidate", {"previous_attempts": previous_ids})
    return new_id, previous_ids


def get_application(session: Session, application_id: int, for_update: bool = False) -> Application:
    stmt = select(Application).where(Application.application_id == application_id)
    if for_update:
        stmt = stmt.with_for_update()
    app = session.scalars(stmt).first()
    if app is None:
        raise NotFoundError(f"Application {application_id} not found")
    return app


def candidate_lookup(session: Session, application_id: int, email: str) -> Application | None:
    """Candidates can only see their own application (id + email must both match)."""
    stmt = select(Application).where(Application.application_id == application_id,
                                     func.lower(Application.email) == email.strip().lower())
    return session.scalars(stmt).first()


def list_applications(session: Session, job_id: int | None = None, screening_status: str | None = None,
                      pipeline_statuses: Iterable[str] | None = None,
                      include_superseded: bool = False) -> list[Application]:
    stmt = select(Application).order_by(Application.created_at.desc())
    if not include_superseded and not pipeline_statuses:
        stmt = stmt.where(Application.pipeline_status != PipelineStatus.SUPERSEDED.value)
    if job_id is not None:
        stmt = stmt.where(Application.job_id == job_id)
    if screening_status:
        stmt = stmt.where(Application.screening_status == screening_status)
    if pipeline_statuses:
        stmt = stmt.where(Application.pipeline_status.in_(list(pipeline_statuses)))
    return list(session.scalars(stmt))


def set_screening_status(session: Session, application_id: int, status: ScreeningStatus,
                         error: str | None = None) -> None:
    values: dict[str, Any] = {"screening_status": status.value}
    if error is not None:
        values["screening_result"] = {"error": error[:500]}
    session.execute(update(Application).where(Application.application_id == application_id).values(**values))


def save_profile(session: Session, application_id: int, profile: ParsedProfile, actor: str = "system") -> None:
    validated = ParsedProfile.model_validate(profile.model_dump(mode="json"))
    experience = validated.experience.completed_years if validated.experience else None
    session.execute(
        update(Application)
        .where(Application.application_id == application_id)
        .values(
            skills=[s.model_dump(mode="json") for s in validated.skills],
            experience=experience,
            profile=validated.model_dump(mode="json"),
        )
    )
    if actor != "system":
        audit(session, "application", application_id, "PROFILE_EDITED", actor,
              {"experience": experience, "skills": [s.canonical_name or s.raw_name for s in validated.skills]})


def save_screening(session: Session, application_id: int, result: ScreeningResult) -> None:
    validated = ScreeningResult.model_validate(result.model_dump(mode="json"))
    session.execute(
        update(Application)
        .where(Application.application_id == application_id)
        .values(screening_status=validated.status, screening_result=validated.model_dump(mode="json"))
    )


def set_pipeline_status(session: Session, application_id: int, new_status: PipelineStatus, actor: str,
                        reason: str | None = None) -> None:
    app = get_application(session, application_id, for_update=True)
    allowed = PIPELINE_TRANSITIONS.get(app.pipeline_status, set())
    if new_status.value not in allowed:
        raise InvalidTransitionError(f"Cannot move application from {app.pipeline_status} to {new_status.value}")
    if app.pipeline_status in CLOSED_STAGES and new_status.value not in CLOSED_STAGES:
        newer = find_active_application(session, app.job_id, app.email)
        if newer is not None and newer.application_id != application_id:
            raise InvalidTransitionError(
                f"Cannot reopen #{application_id}: the candidate has a newer active application #{newer.application_id}")
    old = app.pipeline_status
    app.pipeline_status = new_status.value
    audit(session, "application", application_id, f"PIPELINE_{new_status.value}", actor,
          {"from": old, "to": new_status.value, "reason": reason})


def dashboard_counts(session: Session) -> dict[str, int]:
    counts: dict[str, int] = {
        "jobs": session.scalar(select(func.count()).select_from(Job)) or 0,
        "open_jobs": session.scalar(select(func.count()).select_from(Job).where(Job.status == "OPEN")) or 0,
        "applications": session.scalar(select(func.count()).select_from(Application).where(
            Application.pipeline_status != PipelineStatus.SUPERSEDED.value)) or 0,
        "interviews_confirmed": session.scalar(
            select(func.count()).select_from(Interview).where(Interview.status == "CONFIRMED")) or 0,
    }
    current = Application.pipeline_status != PipelineStatus.SUPERSEDED.value  # count latest attempts only
    for status, n in session.execute(select(Application.screening_status, func.count()).where(current).group_by(
            Application.screening_status)):
        counts[status] = n
    return counts


# ---------------------------------------------------------------------------
# Interviews
# ---------------------------------------------------------------------------
def busy_intervals(session: Session, start: datetime, end: datetime,
                   exclude_interview_id: int | None = None) -> list[tuple[datetime, datetime]]:
    stmt = select(Interview.starts_at, Interview.ends_at).where(
        Interview.status.in_(ACTIVE_INTERVIEW_STATUSES), Interview.starts_at < end, Interview.ends_at > start)
    if exclude_interview_id is not None:
        stmt = stmt.where(Interview.interview_id != exclude_interview_id)
    return [(r.starts_at, r.ends_at) for r in session.execute(stmt)]


def reserve_interview(session: Session, application: Application, starts_at: datetime, ends_at: datetime,
                      tz_name: str, event_id: str) -> Interview:
    """Insert a PENDING_CONFIRMATION row. The exclusion constraint blocks overlapping bookings atomically."""
    row = Interview(job_id=application.job_id, application_id=application.application_id, starts_at=starts_at,
                    ends_at=ends_at, timezone=tz_name, status=InterviewStatus.PENDING_CONFIRMATION.value,
                    google_event_id=event_id)
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError as exc:
        msg = str(exc.orig)
        if "ex_interviews_no_overlap" in msg:
            raise SlotConflictError("That time overlaps another interview.") from exc
        if "ux_interviews_one_active" in msg:
            raise AlreadyScheduledError("This candidate already has an active interview.") from exc
        raise
    audit(session, "interview", row.interview_id, "INTERVIEW_RESERVED", "scheduler",
          {"application_id": application.application_id, "starts_at": starts_at.isoformat()})
    return row


def get_interview(session: Session, interview_id: int, for_update: bool = False) -> Interview:
    stmt = select(Interview).where(Interview.interview_id == interview_id)
    if for_update:
        stmt = stmt.with_for_update()
    row = session.scalars(stmt).first()
    if row is None:
        raise NotFoundError(f"Interview {interview_id} not found")
    return row


def get_active_interview(session: Session, application_id: int) -> Interview | None:
    stmt = select(Interview).where(Interview.application_id == application_id,
                                   Interview.status.in_(ACTIVE_INTERVIEW_STATUSES))
    return session.scalars(stmt).first()


def list_interviews(session: Session, statuses: Iterable[str] | None = None) -> list[Interview]:
    stmt = select(Interview).order_by(Interview.starts_at)
    if statuses:
        stmt = stmt.where(Interview.status.in_(list(statuses)))
    return list(session.scalars(stmt))


def confirm_interview(session: Session, interview_id: int, meeting_link: str | None) -> None:
    row = get_interview(session, interview_id, for_update=True)
    if row.status == InterviewStatus.CONFIRMED.value:
        return  # idempotent
    if row.status != InterviewStatus.PENDING_CONFIRMATION.value:
        raise InvalidTransitionError(f"Interview {interview_id} is {row.status}, cannot confirm")
    row.status = InterviewStatus.CONFIRMED.value
    row.meeting_link = meeting_link
    row.last_error = None
    app = get_application(session, row.application_id, for_update=True)
    if app.pipeline_status == PipelineStatus.APPROVED.value:
        app.pipeline_status = PipelineStatus.INTERVIEW_SCHEDULED.value
    audit(session, "interview", interview_id, "INTERVIEW_CONFIRMED", "scheduler",
          {"application_id": row.application_id})


def cancel_interview(session: Session, interview_id: int, actor: str, reason: str) -> Interview:
    row = get_interview(session, interview_id, for_update=True)
    if row.status in (InterviewStatus.CANCELLED.value, InterviewStatus.COMPLETED.value):
        return row
    was_confirmed = row.status == InterviewStatus.CONFIRMED.value
    row.status = InterviewStatus.CANCELLED.value
    row.last_error = reason[:500] if actor == "scheduler" else row.last_error
    app = get_application(session, row.application_id, for_update=True)
    if was_confirmed and app.pipeline_status == PipelineStatus.INTERVIEW_SCHEDULED.value:
        app.pipeline_status = PipelineStatus.APPROVED.value
    audit(session, "interview", interview_id, "INTERVIEW_CANCELLED", actor, {"reason": reason[:500]})
    return row


def complete_interview(session: Session, interview_id: int, actor: str) -> None:
    row = get_interview(session, interview_id, for_update=True)
    if row.status != InterviewStatus.CONFIRMED.value:
        raise InvalidTransitionError("Only confirmed interviews can be marked completed")
    row.status = InterviewStatus.COMPLETED.value
    app = get_application(session, row.application_id)
    if app.pipeline_status == PipelineStatus.INTERVIEW_SCHEDULED.value:
        set_pipeline_status(session, row.application_id, PipelineStatus.INTERVIEWED, actor, "Interview completed")
    audit(session, "interview", interview_id, "INTERVIEW_COMPLETED", actor)


def complete_past_interviews(session: Session, now: datetime | None = None) -> int:
    """Mark confirmed interviews whose end time has passed as COMPLETED (and the application INTERVIEWED)."""
    now = now or _now()
    ids = list(session.scalars(select(Interview.interview_id).where(
        Interview.status == InterviewStatus.CONFIRMED.value, Interview.ends_at < now)))
    for interview_id in ids:
        complete_interview(session, interview_id, "system")
    return len(ids)


def set_interview_error(session: Session, interview_id: int, error: str) -> None:
    session.execute(update(Interview).where(Interview.interview_id == interview_id).values(last_error=error[:500]))


def stale_pending_interviews(session: Session, older_than_minutes: int = 5) -> list[Interview]:
    cutoff = _now() - timedelta(minutes=older_than_minutes)
    stmt = select(Interview).where(Interview.status == InterviewStatus.PENDING_CONFIRMATION.value,
                                   Interview.updated_at < cutoff)
    return list(session.scalars(stmt))


def claim_email_send(session: Session, interview_id: int, max_attempts: int) -> bool:
    """Atomically claim the right to send the invitation. Prevents duplicate sends across retries/workers."""
    claimed = session.execute(
        text(
            """
            UPDATE interviews
               SET email_status = 'SENDING', email_attempts = email_attempts + 1
             WHERE interview_id = :id
               AND status = 'CONFIRMED'
               AND email_status IN ('NOT_SENT', 'FAILED')
               AND email_attempts < :max_attempts
         RETURNING interview_id
            """
        ),
        {"id": interview_id, "max_attempts": max_attempts},
    ).first()
    return claimed is not None


def mark_email_result(session: Session, interview_id: int, ok: bool, error: str | None = None) -> None:
    values: dict[str, Any] = {"email_status": EmailStatus.SENT.value if ok else EmailStatus.FAILED.value}
    if ok:
        values["email_sent_at"] = _now()
    else:
        values["last_error"] = (error or "email failed")[:500]
    session.execute(update(Interview).where(Interview.interview_id == interview_id).values(**values))
    audit(session, "interview", interview_id, "EMAIL_SENT" if ok else "EMAIL_FAILED", "scheduler",
          {"error": error[:200] if error else None})


def reset_stuck_email(session: Session, interview_id: int, actor: str) -> None:
    """HR action: a crash between claim and send leaves SENDING; HR decides to allow a resend."""
    session.execute(update(Interview).where(Interview.interview_id == interview_id,
                                            Interview.email_status == EmailStatus.SENDING.value)
                    .values(email_status=EmailStatus.FAILED.value))
    audit(session, "interview", interview_id, "EMAIL_RESET", actor)

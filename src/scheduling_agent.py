"""Agent 2 - interview scheduling.

Booking lifecycle (Google Calendar and PostgreSQL share no transaction):

  1. Validate: application APPROVED, no active interview, slot shape valid.
  2. Re-check live calendar availability for the exact slot.
  3. RESERVE in PostgreSQL: insert interviews row PENDING_CONFIRMATION with a
     pre-generated google_event_id. The exclusion constraint makes this the
     atomic, concurrency-safe "lock" on the time range.
  4. CREATE the calendar event with that id (idempotent: a retry gets 409 and
     re-uses the existing event; never a duplicate).
       - failure -> compensate: cancel the reservation (slot released).
  5. CONFIRM in PostgreSQL (status CONFIRMED, application INTERVIEW_SCHEDULED).
       - failure -> row stays PENDING_CONFIRMATION with the event id;
         `reconcile_pending()` later confirms it (event exists) or cancels it.
  6. SEND the invitation only for CONFIRMED rows, guarded by an atomic claim
     (email_status) so retries never send twice. Email failure never touches
     the booking; it is retried separately.

A booking is reported as successful only after step 5 commits.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

from src import repository as repo
from src.calendar_service import CalendarError, CalendarService, generate_slots, new_event_id
from src.config import Settings
from src.database import session_scope
from src.email_service import EmailDeliveryError, EmailSender, booking_details, build_invitation
from src.schemas import BookingResult, EmailStatus, InterviewStatus, PipelineStatus, TimeSlot

log = logging.getLogger(__name__)


class SchedulingError(RuntimeError):
    pass


class SchedulingAgent:
    def __init__(self, settings: Settings, calendar: CalendarService, email_sender: EmailSender,
                 now: Callable[[], datetime] | None = None):
        self.settings = settings
        self.calendar = calendar
        self.email_sender = email_sender
        self._now = now or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------ slots
    def _window(self) -> tuple[datetime, datetime]:
        now = self._now()
        return now + timedelta(hours=self.settings.min_notice_hours), now + timedelta(
            days=self.settings.scheduling_horizon_days)

    def find_available_slots(self, application_id: int, limit: int | None = None) -> list[TimeSlot]:
        with session_scope() as s:
            app = repo.get_application(s, application_id)
            if app.pipeline_status != PipelineStatus.APPROVED.value:
                raise SchedulingError(f"Application {application_id} is {app.pipeline_status}; "
                                      "only APPROVED candidates can be scheduled")
            if repo.get_active_interview(s, application_id) is not None:
                raise SchedulingError("This candidate already has an active interview")
            duration = repo.get_job(s, app.job_id).interview_duration_minutes
            window_start, window_end = self._window()
            db_busy = repo.busy_intervals(s, window_start, window_end)
        calendar_busy = self.calendar.free_busy(window_start, window_end)
        return generate_slots(
            window_start, window_end, duration, self.settings.default_timezone,
            self.settings.working_hours_start, self.settings.working_hours_end, self.settings.slot_step_minutes,
            db_busy + calendar_busy, limit=limit or self.settings.max_proposed_slots,
        )

    def _slot_is_free(self, slot: TimeSlot) -> bool:
        busy = self.calendar.free_busy(slot.starts_at, slot.ends_at)
        return not any(b_start < slot.ends_at and slot.starts_at < b_end for b_start, b_end in busy)

    # ---------------------------------------------------------------- booking
    def book_slot(self, application_id: int, slot: TimeSlot, actor: str = "hr") -> BookingResult:
        # 1. validate
        with session_scope() as s:
            app = repo.get_application(s, application_id)
            job = repo.get_job(s, app.job_id)
            if app.pipeline_status != PipelineStatus.APPROVED.value:
                return BookingResult(outcome="INVALID", message=f"Application is {app.pipeline_status}, not APPROVED")
            if repo.get_active_interview(s, application_id) is not None:
                return BookingResult(outcome="INVALID", message="Candidate already has an active interview")
            expected = timedelta(minutes=job.interview_duration_minutes)
            if slot.ends_at - slot.starts_at != expected:
                return BookingResult(outcome="INVALID", message="Slot length does not match the interview duration")
            if slot.starts_at < self._now() + timedelta(hours=self.settings.min_notice_hours):
                return BookingResult(outcome="INVALID", message="Slot is too soon (minimum notice not met)")
            candidate = {"name": app.name, "email": app.email, "job_title": job.title}

        # 2. live re-check right before booking
        try:
            if not self._slot_is_free(slot):
                return self._conflict(application_id, "The panel is no longer free at that time")
        except CalendarError as exc:
            return BookingResult(outcome="CALENDAR_ERROR", message=str(exc))

        # 3. reserve (atomic, exclusion constraint)
        event_id = new_event_id()
        try:
            with session_scope() as s:
                app = repo.get_application(s, application_id, for_update=True)
                row = repo.reserve_interview(s, app, slot.starts_at, slot.ends_at, slot.timezone, event_id)
                interview_id = row.interview_id
        except repo.SlotConflictError:
            return self._conflict(application_id, "Another interview was booked in that time range")
        except repo.AlreadyScheduledError as exc:
            return BookingResult(outcome="INVALID", message=str(exc))

        # 4. create calendar event (idempotent id)
        try:
            event = self._create_event(event_id, candidate, slot)
        except CalendarError as exc:
            with session_scope() as s:  # compensate: release the reservation
                repo.cancel_interview(s, interview_id, "scheduler", f"Calendar event creation failed: {exc}")
            return BookingResult(outcome="CALENDAR_ERROR", interview_id=interview_id,
                                 message=f"{exc}. The reservation was released; please retry.")

        # 5. confirm in DB
        try:
            with session_scope() as s:
                repo.confirm_interview(s, interview_id, event.meeting_link)
        except Exception as exc:  # DB failure after the external side effect
            log.exception("Confirming interview %s failed after calendar event creation", interview_id)
            try:
                with session_scope() as s:
                    repo.set_interview_error(s, interview_id, f"Confirmation pending reconciliation: {exc}")
            except Exception:
                pass
            return BookingResult(outcome="DB_ERROR", interview_id=interview_id,
                                 message="The calendar event exists but the database update failed. "
                                         "Run 'Reconcile pending bookings' to finish it (no duplicate is created).")
        with session_scope() as s:
            repo.audit(s, "interview", interview_id, "BOOKED_BY", actor, {"application_id": application_id})
        return BookingResult(outcome="CONFIRMED", interview_id=interview_id, message="Interview confirmed")

    def _create_event(self, event_id: str, candidate: dict, slot: TimeSlot):
        attendees = [candidate["email"]] + [c for c in self.settings.google_calendar_ids if "@" in c]
        return self.calendar.create_event(
            event_id=event_id,
            summary=f"Interview: {candidate['name']} - {candidate['job_title']}",
            description=f"Interview for {candidate['job_title']} (scheduled by AI Recruiter)",
            start=slot.starts_at, end=slot.ends_at, tz_name=slot.timezone, attendees=attendees,
        )

    def _conflict(self, application_id: int, message: str) -> BookingResult:
        try:
            alternatives = self.find_available_slots(application_id)
        except (CalendarError, SchedulingError):
            alternatives = []
        return BookingResult(outcome="CONFLICT", message=message, alternative_slots=alternatives)

    # -------------------------------------------------------------- reconcile
    def reconcile_pending(self, older_than_minutes: int = 5) -> dict[str, list[int]]:
        """Resolve PENDING_CONFIRMATION rows left by crashes / DB failures."""
        report: dict[str, list[int]] = {"confirmed": [], "released": [], "skipped": []}
        with session_scope() as s:
            pending = [(i.interview_id, i.google_event_id) for i in repo.stale_pending_interviews(s, older_than_minutes)]
        for interview_id, event_id in pending:
            try:
                event = self.calendar.get_event(event_id) if event_id else None
            except CalendarError:
                report["skipped"].append(interview_id)
                continue
            with session_scope() as s:
                if event is not None and event.status != "cancelled":
                    repo.confirm_interview(s, interview_id, event.meeting_link)
                    report["confirmed"].append(interview_id)
                else:
                    repo.cancel_interview(s, interview_id, "scheduler", "Reconciliation: no calendar event found")
                    report["released"].append(interview_id)
        return report

    # ------------------------------------------------------- cancel / complete
    def cancel(self, interview_id: int, actor: str, reason: str) -> None:
        with session_scope() as s:
            row = repo.get_interview(s, interview_id)
            event_id, status = row.google_event_id, row.status
        if status in (InterviewStatus.CANCELLED.value, InterviewStatus.COMPLETED.value):
            return
        if event_id:
            self.calendar.cancel_event(event_id)  # raise -> DB untouched, state stays consistent
        with session_scope() as s:
            repo.cancel_interview(s, interview_id, actor, reason)

    def reschedule(self, interview_id: int, new_slot: TimeSlot, actor: str) -> BookingResult:
        with session_scope() as s:
            application_id = repo.get_interview(s, interview_id).application_id
        self.cancel(interview_id, actor, "Rescheduled")
        result = self.book_slot(application_id, new_slot, actor)
        if result.outcome == "CONFIRMED" and result.interview_id:
            self.send_invitation(result.interview_id)
        return result

    # ------------------------------------------------------------------ email
    def send_invitation(self, interview_id: int) -> str:
        """Send the invitation at most once. Returns the resulting email_status."""
        with session_scope() as s:
            if not repo.claim_email_send(s, interview_id, self.settings.email_max_attempts):
                row = repo.get_interview(s, interview_id)
                return row.email_status
            row = repo.get_interview(s, interview_id)
            app = repo.get_application(s, row.application_id)
            job = repo.get_job(s, row.job_id)
            details = booking_details(
                candidate_name=app.name, candidate_email=app.email, job_title=job.title,
                starts_at=row.starts_at, ends_at=row.ends_at, tz_name=row.timezone,
                meeting_link=row.meeting_link, contact_email=self.settings.recruiter_contact_email,
                event_id=row.google_event_id or str(interview_id),
            )
        try:
            message = build_invitation(details, self.settings)
            self.email_sender.send(message, interview_id)
        except EmailDeliveryError as exc:
            with session_scope() as s:
                repo.mark_email_result(s, interview_id, ok=False, error=str(exc))
            return EmailStatus.FAILED.value
        with session_scope() as s:
            repo.mark_email_result(s, interview_id, ok=True)
        return EmailStatus.SENT.value

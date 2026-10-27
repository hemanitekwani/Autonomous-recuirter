"""Interview invitation drafting and delivery.

* The email is rendered from a fixed template using only VALIDATED booking
  details (name, job title, date, start/end time, timezone, meeting link),
  plus an .ics calendar attachment.
* Sending is guarded by `repository.claim_email_send` (see scheduling_agent),
  so retries never send twice.
* In mock mode the message is written to the outbox folder instead of sent.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from src.config import Settings
from src.schemas import EmailMessageContent

log = logging.getLogger(__name__)


class EmailDeliveryError(RuntimeError):
    """Delivery failed. Message never includes credentials."""


def booking_details(*, candidate_name: str, candidate_email: str, job_title: str, starts_at: datetime,
                    ends_at: datetime, tz_name: str, meeting_link: str | None, contact_email: str,
                    event_id: str) -> dict[str, Any]:
    tz = ZoneInfo(tz_name)
    start_local, end_local = starts_at.astimezone(tz), ends_at.astimezone(tz)
    return {
        "candidate_name": candidate_name,
        "candidate_email": candidate_email,
        "job_title": job_title,
        "date": start_local.strftime("%A, %d %B %Y"),
        "start_time": start_local.strftime("%H:%M"),
        "end_time": end_local.strftime("%H:%M"),
        "timezone": tz_name,
        "format": "Video interview" + (" (online meeting link below)" if meeting_link else ""),
        "meeting_link": meeting_link,
        "contact_email": contact_email or "the recruiting team (reply to this email)",
        "starts_at_utc": starts_at.astimezone(timezone.utc).isoformat(),
        "ends_at_utc": ends_at.astimezone(timezone.utc).isoformat(),
        "event_id": event_id,
    }


def render_template(d: dict[str, Any]) -> tuple[str, str]:
    subject = f"Interview invitation: {d['job_title']} - {d['date']}"
    link_line = f"Meeting link: {d['meeting_link']}\n" if d.get("meeting_link") else ""
    body = (
        f"Dear {d['candidate_name']},\n\n"
        f"Thank you for applying for the {d['job_title']} position. We are pleased to invite you to an interview.\n\n"
        f"Date: {d['date']}\n"
        f"Time: {d['start_time']} - {d['end_time']} ({d['timezone']})\n"
        f"Format: {d['format']}\n"
        f"{link_line}\n"
        "A calendar invitation (.ics) is attached.\n\n"
        f"If you need to reschedule, please contact {d['contact_email']} at least 24 hours before the interview.\n\n"
        "Best regards,\nRecruiting Team\n"
    )
    return subject, body


def build_ics(d: dict[str, Any], organizer: str) -> str:
    def fmt(iso: str) -> str:
        return datetime.fromisoformat(iso).strftime("%Y%m%dT%H%M%SZ")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    description = f"Interview for {d['job_title']}" + (f"\\nJoin: {d['meeting_link']}" if d.get("meeting_link") else "")
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Autonomous AI Recruiter//EN", "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{d['event_id']}@ai-recruiter",
        f"DTSTAMP:{now}",
        f"DTSTART:{fmt(d['starts_at_utc'])}",
        f"DTEND:{fmt(d['ends_at_utc'])}",
        f"SUMMARY:Interview - {d['job_title']}",
        f"DESCRIPTION:{description}",
        f"ORGANIZER:mailto:{organizer}" if organizer else "ORGANIZER:mailto:noreply@example.invalid",
        f"ATTENDEE;ROLE=REQ-PARTICIPANT;RSVP=TRUE:mailto:{d['candidate_email']}",
    ]
    if d.get("meeting_link"):
        lines.append(f"LOCATION:{d['meeting_link']}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(lines) + "\r\n"


def build_invitation(d: dict[str, Any], settings: Settings) -> EmailMessageContent:
    subject, body = render_template(d)
    try:
        return EmailMessageContent(to=d["candidate_email"], subject=subject, body=body,
                                   ics=build_ics(d, settings.email_from))
    except ValidationError as exc:
        raise EmailDeliveryError("Invitation content failed validation") from exc


class EmailSender(Protocol):
    def send(self, message: EmailMessageContent, interview_id: int) -> None: ...


def _to_mime(message: EmailMessageContent, sender: str) -> EmailMessage:
    mime = EmailMessage()
    mime["From"] = sender or "noreply@example.invalid"
    mime["To"] = message.to
    mime["Subject"] = message.subject
    mime.set_content(message.body)
    if message.ics:
        mime.add_attachment(message.ics.encode("utf-8"), maintype="text", subtype="calendar",
                            filename="interview.ics", params={"method": "REQUEST"})
    return mime


class SmtpEmailSender:
    def __init__(self, settings: Settings):
        self._s = settings

    def send(self, message: EmailMessageContent, interview_id: int) -> None:
        mime = _to_mime(message, self._s.email_from)
        try:
            with smtplib.SMTP(self._s.smtp_host, self._s.smtp_port, timeout=30) as smtp:
                if self._s.smtp_use_tls:
                    smtp.starttls(context=ssl.create_default_context())
                if self._s.smtp_username:
                    smtp.login(self._s.smtp_username, self._s.smtp_password)
                smtp.send_message(mime)
        except smtplib.SMTPAuthenticationError as exc:
            raise EmailDeliveryError("SMTP authentication failed - check SMTP_USERNAME / SMTP_PASSWORD") from exc
        except (smtplib.SMTPException, OSError) as exc:
            raise EmailDeliveryError(f"SMTP delivery failed ({type(exc).__name__})") from exc


class MockEmailSender:
    """Writes .eml files to <MOCK_DATA_DIR>/outbox instead of sending. `fail_next` simulates failures."""

    def __init__(self, settings: Settings, outbox: Path | None = None):
        self._outbox = outbox or (settings.mock_data_dir / "outbox")
        self._from = settings.email_from
        self.fail_next = 0

    def send(self, message: EmailMessageContent, interview_id: int) -> None:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise EmailDeliveryError("Mock email failure (simulated)")
        self._outbox.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = self._outbox / f"interview-{interview_id}-{stamp}.eml"
        path.write_bytes(bytes(_to_mime(message, self._from)))
        log.info("Mock email for interview %s written to %s", interview_id, path)


def get_email_sender(settings: Settings) -> EmailSender:
    return MockEmailSender(settings) if settings.mock_email else SmtpEmailSender(settings)

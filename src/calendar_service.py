"""Google Calendar integration (+ deterministic mock) and slot generation.

The interview panel = the calendars in GOOGLE_CALENDAR_IDS. All panel members
must attend, so availability is the INTERSECTION of their free time (we take
the UNION of their busy blocks). Events are created on the first calendar.

Idempotency: the event id is generated before calling Google and stored in
the interviews row first. A retry re-uses the same id; Google answers 409 for
an existing id and we return that event instead of creating a duplicate.

OAuth setup: see README ("Google Calendar OAuth"). Run once:
    python -m scripts.google_auth
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

from src.config import PROJECT_ROOT, Settings
from src.schemas import TimeSlot

SCOPES = ["https://www.googleapis.com/auth/calendar"]


class CalendarError(RuntimeError):
    """Calendar API failure (message is safe to show; never contains tokens)."""


@dataclass(frozen=True)
class CalendarEvent:
    event_id: str
    status: str  # confirmed | tentative | cancelled
    starts_at: datetime | None
    ends_at: datetime | None
    meeting_link: str | None
    html_link: str | None


class CalendarService(Protocol):
    def free_busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]: ...

    def create_event(self, event_id: str, summary: str, description: str, start: datetime, end: datetime,
                     tz_name: str, attendees: list[str]) -> CalendarEvent: ...

    def get_event(self, event_id: str) -> CalendarEvent | None: ...

    def update_event(self, event_id: str, start: datetime, end: datetime, tz_name: str) -> CalendarEvent: ...

    def cancel_event(self, event_id: str) -> None: ...


def new_event_id() -> str:
    """Google event ids may use base32hex characters (a-v, 0-9); uuid hex is a subset."""
    return "rec" + uuid.uuid4().hex


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# ---------------------------------------------------------------------------
# Slot generation (pure)
# ---------------------------------------------------------------------------
def generate_slots(
    window_start: datetime,
    window_end: datetime,
    duration_minutes: int,
    tz_name: str,
    work_start_hour: int,
    work_end_hour: int,
    step_minutes: int,
    busy: list[tuple[datetime, datetime]],
    limit: int | None = None,
) -> list[TimeSlot]:
    """Weekday working-hour slots in `tz_name` that do not overlap any busy interval."""
    if window_start.tzinfo is None or window_end.tzinfo is None:
        raise ValueError("window datetimes must be timezone-aware")
    tz = ZoneInfo(tz_name)
    duration = timedelta(minutes=duration_minutes)
    step = timedelta(minutes=step_minutes)
    busy_sorted = sorted((b[0].astimezone(timezone.utc), b[1].astimezone(timezone.utc)) for b in busy)
    slots: list[TimeSlot] = []
    day: date = window_start.astimezone(tz).date()
    last_day = window_end.astimezone(tz).date()
    while day <= last_day:
        if day.weekday() < 5:
            cursor = datetime.combine(day, time(work_start_hour), tzinfo=tz)
            day_end = (datetime.combine(day, time(0), tzinfo=tz) + timedelta(hours=work_end_hour))
            while cursor + duration <= day_end:
                start_utc = cursor.astimezone(timezone.utc)
                end_utc = start_utc + duration
                if start_utc >= window_start and end_utc <= window_end and not any(
                    b_start < end_utc and start_utc < b_end for b_start, b_end in busy_sorted
                ):
                    slots.append(TimeSlot(starts_at=cursor, ends_at=(cursor + duration), timezone=tz_name))
                    if limit is not None and len(slots) >= limit:
                        return slots
                cursor = cursor + step
        day += timedelta(days=1)
    return slots


# ---------------------------------------------------------------------------
# Google implementation
# ---------------------------------------------------------------------------
def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def authorize_interactively(settings: Settings) -> Path:
    """Run the OAuth consent flow in a local browser and store token.json (one-time setup)."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = _resolve(settings.google_client_secrets_file)
    if not secrets.exists():
        raise CalendarError(f"OAuth client secrets file not found: {secrets}")
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
    creds = flow.run_local_server(port=0)
    token_path = _resolve(settings.google_token_file)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return token_path


class GoogleCalendarService:
    def __init__(self, settings: Settings):
        if not settings.google_calendar_ids:
            raise CalendarError("GOOGLE_CALENDAR_IDS is not configured")
        self._calendar_ids = list(settings.google_calendar_ids)
        self._organizer_calendar = self._calendar_ids[0]
        self._token_path = _resolve(settings.google_token_file)
        self._create_meet = settings.google_create_meet_link
        self._service = None

    def _credentials(self):
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if not self._token_path.exists():
            raise CalendarError("Google token not found - run `python -m scripts.google_auth` once")
        creds = Credentials.from_authorized_user_file(str(self._token_path), SCOPES)
        if not creds.valid:
            if creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except RefreshError as exc:
                    raise CalendarError("Google token refresh failed - re-run `python -m scripts.google_auth`") from exc
                self._token_path.write_text(creds.to_json(), encoding="utf-8")
            else:
                raise CalendarError("Google credentials are invalid - re-run `python -m scripts.google_auth`")
        return creds

    def _api(self):
        if self._service is None:
            from googleapiclient.discovery import build

            self._service = build("calendar", "v3", credentials=self._credentials(), cache_discovery=False)
        return self._service

    @staticmethod
    def _http_status(exc: Exception) -> int | None:
        resp = getattr(exc, "resp", None)
        return int(resp.status) if resp is not None and getattr(resp, "status", None) else None

    def _call(self, request, action: str):
        from googleapiclient.errors import HttpError

        try:
            return request.execute(num_retries=2)
        except HttpError as exc:
            status = self._http_status(exc)
            err = CalendarError(f"Google Calendar {action} failed (HTTP {status})")
            err.status = status  # type: ignore[attr-defined]
            raise err from exc
        except (OSError, TimeoutError) as exc:
            raise CalendarError(f"Google Calendar {action} failed (network error)") from exc

    @staticmethod
    def _to_event(data: dict) -> CalendarEvent:
        link = data.get("hangoutLink")
        if not link:
            for ep in (data.get("conferenceData") or {}).get("entryPoints", []):
                if ep.get("entryPointType") == "video":
                    link = ep.get("uri")
        return CalendarEvent(
            event_id=data["id"],
            status=data.get("status", "confirmed"),
            starts_at=_parse_dt((data.get("start") or {}).get("dateTime")),
            ends_at=_parse_dt((data.get("end") or {}).get("dateTime")),
            meeting_link=link,
            html_link=data.get("htmlLink"),
        )

    def free_busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        body = {"timeMin": start.isoformat(), "timeMax": end.isoformat(),
                "items": [{"id": cid} for cid in self._calendar_ids]}
        data = self._call(self._api().freebusy().query(body=body), "free/busy query")
        busy: list[tuple[datetime, datetime]] = []
        for cid, info in (data.get("calendars") or {}).items():
            if info.get("errors"):
                raise CalendarError(f"Cannot read availability for calendar {cid}: {info['errors'][0].get('reason')}")
            for block in info.get("busy", []):
                busy.append((_parse_dt(block["start"]), _parse_dt(block["end"])))
        return busy

    def get_event(self, event_id: str) -> CalendarEvent | None:
        try:
            data = self._call(self._api().events().get(calendarId=self._organizer_calendar, eventId=event_id),
                              "event lookup")
        except CalendarError as exc:
            if getattr(exc, "status", None) in (404, 410):
                return None
            raise
        return self._to_event(data)

    def create_event(self, event_id: str, summary: str, description: str, start: datetime, end: datetime,
                     tz_name: str, attendees: list[str]) -> CalendarEvent:
        body = {
            "id": event_id,
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": tz_name},
            "end": {"dateTime": end.isoformat(), "timeZone": tz_name},
            "attendees": [{"email": a} for a in attendees],
        }
        if self._create_meet:
            body["conferenceData"] = {"createRequest": {"requestId": event_id,
                                                        "conferenceSolutionKey": {"type": "hangoutsMeet"}}}
        try:
            data = self._call(
                self._api().events().insert(calendarId=self._organizer_calendar, body=body,
                                            conferenceDataVersion=1 if self._create_meet else 0, sendUpdates="none"),
                "event creation",
            )
        except CalendarError as exc:
            if getattr(exc, "status", None) == 409:  # already created by an earlier attempt
                existing = self.get_event(event_id)
                if existing is not None and existing.status != "cancelled":
                    return existing
            raise
        return self._to_event(data)

    def update_event(self, event_id: str, start: datetime, end: datetime, tz_name: str) -> CalendarEvent:
        body = {"start": {"dateTime": start.isoformat(), "timeZone": tz_name},
                "end": {"dateTime": end.isoformat(), "timeZone": tz_name}}
        data = self._call(self._api().events().patch(calendarId=self._organizer_calendar, eventId=event_id,
                                                      body=body, sendUpdates="none"), "event update")
        return self._to_event(data)

    def cancel_event(self, event_id: str) -> None:
        try:
            self._call(self._api().events().delete(calendarId=self._organizer_calendar, eventId=event_id,
                                                   sendUpdates="none"), "event cancellation")
        except CalendarError as exc:
            if getattr(exc, "status", None) not in (404, 410):
                raise


# ---------------------------------------------------------------------------
# Mock implementation (explicitly generated demo data)
# ---------------------------------------------------------------------------
class MockCalendarService:
    """File-backed fake calendar. Busy blocks are synthetic and labelled as mock data.

    Synthetic busy pattern (panel local time): weekdays 13:00-14:00 (lunch),
    Tue/Thu 11:00-12:00 (team meeting), plus every event created through it.
    `fail_next` can be set by tests to simulate API failures.
    """

    _lock = threading.Lock()

    def __init__(self, settings: Settings, store_path: Path | None = None):
        self._tz = ZoneInfo(settings.default_timezone)
        self._path = store_path or (settings.mock_data_dir / "calendar.json")
        self.fail_next: dict[str, int] = {}

    def _maybe_fail(self, op: str) -> None:
        if self.fail_next.get(op, 0) > 0:
            self.fail_next[op] -= 1
            raise CalendarError(f"Mock calendar {op} failure (simulated)")

    def _load(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {"events": {}}

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def _synthetic_busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        busy = []
        day = start.astimezone(self._tz).date()
        while day <= end.astimezone(self._tz).date():
            if day.weekday() < 5:
                busy.append((datetime.combine(day, time(13), self._tz), datetime.combine(day, time(14), self._tz)))
                if day.weekday() in (1, 3):
                    busy.append((datetime.combine(day, time(11), self._tz), datetime.combine(day, time(12), self._tz)))
            day += timedelta(days=1)
        return busy

    def free_busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        self._maybe_fail("free_busy")
        with self._lock:
            events = self._load()["events"].values()
        busy = self._synthetic_busy(start, end)
        for ev in events:
            if ev["status"] != "cancelled":
                busy.append((_parse_dt(ev["start"]), _parse_dt(ev["end"])))
        return [(s, e) for s, e in busy if s < end and e > start]

    @staticmethod
    def _to_event(ev: dict) -> CalendarEvent:
        return CalendarEvent(event_id=ev["id"], status=ev["status"], starts_at=_parse_dt(ev["start"]),
                             ends_at=_parse_dt(ev["end"]), meeting_link=ev.get("meeting_link"), html_link=None)

    def get_event(self, event_id: str) -> CalendarEvent | None:
        self._maybe_fail("get_event")
        with self._lock:
            ev = self._load()["events"].get(event_id)
        return self._to_event(ev) if ev else None

    def create_event(self, event_id: str, summary: str, description: str, start: datetime, end: datetime,
                     tz_name: str, attendees: list[str]) -> CalendarEvent:
        self._maybe_fail("create_event")
        with self._lock:
            data = self._load()
            if event_id in data["events"] and data["events"][event_id]["status"] != "cancelled":
                return self._to_event(data["events"][event_id])  # idempotent like Google's 409
            data["events"][event_id] = {
                "id": event_id, "summary": summary, "description": description, "status": "confirmed",
                "start": start.isoformat(),
                "end": end.isoformat(), "timezone": tz_name, "attendees": attendees,
                "meeting_link": f"https://meet.example.invalid/{event_id[-10:]}",
            }
            self._save(data)
            return self._to_event(data["events"][event_id])

    def update_event(self, event_id: str, start: datetime, end: datetime, tz_name: str) -> CalendarEvent:
        self._maybe_fail("update_event")
        with self._lock:
            data = self._load()
            ev = data["events"].get(event_id)
            if ev is None:
                raise CalendarError("Mock event not found")
            ev.update(start=start.isoformat(), end=end.isoformat(), timezone=tz_name)
            self._save(data)
            return self._to_event(ev)

    def cancel_event(self, event_id: str) -> None:
        self._maybe_fail("cancel_event")
        with self._lock:
            data = self._load()
            if event_id in data["events"]:
                data["events"][event_id]["status"] = "cancelled"
                self._save(data)


def get_calendar_service(settings: Settings) -> CalendarService:
    return MockCalendarService(settings) if settings.mock_calendar else GoogleCalendarService(settings)

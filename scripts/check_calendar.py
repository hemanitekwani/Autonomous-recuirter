"""Check the real Google Calendar connection (uses .env settings).

  python -m scripts.check_calendar            # read-only: show busy times + free slots for the next days
  python -m scripts.check_calendar --write    # also create a test event 2 days ahead and delete it again
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from src.calendar_service import CalendarError, GoogleCalendarService, generate_slots, new_event_id
from src.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="create and delete one test event")
    args = parser.parse_args()

    s = get_settings()
    print(f"Calendars: {', '.join(s.google_calendar_ids) or '(none - set GOOGLE_CALENDAR_IDS)'}\n")
    cal = GoogleCalendarService(s)
    tz = s.tz

    now = datetime.now(timezone.utc)
    start, end = now, now + timedelta(days=3)
    try:
        busy = cal.free_busy(start, end)
    except CalendarError as exc:
        hint = "Did you run `python -m scripts.google_auth`? Is the Google Calendar API enabled?"
        raise SystemExit(f"FAILED reading free/busy: {exc}\n-> {hint}")
    print(f"[read] busy blocks in the next 3 days: {len(busy)}")
    for b_start, b_end in sorted(busy)[:10]:
        print(f"   busy {b_start.astimezone(tz):%a %d %b %H:%M} - {b_end.astimezone(tz):%H:%M}")
    slots = generate_slots(start + timedelta(hours=s.min_notice_hours), end, s.interview_duration_minutes,
                           s.default_timezone, s.working_hours_start, s.working_hours_end, s.slot_step_minutes,
                           busy, limit=5)
    print("[read] first free interview slots:")
    for slot in slots:
        print(f"   {slot.starts_at:%a %d %b %H:%M}-{slot.ends_at:%H:%M} ({s.default_timezone})")

    if args.write:
        event_id = new_event_id()
        t0 = (datetime.now(tz) + timedelta(days=2)).replace(hour=s.working_hours_start, minute=0, second=0,
                                                           microsecond=0)
        try:
            event = cal.create_event(event_id, "AI Recruiter connection test (safe to delete)",
                                     "Created by scripts/check_calendar.py", t0, t0 + timedelta(minutes=15),
                                     s.default_timezone, [])
            print(f"\n[write] created test event {event.event_id} at {t0:%a %d %b %H:%M} - check your calendar")
            again = cal.create_event(event_id, "duplicate?", "", t0, t0 + timedelta(minutes=15), s.default_timezone, [])
            print(f"[write] retry with same id returned the same event (no duplicate): {again.event_id == event_id}")
        except CalendarError as exc:
            raise SystemExit(f"FAILED creating event: {exc}\n-> Share permission must be 'Make changes to events'.")
        cal.cancel_event(event_id)
        print("[write] test event deleted")
    print("\nOK - Google Calendar connection is working.")


if __name__ == "__main__":
    main()

"""Periodic maintenance - run from cron / Task Scheduler (e.g. every 10 minutes).

  * reconcile PENDING_CONFIRMATION bookings (calendar vs DB)
  * mark finished interviews COMPLETED (application -> INTERVIEWED)
  * retry FAILED invitation emails (bounded by EMAIL_MAX_ATTEMPTS)

Usage:  python -m scripts.maintenance
"""
from __future__ import annotations

from src import repository as repo
from src.calendar_service import get_calendar_service
from src.config import get_settings
from src.database import session_scope
from src.email_service import get_email_sender
from src.scheduling_agent import SchedulingAgent


def main() -> None:
    settings = get_settings()
    agent = SchedulingAgent(settings, get_calendar_service(settings), get_email_sender(settings))
    print("reconcile:", agent.reconcile_pending(older_than_minutes=5))
    with session_scope() as s:
        print("completed:", repo.complete_past_interviews(s))
        retry_ids = [i.interview_id for i in repo.list_interviews(s, ["CONFIRMED"]) if i.email_status == "FAILED"]
    for interview_id in retry_ids:
        print(f"email retry #{interview_id}:", agent.send_invitation(interview_id))


if __name__ == "__main__":
    main()

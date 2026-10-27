"""Slot generation + booking lifecycle (tests 19-23)."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.calendar_service import MockCalendarService, generate_slots
from src.email_service import MockEmailSender
from src.schemas import TimeSlot

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 11, 2, 9, 0, tzinfo=IST)  # a Monday


def slot(day: int, hour: int, minute: int = 0, minutes: int = 45) -> TimeSlot:
    start = datetime(2026, 11, day, hour, minute, tzinfo=IST)
    return TimeSlot(starts_at=start, ends_at=start + timedelta(minutes=minutes), timezone="Asia/Kolkata")


# ---------------------------------------------------------------------------
# Unit: slot generation
# ---------------------------------------------------------------------------
def test_generate_slots_respects_working_hours_weekends_and_busy():
    busy = [(datetime(2026, 11, 6, 10, 0, tzinfo=IST), datetime(2026, 11, 6, 12, 0, tzinfo=IST))]
    slots = generate_slots(datetime(2026, 11, 6, 0, 0, tzinfo=IST), datetime(2026, 11, 9, 23, 0, tzinfo=IST),
                           60, "Asia/Kolkata", 10, 13, 30, busy)
    starts = [s.starts_at.astimezone(IST) for s in slots]
    assert all(s.weekday() < 5 for s in starts)  # Sat 7 / Sun 8 skipped
    friday = [s for s in starts if s.day == 6]
    assert [s.hour * 60 + s.minute for s in friday] == [12 * 60]  # 10-12 busy, last start 12:00 (ends 13:00)
    monday = [s for s in starts if s.day == 9]
    assert monday[0].hour == 10 and monday[-1].hour == 12


def test_generate_slots_requires_aware_datetimes():
    with pytest.raises(ValueError):
        generate_slots(datetime(2026, 11, 6), datetime(2026, 11, 7), 45, "Asia/Kolkata", 10, 18, 30, [])


def test_slot_must_end_after_start():
    with pytest.raises(ValueError):
        TimeSlot(starts_at=NOW, ends_at=NOW, timezone="Asia/Kolkata")


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
@pytest.fixture
def agent(settings):
    from src.scheduling_agent import SchedulingAgent

    return SchedulingAgent(settings, MockCalendarService(settings), MockEmailSender(settings),
                           now=lambda: NOW.astimezone(timezone.utc))


@pytest.fixture
def approved(submit):
    from src import repository as repo
    from src.database import session_scope
    from src.schemas import PipelineStatus

    def _approved(sample: str = "priya_sharma", email: str | None = None) -> int:
        app_id = submit(sample, email=email)
        with session_scope() as s:
            repo.set_pipeline_status(s, app_id, PipelineStatus.APPROVED, "test")
        return app_id

    return _approved


def _interview(interview_id):
    from src import repository as repo
    from src.database import session_scope

    with session_scope() as s:
        row = repo.get_interview(s, interview_id)
        app = repo.get_application(s, row.application_id)
        return row.status, row.email_status, row.google_event_id, app.pipeline_status


@pytest.mark.integration
def test_find_slots_excludes_synthetic_busy_and_requires_approval(db, agent, approved, submit):
    from src.scheduling_agent import SchedulingError

    app_id = approved()
    slots = agent.find_available_slots(app_id, limit=50)
    assert slots and all(s.starts_at >= NOW + timedelta(hours=24) for s in slots)
    assert not any(s.starts_at.astimezone(IST).hour == 13 for s in slots)  # mock lunch block
    with pytest.raises(SchedulingError):
        agent.find_available_slots(submit("rahul_verma"))  # still SCREENING


@pytest.mark.integration
def test_successful_booking_and_single_email(db, agent, approved, settings):
    app_id = approved()
    result = agent.book_slot(app_id, slot(3, 10))
    assert result.outcome == "CONFIRMED"
    assert agent.send_invitation(result.interview_id) == "SENT"
    assert agent.send_invitation(result.interview_id) == "SENT"  # retry is a no-op
    status, email_status, event_id, pipeline = _interview(result.interview_id)
    assert (status, email_status, pipeline) == ("CONFIRMED", "SENT", "INTERVIEW_SCHEDULED")
    assert len(list((settings.mock_data_dir / "outbox").glob("*.eml"))) == 1
    assert MockCalendarService(settings).get_event(event_id) is not None


@pytest.mark.integration
def test_slot_conflict_returns_alternatives(db, agent, approved):
    first = agent.book_slot(approved("priya_sharma"), slot(3, 10))
    assert first.outcome == "CONFIRMED"
    second = agent.book_slot(approved("rahul_verma"), slot(3, 10, 15))  # overlaps 10:00-10:45
    assert second.outcome == "CONFLICT"
    assert second.alternative_slots and all(
        not (a.starts_at < slot(3, 10).ends_at and slot(3, 10).starts_at < a.ends_at) for a in second.alternative_slots)


@pytest.mark.integration
def test_concurrent_reservations_only_one_wins(db, approved):
    from src import repository as repo
    from src.calendar_service import new_event_id
    from src.database import session_scope

    apps = [approved("priya_sharma"), approved("rahul_verma"), approved("anita_rao")]
    target = slot(4, 15)
    barrier = threading.Barrier(len(apps))
    outcomes: list[str] = []

    def attempt(app_id):
        barrier.wait()
        try:
            with session_scope() as s:
                app = repo.get_application(s, app_id)
                repo.reserve_interview(s, app, target.starts_at, target.ends_at, target.timezone, new_event_id())
            outcomes.append("ok")
        except repo.SlotConflictError:
            outcomes.append("conflict")

    threads = [threading.Thread(target=attempt, args=(a,)) for a in apps]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["conflict", "conflict", "ok"]


@pytest.mark.integration
def test_one_active_interview_per_application(db, agent, approved):
    app_id = approved()
    assert agent.book_slot(app_id, slot(3, 10)).outcome == "CONFIRMED"
    assert agent.book_slot(app_id, slot(5, 10)).outcome == "INVALID"


@pytest.mark.integration
def test_calendar_failure_releases_reservation(db, agent, approved):
    app_id = approved()
    agent.calendar.fail_next["create_event"] = 1
    failed = agent.book_slot(app_id, slot(3, 10))
    assert failed.outcome == "CALENDAR_ERROR"
    status, _, _, pipeline = _interview(failed.interview_id)
    assert status == "CANCELLED" and pipeline == "APPROVED"
    retry = agent.book_slot(app_id, slot(3, 10))  # same slot is free again
    assert retry.outcome == "CONFIRMED"


@pytest.mark.integration
def test_calendar_unavailable_during_recheck(db, agent, approved):
    agent.calendar.fail_next["free_busy"] = 1
    assert agent.book_slot(approved(), slot(3, 10)).outcome == "CALENDAR_ERROR"


@pytest.mark.integration
def test_db_failure_after_event_creation_is_reconciled(db, agent, approved, monkeypatch, settings):
    from src import repository as repo

    app_id = approved()
    real_confirm = repo.confirm_interview
    calls = {"n": 0}

    def flaky_confirm(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated database outage")
        return real_confirm(*args, **kwargs)

    monkeypatch.setattr(repo, "confirm_interview", flaky_confirm)
    result = agent.book_slot(app_id, slot(3, 10))
    assert result.outcome == "DB_ERROR"
    status, _, event_id, _ = _interview(result.interview_id)
    assert status == "PENDING_CONFIRMATION"
    assert agent.send_invitation(result.interview_id) == "NOT_SENT"  # no email before confirmation

    report = agent.reconcile_pending(older_than_minutes=0)
    assert report["confirmed"] == [result.interview_id]
    status, _, _, pipeline = _interview(result.interview_id)
    assert (status, pipeline) == ("CONFIRMED", "INTERVIEW_SCHEDULED")
    events = MockCalendarService(settings)._load()["events"]
    assert list(events) == [event_id]  # no duplicate event


@pytest.mark.integration
def test_email_failure_keeps_booking_and_retries(db, agent, approved, settings):
    result = agent.book_slot(approved(), slot(3, 10))
    agent.email_sender.fail_next = 1
    assert agent.send_invitation(result.interview_id) == "FAILED"
    status, email_status, _, _ = _interview(result.interview_id)
    assert (status, email_status) == ("CONFIRMED", "FAILED")
    assert agent.send_invitation(result.interview_id) == "SENT"
    assert len(list((settings.mock_data_dir / "outbox").glob("*.eml"))) == 1


@pytest.mark.integration
def test_email_retries_are_bounded(db, agent, approved, settings):
    result = agent.book_slot(approved(), slot(3, 10))
    agent.email_sender.fail_next = 10
    statuses = [agent.send_invitation(result.interview_id) for _ in range(settings.email_max_attempts + 2)]
    assert statuses.count("FAILED") == len(statuses)
    from src import repository as repo
    from src.database import session_scope

    with session_scope() as s:
        assert repo.get_interview(s, result.interview_id).email_attempts == settings.email_max_attempts


@pytest.mark.integration
def test_cancel_and_complete_flow(db, agent, approved):
    from src import repository as repo
    from src.database import session_scope

    app_id = approved()
    booked = agent.book_slot(app_id, slot(3, 10))
    agent.cancel(booked.interview_id, "hr:test", "Candidate asked to move")
    assert _interview(booked.interview_id)[0] == "CANCELLED"
    assert _interview(booked.interview_id)[3] == "APPROVED"
    rebooked = agent.book_slot(app_id, slot(3, 10))  # cancelled slot is reusable
    assert rebooked.outcome == "CONFIRMED"
    with session_scope() as s:
        n = repo.complete_past_interviews(s, now=datetime(2026, 11, 4, tzinfo=IST))
    assert n == 1 and _interview(rebooked.interview_id)[0] == "COMPLETED"
    assert _interview(rebooked.interview_id)[3] == "INTERVIEWED"

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def test_scheduler_triggers_due_activation_and_retries_after_five_minutes():
    from services.rag_api.index_scheduler import IndexScheduler

    due = datetime(2026, 7, 11, 1, 0, tzinfo=timezone.utc)
    calls = []
    scheduler = IndexScheduler(
        status_loader=lambda: {"next_activation_at": "2026-07-11T01:00:00Z"},
        trigger=lambda: calls.append("triggered") or {"status": "queued"},
    )

    scheduler.check_once(due)
    scheduler.check_once(due + timedelta(minutes=4, seconds=59))
    scheduler.check_once(due + timedelta(minutes=5))

    assert calls == ["triggered", "triggered"]
    assert scheduler.status()["last_attempt_at"] == "2026-07-11T01:05:00Z"


def test_scheduler_does_not_trigger_future_activation():
    from services.rag_api.index_scheduler import IndexScheduler

    calls = []
    scheduler = IndexScheduler(
        status_loader=lambda: {"next_activation_at": "2026-07-11T02:00:00Z"},
        trigger=lambda: calls.append("triggered"),
    )

    scheduler.check_once(datetime(2026, 7, 11, 1, 0, tzinfo=timezone.utc))

    assert calls == []


def test_scheduler_runs_cleanup_at_most_once_per_day():
    from services.rag_api.index_scheduler import IndexScheduler

    start = datetime(2026, 7, 11, 1, 0, tzinfo=timezone.utc)
    cleanups = []
    scheduler = IndexScheduler(status_loader=lambda: {}, trigger=lambda: None, cleanup=lambda: cleanups.append("cleanup"))

    scheduler.check_once(start)
    scheduler.check_once(start + timedelta(hours=23))
    scheduler.check_once(start + timedelta(hours=24))

    assert cleanups == ["cleanup", "cleanup"]

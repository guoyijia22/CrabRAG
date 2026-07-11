from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from services.rag_api import index_generation


RETRY_INTERVAL = timedelta(minutes=5)


class IndexScheduler:
    def __init__(
        self,
        *,
        status_loader: Callable[[], dict[str, Any]] | None = None,
        trigger: Callable[[], Any] | None = None,
        cleanup: Callable[[], Any] | None = None,
        poll_seconds: int = 30,
    ) -> None:
        self._status_loader = status_loader or _active_generation_status
        self._trigger = trigger or _trigger_incremental_ingest
        self._cleanup = cleanup or _cleanup_generations
        self._poll_seconds = poll_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_activation: str | None = None
        self._last_attempt: datetime | None = None
        self._last_error: str | None = None
        self._last_cleanup: datetime | None = None
        self._last_cleanup_result: Any = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="crabrag-index-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def check_once(self, now: datetime | None = None) -> None:
        current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        if self._last_cleanup is None or current_time - self._last_cleanup >= timedelta(days=1):
            self._last_cleanup = current_time
            try:
                self._last_cleanup_result = self._cleanup()
            except Exception as exc:  # noqa: BLE001
                self._last_cleanup_result = {"errors": [str(exc)]}
        status = self._status_loader() or {}
        activation_text = str(status.get("next_activation_at") or "")
        if not activation_text:
            return
        activation = _parse_timestamp(activation_text)
        if activation > current_time:
            return
        if self._last_activation != activation_text:
            self._last_activation = activation_text
            self._last_attempt = None
        if self._last_attempt is not None and current_time - self._last_attempt < RETRY_INTERVAL:
            return
        self._last_attempt = current_time
        try:
            self._trigger()
            self._last_error = None
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)

    def status(self) -> dict[str, Any]:
        source = self._status_loader() or {}
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "next_activation_at": source.get("next_activation_at"),
            "last_attempt_at": _format_timestamp(self._last_attempt) if self._last_attempt else None,
            "last_error": self._last_error,
            "retry_seconds": int(RETRY_INTERVAL.total_seconds()),
            "last_cleanup_at": _format_timestamp(self._last_cleanup) if self._last_cleanup else None,
            "last_cleanup": self._last_cleanup_result,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.check_once()
            self._stop_event.wait(self._poll_seconds)


def _active_generation_status() -> dict[str, Any]:
    generation_id = index_generation.active_generation_id()
    if not generation_id:
        return {}
    try:
        return index_generation.load_generation_manifest(generation_id)
    except ValueError:
        return {}


def _trigger_incremental_ingest() -> dict:
    from services.rag_api.document.ingest_tasks import start_ingest_run

    return start_ingest_run(full_rebuild=False)


def _cleanup_generations() -> dict:
    from services.rag_api.vector.chroma_store import cleanup_obsolete_generations

    return cleanup_obsolete_generations()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("activation timestamp must contain timezone")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


INDEX_SCHEDULER = IndexScheduler()

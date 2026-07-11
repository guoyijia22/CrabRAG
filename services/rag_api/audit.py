from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from services.rag_api.runtime_environment import PROJECT_DIR
from services.rag_api.security import PrincipalContext


AUDIT_SCHEMA_VERSION = 1
ZERO_HASH = "0" * 64
_AUDIT_LOCK = threading.RLock()
_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "content",
    "body",
    "password",
    "prompt",
    "question",
    "answer",
    "secret",
    "token",
)


class AuditError(RuntimeError):
    pass


class AuditIntegrityError(AuditError):
    pass


class AuditWriteError(AuditError):
    pass


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.anchor_path = path.with_suffix(f"{path.suffix}.anchor.json")
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")

    def append(
        self,
        event_type: str,
        *,
        principal: PrincipalContext | None = None,
        subject: str | None = None,
        permission_revision: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = str(event_type).strip()
        if not event or len(event) > 128:
            raise AuditWriteError("audit event type is invalid")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with _AUDIT_LOCK, _exclusive_file_lock(self.lock_path):
                sequence, previous_hash = self._tail_state()
                record: dict[str, Any] = {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "sequence": sequence + 1,
                    "timestamp": _utc_now(),
                    "event_type": event,
                    "subject": _bounded_text(subject or (principal.subject if principal else "system"), 256),
                    "permission_revision": _bounded_text(
                        permission_revision or (principal.permission_revision if principal else "system"),
                        128,
                    ),
                    "details": _sanitize(details or {}),
                    "previous_hash": previous_hash,
                }
                record["record_hash"] = _record_hash(record)
                encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._write_anchor(record["sequence"], record["record_hash"])
                return record
        except AuditError:
            raise
        except OSError as exc:
            raise AuditWriteError("security audit could not be written") from exc

    def verify(self) -> dict[str, Any]:
        if not self.path.exists() and not self.anchor_path.exists():
            return {"valid": True, "record_count": 0, "last_hash": ZERO_HASH}
        try:
            with _AUDIT_LOCK, _exclusive_file_lock(self.lock_path):
                if not self.path.exists():
                    if self.anchor_path.exists():
                        raise AuditIntegrityError("audit log is missing while its anchor exists")
                    return {"valid": True, "record_count": 0, "last_hash": ZERO_HASH}
                previous_hash = ZERO_HASH
                record_count = 0
                for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        raise AuditIntegrityError(f"audit record {line_number} is empty")
                    record = _parse_record(line, line_number)
                    if record.get("sequence") != line_number:
                        raise AuditIntegrityError(f"audit sequence is invalid at record {line_number}")
                    if record.get("previous_hash") != previous_hash:
                        raise AuditIntegrityError(f"audit chain is broken at record {line_number}")
                    expected_hash = _record_hash(record)
                    if record.get("record_hash") != expected_hash:
                        raise AuditIntegrityError(f"audit hash is invalid at record {line_number}")
                    previous_hash = expected_hash
                    record_count = line_number
                anchor = self._read_anchor(required=record_count > 0)
                if record_count == 0:
                    if anchor is not None:
                        raise AuditIntegrityError("empty audit log has a non-empty anchor")
                elif anchor != {"schema_version": AUDIT_SCHEMA_VERSION, "record_count": record_count, "last_hash": previous_hash}:
                    raise AuditIntegrityError("audit anchor does not match the log")
                return {"valid": True, "record_count": record_count, "last_hash": previous_hash}
        except AuditError:
            raise
        except (OSError, UnicodeError) as exc:
            raise AuditIntegrityError("security audit could not be verified") from exc

    def _tail_state(self) -> tuple[int, str]:
        anchor = self._read_anchor(required=False)
        if not self.path.exists() or self.path.stat().st_size == 0:
            if anchor is not None:
                raise AuditIntegrityError("audit anchor exists without matching records")
            return 0, ZERO_HASH
        if anchor is None:
            raise AuditIntegrityError("audit log exists without its anchor")
        line = _last_record_line(self.path)
        if not line:
            raise AuditIntegrityError("audit log has no valid tail record")
        record = _parse_record(line, int(anchor.get("record_count") or 0))
        expected_hash = _record_hash(record)
        if record.get("record_hash") != expected_hash:
            raise AuditIntegrityError("audit tail hash is invalid")
        if anchor != {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "record_count": record.get("sequence"),
            "last_hash": expected_hash,
        }:
            raise AuditIntegrityError("audit tail does not match its anchor")
        return int(record["sequence"]), expected_hash

    def _read_anchor(self, *, required: bool) -> dict[str, Any] | None:
        if not self.anchor_path.exists():
            if required:
                raise AuditIntegrityError("audit anchor is missing")
            return None
        try:
            value = json.loads(self.anchor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AuditIntegrityError("audit anchor is invalid") from exc
        if not isinstance(value, dict):
            raise AuditIntegrityError("audit anchor is invalid")
        record_count = value.get("record_count")
        last_hash = value.get("last_hash")
        if (
            value.get("schema_version") != AUDIT_SCHEMA_VERSION
            or not isinstance(record_count, int)
            or record_count < 1
            or not isinstance(last_hash, str)
            or len(last_hash) != 64
            or any(character not in "0123456789abcdef" for character in last_hash)
        ):
            raise AuditIntegrityError("audit anchor is invalid")
        return value

    def _write_anchor(self, record_count: int, last_hash: str) -> None:
        anchor = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "record_count": record_count,
            "last_hash": last_hash,
        }
        temporary = self.anchor_path.with_name(f".{self.anchor_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(anchor, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.anchor_path)
            _fsync_directory(self.anchor_path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def default_audit_log(project_root: Path | None = None) -> AuditLog:
    root = (project_root or PROJECT_DIR).resolve()
    return AuditLog(root / "data" / "audit" / "security-audit.jsonl")


SECURITY_AUDIT = default_audit_log()


def _parse_record(line: str, line_number: int) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise AuditIntegrityError(f"audit record {line_number} is invalid JSON") from exc
    if not isinstance(record, dict) or record.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise AuditIntegrityError(f"audit record {line_number} has an invalid schema")
    return record


def _record_hash(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_hash"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sanitize(value: Any, *, key: str = "") -> Any:
    normalized_key = key.lower().replace("-", "_")
    if any(marker in normalized_key for marker in _SENSITIVE_KEY_MARKERS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _sanitize(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _bounded_text(value, 1024) if isinstance(value, str) else value
    return _bounded_text(str(value), 1024)


def _bounded_text(value: str, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _last_record_line(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell() - 1
        while position >= 0:
            handle.seek(position)
            if handle.read(1) not in {b"\n", b"\r"}:
                break
            position -= 1
        end = position + 1
        while position >= 0:
            handle.seek(position)
            if handle.read(1) == b"\n":
                position += 1
                break
            position -= 1
        start = max(position, 0)
        handle.seek(start)
        return handle.read(end - start).decode("utf-8").strip()


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

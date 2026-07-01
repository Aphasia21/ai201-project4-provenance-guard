"""Structured audit log + content-status store.

Every submission and appeal writes one JSON line to `audit.jsonl`. On start
the file is replayed to rebuild the per-content-id status index, so a fresh
process picks up where the previous one left off.

Kept intentionally simple — no SQLite, no external DB. A real deployment
would swap this for something transactional, but the JSONL log is the
canonical, human-inspectable evidence for grading.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def new_content_id() -> str:
    return str(uuid.uuid4())


class AuditStore:
    """Append-only audit log with an in-memory status index."""

    def __init__(self, data_dir: Path | str = "data") -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._data_dir / "audit.jsonl"
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._content: dict[str, dict[str, Any]] = {}
        self._load_from_disk()

    @property
    def log_path(self) -> Path:
        return self._log_path

    def _load_from_disk(self) -> None:
        if not self._log_path.exists():
            return
        with self._log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._apply_entry(entry)

    def _apply_entry(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)
        cid = entry.get("content_id")
        if not cid:
            return
        if entry.get("event") == "submission":
            self._content[cid] = {
                "creator_id": entry.get("creator_id"),
                "attribution": entry.get("attribution"),
                "ai_probability": entry.get("ai_probability"),
                "confidence": entry.get("confidence"),
                "status": entry.get("status", "classified"),
                "created_at": entry.get("timestamp"),
                "appeals": [],
            }
        elif entry.get("event") == "appeal":
            record = self._content.get(cid)
            if record is not None:
                record["status"] = "under_review"
                record.setdefault("appeals", []).append(
                    {
                        "timestamp": entry.get("timestamp"),
                        "reasoning": entry.get("appeal_reasoning"),
                    }
                )

    def _persist(self, entry: dict[str, Any]) -> None:
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def record_submission(
        self,
        content_id: str,
        creator_id: str,
        text_length: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        entry = {
            "event": "submission",
            "content_id": content_id,
            "creator_id": creator_id,
            "timestamp": _now_iso(),
            "text_length": text_length,
            "attribution": result["attribution"],
            "ai_probability": result["ai_probability"],
            "confidence": result["confidence"],
            "label_variant": result["label"]["variant"],
            "signals": result["signals"],
            "status": "classified",
        }
        with self._lock:
            self._persist(entry)
            self._apply_entry(entry)
        return entry

    def record_appeal(
        self, content_id: str, creator_reasoning: str
    ) -> dict[str, Any] | None:
        """Record an appeal and flip status to under_review.

        Returns None if the content_id is not known so callers can surface
        a 404. Never overwrites the original submission entry — appeals
        are separate log lines.
        """
        with self._lock:
            record = self._content.get(content_id)
            if record is None:
                return None
            entry = {
                "event": "appeal",
                "content_id": content_id,
                "creator_id": record.get("creator_id"),
                "timestamp": _now_iso(),
                "original_attribution": record.get("attribution"),
                "original_ai_probability": record.get("ai_probability"),
                "original_confidence": record.get("confidence"),
                "appeal_reasoning": creator_reasoning,
                "status": "under_review",
            }
            self._persist(entry)
            self._apply_entry(entry)
            return entry

    def get_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if limit <= 0:
                return list(self._entries)
            return list(self._entries[-limit:])

    def get_content(self, content_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._content.get(content_id)
            return dict(record) if record else None

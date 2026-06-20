"""CalendarTool — local SQLite-backed scheduling assistant.

All events are stored in a single SQLite database (default: ``data/calendar.db``).
The schema is created automatically on first use.

Datetime format
---------------
All ``start_dt`` and ``end_dt`` values are stored and returned as ISO-8601
strings (``YYYY-MM-DDTHH:MM:SS``).  The tool accepts both date-only strings
(``2026-06-20``) and full datetime strings.
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base import BaseTool

logger = logging.getLogger(__name__)

_VALID_OPS = {"add", "get", "search", "update", "delete", "upcoming"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    start_dt    TEXT NOT NULL,
    end_dt      TEXT NOT NULL,
    description TEXT DEFAULT '',
    location    TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_start ON events (start_dt);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _normalise_dt(value: str) -> str:
    """Accept date-only or full datetime; return ISO-8601 datetime string."""
    value = value.strip()
    if len(value) == 10:  # date only: YYYY-MM-DD
        value = value + "T00:00:00"
    return value


class CalendarTool(BaseTool):
    """Manage local calendar events stored in SQLite.

    Parameters (passed as ``**kwargs`` from the Executor):

    op : str
        One of ``add``, ``get``, ``search``, ``update``, ``delete``, ``upcoming``.

    add
        title (str), start_dt (str), end_dt (str), description (str), location (str)
    get
        event_id (str)
    search
        query (str), from_dt (str, optional), to_dt (str, optional)
    update
        event_id (str), + any of: title, start_dt, end_dt, description, location
    delete
        event_id (str)
    upcoming
        n (int, default 5)
    """

    name = "calendar"
    description = "Add, view, search, update, and delete local calendar events."

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # BaseTool implementation
    # ------------------------------------------------------------------

    def run(self, *, op: str, **kwargs: Any) -> str:
        op = op.lower().strip()
        if op not in _VALID_OPS:
            return f"[calendar] Unknown op '{op}'. Valid ops: {sorted(_VALID_OPS)}"
        try:
            if op == "add":
                return self._add(**kwargs)
            if op == "get":
                return self._get(**kwargs)
            if op == "search":
                return self._search(**kwargs)
            if op == "update":
                return self._update(**kwargs)
            if op == "delete":
                return self._delete(**kwargs)
            if op == "upcoming":
                return self._upcoming(**kwargs)
        except sqlite3.Error as exc:
            logger.error("CalendarTool DB error (op=%s): %s", op, exc)
            return f"[calendar] Database error: {exc}"
        return "[calendar] Unexpected state"

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _add(
        self,
        *,
        title: str = "",
        start_dt: str = "",
        end_dt: str = "",
        description: str = "",
        location: str = "",
        **_: Any,
    ) -> str:
        if not title or not start_dt or not end_dt:
            return "[calendar] 'title', 'start_dt', and 'end_dt' are required for add."
        event_id = str(uuid.uuid4())
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (id, title, start_dt, end_dt, description, location, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    title,
                    _normalise_dt(start_dt),
                    _normalise_dt(end_dt),
                    description,
                    location,
                    now,
                ),
            )
        return f"[calendar] Event added (id={event_id}): '{title}' from {start_dt} to {end_dt}"

    def _get(self, *, event_id: str, **_: Any) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, start_dt, end_dt, description, location FROM events WHERE id=?",
                (event_id,),
            ).fetchone()
        if row is None:
            return f"[calendar] No event found with id={event_id}"
        return self._format_row(row)

    def _search(
        self,
        *,
        query: str = "",
        from_dt: str = "",
        to_dt: str = "",
        **_: Any,
    ) -> str:
        sql = (
            "SELECT id, title, start_dt, end_dt, description, location FROM events WHERE 1=1"
        )
        params: List[Any] = []
        if query:
            sql += " AND (title LIKE ? OR description LIKE ? OR location LIKE ?)"
            like = f"%{query}%"
            params.extend([like, like, like])
        if from_dt:
            sql += " AND start_dt >= ?"
            params.append(_normalise_dt(from_dt))
        if to_dt:
            sql += " AND start_dt <= ?"
            params.append(_normalise_dt(to_dt))
        sql += " ORDER BY start_dt LIMIT 50"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return "[calendar] No events found."
        return "\n\n".join(self._format_row(r) for r in rows)

    def _update(self, *, event_id: str, **fields: Any) -> str:
        allowed_fields = {"title", "start_dt", "end_dt", "description", "location"}
        updates = {k: v for k, v in fields.items() if k in allowed_fields}
        if not updates:
            return "[calendar] No valid fields to update."
        # Normalise datetime fields.
        for dt_field in ("start_dt", "end_dt"):
            if dt_field in updates:
                updates[dt_field] = _normalise_dt(updates[dt_field])
        set_clause = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [event_id]
        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE events SET {set_clause} WHERE id=?", values  # noqa: S608
            )
        if cur.rowcount == 0:
            return f"[calendar] No event found with id={event_id}"
        return f"[calendar] Event {event_id} updated: {list(updates.keys())}"

    def _delete(self, *, event_id: str, **_: Any) -> str:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM events WHERE id=?", (event_id,))
        if cur.rowcount == 0:
            return f"[calendar] No event found with id={event_id}"
        return f"[calendar] Event {event_id} deleted."

    def _upcoming(self, *, n: int = 5, **_: Any) -> str:
        now = _now_iso()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, start_dt, end_dt, description, location "
                "FROM events WHERE start_dt >= ? ORDER BY start_dt LIMIT ?",
                (now, int(n)),
            ).fetchall()
        if not rows:
            return "[calendar] No upcoming events."
        return "\n\n".join(self._format_row(r) for r in rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    @staticmethod
    def _format_row(row: tuple) -> str:
        event_id, title, start_dt, end_dt, description, location = row
        parts = [f"ID:    {event_id}", f"Title: {title}", f"Start: {start_dt}", f"End:   {end_dt}"]
        if description:
            parts.append(f"Desc:  {description}")
        if location:
            parts.append(f"Loc:   {location}")
        return "\n".join(parts)


# ------------------------------------------------------------------
# Factory consumed by ToolLoader
# ------------------------------------------------------------------

def create_tool(config: Any) -> CalendarTool:
    db_path = Path("data/calendar.db")
    try:
        db_path = Path(config.tools.calendar.db_path)
    except AttributeError:
        pass
    return CalendarTool(db_path)

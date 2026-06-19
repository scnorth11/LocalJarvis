import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class SQLiteStore:
    """Persistent memory store backed by SQLite.

    Usage::

        store = SQLiteStore("data/memory.db")
        store.connect()
        store.write("general", "Paris is the capital of France", [], ["geography"])
        records = store.read("general")
        matches = store.search("general", "capital")
        store.close()
    """

    def __init__(self, db_path: Union[str, Path] = "data/memory.db") -> None:
        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open the database connection and apply schema migrations."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._apply_schema()
        logger.debug("SQLiteStore connected: %s", self._db_path)

    def _apply_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            self._conn.executescript(schema_path.read_text(encoding="utf-8"))
        else:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL DEFAULT '[]',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
            """)
        self._conn.commit()

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteStore not connected — call connect() first.")
        return self._conn

    def write(
        self,
        namespace: str,
        text: str,
        embedding: List[float],
        tags: List[str],
    ) -> int:
        """Insert a memory record and return the new row id."""
        conn = self._require_conn()
        cursor = conn.execute(
            "INSERT INTO memories (namespace, text, embedding_json, tags_json) VALUES (?, ?, ?, ?)",
            (namespace, text, json.dumps(embedding), json.dumps(tags)),
        )
        conn.commit()
        return cursor.lastrowid

    def read(self, namespace: str) -> List[Dict[str, Any]]:
        """Return all records in a namespace, newest first."""
        conn = self._require_conn()
        rows = conn.execute(
            "SELECT id, namespace, text, embedding_json, tags_json, created_at"
            " FROM memories WHERE namespace = ? ORDER BY created_at DESC",
            (namespace,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search(self, namespace: str, query: str) -> List[Dict[str, Any]]:
        """Full-text substring search within a namespace."""
        conn = self._require_conn()
        rows = conn.execute(
            "SELECT id, namespace, text, embedding_json, tags_json, created_at"
            " FROM memories WHERE namespace = ? AND text LIKE ? ORDER BY created_at DESC",
            (namespace, f"%{query}%"),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete(self, namespace: str, record_id: int) -> None:
        """Delete a single record by id."""
        conn = self._require_conn()
        conn.execute("DELETE FROM memories WHERE namespace = ? AND id = ?", (namespace, record_id))
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("SQLiteStore closed: %s", self._db_path)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "namespace": row["namespace"],
            "text": row["text"],
            "embedding": json.loads(row["embedding_json"]),
            "tags": json.loads(row["tags_json"]),
            "created_at": row["created_at"],
        }

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.errors import AgentError
from core.schema import AgentErrorSchema, AgentMessage, MemoryData, MemoryMessage
from memory.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class MemoryAgent:
    name = "memory"

    def __init__(self, store: Optional[SQLiteStore] = None) -> None:
        self.config = None
        self.enforcer = None
        self.invoke_tool = None
        self._store: Optional[SQLiteStore] = store
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    async def initialize(self, config: Any, enforcer: Any, invoke_tool: Any) -> None:
        self.config = config
        self.enforcer = enforcer
        self.invoke_tool = invoke_tool
        if self._store is None:
            db_path = Path(config.paths.data_dir) / "memory.db"
            self._store = SQLiteStore(db_path)
            self._store.connect()
        logger.debug("MemoryAgent initialized with store: %s", getattr(self._store, "_db_path", "in-memory"))

    async def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()

    # ------------------------------------------------------------------
    # Direct API — used by ExecutorAgent for memory_search steps.
    # ------------------------------------------------------------------

    async def write_direct(
        self, namespace: str, text: str, embedding: List[float], tags: List[str]
    ) -> Dict[str, Any]:
        """Write a record directly without going through the message envelope."""
        if self._store is not None:
            row_id = self._store.write(namespace, text, embedding, tags)
            return {"status": "stored", "id": row_id, "namespace": namespace}
        self._cache.setdefault(namespace, []).append(
            {"text": text, "embedding": embedding, "tags": tags}
        )
        return {"status": "stored", "namespace": namespace}

    async def read_direct(self, namespace: str) -> List[Dict[str, Any]]:
        """Read all records in a namespace directly."""
        if self._store is not None:
            return self._store.read(namespace)
        return self._cache.get(namespace, [])

    async def search_direct(self, namespace: str, query: str) -> List[Dict[str, Any]]:
        """Substring search within a namespace directly."""
        if self._store is not None:
            return self._store.search(namespace, query)
        return [
            item
            for item in self._cache.get(namespace, [])
            if query.lower() in item.get("text", "").lower()
        ]

    async def delete_direct(self, namespace: str, record_id: int) -> None:
        """Delete a record by id directly."""
        if self._store is not None:
            self._store.delete(namespace, record_id)

    # ------------------------------------------------------------------
    # Message-envelope API — used for agent-to-agent MemoryMessage routing.
    # ------------------------------------------------------------------

    async def handle(self, message: AgentMessage) -> AgentMessage:
        if not isinstance(message.payload, MemoryMessage):
            return self._error_message(
                message,
                AgentError(
                    "ValidationError",
                    "MemoryAgent expects MemoryMessage",
                    agent=self.name,
                    recoverable=False,
                ),
            )

        try:
            results = await self._dispatch(message.payload)
        except AgentError as exc:
            return self._error_message(message, exc)
        except Exception as exc:
            return self._error_message(
                message,
                AgentError("Unknown", str(exc), agent=self.name, recoverable=True),
            )

        response_payload = MemoryMessage(
            operation=message.payload.operation,
            namespace=message.payload.namespace,
            query=message.payload.query,
            data=message.payload.data,
            results=results,
        )
        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target=message.source,
            type="memory.response",
            payload=response_payload,
            metadata=message.metadata,
        )

    async def _dispatch(self, payload: MemoryMessage) -> List[Dict[str, Any]]:
        op = payload.operation
        ns = payload.namespace

        if op == "write":
            if payload.data is None:
                raise AgentError(
                    "ValidationError",
                    "write operation requires data",
                    agent=self.name,
                    recoverable=False,
                )
            data: MemoryData = payload.data
            result = await self.write_direct(ns, data.text, list(data.embedding), list(data.tags))
            return [result]

        if op == "read":
            return await self.read_direct(ns)

        if op == "search":
            if not payload.query:
                raise AgentError(
                    "ValidationError",
                    "search operation requires a query",
                    agent=self.name,
                    recoverable=False,
                )
            return await self.search_direct(ns, payload.query)

        if op == "delete":
            record_id = payload.data and getattr(payload.data, "id", None)
            if record_id is None:
                raise AgentError(
                    "ValidationError",
                    "delete operation requires data.id",
                    agent=self.name,
                    recoverable=False,
                )
            await self.delete_direct(ns, int(record_id))
            return [{"status": "deleted", "namespace": ns}]

        raise AgentError(
            "ValidationError",
            f"Unsupported memory operation: {op!r}",
            agent=self.name,
            recoverable=False,
        )

    def _error_message(self, original: AgentMessage, exc: AgentError) -> AgentMessage:
        logger.error("MemoryAgent error: %s — %s", exc.error_type, exc.message)
        return AgentMessage(
            id=original.id,
            timestamp=original.timestamp,
            source=self.name,
            target=original.source,
            type="error",
            payload=AgentErrorSchema(
                error_type=exc.error_type,
                message=exc.message,
                agent=self.name,
                recoverable=exc.recoverable,
            ),
            metadata=original.metadata,
        )

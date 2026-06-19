from pathlib import Path
from typing import Any, Dict, List, Optional

from core.schema import AgentMessage, MemoryMessage
from memory.sqlite_store import SQLiteStore


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

    async def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()

    async def handle(self, message: AgentMessage) -> AgentMessage:
        if not isinstance(message.payload, MemoryMessage):
            raise TypeError("MemoryAgent expects MemoryMessage in message.payload")

        operation = message.payload.operation
        namespace = message.payload.namespace
        results: List[Dict[str, Any]]

        if operation == "write" and message.payload.data is not None:
            data = message.payload.data
            if self._store is not None:
                self._store.write(namespace, data.text, list(data.embedding), list(data.tags))
            else:
                self._cache.setdefault(namespace, []).append(
                    {"text": data.text, "embedding": list(data.embedding), "tags": list(data.tags)}
                )
            results = [{"status": "stored", "namespace": namespace}]

        elif operation == "read":
            results = (
                self._store.read(namespace)
                if self._store is not None
                else self._cache.get(namespace, [])
            )

        elif operation == "search" and message.payload.query is not None:
            query = message.payload.query
            results = (
                self._store.search(namespace, query)
                if self._store is not None
                else [
                    item
                    for item in self._cache.get(namespace, [])
                    if query in item.get("text", "")
                ]
            )

        else:
            raise ValueError(f"Unsupported memory operation: {operation}")

        response_payload = MemoryMessage(
            operation=operation,
            namespace=namespace,
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

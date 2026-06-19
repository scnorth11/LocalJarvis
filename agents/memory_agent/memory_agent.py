from typing import Any, Dict, List, Optional

from core.schema import AgentMessage, MemoryMessage


class MemoryAgent:
    name = "memory_agent"

    def __init__(self) -> None:
        self.config = None
        self.enforcer = None
        self.invoke_tool = None
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    async def initialize(self, config: Any, enforcer: Any, invoke_tool: Any) -> None:
        self.config = config
        self.enforcer = enforcer
        self.invoke_tool = invoke_tool

    async def shutdown(self) -> None:
        return None

    async def handle(self, message: AgentMessage) -> AgentMessage:
        if not isinstance(message.payload, MemoryMessage):
            raise TypeError("MemoryAgent expects MemoryMessage in message.payload")

        operation = message.payload.operation
        namespace = message.payload.namespace

        if operation == "write" and message.payload.data is not None:
            self._store.setdefault(namespace, []).append({
                "text": message.payload.data.text,
                "embedding": message.payload.data.embedding,
                "tags": message.payload.data.tags,
            })
            results = [{"status": "stored", "namespace": namespace}]
        elif operation == "read":
            results = self._store.get(namespace, [])
        elif operation == "search" and message.payload.query is not None:
            found = [item for item in self._store.get(namespace, []) if message.payload.query in item.get("text", "")]
            results = found
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
            target=message.target,
            type="memory.response",
            payload=response_payload,
            metadata=message.metadata,
        )

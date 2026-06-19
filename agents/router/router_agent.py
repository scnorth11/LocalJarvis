from typing import Any

from core.schema import AgentMessage, AgentErrorSchema, TaskPayload


class RouterAgent:
    name = "router_agent"

    def __init__(self) -> None:
        self.config = None
        self.enforcer = None
        self.invoke_tool = None

    async def initialize(self, config: Any, enforcer: Any, invoke_tool: Any) -> None:
        self.config = config
        self.enforcer = enforcer
        self.invoke_tool = invoke_tool

    async def shutdown(self) -> None:
        return None

    async def handle(self, message: AgentMessage) -> AgentMessage:
        if not isinstance(message.payload, TaskPayload):
            raise TypeError("RouterAgent expects TaskPayload in message.payload")

        selected_model = getattr(self.config.models, "default", "phi-3-mini")
        enriched_payload = TaskPayload(
            user_intent=message.payload.user_intent,
            input_text=message.payload.input_text,
            selected_model=selected_model,
            intent_confidence=message.payload.intent_confidence,
            context=message.payload.context,
            constraints=message.payload.constraints,
        )

        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target="planner_agent",
            type="task.routing",
            payload=enriched_payload,
            metadata=message.metadata,
        )

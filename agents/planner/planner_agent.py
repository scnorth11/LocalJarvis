from typing import Any

from core.schema import AgentMessage, PlanPayload, PlanStep, TaskPayload


class PlannerAgent:
    name = "planner"

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
            raise TypeError("PlannerAgent expects TaskPayload in message.payload")

        step = PlanStep(
            id="step-1",
            action="generate_response",
            args={"text": message.payload.input_text, "model": message.payload.selected_model},
            model=message.payload.selected_model,
        )

        plan_payload = PlanPayload(steps=[step], expected_outputs=["response_text"])

        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target="executor",
            type="plan.created",
            payload=plan_payload,
            metadata=message.metadata,
        )

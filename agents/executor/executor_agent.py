from typing import Any

from core.schema import AgentMessage, ExecutionResult, PlanPayload


class ExecutorAgent:
    name = "executor"

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
        if not isinstance(message.payload, PlanPayload):
            raise TypeError("ExecutorAgent expects PlanPayload in message.payload")

        results = {
            step.id: {"status": "completed", "output": f"Executed action {step.action}"}
            for step in message.payload.steps
        }

        execution_result = ExecutionResult(results=results, final_output="Execution placeholder output")

        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target="persona",
            type="execution.completed",
            payload=execution_result,
            metadata=message.metadata,
        )

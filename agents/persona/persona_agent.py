from typing import Any

from core.schema import AgentMessage, ExecutionResult


class PersonaAgent:
    name = "persona"

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
        if not isinstance(message.payload, ExecutionResult):
            raise TypeError("PersonaAgent expects ExecutionResult in message.payload")

        voice_style = getattr(self.config.voice, "default_voice", "neutral")
        enriched_text = f"[{voice_style}] {message.payload.final_output}"

        enriched_result = ExecutionResult(
            results=message.payload.results,
            final_output=enriched_text,
        )

        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target="router",
            type="persona.enriched",
            payload=enriched_result,
            metadata=message.metadata,
        )

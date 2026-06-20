import logging
from typing import Any

from core.errors import AgentError
from core.schema import AgentErrorSchema, AgentMessage, ExecutionResult

logger = logging.getLogger(__name__)

# Maps voice style → persona display name used in formatted output.
_PERSONA_NAMES = {
    "alloy": "Jarvis",
    "neutral": "Assistant",
    "echo": "System",
}


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
        voice_style = getattr(self.config.voice, "default_voice", "alloy")
        persona_name = _PERSONA_NAMES.get(voice_style, "Jarvis")

        # Surface upstream errors as a natural-language response rather than crashing.
        if isinstance(message.payload, AgentErrorSchema):
            error = message.payload
            logger.warning(
                "PersonaAgent: surfacing upstream error [%s]: %s",
                error.error_type,
                error.message,
            )
            final_text = (
                f"{persona_name}: I encountered an issue while processing your request. "
                f"{error.message}"
            )
            return AgentMessage(
                id=message.id,
                timestamp=message.timestamp,
                source=self.name,
                target="router",
                type="persona.enriched",
                payload=ExecutionResult(results={}, final_output=final_text),
                metadata=message.metadata,
            )

        if not isinstance(message.payload, ExecutionResult):
            exc = AgentError(
                "ValidationError",
                "PersonaAgent expects ExecutionResult",
                agent=self.name,
                recoverable=False,
            )
            logger.error("PersonaAgent error: %s — %s", exc.error_type, exc.message)
            final_text = f"{persona_name}: I'm unable to format the response at this time."
            return AgentMessage(
                id=message.id,
                timestamp=message.timestamp,
                source=self.name,
                target="router",
                type="persona.enriched",
                payload=ExecutionResult(results={}, final_output=final_text),
                metadata=message.metadata,
            )

        raw = message.payload.final_output.strip()
        formatted = (
            f"{persona_name}: {raw}"
            if raw
            else f"{persona_name}: I wasn't able to produce a response."
        )

        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target="router",
            type="persona.enriched",
            payload=ExecutionResult(
                results=message.payload.results,
                final_output=formatted,
            ),
            metadata=message.metadata,
        )

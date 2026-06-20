from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from .errors import Result, AgentError, Success
from .schema import AgentErrorSchema, AgentMessage


class Agent(ABC):
    def __init__(self, name: str):
        self._name = name
        self._initialized = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> Result[None, AgentError]:
        self._initialized = True
        return Success(None)

    async def shutdown(self) -> Result[None, AgentError]:
        self._initialized = False
        return Success(None)

    @abstractmethod
    async def handle(self, message: AgentMessage) -> Result[AgentMessage, AgentError]:
        pass

    async def validate_message(self, message: AgentMessage) -> Result[AgentMessage, AgentError]:
        return Success(message)


class AgentBase:
    """Mixin providing shared helpers for concrete pipeline agents.

    Concrete agents (RouterAgent, PlannerAgent, ExecutorAgent, etc.) inherit
    this class to avoid repeating boilerplate.
    """

    name: str = ""

    @staticmethod
    def _make_error_reply(
        original: AgentMessage,
        exc: AgentError,
        source: str,
        target: str,
    ) -> AgentMessage:
        """Build a standard error :class:`AgentMessage` from an :class:`AgentError`.

        Parameters
        ----------
        original:
            The incoming message whose ``id``, ``timestamp``, and ``metadata``
            are preserved on the error reply.
        exc:
            The error that occurred.
        source:
            The agent emitting the error (usually ``self.name``).
        target:
            The downstream agent that should receive the error message.
        """
        import logging

        logging.getLogger(__name__).error(
            "%s error: %s — %s", source, exc.error_type, exc.message
        )
        return AgentMessage(
            id=original.id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            target=target,
            type="error",
            payload=AgentErrorSchema(
                error_type=exc.error_type,
                message=exc.message,
                agent=source,
                recoverable=exc.recoverable,
            ),
            metadata=original.metadata,
        )


from abc import ABC, abstractmethod
from .errors import Result, AgentError, Success
from .schema import AgentMessage


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

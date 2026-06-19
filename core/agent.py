from abc import ABC, abstractmethod
from .errors import Result, AgentError
from .schema import AgentMessage

class Agent(ABC):
    def __init__(self, name: str):
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def initialized(self) -> bool:
        pass

    async def initialize(self) -> Result[None, AgentError]:
        pass

    async def shutdown(self) -> Result[None, AgentError]:
        pass

    @abstractmethod
    async def handle(self, message: AgentMessage) -> Result[AgentMessage, AgentError]:
        pass

    async def validate_message(self, message: AgentMessage) -> Result[AgentMessage, AgentError]:
        pass

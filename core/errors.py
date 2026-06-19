from dataclasses import dataclass
from typing import Generic, TypeVar, Union, Optional, Any, Dict

T = TypeVar("T")
E = TypeVar("E")

@dataclass(frozen=True)
class Success(Generic[T]):
    value: T

@dataclass(frozen=True)
class Failure(Generic[E]):
    error: "AgentError"

Result = Union[Success[T], Failure[E]]

class AgentError(Exception):
    def __init__(
        self,
        error_type: str,
        message: str,
        agent: Optional[str] = None,
        step_id: Optional[str] = None,
        recoverable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ):
        pass

class ToolFailure(AgentError):
    pass

class ModelError(AgentError):
    pass

class TimeoutError_(AgentError):
    pass

class ValidationError_(AgentError):
    pass

class UnknownError(AgentError):
    pass

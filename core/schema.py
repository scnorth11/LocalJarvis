from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

@dataclass(frozen=True)
class MessageMetadata:
    correlation_id: UUID
    session_id: UUID
    logging_breakpoint: bool = False

@dataclass(frozen=True)
class TaskConstraints:
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout_ms: int = 5000

@dataclass(frozen=True)
class TaskPayload:
    user_intent: str
    input_text: str
    selected_model: str
    intent_confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    constraints: TaskConstraints = field(default_factory=TaskConstraints)

@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_ms: int = 100
    retry_on: List[str] = field(default_factory=lambda: ["ToolFailure", "Timeout"])

@dataclass(frozen=True)
class PlanStep:
    id: str
    action: str
    args: Dict[str, Any] = field(default_factory=dict)
    model: Optional[str] = None
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

@dataclass(frozen=True)
class PlanPayload:
    steps: List[PlanStep]
    expected_outputs: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class ExecutionResult:
    results: Dict[str, Dict[str, Any]]
    final_output: str = ""

@dataclass(frozen=True)
class MemoryData:
    text: str
    embedding: List[float] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class MemoryMessage:
    operation: str
    namespace: str
    query: Optional[str] = None
    data: Optional[MemoryData] = None
    results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass(frozen=True)
class AgentErrorSchema:
    error_type: str
    message: str
    agent: Optional[str] = None
    step_id: Optional[str] = None
    recoverable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AgentMessage:
    id: UUID
    timestamp: str
    source: str
    target: str
    type: str
    payload: Union[TaskPayload, PlanPayload, ExecutionResult, MemoryMessage, AgentErrorSchema, Dict]
    metadata: MessageMetadata = field(default_factory=lambda: MessageMetadata(uuid4(), uuid4()))

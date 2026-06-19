from typing import Literal, Dict, Set
from uuid import UUID

MessageType = Literal["task_request", "plan", "execution_result", "memory_op", "error"]
AgentName = Literal["router", "planner", "executor", "persona", "memory"]
ActionType = Literal["memory_search", "model_inference", "tool_call"]
OperationType = Literal["write", "search", "read", "delete"]
ExecutionStatus = Literal["success", "failed"]
ErrorType = Literal["ToolFailure", "ModelError", "Timeout", "Unknown"]

ALLOWED_ROUTES: Dict[str, Set[str]] = {
    "router": {"planner"},
    "planner": {"executor", "memory"},
    "executor": {"persona", "memory"},
    "persona": {"router"},
    "memory": {"router", "planner", "executor"},
}

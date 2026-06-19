import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict
from uuid import UUID
from .errors import Result, AgentError, Success, Failure
from .schema import (
    AgentMessage, MessageMetadata, TaskPayload, TaskConstraints, RetryPolicy,
    PlanStep, PlanPayload, ExecutionResult, MemoryMessage, MemoryData, AgentErrorSchema
)
from .validators import validate_agent_message

class AgentMessageEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


def payload_to_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if is_dataclass(payload):
        return asdict(payload)
    return {"value": payload}


def to_dict(obj: AgentMessage) -> Dict[str, Any]:
    data = asdict(obj)
    data["payload"] = payload_to_dict(obj.payload)
    return data


def to_json(obj: AgentMessage) -> str:
    return json.dumps(to_dict(obj), cls=AgentMessageEncoder)


def dict_to_payload(data: Dict[str, Any], msg_type: str) -> Result[Any, AgentError]:
    if not isinstance(data, dict):
        return Failure(AgentError("ValidationError", "Invalid payload"))
    if msg_type == "task_request":
        constraints = data.get("constraints")
        if isinstance(constraints, dict):
            data["constraints"] = TaskConstraints(**constraints)
        return Success(TaskPayload(**data))
    if msg_type == "plan":
        steps = []
        for step in data.get("steps", []):
            retry = step.get("retry_policy")
            if isinstance(retry, dict):
                retry = RetryPolicy(**retry)
            steps.append(PlanStep(
                id=step["id"],
                action=step["action"],
                args=step.get("args", {}),
                model=step.get("model"),
                retry_policy=retry if isinstance(retry, RetryPolicy) else RetryPolicy()
            ))
        return Success(PlanPayload(steps=steps, expected_outputs=data.get("expected_outputs", [])))
    if msg_type == "execution_result":
        return Success(ExecutionResult(results=data.get("results", {}), final_output=data.get("final_output", "")))
    if msg_type == "memory_op":
        memory_data = data.get("data")
        if isinstance(memory_data, dict):
            memory_data = MemoryData(**memory_data)
        return Success(MemoryMessage(
            operation=data["operation"],
            namespace=data["namespace"],
            query=data.get("query"),
            data=memory_data,
            results=data.get("results", [])
        ))
    if msg_type == "error":
        return Success(AgentErrorSchema(
            error_type=data["error_type"],
            message=data["message"],
            agent=data.get("agent"),
            step_id=data.get("step_id"),
            recoverable=data.get("recoverable", False),
            details=data.get("details", {})
        ))
    return Failure(AgentError("ValidationError", "Invalid message type"))


def from_dict(data: Dict[str, Any], validate: bool = True) -> Result[AgentMessage, AgentError]:
    if not isinstance(data, dict):
        return Failure(AgentError("ValidationError", "Invalid message"))
    try:
        message_id = UUID(str(data["id"]))
        metadata = data["metadata"]
        correlation = UUID(str(metadata["correlation_id"]))
        session = UUID(str(metadata["session_id"]))
        metadata_obj = MessageMetadata(
            correlation_id=correlation,
            session_id=session,
            logging_breakpoint=bool(metadata.get("logging_breakpoint", False))
        )
        payload_res = dict_to_payload(data["payload"], data["type"])
        if isinstance(payload_res, Failure):
            return payload_res
        message = AgentMessage(
            id=message_id,
            timestamp=data["timestamp"],
            source=data["source"],
            target=data["target"],
            type=data["type"],
            payload=payload_res.value,
            metadata=metadata_obj
        )
        if validate:
            validated = validate_agent_message(message)
            if isinstance(validated, Failure):
                return validated
        return Success(message)
    except Exception:
        return Failure(AgentError("ValidationError", "Invalid message"))


def from_json(json_str: str, validate: bool = True) -> Result[AgentMessage, AgentError]:
    try:
        data = json.loads(json_str)
    except Exception:
        return Failure(AgentError("ValidationError", "Invalid json"))
    return from_dict(data, validate=validate)


def roundtrip_dict(data: Dict[str, Any]) -> Result[Dict[str, Any], AgentError]:
    parsed = from_dict(data)
    if isinstance(parsed, Failure):
        return parsed
    return Success(to_dict(parsed.value))


def roundtrip_json(json_str: str) -> Result[str, AgentError]:
    parsed = from_json(json_str)
    if isinstance(parsed, Failure):
        return parsed
    return Success(to_json(parsed.value))

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set
from uuid import UUID
from .errors import Result, AgentError, Success, Failure
from .schema import AgentMessage
from .types import ALLOWED_ROUTES

MESSAGE_TYPES = [
    # canonical envelope types
    "task_request", "plan", "execution_result", "memory_op", "error",
    # agent pipeline types (dot-notation)
    "task.routing", "plan.created", "execution.completed", "memory.response", "persona.enriched",
]

# Maps dot-notation pipeline types to their canonical validation category.
_TYPE_CANONICAL: dict = {
    "task.routing": "task_request",
    "plan.created": "plan",
    "execution.completed": "execution_result",
    "memory.response": "memory_op",
    "persona.enriched": "execution_result",
}


def validate_required_fields(data: Dict[str, Any], required: list) -> Result[Dict[str, Any], AgentError]:
    for field in required:
        if field not in data:
            return Failure(AgentError("ValidationError", f"Missing {field}"))
    return Success(data)


def validate_enum(value: str, allowed: list, field_name: str) -> Result[str, AgentError]:
    if value not in allowed:
        return Failure(AgentError("ValidationError", f"Invalid {field_name}"))
    return Success(value)


def validate_timestamp(timestamp_str: str, max_age_ms: int = 300000) -> Result[str, AgentError]:
    if not isinstance(timestamp_str, str):
        return Failure(AgentError("ValidationError", "Invalid timestamp"))
    try:
        parsed = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except ValueError:
        return Failure(AgentError("ValidationError", "Invalid timestamp"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_ms = (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() * 1000
    if age_ms > max_age_ms or age_ms < -60000:
        return Failure(AgentError("ValidationError", "Timestamp out of range"))
    return Success(timestamp_str)


def validate_message_metadata(metadata: Any) -> Result[Any, AgentError]:
    if isinstance(metadata, dict):
        correlation = metadata.get("correlation_id")
        session = metadata.get("session_id")
        logging_breakpoint = metadata.get("logging_breakpoint")
    else:
        correlation = getattr(metadata, "correlation_id", None)
        session = getattr(metadata, "session_id", None)
        logging_breakpoint = getattr(metadata, "logging_breakpoint", None)
    if not isinstance(correlation, UUID):
        return Failure(AgentError("ValidationError", "Invalid metadata"))
    if not isinstance(session, UUID):
        return Failure(AgentError("ValidationError", "Invalid metadata"))
    if not isinstance(logging_breakpoint, bool):
        return Failure(AgentError("ValidationError", "Invalid metadata"))
    return Success(metadata)


def validate_message_type(msg_type: str, source: str, target: str) -> Result[str, AgentError]:
    return validate_enum(msg_type, MESSAGE_TYPES, "type")


def validate_payload_structure(msg_type: str, payload: Any) -> Result[Any, AgentError]:
    canonical = _TYPE_CANONICAL.get(msg_type, msg_type)

    def get(field: str):
        if isinstance(payload, dict):
            return payload.get(field)
        return getattr(payload, field, None)

    if canonical == "task_request":
        # task.routing (post-router) must have a model selected; the initial
        # task_request envelope only requires user_intent.
        if not get("user_intent"):
            return Failure(AgentError("ValidationError", "Invalid payload"))
        if msg_type == "task.routing" and not get("selected_model"):
            return Failure(AgentError("ValidationError", "Invalid payload"))
    elif canonical == "plan":
        if not get("steps"):
            return Failure(AgentError("ValidationError", "Invalid payload"))
    elif canonical == "execution_result":
        if get("results") is None:
            return Failure(AgentError("ValidationError", "Invalid payload"))
    elif canonical == "memory_op":
        if not get("operation") or not get("namespace"):
            return Failure(AgentError("ValidationError", "Invalid payload"))
    elif canonical == "error":
        if not get("error_type") or not get("message"):
            return Failure(AgentError("ValidationError", "Invalid payload"))
    else:
        return Failure(AgentError("ValidationError", "Invalid payload type"))
    return Success(payload)


def validate_routing(source: str, target: str, allowed_routes: Optional[Dict[str, Set[str]]] = None) -> Result[tuple, AgentError]:
    routes = allowed_routes if allowed_routes is not None else ALLOWED_ROUTES
    if source not in routes or target not in routes.get(source, set()):
        return Failure(AgentError("ValidationError", "Routing not allowed"))
    return Success((source, target))


def validate_envelope_structure(data: Dict[str, Any], max_age_ms: int = 300000) -> Result[Dict[str, Any], AgentError]:
    required = ["id", "timestamp", "source", "target", "type", "payload", "metadata"]
    if not isinstance(data, dict):
        return Failure(AgentError("ValidationError", "Invalid envelope"))
    fields = validate_required_fields(data, required)
    if isinstance(fields, Failure):
        return fields
    if not isinstance(data["source"], str) or not isinstance(data["target"], str):
        return Failure(AgentError("ValidationError", "Invalid envelope"))
    ts = validate_timestamp(data["timestamp"], max_age_ms)
    if isinstance(ts, Failure):
        return ts
    mt = validate_message_type(data["type"], data["source"], data["target"])
    if isinstance(mt, Failure):
        return mt
    metadata = validate_message_metadata(data["metadata"])
    if isinstance(metadata, Failure):
        return metadata
    routing = validate_routing(data["source"], data["target"])
    if isinstance(routing, Failure):
        return routing
    return Success(data)


def validate_agent_message(msg: AgentMessage, max_age_ms: int = 300000) -> Result[AgentMessage, AgentError]:
    if not isinstance(msg, AgentMessage):
        return Failure(AgentError("ValidationError", "Invalid message"))
    ts = validate_timestamp(msg.timestamp, max_age_ms)
    if isinstance(ts, Failure):
        return ts
    mt = validate_message_type(msg.type, msg.source, msg.target)
    if isinstance(mt, Failure):
        return mt
    md = validate_message_metadata(msg.metadata)
    if isinstance(md, Failure):
        return md
    payload = validate_payload_structure(msg.type, msg.payload)
    if isinstance(payload, Failure):
        return payload
    routing = validate_routing(msg.source, msg.target)
    if isinstance(routing, Failure):
        return routing
    return Success(msg)

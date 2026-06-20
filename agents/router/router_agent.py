import logging
import re
from typing import Any, List, Optional

from core.agent import AgentBase
from core.errors import AgentError
from core.schema import AgentErrorSchema, AgentMessage, MessageMetadata, TaskPayload
from models.selection_logic import select_model

logger = logging.getLogger(__name__)

# Maps user intent regex patterns → workflow names.
# Evaluated in order; first match wins.
_WORKFLOW_PATTERNS: List[tuple] = [
    (re.compile(r"\b(daily briefing|morning briefing|what'?s? (on |happening )?today|start (my )?day)\b", re.I), "daily_briefing"),
    (re.compile(r"\b(research|look up|find info about|search for).{0,60}\b(report|save|write|document|summarize)\b", re.I), "research"),
    (re.compile(r"\b(write code|create (a |the )?script|implement (a |the )?|code (a |the )?|generate (a |the )?(function|class|module|script|program))\b", re.I), "coding"),
    (re.compile(r"\b(add task|new task|create task|list tasks|show tasks|manage tasks|complete task|mark (task|todo)|task (list|management))\b", re.I), "task_management"),
]


class RouterAgent(AgentBase):
    name = "router"

    def __init__(self) -> None:
        self.config = None
        self.enforcer = None
        self.invoke_tool = None

    @classmethod
    def _detect_workflow(cls, user_intent: str) -> Optional[str]:
        """Return a workflow name if the intent matches a known workflow pattern, else None."""
        for pattern, workflow_name in _WORKFLOW_PATTERNS:
            if pattern.search(user_intent):
                return workflow_name
        return None

    async def initialize(self, config: Any, enforcer: Any, invoke_tool: Any) -> None:
        self.config = config
        self.enforcer = enforcer
        self.invoke_tool = invoke_tool

    async def shutdown(self) -> None:
        return None

    async def handle(self, message: AgentMessage) -> AgentMessage:
        if not isinstance(message.payload, TaskPayload):
            return self._error_message(
                message,
                AgentError(
                    "ValidationError",
                    "RouterAgent expects TaskPayload",
                    agent=self.name,
                    recoverable=False,
                ),
            )

        try:
            routing = select_model(message.payload.user_intent, self.config)
        except Exception as exc:
            return self._error_message(
                message,
                AgentError(
                    "ModelError",
                    f"Model selection failed: {exc}",
                    agent=self.name,
                    recoverable=True,
                ),
            )

        logger.info(
            "Router: tier=%s model=%s confidence=%.2f rationale=%r",
            routing.tier,
            routing.model,
            routing.confidence,
            routing.rationale,
        )

        workflow_name = self._detect_workflow(message.payload.user_intent)
        if workflow_name:
            logger.info("Router: detected workflow=%s", workflow_name)

        enriched_context = {
            **message.payload.context,
            "routing_tier": routing.tier,
            "routing_rationale": routing.rationale,
        }
        if workflow_name:
            enriched_context["workflow_name"] = workflow_name

        enriched_payload = TaskPayload(
            user_intent=message.payload.user_intent,
            input_text=message.payload.input_text,
            selected_model=routing.model,
            intent_confidence=routing.confidence,
            context=enriched_context,
            constraints=message.payload.constraints,
        )

        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target="planner",
            type="task.routing",
            payload=enriched_payload,
            metadata=message.metadata,
        )

    def _error_message(self, original: AgentMessage, exc: AgentError) -> AgentMessage:
        return self._make_error_reply(original, exc, source=self.name, target="planner")

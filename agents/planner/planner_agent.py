import logging
import re
from typing import Any, List, Optional

from core.agent import AgentBase
from core.errors import AgentError
from core.schema import (
    AgentErrorSchema,
    AgentMessage,
    PlanPayload,
    PlanStep,
    RetryPolicy,
    TaskPayload,
)

logger = logging.getLogger(__name__)

# Shared retry policies reused across plan steps.
_FAST_RETRY = RetryPolicy(max_retries=1, backoff_ms=0, retry_on=["ToolFailure"])
_MODEL_RETRY = RetryPolicy(max_retries=2, backoff_ms=500, retry_on=["ModelError", "Timeout"])


class PlannerAgent(AgentBase):
    name = "planner"

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
        # Forward errors downstream without modification.
        if isinstance(message.payload, AgentErrorSchema):
            return AgentMessage(
                id=message.id,
                timestamp=message.timestamp,
                source=self.name,
                target="executor",
                type="error",
                payload=message.payload,
                metadata=message.metadata,
            )

        if not isinstance(message.payload, TaskPayload):
            return self._error_message(
                message,
                AgentError(
                    "ValidationError",
                    "PlannerAgent expects TaskPayload",
                    agent=self.name,
                    recoverable=False,
                ),
            )

        try:
            steps = self._decompose(message.payload)
        except Exception as exc:
            return self._error_message(
                message,
                AgentError("Unknown", f"Planning failed: {exc}", agent=self.name, recoverable=True),
            )

        tier = message.payload.context.get("routing_tier", "?")
        logger.info("Planner: %d step(s) for tier=%s", len(steps), tier)

        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target="executor",
            type="plan.created",
            payload=PlanPayload(steps=steps, expected_outputs=["response_text"]),
            metadata=message.metadata,
        )

    def _decompose(self, payload: TaskPayload) -> List[PlanStep]:
        """Decompose the task into ordered PlanSteps based on routing tier."""
        tier = payload.context.get("routing_tier", "small")
        model = payload.selected_model
        steps: List[PlanStep] = []

        if tier == "large":
            # Fetch relevant context from memory before generating a response.
            steps.append(
                PlanStep(
                    id="step-1-memory",
                    action="memory_search",
                    args={"query": payload.user_intent, "namespace": "general"},
                    model=None,
                    retry_policy=_FAST_RETRY,
                )
            )

        # Detect whether the intent maps to a specific tool and prepend a tool_call step.
        tool_name = self._map_intent_to_tool(payload.user_intent)
        if tool_name:
            step_num = len(steps) + 1
            tool_args = self._build_tool_args(tool_name, payload)
            steps.append(
                PlanStep(
                    id=f"step-{step_num}-tool",
                    action="tool_call",
                    args=tool_args,
                    model=None,
                    retry_policy=_FAST_RETRY,
                )
            )

        step_num = len(steps) + 1
        steps.append(
            PlanStep(
                id=f"step-{step_num}-generate",
                action="generate_response",
                args={"text": payload.input_text, "model": model},
                model=model,
                retry_policy=_MODEL_RETRY,
            )
        )

        return steps

    # ------------------------------------------------------------------
    # Intent → tool mapping (keyword-based stub; replaced by LLM planner later)
    # ------------------------------------------------------------------

    _INTENT_TOOL_PATTERNS: List[tuple] = [
        # (compiled regex, tool_name)
        (re.compile(r"\b(write|create file|save draft|save to file|new document)\b", re.I), "writing"),
        (re.compile(r"\b(find file|search file|read file|look in file|open file|list (files|dir))\b", re.I), "file_search"),
        (re.compile(r"\b(research|search (for|the web|online)|look up|find info|google|duck|ddg)\b", re.I), "research"),
        (re.compile(r"\b(calculat\w*|comput\w*|solv\w*|math|equation|formula|sqrt|factorial)\b", re.I), "math"),
        (re.compile(r"\b(copy file|move file|rename|mkdir|make dir|delete file|remove file)\b", re.I), "file_ops"),
        (re.compile(r"\b(calendar|schedule|event|remind|appointment|meeting|upcoming)\b", re.I), "calendar"),
        (re.compile(r"\b(spotify|play (music|song|track|album)|pause music|skip (track|song)|queue|playlist|dj mode|now playing)\b", re.I), "spotify"),
    ]

    @classmethod
    def _map_intent_to_tool(cls, user_intent: str) -> Optional[str]:
        """Return the tool name if the intent matches a known tool pattern, else None."""
        for pattern, tool_name in cls._INTENT_TOOL_PATTERNS:
            if pattern.search(user_intent):
                return tool_name
        return None

    @staticmethod
    def _build_tool_args(tool_name: str, payload: TaskPayload) -> dict:
        """Build a minimal tool_call args dict for the detected tool.

        The downstream Executor passes these kwargs directly to the tool's
        ``run()`` method.  For most tools we only populate the universal
        arguments here; the LLM will flesh out specifics in future phases.
        """
        base = {"tool": tool_name}
        text = payload.input_text

        if tool_name == "writing":
            base.update({"op": "create", "path": "data/draft.txt", "content": text})
        elif tool_name == "file_search":
            base.update({"op": "search_content", "query": text})
        elif tool_name == "research":
            base.update({"query": text})
        elif tool_name == "math":
            base.update({"expression": text})
        elif tool_name == "file_ops":
            base.update({"op": "list", "src": "data"})
        elif tool_name == "calendar":
            base.update({"op": "upcoming", "n": 5})
        elif tool_name == "spotify":
            base.update({"op": "current_track"})

        return base

    def _error_message(self, original: AgentMessage, exc: AgentError) -> AgentMessage:
        return self._make_error_reply(original, exc, source=self.name, target="executor")

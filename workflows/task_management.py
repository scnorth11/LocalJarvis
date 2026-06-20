"""Task management workflow.

Detects the requested task operation (add / list / complete / upcoming),
executes it via the calendar tool through the pipeline, and returns a
confirmation message.
"""

import logging
import re
from typing import Any, Dict, List

from workflows.engine import BaseWorkflow, WorkflowStep

logger = logging.getLogger(__name__)

# Minimal patterns to detect the calendar operation from raw user input.
_OP_PATTERNS = [
    (re.compile(r"\b(add|create|new|set)\b.*\b(task|todo|reminder|event|appointment)\b", re.I), "add"),
    (re.compile(r"\b(complete|done|finish|check off|mark)\b.*\b(task|todo)\b", re.I), "complete"),
    (re.compile(r"\b(list|show|view|what|display)\b.*\b(tasks?|todos?|events?)\b", re.I), "list"),
    (re.compile(r"\b(upcoming|next|schedule|agenda|calendar)\b", re.I), "upcoming"),
]


def _detect_operation(text: str) -> str:
    """Return the calendar operation implied by *text*, defaulting to 'upcoming'."""
    for pattern, op in _OP_PATTERNS:
        if pattern.search(text):
            return op
    return "upcoming"


class TaskManagementWorkflow(BaseWorkflow):
    name = "task_management"

    def build_steps(self, context: Dict[str, Any]) -> List[WorkflowStep]:
        return [
            WorkflowStep(
                name="detect_operation",
                description="Determine which calendar operation the user wants",
                action=self._detect_operation,
            ),
            WorkflowStep(
                name="execute_task_op",
                description="Execute the calendar operation via the pipeline",
                action=self._execute_task_op,
            ),
            WorkflowStep(
                name="confirm",
                description="Generate a confirmation message",
                action=self._confirm,
            ),
            WorkflowStep(
                name="format_output",
                description="Store final output in context",
                action=self._format_output,
            ),
        ]

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    @staticmethod
    async def _detect_operation(context: Dict[str, Any]) -> str:
        user_input = context.get("input", context.get("input_text", ""))
        op = _detect_operation(user_input)
        context["task_op"] = op
        logger.debug("task_management: detected operation=%s", op)
        return op

    @staticmethod
    async def _execute_task_op(context: Dict[str, Any]) -> str:
        user_input = context.get("input", context.get("input_text", ""))
        op = context.get("task_op", "upcoming")
        pipeline = context["_pipeline"]

        if op == "add":
            prompt = f"add calendar event: {user_input}"
        elif op == "complete":
            prompt = f"mark task as complete: {user_input}"
        elif op == "list":
            prompt = "show my upcoming calendar events"
        else:  # upcoming
            prompt = "show upcoming calendar events for this week"

        return await pipeline(prompt)

    @staticmethod
    async def _confirm(context: Dict[str, Any]) -> str:
        result = context.get("execute_task_op", "").strip()
        op = context.get("task_op", "upcoming")
        pipeline = context["_pipeline"]

        if result:
            return await pipeline(f"Summarise this task management result briefly: {result}")
        return f"Task operation '{op}' completed with no output."

    @staticmethod
    async def _format_output(context: Dict[str, Any]) -> str:
        output = context.get("confirm", context.get("execute_task_op", "")).strip()
        if not output:
            output = "Task management operation completed."
        context["output"] = output
        return output

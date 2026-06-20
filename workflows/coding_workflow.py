"""Coding workflow.

Clarifies a coding requirement, generates the implementation, and saves
the resulting file.
"""

import logging
import re
from typing import Any, Dict, List

from workflows.engine import BaseWorkflow, WorkflowStep

logger = logging.getLogger(__name__)


def _derive_filename(requirement: str) -> str:
    """Guess a reasonable Python filename from the requirement text."""
    # Look for explicit 'for a <name>' or 'called <name>' pattern first.
    match = re.search(
        r"(?:for|called|named|a|an)\s+([\w_]+)\s*(?:function|class|module|script|program)?",
        requirement,
        re.I,
    )
    if match:
        name = re.sub(r"[^\w]", "_", match.group(1).lower())[:30]
        return f"data/{name}.py"
    slug = re.sub(r"[^\w\s]", "", requirement.lower())
    slug = re.sub(r"\s+", "_", slug.strip())[:30].strip("_")
    return f"data/{slug or 'output'}.py"


class CodingWorkflow(BaseWorkflow):
    name = "coding"

    def build_steps(self, context: Dict[str, Any]) -> List[WorkflowStep]:
        return [
            WorkflowStep(
                name="clarify_requirement",
                description="Clarify and expand the coding requirement",
                action=self._clarify_requirement,
                required=False,
            ),
            WorkflowStep(
                name="generate_code",
                description="Generate the code implementation",
                action=self._generate_code,
            ),
            WorkflowStep(
                name="save_file",
                description="Save generated code to a .py file",
                action=self._save_file,
                required=False,
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
    async def _clarify_requirement(context: Dict[str, Any]) -> str:
        requirement = context.get("requirement", context.get("input_text", ""))
        pipeline = context["_pipeline"]
        try:
            return await pipeline(
                f"Briefly clarify and restate this coding requirement in one paragraph: {requirement}"
            )
        except Exception as exc:
            logger.warning("coding_workflow: clarify_requirement failed: %s", exc)
            return requirement

    @staticmethod
    async def _generate_code(context: Dict[str, Any]) -> str:
        requirement = context.get("requirement", context.get("input_text", ""))
        clarification = context.get("clarify_requirement", "").strip()
        pipeline = context["_pipeline"]

        prompt_body = clarification if clarification else requirement
        return await pipeline(f"Write Python code for: {prompt_body}")

    @staticmethod
    async def _save_file(context: Dict[str, Any]) -> str:
        code = context.get("generate_code", "").strip()
        requirement = context.get("requirement", context.get("input_text", ""))
        path = context.get("output_path") or _derive_filename(requirement)
        pipeline = context["_pipeline"]

        if not code:
            return ""

        try:
            result = await pipeline(f"write to file {path}: {code}")
            context["file_path"] = path
            return result
        except Exception as exc:
            logger.warning("coding_workflow: save_file failed: %s", exc)
            return ""

    @staticmethod
    async def _format_output(context: Dict[str, Any]) -> str:
        code = context.get("generate_code", "").strip()
        file_path = context.get("file_path", "")
        if code:
            output = code
            if file_path:
                output += f"\n\n[Saved to {file_path}]"
        else:
            output = "Code generation did not produce a result."
        context["output"] = output
        return output

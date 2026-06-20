"""Research workflow.

Runs two research queries for breadth, synthesises the results via the
agent pipeline, then saves the report to a file.
"""

import logging
import re
from typing import Any, Dict, List

from workflows.engine import BaseWorkflow, WorkflowStep

logger = logging.getLogger(__name__)


def _slugify(text: str) -> str:
    """Return a safe filename slug (max 40 chars)."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:40] or "research"


class ResearchWorkflow(BaseWorkflow):
    name = "research"

    def build_steps(self, context: Dict[str, Any]) -> List[WorkflowStep]:
        return [
            WorkflowStep(
                name="search_primary",
                description="Primary research query",
                action=self._search_primary,
            ),
            WorkflowStep(
                name="search_secondary",
                description="Secondary research query for broader coverage",
                action=self._search_secondary,
                required=False,
            ),
            WorkflowStep(
                name="synthesize",
                description="Synthesise results into a coherent report",
                action=self._synthesize,
            ),
            WorkflowStep(
                name="save_report",
                description="Save the report to data/",
                action=self._save_report,
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
    async def _search_primary(context: Dict[str, Any]) -> str:
        query = context.get("query", context.get("input_text", ""))
        pipeline = context["_pipeline"]
        return await pipeline(f"research {query}")

    @staticmethod
    async def _search_secondary(context: Dict[str, Any]) -> str:
        query = context.get("query", context.get("input_text", ""))
        pipeline = context["_pipeline"]
        try:
            return await pipeline(f"research {query} overview background")
        except Exception as exc:
            logger.warning("research_workflow: secondary search failed: %s", exc)
            return ""

    @staticmethod
    async def _synthesize(context: Dict[str, Any]) -> str:
        primary = context.get("search_primary", "").strip()
        secondary = context.get("search_secondary", "").strip()
        query = context.get("query", "")
        pipeline = context["_pipeline"]

        parts = [p for p in (primary, secondary) if p]
        combined = "\n\n".join(parts) if parts else ""

        if combined:
            prompt = (
                f"Synthesise the following research into a well-structured report on '{query}'.\n\n"
                + combined
            )
        else:
            prompt = f"Write a comprehensive research report on '{query}'."

        return await pipeline(prompt)

    @staticmethod
    async def _save_report(context: Dict[str, Any]) -> str:
        pipeline = context["_pipeline"]
        synthesis = context.get("synthesize", "").strip()
        query = context.get("query", "research")
        slug = _slugify(query)
        path = f"data/research_{slug}.txt"

        if not synthesis:
            return ""

        try:
            result = await pipeline(
                f"write to file {path}: {synthesis}"
            )
            context["report_path"] = path
            return result
        except Exception as exc:
            logger.warning("research_workflow: save_report failed: %s", exc)
            return ""

    @staticmethod
    async def _format_output(context: Dict[str, Any]) -> str:
        synthesis = context.get("synthesize", "").strip()
        report_path = context.get("report_path", "")
        if synthesis:
            output = synthesis
            if report_path:
                output += f"\n\n[Report saved to {report_path}]"
        else:
            output = "Research completed but no results were retrieved."
        context["output"] = output
        return output

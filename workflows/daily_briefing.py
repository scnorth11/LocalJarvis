"""Daily briefing workflow.

Fetches upcoming calendar events and a news snapshot, then generates a
formatted morning briefing via the agent pipeline.
"""

import logging
from typing import Any, Dict, List

from workflows.engine import BaseWorkflow, WorkflowStep

logger = logging.getLogger(__name__)


class DailyBriefingWorkflow(BaseWorkflow):
    name = "daily_briefing"

    def build_steps(self, context: Dict[str, Any]) -> List[WorkflowStep]:
        return [
            WorkflowStep(
                name="fetch_calendar",
                description="Retrieve upcoming calendar events",
                action=self._fetch_calendar,
                required=False,
            ),
            WorkflowStep(
                name="fetch_news",
                description="Retrieve today's top tech/AI headlines",
                action=self._fetch_news,
                required=False,
            ),
            WorkflowStep(
                name="generate_briefing",
                description="Generate a natural-language briefing via the pipeline",
                action=self._generate_briefing,
            ),
            WorkflowStep(
                name="format_output",
                description="Assemble final briefing string and store in context['output']",
                action=self._format_output,
            ),
        ]

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    @staticmethod
    async def _fetch_calendar(context: Dict[str, Any]) -> str:
        pipeline = context["_pipeline"]
        try:
            result = await pipeline("show my upcoming calendar events for today and this week")
            return result
        except Exception as exc:
            logger.warning("daily_briefing: fetch_calendar failed: %s", exc)
            return ""

    @staticmethod
    async def _fetch_news(context: Dict[str, Any]) -> str:
        pipeline = context["_pipeline"]
        try:
            result = await pipeline("research today's top technology and AI news")
            return result
        except Exception as exc:
            logger.warning("daily_briefing: fetch_news failed: %s", exc)
            return ""

    @staticmethod
    async def _generate_briefing(context: Dict[str, Any]) -> str:
        pipeline = context["_pipeline"]
        calendar_text = context.get("fetch_calendar", "").strip()
        news_text = context.get("fetch_news", "").strip()

        sections: List[str] = []
        if calendar_text:
            sections.append(f"Calendar:\n{calendar_text}")
        if news_text:
            sections.append(f"News:\n{news_text}")

        if sections:
            summary_prompt = (
                "Generate a concise morning briefing from the following information.\n\n"
                + "\n\n".join(sections)
            )
        else:
            summary_prompt = "Give me a general good morning briefing."

        return await pipeline(summary_prompt)

    @staticmethod
    async def _format_output(context: Dict[str, Any]) -> str:
        briefing = context.get("generate_briefing", "").strip()
        output = briefing if briefing else "Good morning! No briefing data was available."
        context["output"] = output
        return output

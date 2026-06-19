#!/usr/bin/env python3
"""LocalJarvis CLI driver.

Usage::

    python main.py --text "your query here"   # single-shot mode
    python main.py                            # interactive REPL
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from uuid import uuid4

from config import load_config
from core.schema import AgentMessage, MessageMetadata, TaskPayload
from core.registry import AgentRegistry
from agents import RouterAgent, PlannerAgent, ExecutorAgent, MemoryAgent, PersonaAgent

logger = logging.getLogger("jarvis")


async def build_registry(config) -> AgentRegistry:
    """Instantiate and register all pipeline agents."""
    registry = AgentRegistry(config=config)
    await registry.register_agent("router", RouterAgent())
    await registry.register_agent("planner", PlannerAgent())
    await registry.register_agent("executor", ExecutorAgent())
    await registry.register_agent("memory", MemoryAgent())
    await registry.register_agent("persona", PersonaAgent())
    return registry


async def run_pipeline(registry: AgentRegistry, user_text: str) -> str:
    """Pass *user_text* through the Router → Planner → Executor → Persona pipeline.

    Returns the final persona-enriched output string.
    """
    now = datetime.now(timezone.utc).isoformat()
    initial = AgentMessage(
        id=uuid4(),
        timestamp=now,
        source="user",
        target="router",
        type="task_request",
        payload=TaskPayload(
            user_intent=user_text,
            input_text=user_text,
            selected_model="",
            intent_confidence=1.0,
        ),
        metadata=MessageMetadata(
            correlation_id=uuid4(),
            session_id=uuid4(),
        ),
    )

    logger.debug("Pipeline start | input=%r", user_text)

    routed = await registry.get("router").handle(initial)
    logger.debug("Router selected model: %s", routed.payload.selected_model)

    planned = await registry.get("planner").handle(routed)
    logger.debug("Planner produced %d step(s)", len(planned.payload.steps))

    executed = await registry.get("executor").handle(planned)
    logger.debug("Executor finished")

    enriched = await registry.get("persona").handle(executed)
    logger.debug("Pipeline end")

    return enriched.payload.final_output


async def main() -> None:
    parser = argparse.ArgumentParser(description="LocalJarvis voice assistant")
    parser.add_argument("--text", metavar="QUERY", help="Run a single query and exit")
    args = parser.parse_args()

    # load_config also sets up the logging framework (logging.yaml).
    config = load_config()

    logger.info("LocalJarvis starting up")
    registry = await build_registry(config)

    if args.text:
        result = await run_pipeline(registry, args.text)
        print(result)
        await registry.shutdown_all()
        return

    # Interactive REPL
    print("LocalJarvis ready. Type 'quit' or press Ctrl+D to exit.")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        try:
            result = await run_pipeline(registry, user_input)
            print(result)
        except Exception as exc:
            logger.exception("Pipeline error")
            print(f"Error: {exc}", file=sys.stderr)

    await registry.shutdown_all()
    logger.info("LocalJarvis shutdown")


if __name__ == "__main__":
    asyncio.run(main())

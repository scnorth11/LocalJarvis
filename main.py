#!/usr/bin/env python3
"""LocalJarvis CLI driver.

Usage::

    python main.py --text "your query here"   # single-shot mode
    python main.py                            # interactive REPL
"""
import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Load .env from project root before any other imports that may read env vars.
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    with _ENV_FILE.open() as _fh:
        for _line in _fh:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from config import load_config
from core.schema import AgentMessage, MessageMetadata, TaskPayload
from core.registry import AgentRegistry
from agents import RouterAgent, PlannerAgent, ExecutorAgent, MemoryAgent, PersonaAgent
from workflows.engine import WorkflowEngine
from workflows.daily_briefing import DailyBriefingWorkflow
from workflows.research_workflow import ResearchWorkflow
from workflows.coding_workflow import CodingWorkflow
from workflows.task_management import TaskManagementWorkflow
from workflows.voice_command import VoiceCommandWorkflow

logger = logging.getLogger("jarvis")

# Maps workflow names (injected by RouterAgent) to workflow classes.
_WORKFLOW_MAP = {
    "daily_briefing": DailyBriefingWorkflow,
    "research": ResearchWorkflow,
    "coding": CodingWorkflow,
    "task_management": TaskManagementWorkflow,
}


async def build_registry(config) -> AgentRegistry:
    """Instantiate and register all pipeline agents."""
    registry = AgentRegistry(config=config)

    memory_agent = MemoryAgent()
    executor_agent = ExecutorAgent()

    await registry.register_agent("router", RouterAgent())
    await registry.register_agent("planner", PlannerAgent())
    await registry.register_agent("executor", executor_agent)
    await registry.register_agent("memory", memory_agent)
    await registry.register_agent("persona", PersonaAgent())

    # Give the executor a direct reference to memory so memory_search
    # steps work without going through the security-gated tool path.
    executor_agent.set_memory_agent(memory_agent)

    return registry


async def run_voice_loop(
    registry: AgentRegistry,
    engine: WorkflowEngine,
    config,
) -> None:
    """Continuous hybrid voice loop: listen → respond → speak → repeat.

    Uses the voice settings from *config* (wake phrase, Piper voice, Whisper
    model, voice mode, PTT key).  Press Ctrl+C to exit.
    """
    voice_cfg = getattr(config, "voice", None)
    wake_phrase = getattr(voice_cfg, "wake_phrase", "jarvis") if voice_cfg else "jarvis"
    voice_mode = getattr(voice_cfg, "voice_mode", "hybrid") if voice_cfg else "hybrid"
    ptt_key = getattr(voice_cfg, "ptt_key", "f12") if voice_cfg else "f12"

    print(
        f"Voice mode active [{voice_mode}]. "
        f"Say '{wake_phrase}' or hold {ptt_key.upper()} to speak. "
        "Press Ctrl+C to quit."
    )

    while True:
        try:
            workflow = VoiceCommandWorkflow(config=voice_cfg)
            run = await engine.run(workflow, {})
            output = run.context.get("output") or run.error or "No response."
            print(f"\nJarvis: {output}\n")
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as exc:
            logger.exception("Voice loop iteration failed")
            print(f"[error] {exc}", file=sys.stderr)


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

    routed = await registry.route(initial)
    planned = await registry.route(routed)
    executed = await registry.route(planned)
    enriched = await registry.route(executed)

    logger.debug("Pipeline end")
    return enriched.payload.final_output


async def run_with_workflow(
    registry: AgentRegistry,
    engine: WorkflowEngine,
    user_text: str,
) -> str:
    """Route *user_text* through the pipeline, dispatching to a workflow if detected.

    The Router runs first in all cases.  If the routing result contains a
    ``workflow_name`` in its context, the corresponding :class:`BaseWorkflow`
    is executed via the engine.  Otherwise the standard pipeline continues.
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

    routed = await registry.route(initial)

    workflow_name = None
    if hasattr(routed.payload, "context"):
        workflow_name = routed.payload.context.get("workflow_name")

    if workflow_name and workflow_name in _WORKFLOW_MAP:
        logger.info("Dispatching to workflow: %s", workflow_name)
        workflow_cls = _WORKFLOW_MAP[workflow_name]
        workflow = workflow_cls()
        # Pass routing context + raw input into the workflow initial context.
        wf_context: dict = {
            "input_text": user_text,
            "query": user_text,
            "requirement": user_text,
            "input": user_text,
        }
        if hasattr(routed.payload, "context"):
            wf_context.update({
                k: v for k, v in routed.payload.context.items()
                if not k.startswith("_")
            })

        run = await engine.run(workflow, wf_context)
        return run.context.get("output", run.error or "Workflow completed with no output.")

    # No workflow matched — continue standard pipeline.
    logger.debug("No workflow detected, running standard pipeline")
    planned = await registry.route(routed)
    executed = await registry.route(planned)
    enriched = await registry.route(executed)
    return enriched.payload.final_output


async def main() -> None:
    parser = argparse.ArgumentParser(description="LocalJarvis voice assistant")
    parser.add_argument("--text", metavar="QUERY", help="Run a single query and exit")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Start continuous hybrid voice mode (wake word + push-to-talk)",
    )
    args = parser.parse_args()

    # load_config also sets up the logging framework (logging.yaml).
    config = load_config()

    logger.info("LocalJarvis starting up")
    registry = await build_registry(config)

    # Build a pipeline callable that the WorkflowEngine can pass to workflow steps.
    async def _pipeline(text: str) -> str:
        return await run_pipeline(registry, text)

    engine = WorkflowEngine(pipeline=_pipeline, db_path="data/workflows.db")
    engine.initialize()

    if args.text:
        result = await run_with_workflow(registry, engine, args.text)
        print(result)
        await registry.shutdown_all()
        engine.close()
        return

    if args.voice:
        try:
            await run_voice_loop(registry, engine, config)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print()
        await registry.shutdown_all()
        engine.close()
        logger.info("LocalJarvis shutdown")
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
            result = await run_with_workflow(registry, engine, user_input)
            print(result)
        except Exception as exc:
            logger.exception("Pipeline error")
            print(f"Error: {exc}", file=sys.stderr)

    await registry.shutdown_all()
    engine.close()
    logger.info("LocalJarvis shutdown")


if __name__ == "__main__":
    asyncio.run(main())


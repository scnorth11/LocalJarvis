"""Agent contract and pipeline integration tests.

Run with::

    pytest agents/test_agents.py -v

or as a standalone script::

    python agents/test_agents.py
"""
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

# ---------------------------------------------------------------------------
# Minimal stubs — no external dependencies required.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ModelsConfig:
    default: str = "small"
    tiers: Dict[str, str] = field(default_factory=lambda: {"small": "phi-3-mini", "large": "llama-3.1-8b"})

@dataclass(frozen=True)
class _PathsConfig:
    data_dir: str = "data"
    cache_dir: str = ".cache"
    log_dir: str = "logs"

@dataclass(frozen=True)
class _VoiceConfig:
    default_voice: str = "alloy"
    tts_engine: str = "piper"
    stt_engine: str = "whisper"

@dataclass(frozen=True)
class _OllamaConfig:
    base_url: str = "http://localhost:11434"
    model_names: Dict[str, str] = field(default_factory=lambda: {"phi-3-mini": "phi3:mini", "llama-3.1-8b": "llama3.1:8b"})

@dataclass(frozen=True)
class _FakeConfig:
    models: _ModelsConfig = field(default_factory=_ModelsConfig)
    paths: _PathsConfig = field(default_factory=_PathsConfig)
    voice: _VoiceConfig = field(default_factory=_VoiceConfig)
    allowed_tools: List[str] = field(default_factory=list)
    ollama: _OllamaConfig = field(default_factory=_OllamaConfig)
    timeouts: Dict[str, Any] = field(default_factory=lambda: {"model_call_seconds": 5})

_FAKE_CONFIG = _FakeConfig()

# ---------------------------------------------------------------------------
# Helpers to build AgentMessage fixtures.
# ---------------------------------------------------------------------------

from core.schema import (
    AgentErrorSchema,
    AgentMessage,
    ExecutionResult,
    MemoryData,
    MemoryMessage,
    MessageMetadata,
    PlanPayload,
    PlanStep,
    RetryPolicy,
    TaskPayload,
)
from core.errors import AgentError


def _meta() -> MessageMetadata:
    return MessageMetadata(correlation_id=uuid4(), session_id=uuid4())


def _task_msg(intent: str, selected_model: str = "", tier: str = "") -> AgentMessage:
    ctx: Dict[str, Any] = {}
    if tier:
        ctx["routing_tier"] = tier
    return AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="user",
        target="router",
        type="task_request",
        payload=TaskPayload(
            user_intent=intent,
            input_text=intent,
            selected_model=selected_model,
            intent_confidence=1.0,
            context=ctx,
        ),
        metadata=_meta(),
    )


def _routed_msg(intent: str, model: str, tier: str) -> AgentMessage:
    return AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="router",
        target="planner",
        type="task.routing",
        payload=TaskPayload(
            user_intent=intent,
            input_text=intent,
            selected_model=model,
            intent_confidence=0.9,
            context={"routing_tier": tier, "routing_rationale": "test"},
        ),
        metadata=_meta(),
    )


def _plan_msg(steps: List[PlanStep]) -> AgentMessage:
    return AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="planner",
        target="executor",
        type="plan.created",
        payload=PlanPayload(steps=steps, expected_outputs=["response_text"]),
        metadata=_meta(),
    )


def _exec_msg(results: Dict[str, Any], final_output: str) -> AgentMessage:
    return AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="executor",
        target="persona",
        type="execution.completed",
        payload=ExecutionResult(results=results, final_output=final_output),
        metadata=_meta(),
    )


def _error_msg(source: str, target: str, error_type: str, message: str) -> AgentMessage:
    return AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=source,
        target=target,
        type="error",
        payload=AgentErrorSchema(
            error_type=error_type,
            message=message,
            agent=source,
            recoverable=True,
        ),
        metadata=_meta(),
    )


# ---------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------

_PASSED: List[str] = []
_FAILED: List[str] = []


def _ok(name: str) -> None:
    _PASSED.append(name)
    print(f"  PASS  {name}")


def _fail(name: str, reason: str) -> None:
    _FAILED.append(name)
    print(f"  FAIL  {name}: {reason}")


def _assert(name: str, condition: bool, reason: str = "") -> None:
    if condition:
        _ok(name)
    else:
        _fail(name, reason or "assertion failed")


# ---------------------------------------------------------------------------
# 1. Model selection logic
# ---------------------------------------------------------------------------

async def test_selection_logic() -> None:
    from models.selection_logic import select_model, RoutingResult

    result = select_model("what time is it", _FAKE_CONFIG)
    _assert("select_model/simple → small tier", result.tier == "small")
    _assert("select_model/simple → phi-3-mini", result.model == "phi-3-mini")
    _assert("select_model/simple confidence ≥ 0.8", result.confidence >= 0.8)

    result = select_model("write a Python script to parse JSON", _FAKE_CONFIG)
    _assert("select_model/complex → large tier", result.tier == "large")
    _assert("select_model/complex → llama-3.1-8b", result.model == "llama-3.1-8b")
    _assert("select_model/complex confidence > 0.6", result.confidence > 0.6)
    _assert("select_model/complex is RoutingResult", isinstance(result, RoutingResult))

    result = select_model("analyze this research paper", _FAKE_CONFIG)
    _assert("select_model/research → large tier", result.tier == "large")


# ---------------------------------------------------------------------------
# 2. Router agent
# ---------------------------------------------------------------------------

async def test_router_agent() -> None:
    from agents.router.router_agent import RouterAgent

    agent = RouterAgent()
    await agent.initialize(_FAKE_CONFIG, None, None)

    # Valid simple intent
    msg = _task_msg("hello")
    out = await agent.handle(msg)
    _assert("router/simple returns task.routing type", out.type == "task.routing")
    _assert("router/simple target is planner", out.target == "planner")
    _assert("router/simple model is set", bool(out.payload.selected_model))
    _assert("router/simple tier in context", out.payload.context.get("routing_tier") == "small")

    # Valid complex intent
    msg = _task_msg("write a complex Python class")
    out = await agent.handle(msg)
    _assert("router/complex tier is large", out.payload.context.get("routing_tier") == "large")
    _assert("router/complex model is llama", "llama" in out.payload.selected_model.lower())

    # Wrong payload type → structured error
    bad_msg = AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="user",
        target="router",
        type="error",
        payload=AgentErrorSchema(error_type="X", message="bad", agent="test", recoverable=False),
        metadata=_meta(),
    )
    out = await agent.handle(bad_msg)
    _assert("router/bad payload → error type", out.type == "error")
    _assert("router/bad payload → AgentErrorSchema", isinstance(out.payload, AgentErrorSchema))


# ---------------------------------------------------------------------------
# 3. Planner agent
# ---------------------------------------------------------------------------

async def test_planner_agent() -> None:
    from agents.planner.planner_agent import PlannerAgent

    agent = PlannerAgent()
    await agent.initialize(_FAKE_CONFIG, None, None)

    # Simple tier → single generate_response step
    msg = _routed_msg("what is 2+2", "phi-3-mini", "small")
    out = await agent.handle(msg)
    _assert("planner/simple → plan.created", out.type == "plan.created")
    steps = out.payload.steps
    _assert("planner/simple → 1 step", len(steps) == 1)
    _assert("planner/simple step is generate_response", steps[0].action == "generate_response")
    _assert("planner/simple step has model", steps[0].model == "phi-3-mini")

    # Complex tier → memory_search + generate_response
    msg = _routed_msg("write a sorting algorithm", "llama-3.1-8b", "large")
    out = await agent.handle(msg)
    steps = out.payload.steps
    _assert("planner/complex → 2 steps", len(steps) == 2)
    _assert("planner/complex step[0] is memory_search", steps[0].action == "memory_search")
    _assert("planner/complex step[1] is generate_response", steps[1].action == "generate_response")

    # Error pass-through
    err_msg = _error_msg("router", "planner", "ModelError", "router failed")
    out = await agent.handle(err_msg)
    _assert("planner/error pass-through type", out.type == "error")
    _assert("planner/error pass-through target", out.target == "executor")
    _assert("planner/error pass-through payload", isinstance(out.payload, AgentErrorSchema))


# ---------------------------------------------------------------------------
# 4. Executor agent
# ---------------------------------------------------------------------------

async def test_executor_agent() -> None:
    from agents.executor.executor_agent import ExecutorAgent

    agent = ExecutorAgent()
    await agent.initialize(_FAKE_CONFIG, None, None)

    # Single generate_response step
    step = PlanStep(
        id="step-1-generate",
        action="generate_response",
        args={"text": "hello world", "model": "phi-3-mini"},
        model="phi-3-mini",
        retry_policy=RetryPolicy(),
    )
    msg = _plan_msg([step])
    out = await agent.handle(msg)
    _assert("executor/generate → execution.completed", out.type == "execution.completed")
    _assert("executor/generate step succeeded", out.payload.results["step-1-generate"]["status"] == "success")
    _assert("executor/generate final_output not empty", bool(out.payload.final_output))

    # memory_search step without memory agent → skipped gracefully
    mem_step = PlanStep(
        id="step-1-memory",
        action="memory_search",
        args={"query": "test", "namespace": "general"},
        model=None,
        retry_policy=RetryPolicy(),
    )
    msg = _plan_msg([mem_step, step])
    out = await agent.handle(msg)
    _assert("executor/mem+generate both have results", len(out.payload.results) == 2)
    _assert("executor/generate step still succeeds", out.payload.results["step-1-generate"]["status"] == "success")

    # Error pass-through
    err_msg = _error_msg("planner", "executor", "ValidationError", "plan failed")
    out = await agent.handle(err_msg)
    _assert("executor/error pass-through type", out.type == "error")
    _assert("executor/error pass-through target", out.target == "persona")

    # Unknown action → step marked failed, not a crash
    bad_step = PlanStep(
        id="step-bad",
        action="nonexistent_action",
        args={},
        model=None,
        retry_policy=RetryPolicy(),
    )
    msg = _plan_msg([bad_step])
    out = await agent.handle(msg)
    _assert("executor/unknown action → failed step", out.payload.results["step-bad"]["status"] == "failed")
    _assert("executor/unknown action → still returns ExecutionResult", isinstance(out.payload, ExecutionResult))


# ---------------------------------------------------------------------------
# 5. Memory agent
# ---------------------------------------------------------------------------

async def test_memory_agent() -> None:
    from agents.memory_agent.memory_agent import MemoryAgent
    from memory.sqlite_store import SQLiteStore

    # Use in-memory mode (no file path) by injecting a disconnected-store bypass.
    agent = MemoryAgent(store=None)
    agent.config = _FAKE_CONFIG
    # Skip calling initialize() to avoid creating a real DB file; use cache path.

    # Direct API: write
    result = await agent.write_direct("test_ns", "Paris is the capital of France", [], ["geo"])
    _assert("memory/write_direct returns status:stored", result.get("status") == "stored")

    # Direct API: read
    records = await agent.read_direct("test_ns")
    _assert("memory/read_direct returns 1 record", len(records) == 1)
    _assert("memory/read_direct text matches", records[0]["text"] == "Paris is the capital of France")

    # Direct API: search
    hits = await agent.search_direct("test_ns", "capital")
    _assert("memory/search_direct finds match", len(hits) == 1)

    hits = await agent.search_direct("test_ns", "nonexistent_xyz")
    _assert("memory/search_direct no match returns []", hits == [])

    # Message-envelope API: write
    mem_msg = AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="executor",
        target="memory",
        type="memory_op",
        payload=MemoryMessage(
            operation="write",
            namespace="env_ns",
            data=MemoryData(text="Berlin is in Germany", embedding=[], tags=["geo"]),
        ),
        metadata=_meta(),
    )
    out = await agent.handle(mem_msg)
    _assert("memory/handle write → memory.response", out.type == "memory.response")
    _assert("memory/handle write result stored", out.payload.results[0].get("status") == "stored")

    # Message-envelope API: search
    search_msg = AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="executor",
        target="memory",
        type="memory_op",
        payload=MemoryMessage(operation="search", namespace="env_ns", query="Berlin"),
        metadata=_meta(),
    )
    out = await agent.handle(search_msg)
    _assert("memory/handle search returns result", len(out.payload.results) == 1)

    # Bad operation → structured error
    bad_msg = AgentMessage(
        id=uuid4(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        source="executor",
        target="memory",
        type="memory_op",
        payload=MemoryMessage(operation="explode", namespace="test"),
        metadata=_meta(),
    )
    out = await agent.handle(bad_msg)
    _assert("memory/bad op → error response", out.type == "error")
    _assert("memory/bad op → AgentErrorSchema", isinstance(out.payload, AgentErrorSchema))


# ---------------------------------------------------------------------------
# 6. Persona agent
# ---------------------------------------------------------------------------

async def test_persona_agent() -> None:
    from agents.persona.persona_agent import PersonaAgent

    agent = PersonaAgent()
    await agent.initialize(_FAKE_CONFIG, None, None)

    # Normal execution result
    msg = _exec_msg({"s1": {"status": "success", "output": "hello"}}, "hello there")
    out = await agent.handle(msg)
    _assert("persona/normal → persona.enriched", out.type == "persona.enriched")
    _assert("persona/normal prefixes with name", out.payload.final_output.startswith("Jarvis:"))
    _assert("persona/normal preserves results", len(out.payload.results) == 1)

    # Empty final output → fallback message
    msg = _exec_msg({}, "")
    out = await agent.handle(msg)
    _assert("persona/empty output → fallback", "wasn't able" in out.payload.final_output)

    # Upstream error → natural-language error response
    err_msg = _error_msg("executor", "persona", "ToolFailure", "tool X failed")
    out = await agent.handle(err_msg)
    _assert("persona/error → persona.enriched", out.type == "persona.enriched")
    _assert("persona/error → error text contains message", "tool X failed" in out.payload.final_output)


# ---------------------------------------------------------------------------
# 7. End-to-end pipeline
# ---------------------------------------------------------------------------

async def test_pipeline_integration() -> None:
    """Full Router → Planner → Executor → Persona loop through the registry."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from config.loader import load_config
    from core.registry import AgentRegistry
    from agents import RouterAgent, PlannerAgent, ExecutorAgent, MemoryAgent, PersonaAgent
    from core.schema import AgentMessage, MessageMetadata, TaskPayload

    try:
        config = load_config()
    except Exception as exc:
        _fail("pipeline/config loaded", f"load_config() raised: {exc}")
        return
    _ok("pipeline/config loaded")

    registry = AgentRegistry(config=config)
    memory_agent = MemoryAgent()
    executor_agent = ExecutorAgent()
    await registry.register_agent("router", RouterAgent())
    await registry.register_agent("planner", PlannerAgent())
    await registry.register_agent("executor", executor_agent)
    await registry.register_agent("memory", memory_agent)
    await registry.register_agent("persona", PersonaAgent())
    executor_agent.set_memory_agent(memory_agent)
    _ok("pipeline/registry built")

    async def _run(text: str) -> str:
        now = datetime.now(timezone.utc).isoformat()
        msg = AgentMessage(
            id=uuid4(),
            timestamp=now,
            source="user",
            target="router",
            type="task_request",
            payload=TaskPayload(
                user_intent=text,
                input_text=text,
                selected_model="",
                intent_confidence=1.0,
            ),
            metadata=MessageMetadata(correlation_id=uuid4(), session_id=uuid4()),
        )
        routed = await registry.get("router").handle(msg)
        planned = await registry.get("planner").handle(routed)
        executed = await registry.get("executor").handle(planned)
        enriched = await registry.get("persona").handle(executed)
        return enriched.payload.final_output

    # Simple query
    out = await _run("what is the time")
    _assert("pipeline/simple returns non-empty string", bool(out))
    _assert("pipeline/simple starts with Jarvis", out.startswith("Jarvis:"))
    _assert("pipeline/simple uses phi-3-mini", "phi-3-mini" in out)

    # Complex query → memory_search step prepended, large model
    out = await _run("write a Python class for a binary tree")
    _assert("pipeline/complex returns non-empty string", bool(out))
    _assert("pipeline/complex uses llama", "llama" in out)

    await registry.shutdown_all()
    _ok("pipeline/shutdown clean")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def _run_all() -> None:
    print("\n── Model Selection ──────────────────────────────────")
    await test_selection_logic()
    print("\n── Router ───────────────────────────────────────────")
    await test_router_agent()
    print("\n── Planner ──────────────────────────────────────────")
    await test_planner_agent()
    print("\n── Executor ─────────────────────────────────────────")
    await test_executor_agent()
    print("\n── Memory ───────────────────────────────────────────")
    await test_memory_agent()
    print("\n── Persona ──────────────────────────────────────────")
    await test_persona_agent()
    print("\n── End-to-end Pipeline ──────────────────────────────")
    await test_pipeline_integration()

    print(f"\n{'─'*52}")
    total = len(_PASSED) + len(_FAILED)
    print(f"Results: {len(_PASSED)}/{total} passed", end="")
    if _FAILED:
        print(f"  |  FAILED: {', '.join(_FAILED)}")
    else:
        print("  — all green")


# pytest entry-points — each async function is wrapped for pytest-asyncio-free
# use via a sync wrapper collected by pytest.
def test_model_selection_logic():   asyncio.run(test_selection_logic())
def test_router():                  asyncio.run(test_router_agent())
def test_planner():                 asyncio.run(test_planner_agent())
def test_executor():                asyncio.run(test_executor_agent())
def test_memory():                  asyncio.run(test_memory_agent())
def test_persona():                 asyncio.run(test_persona_agent())
def test_pipeline():                asyncio.run(test_pipeline_integration())


if __name__ == "__main__":
    asyncio.run(_run_all())
    sys.exit(1 if _FAILED else 0)

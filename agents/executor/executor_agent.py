import logging
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from core.agent import AgentBase
from core.errors import AgentError, ToolFailure
from core.schema import (
    AgentErrorSchema,
    AgentMessage,
    ExecutionResult,
    PlanPayload,
    PlanStep,
)
from tools.tool_loader import ToolLoader

logger = logging.getLogger(__name__)


class ExecutorAgent(AgentBase):
    name = "executor"

    def __init__(self) -> None:
        self.config = None
        self.enforcer = None
        self.invoke_tool = None
        self._memory_agent: Optional[Any] = None  # injected via set_memory_agent()
        self.tools: Dict[str, Callable[..., str]] = {}  # populated in initialize()

    def set_memory_agent(self, agent: Any) -> None:
        """Inject a memory agent reference for memory_search steps."""
        self._memory_agent = agent

    async def initialize(self, config: Any, enforcer: Any, invoke_tool: Any) -> None:
        self.config = config
        self.enforcer = enforcer
        self.invoke_tool = invoke_tool
        self.tools = ToolLoader().load(config)
        logger.info("ExecutorAgent: loaded tools: %s", list(self.tools.keys()))

    async def shutdown(self) -> None:
        return None

    async def handle(self, message: AgentMessage) -> AgentMessage:
        # Forward errors downstream without modification.
        if isinstance(message.payload, AgentErrorSchema):
            return AgentMessage(
                id=message.id,
                timestamp=message.timestamp,
                source=self.name,
                target="persona",
                type="error",
                payload=message.payload,
                metadata=message.metadata,
            )

        if not isinstance(message.payload, PlanPayload):
            return self._error_message(
                message,
                AgentError(
                    "ValidationError",
                    "ExecutorAgent expects PlanPayload",
                    agent=self.name,
                    recoverable=False,
                ),
            )

        steps = message.payload.steps
        results: Dict[str, Dict[str, Any]] = {}
        aborted = False

        for step in steps:
            if aborted:
                results[step.id] = {
                    "status": "skipped",
                    "output": "",
                    "error": "aborted due to prior non-recoverable failure",
                }
                continue

            try:
                output = await self._run_step(step, results)
                results[step.id] = {"status": "success", "output": output}
                logger.debug("Executor: step %s completed", step.id)
            except AgentError as exc:
                results[step.id] = {
                    "status": "failed",
                    "output": "",
                    "error": exc.message,
                    "recoverable": exc.recoverable,
                }
                logger.warning(
                    "Executor: step %s failed (%s): %s", step.id, exc.error_type, exc.message
                )
                if not exc.recoverable:
                    aborted = True
            except Exception as exc:
                results[step.id] = {
                    "status": "failed",
                    "output": "",
                    "error": str(exc),
                    "recoverable": False,
                }
                logger.warning("Executor: step %s unexpected error: %s", step.id, exc)
                aborted = True

        final_output = self._build_final_output(results, steps)
        return AgentMessage(
            id=message.id,
            timestamp=message.timestamp,
            source=self.name,
            target="persona",
            type="execution.completed",
            payload=ExecutionResult(results=results, final_output=final_output),
            metadata=message.metadata,
        )

    async def _run_step(self, step: PlanStep, prior_results: Dict[str, Any]) -> str:
        """Dispatch a single plan step and return its string output."""
        if step.action == "memory_search":
            return await self._step_memory_search(step)
        if step.action == "generate_response":
            return self._step_generate_response(step, prior_results)
        if step.action == "tool_call":
            return self._step_tool_call(step)
        raise AgentError(
            "ToolFailure",
            f"Unknown step action: {step.action!r}",
            agent=self.name,
            step_id=step.id,
            recoverable=False,
        )

    async def _step_memory_search(self, step: PlanStep) -> str:
        if self._memory_agent is None:
            logger.debug("Executor: memory agent not wired, skipping memory_search")
            return ""
        namespace = step.args.get("namespace", "general")
        query = step.args.get("query", "")
        try:
            results = await self._memory_agent.search_direct(namespace, query)
            snippets = [r.get("text", "") for r in results[:3] if r.get("text")]
            return "\n".join(snippets)
        except Exception as exc:
            raise ToolFailure(
                "ToolFailure",
                f"Memory search failed: {exc}",
                agent=self.name,
                step_id=step.id,
                recoverable=True,
            )

    def _step_generate_response(self, step: PlanStep, prior_results: Dict[str, Any]) -> str:
        text = step.args.get("text", "")
        model_tier = step.args.get("model") or step.model or "unknown"

        # Resolve tier name → Ollama model identifier via config.
        ollama_model = model_tier
        if self.config is not None and hasattr(self.config, "ollama"):
            ollama_model = self.config.ollama.model_names.get(model_tier, model_tier)

        # Build prompt — inject context from successful prior steps (memory / tools).
        prior_outputs = [
            r["output"]
            for r in prior_results.values()
            if r.get("status") == "success" and r.get("output")
        ]
        system_prompt = (
            "You are Jarvis, a helpful and concise local AI assistant. "
            "Respond directly and naturally, as if speaking aloud."
        )
        if prior_outputs:
            context_block = "\n".join(prior_outputs)
            prompt = (
                f"{system_prompt}\n\n"
                f"Context:\n{context_block}\n\n"
                f"User: {text}"
            )
        else:
            prompt = f"{system_prompt}\n\nUser: {text}"

        if not shutil.which("ollama"):
            logger.warning("ExecutorAgent: ollama binary not found — returning placeholder")
            return f"[Ollama unavailable] {text}"

        timeout = 30
        if self.config is not None and hasattr(self.config, "timeouts"):
            timeout = int(self.config.timeouts.get("model_call_seconds", 30))

        try:
            result = subprocess.run(
                ["ollama", "run", ollama_model],
                input=prompt.encode(),
                capture_output=True,
                timeout=timeout,
            )
            response = result.stdout.decode(errors="replace").strip()
            if result.returncode != 0:
                err = result.stderr.decode(errors="replace").strip()
                logger.error("ExecutorAgent: ollama exited %d: %s", result.returncode, err)
                return f"[model error] {err or text}"
            return response or f"[empty response from {ollama_model}]"
        except subprocess.TimeoutExpired:
            logger.error("ExecutorAgent: ollama timed out after %ds", timeout)
            return "[timeout] The model took too long to respond."
        except Exception as exc:
            logger.error("ExecutorAgent: ollama call failed: %s", exc)
            return f"[error] {exc}"

    def _step_tool_call(self, step: PlanStep) -> str:
        if self.invoke_tool is None:
            raise ToolFailure(
                "ToolFailure",
                "invoke_tool not available",
                agent=self.name,
                step_id=step.id,
                recoverable=False,
            )
        tool_name = step.args.get("tool", "")
        if not tool_name:
            raise ToolFailure(
                "ToolFailure",
                "step.args['tool'] is required for tool_call action",
                agent=self.name,
                step_id=step.id,
                recoverable=False,
            )
        tool_args = {k: v for k, v in step.args.items() if k not in ("tool",)}
        try:
            return str(self.invoke_tool(tool_name, **tool_args))
        except PermissionError as exc:
            raise ToolFailure(
                "ToolFailure",
                f"Tool access denied: {exc}",
                agent=self.name,
                step_id=step.id,
                recoverable=False,
            )
        except LookupError as exc:
            raise ToolFailure(
                "ToolFailure",
                f"Tool not found: {exc}",
                agent=self.name,
                step_id=step.id,
                recoverable=False,
            )

    @staticmethod
    def _build_final_output(
        results: Dict[str, Dict[str, Any]], steps: List[PlanStep]
    ) -> str:
        """Return the last successful generate_response output, or any successful output."""
        for step in reversed(steps):
            if step.action == "generate_response":
                r = results.get(step.id, {})
                if r.get("status") == "success":
                    return r.get("output", "")
        for step in reversed(steps):
            r = results.get(step.id, {})
            if r.get("status") == "success":
                return r.get("output", "")
        return ""

    def _error_message(self, original: AgentMessage, exc: AgentError) -> AgentMessage:
        return self._make_error_reply(original, exc, source=self.name, target="persona")

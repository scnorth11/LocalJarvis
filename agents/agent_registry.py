from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, Protocol

from config.config import RestrictedConfig
from config.loader import load_config
from security.capabilities import build_capability_map
from security.enforcement import SecurityEnforcer


class AgentContract(Protocol):
    name: str
    config: Any
    enforcer: SecurityEnforcer
    invoke_tool: Callable[..., Any]

    async def initialize(self, config: RestrictedConfig, enforcer: SecurityEnforcer, invoke_tool: Callable[..., Any]) -> Any: ...
    async def shutdown(self) -> Any: ...
    async def handle(self, message: Any) -> Any: ...


@dataclass
class AgentRecord:
    name: str
    instance: AgentContract
    config: RestrictedConfig
    tool_invoker: Callable[..., Any]


class AgentProxy:
    def __init__(self, record: AgentRecord):
        self._record = record

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def config(self) -> RestrictedConfig:
        return self._record.config

    async def handle(self, message: Any) -> Any:
        handler = getattr(self._record.instance, "handle", None)
        if not callable(handler):
            raise AttributeError(f"Agent '{self._record.name}' has no handle(message) method")
        result = handler(message)
        if inspect.isawaitable(result):
            return await result
        return result

    async def shutdown(self) -> Any:
        shutdown = getattr(self._record.instance, "shutdown", None)
        if not callable(shutdown):
            return None
        result = shutdown()
        if inspect.isawaitable(result):
            return await result
        return result


class AgentRegistry:
    """Minimal agent registry that wires config + security for agents.

    - Loads config once at startup
    - Builds a CapabilityMap and SecurityEnforcer
    - Supplies each agent with a restricted config view and a tool invoker
    """

    def __init__(self) -> None:
        self._config = load_config()
        capability_map = build_capability_map(self._config)
        self.enforcer = SecurityEnforcer(capability_map)
        self._agents: Dict[str, AgentRecord] = {}

    async def register_agent(self, name: str, agent_instance: AgentContract) -> None:
        """Register an agent instance under `name`.

        The registry will:
        - create a restricted config view via `config.for_agent(name)`
        - create a `tool_invoker` that enforces permissions before calling tools
        - inject `config`, `enforcer`, and `invoke_tool` into the agent when possible
        """
        restricted_config = self._config.for_agent(name)

        def tool_invoker(agent_name: str, tool_name: str, *args: Any, **kwargs: Any) -> Any:
            # Enforce capability first
            self.enforcer.enforce(agent_name, tool_name)

            # Prefer an explicit tools dict on the agent
            tools = getattr(agent_instance, "tools", None)
            if isinstance(tools, dict) and tool_name in tools:
                func = tools[tool_name]
                if callable(func):
                    return func(*args, **kwargs)

            # Fallback to method named `tool_<tool_name>` on the agent
            fallback = getattr(agent_instance, f"tool_{tool_name}", None)
            if callable(fallback):
                return fallback(*args, **kwargs)

            raise LookupError(f"Tool '{tool_name}' not found for agent '{agent_name}'")

        record = AgentRecord(name=name, instance=agent_instance, config=restricted_config, tool_invoker=tool_invoker)

        # Inject common attributes so agents can use them instead of loading config directly.
        try:
            setattr(agent_instance, "config", restricted_config)
            setattr(agent_instance, "enforcer", self.enforcer)
            setattr(agent_instance, "invoke_tool", lambda tool_name, *a, **k: tool_invoker(name, tool_name, *a, **k))

            init = getattr(agent_instance, "initialize", None)
            if callable(init):
                init_result = init(restricted_config, self.enforcer, lambda t, *a, **k: tool_invoker(name, t, *a, **k))
                if inspect.isawaitable(init_result):
                    await init_result
        except Exception:
            # Keep registration resilient; attribute injection is best-effort.
            pass

        self._agents[name] = record

    async def shutdown_agent(self, name: str) -> None:
        rec = self._agents.get(name)
        if rec is None:
            raise KeyError(f"Agent not registered: {name}")
        shutdown = getattr(rec.instance, "shutdown", None)
        if not callable(shutdown):
            return
        result = shutdown()
        if inspect.isawaitable(result):
            await result

    def get(self, name: str) -> AgentProxy:
        """Retrieve an agent proxy by name."""
        rec = self._agents.get(name)
        if rec is None:
            raise KeyError(f"Agent not registered: {name}")
        return AgentProxy(rec)

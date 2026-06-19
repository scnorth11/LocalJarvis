import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class AgentContract(Protocol):
    """Structural interface every registered agent must satisfy."""

    name: str
    config: Any
    enforcer: Any
    invoke_tool: Callable[..., Any]

    async def initialize(self, config: Any, enforcer: Any, invoke_tool: Callable[..., Any]) -> Any: ...
    async def shutdown(self) -> Any: ...
    async def handle(self, message: Any) -> Any: ...


@dataclass
class AgentRecord:
    name: str
    instance: AgentContract
    config: Any
    tool_invoker: Callable[..., Any]


class AgentProxy:
    """Thin wrapper around an AgentRecord that exposes a clean async API."""

    def __init__(self, record: AgentRecord) -> None:
        self._record = record

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def config(self) -> Any:
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
        fn = getattr(self._record.instance, "shutdown", None)
        if not callable(fn):
            return None
        result = fn()
        if inspect.isawaitable(result):
            return await result
        return result


class AgentRegistry:
    """Runtime registry that wires config + security for each registered agent.

    Accepts an optional pre-loaded :class:`~config.config.Config` so that the
    caller controls when config loading (and logging setup) happens.  When no
    config is supplied, the registry loads ``config/settings.yaml`` itself.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        from config.loader import load_config
        from security.capabilities import build_capability_map
        from security.enforcement import SecurityEnforcer

        self._config = config if config is not None else load_config()
        capability_map = build_capability_map(self._config)
        self.enforcer = SecurityEnforcer(capability_map)
        self._agents: Dict[str, AgentRecord] = {}

    async def register_agent(self, name: str, agent_instance: AgentContract) -> None:
        """Register *agent_instance* under *name*.

        Injects a restricted config view, the shared enforcer, and a
        permission-gated tool invoker into the agent instance.
        """
        restricted_config = self._config.for_agent(name)

        def tool_invoker(agent_name: str, tool_name: str, *args: Any, **kwargs: Any) -> Any:
            self.enforcer.enforce(agent_name, tool_name)
            tools = getattr(agent_instance, "tools", None)
            if isinstance(tools, dict) and tool_name in tools:
                fn = tools[tool_name]
                if callable(fn):
                    return fn(*args, **kwargs)
            fallback = getattr(agent_instance, f"tool_{tool_name}", None)
            if callable(fallback):
                return fallback(*args, **kwargs)
            raise LookupError(f"Tool '{tool_name}' not found for agent '{agent_name}'")

        record = AgentRecord(
            name=name,
            instance=agent_instance,
            config=restricted_config,
            tool_invoker=tool_invoker,
        )

        # Best-effort attribute injection so agents can rely on self.config etc.
        try:
            agent_instance.config = restricted_config  # type: ignore[attr-defined]
            agent_instance.enforcer = self.enforcer  # type: ignore[attr-defined]
            agent_instance.invoke_tool = lambda t, *a, **k: tool_invoker(name, t, *a, **k)  # type: ignore[attr-defined]

            init = getattr(agent_instance, "initialize", None)
            if callable(init):
                result = init(restricted_config, self.enforcer, lambda t, *a, **k: tool_invoker(name, t, *a, **k))
                if inspect.isawaitable(result):
                    await result
        except Exception as exc:  # pragma: no cover
            logger.warning("Attribute injection failed for agent '%s': %s", name, exc)

        self._agents[name] = record
        logger.debug("Registered agent: %s", name)

    def get(self, name: str) -> AgentProxy:
        """Return a proxy for *name*, raising :exc:`KeyError` if not found."""
        rec = self._agents.get(name)
        if rec is None:
            raise KeyError(f"Agent not registered: {name}")
        return AgentProxy(rec)

    def list_agents(self) -> list:
        return list(self._agents.keys())

    async def shutdown_agent(self, name: str) -> None:
        rec = self._agents.get(name)
        if rec is None:
            raise KeyError(f"Agent not registered: {name}")
        fn = getattr(rec.instance, "shutdown", None)
        if callable(fn):
            result = fn()
            if inspect.isawaitable(result):
                await result

    async def shutdown_all(self) -> None:
        for name in list(self._agents):
            await self.shutdown_agent(name)

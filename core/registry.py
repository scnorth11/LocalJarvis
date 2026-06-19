from typing import Dict, Optional, Set
from .agent import Agent
from .errors import Result, AgentError, Success, Failure
from .schema import AgentMessage, AgentErrorSchema
from .types import ALLOWED_ROUTES
from .validators import validate_agent_message, validate_routing

class AgentRegistry:
    def __init__(self, allowed_routes: Optional[Dict[str, Set[str]]] = None, max_message_age_ms: int = 300000):
        self._agents: Dict[str, Agent] = {}
        self._allowed_routes = allowed_routes if allowed_routes is not None else ALLOWED_ROUTES
        self._max_message_age_ms = max_message_age_ms
        self._correlation_map: Dict[str, AgentMessage] = {}

    async def register_agent(self, agent: Agent) -> Result[None, AgentError]:
        if agent.name in self._agents:
            return Failure(AgentError("ValidationError", "Agent already registered"))
        self._agents[agent.name] = agent
        return Success(None)

    def get_agent(self, name: str) -> Result[Agent, AgentError]:
        if name not in self._agents:
            return Failure(AgentError("ValidationError", "Agent not found"))
        return Success(self._agents[name])

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    async def validate_message(self, message: AgentMessage) -> Result[AgentMessage, AgentError]:
        validated = validate_agent_message(message, self._max_message_age_ms)
        if isinstance(validated, Failure):
            return validated
        routing = self.validate_routing(message.source, message.target)
        if isinstance(routing, Failure):
            return routing
        self._correlation_map[str(message.metadata.correlation_id)] = message
        return Success(message)

    def validate_routing(self, source: str, target: str) -> Result[tuple, AgentError]:
        return validate_routing(source, target, self._allowed_routes)

    async def dispatch(self, message: AgentMessage) -> Result[AgentMessage, AgentError]:
        validated = await self.validate_message(message)
        if isinstance(validated, Failure):
            return validated
        target_agent = self.get_agent(message.target)
        if isinstance(target_agent, Failure):
            return target_agent
        agent = target_agent.value
        if hasattr(agent, "validate_message"):
            agent_valid = await agent.validate_message(message)
            if isinstance(agent_valid, Failure):
                return agent_valid
        result = await agent.handle(message)
        if isinstance(result, Failure):
            error = result.error
            schema = AgentErrorSchema(
                error_type=getattr(error, "error_type", "UnknownError"),
                message=str(error),
                agent=getattr(error, "agent", None),
                step_id=getattr(error, "step_id", None),
                recoverable=getattr(error, "recoverable", False),
                details=getattr(error, "details", {}) or {}
            )
            return Failure(schema)
        return result

    def get_routing_rules(self) -> Dict[str, Set[str]]:
        return self._allowed_routes

    def update_routing_rules(self, new_routes: Dict[str, Set[str]]) -> Result[None, AgentError]:
        if not isinstance(new_routes, dict):
            return Failure(AgentError("ValidationError", "Invalid routing rules"))
        self._allowed_routes = new_routes
        return Success(None)

    def get_message_correlation(self, correlation_id: str) -> Optional[AgentMessage]:
        return self._correlation_map.get(correlation_id)

    async def shutdown(self) -> Result[None, AgentError]:
        for agent in self._agents.values():
            result = await agent.shutdown()
            if isinstance(result, Failure):
                return result
        return Success(None)

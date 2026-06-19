from dataclasses import dataclass
from typing import Dict, Set

from .capabilities import CapabilityMap


@dataclass(frozen=True)
class SecurityEnforcer:
    capability_map: CapabilityMap

    def enforce(self, agent_name: str, tool_name: str) -> None:
        if agent_name not in self.capability_map:
            raise PermissionError(f"Agent '{agent_name}' is not permitted to use any tools")
        allowed_tools = self.capability_map[agent_name]
        if tool_name not in allowed_tools:
            raise PermissionError(f"Agent '{agent_name}' may not use tool '{tool_name}'")

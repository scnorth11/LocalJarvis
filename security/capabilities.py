from typing import Dict, Set

from config.schema import AppConfig

CapabilityMap = Dict[str, Set[str]]


def build_capability_map(app_config: AppConfig) -> CapabilityMap:
    return {agent: set(tools) for agent, tools in app_config.security.allowed_tools.items()}

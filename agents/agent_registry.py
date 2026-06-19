# Re-export the canonical Protocol-based registry from core.
# This module is kept for backward-compatible imports.
from core.registry import AgentContract, AgentRecord, AgentProxy, AgentRegistry

__all__ = ["AgentContract", "AgentRecord", "AgentProxy", "AgentRegistry"]

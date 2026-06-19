from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class AuditEntry:
    timestamp: str
    agent_name: str
    tool_name: str
    action: str
    allowed: bool
    detail: Optional[str] = None


@dataclass
class AuditLogger:
    entries: List[AuditEntry] = field(default_factory=list)

    def log(self, agent_name: str, tool_name: str, action: str, allowed: bool, detail: str | None = None) -> AuditEntry:
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            agent_name=agent_name,
            tool_name=tool_name,
            action=action,
            allowed=allowed,
            detail=detail,
        )
        self.entries.append(entry)
        return entry

    def get_entries(self) -> List[AuditEntry]:
        return list(self.entries)

    def clear(self) -> None:
        self.entries.clear()

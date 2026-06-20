from dataclasses import dataclass
from typing import Dict, List

from config.config import RestrictedConfig


@dataclass(frozen=True)
class RoutingResult:
    """Model selection decision returned by :func:`select_model`."""

    model: str
    tier: str
    confidence: float
    rationale: str


# Keywords that strongly suggest complex / generative tasks.
_COMPLEX_KEYWORDS = frozenset({
    "write", "create", "generate", "analyze", "analyse", "summarize", "summarise",
    "explain", "compare", "research", "code", "debug", "plan", "build", "design",
    "implement", "review", "critique", "translate", "refactor", "program", "develop",
    "draft", "compose", "architect", "optimize", "optimise", "script", "calculate",
})

# Sub-category signals for a richer rationale string.
_CODING_KEYWORDS = frozenset({
    "code", "debug", "implement", "refactor", "program", "function", "script",
    "develop", "build", "class", "method",
})
_RESEARCH_KEYWORDS = frozenset({
    "research", "analyze", "analyse", "compare", "review", "critique", "study",
    "summarize", "summarise",
})


def select_model(intent: str, config: RestrictedConfig) -> RoutingResult:
    """Select a model tier based on intent complexity.

    Uses keyword heuristics to score complexity and returns a
    :class:`RoutingResult` containing the resolved model name, chosen tier,
    confidence score, and a human-readable rationale string.
    """
    tiers: Dict[str, str] = config.models.tiers
    default_tier: str = config.models.default

    lower = intent.lower()
    hits: List[str] = [kw for kw in _COMPLEX_KEYWORDS if kw in lower]

    if hits:
        chosen_tier = "large"
        # Confidence increases with the number of matched keywords, capped at 0.95.
        confidence = min(0.60 + 0.05 * len(hits), 0.95)
        if any(kw in lower for kw in _CODING_KEYWORDS):
            rationale = f"coding task ({', '.join(hits[:3])})"
        elif any(kw in lower for kw in _RESEARCH_KEYWORDS):
            rationale = f"research/analysis task ({', '.join(hits[:3])})"
        else:
            rationale = f"complex generative task ({', '.join(hits[:3])})"
    else:
        chosen_tier = "small"
        confidence = 0.90
        rationale = "simple query — no complexity keywords found"

    model = tiers.get(chosen_tier) or tiers.get(default_tier) or next(iter(tiers.values()))
    return RoutingResult(model=model, tier=chosen_tier, confidence=confidence, rationale=rationale)

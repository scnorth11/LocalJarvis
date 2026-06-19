from typing import Dict

from config.config import RestrictedConfig

# Keywords that indicate a complex query requiring the large model tier.
_COMPLEX_KEYWORDS = frozenset({
    "write", "create", "generate", "analyze", "analyse", "summarize", "summarise",
    "explain", "compare", "research", "code", "debug", "plan", "build", "design",
    "implement", "review", "critique", "translate", "refactor",
})


def select_model(intent: str, config: RestrictedConfig) -> str:
    """Select a model name based on intent complexity.

    Uses a keyword heuristic: if the intent contains any complexity-indicating
    words, the large model tier is used; otherwise the small tier is used.

    Returns the actual model name (e.g. ``"phi-3-mini"`` or ``"llama-3.1-8b"``)
    resolved through ``config.models.tiers``.
    """
    tiers: Dict[str, str] = config.models.tiers
    default_tier: str = config.models.default

    lower = intent.lower()
    chosen_tier = "large" if any(kw in lower for kw in _COMPLEX_KEYWORDS) else "small"

    # Resolve tier name → model name; fall back to the configured default tier.
    return tiers.get(chosen_tier) or tiers.get(default_tier) or next(iter(tiers.values()))

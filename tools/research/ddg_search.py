"""ResearchTool — web search via DuckDuckGo with domain-quality annotations.

Uses the ``duckduckgo-search`` library (``pip install duckduckgo-search``).
Every result is annotated with a domain category so the downstream LLM can
weight sources appropriately.  A ``source_guidance`` string from config is
prepended to the output as a system hint.

Domain categories
-----------------
academic    — arXiv, PubMed, JSTOR, Springer, Nature, IEEE, ACM, Semantic Scholar, …
reference   — Wikipedia, Wiktionary, Encyclopedia Britannica, …
dev         — GitHub, Stack Overflow, MDN, PyPI, crates.io, …
social      — Reddit, Hacker News, Stack Exchange communities, …
gov         — .gov and .mil domains
news        — Reuters, AP, BBC, NPR, The Guardian, …
tabloid     — TMZ, Daily Mail, BuzzFeed, Breitbart, InfoWars, … (flagged as low-trust)
unknown     — anything not matched above
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List
from urllib.parse import urlparse

from tools.base import BaseTool

logger = logging.getLogger(__name__)

try:
    from duckduckgo_search import DDGS
except ImportError:  # optional dependency
    DDGS = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Domain → category mapping.
# Checked as: does the result URL's netloc END WITH the key?
# ---------------------------------------------------------------------------
_DOMAIN_MAP: Dict[str, str] = {
    # Academic / scholarly
    "arxiv.org": "academic",
    "pubmed.ncbi.nlm.nih.gov": "academic",
    "ncbi.nlm.nih.gov": "academic",
    "scholar.google.com": "academic",
    "semanticscholar.org": "academic",
    "jstor.org": "academic",
    "springer.com": "academic",
    "nature.com": "academic",
    "science.org": "academic",
    "sciencedirect.com": "academic",
    "ieee.org": "academic",
    "acm.org": "academic",
    "researchgate.net": "academic",
    "biorxiv.org": "academic",
    "medrxiv.org": "academic",
    "plos.org": "academic",
    "oup.com": "academic",
    "cambridge.org": "academic",
    "tandfonline.com": "academic",
    "wiley.com": "academic",
    # Reference
    "wikipedia.org": "reference",
    "wiktionary.org": "reference",
    "britannica.com": "reference",
    "merriam-webster.com": "reference",
    "wolframalpha.com": "reference",
    # Developer / technical
    "github.com": "dev",
    "stackoverflow.com": "dev",
    "stackexchange.com": "dev",
    "developer.mozilla.org": "dev",
    "mdn.io": "dev",
    "pypi.org": "dev",
    "npmjs.com": "dev",
    "crates.io": "dev",
    "docs.python.org": "dev",
    "readthedocs.io": "dev",
    "readthedocs.org": "dev",
    # Social / community
    "reddit.com": "social",
    "news.ycombinator.com": "social",
    "lobste.rs": "social",
    "quora.com": "social",
    # Reputable news
    "reuters.com": "news",
    "apnews.com": "news",
    "bbc.com": "news",
    "bbc.co.uk": "news",
    "npr.org": "news",
    "theguardian.com": "news",
    "nytimes.com": "news",
    "washingtonpost.com": "news",
    "economist.com": "news",
    "ft.com": "news",
    "wired.com": "news",
    "arstechnica.com": "news",
    "theatlantic.com": "news",
    "vox.com": "news",
    "propublica.org": "news",
    # Low-trust tabloid / gossip — flagged so LLM can deprioritise
    "tmz.com": "tabloid",
    "dailymail.co.uk": "tabloid",
    "nypost.com": "tabloid",
    "buzzfeed.com": "tabloid",
    "breitbart.com": "tabloid",
    "infowars.com": "tabloid",
    "thesun.co.uk": "tabloid",
    "mirror.co.uk": "tabloid",
    "nationalenquirer.com": "tabloid",
}

_DEFAULT_GUIDANCE = (
    "SOURCE QUALITY GUIDANCE: Prefer academic, reference, dev, and reputable news "
    "sources when available.  Social sources (Reddit, HN) can be useful for niche "
    "topics but should be corroborated.  Treat 'tabloid' sources with high scepticism "
    "and do not present them as authoritative."
)


def _categorise(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:  # noqa: BLE001
        return "unknown"
    # Check exact match first, then suffix match.
    if netloc in _DOMAIN_MAP:
        return _DOMAIN_MAP[netloc]
    for domain, category in _DOMAIN_MAP.items():
        if netloc.endswith("." + domain) or netloc == domain:
            return category
    # Government domains
    if netloc.endswith(".gov") or netloc.endswith(".mil"):
        return "gov"
    return "unknown"


class ResearchTool(BaseTool):
    """Search the web via DuckDuckGo and return quality-annotated results.

    Parameters (passed as ``**kwargs`` from the Executor):

    query : str
        Search query.
    max_results : int, optional
        Maximum number of results to return (default 10).
    search_type : str, optional
        ``"text"`` (default) or ``"news"``.
    """

    name = "research"
    description = "Search the web via DuckDuckGo with domain-quality annotations."

    def __init__(self, source_guidance: str, default_max: int) -> None:
        self._guidance = source_guidance
        self._default_max = default_max

    def run(
        self,
        *,
        query: str,
        max_results: int = 0,
        search_type: str = "text",
        **_: Any,
    ) -> str:
        if not query or not query.strip():
            return "[research] No query provided."

        limit = max_results if max_results > 0 else self._default_max

        if DDGS is None:
            return "[research] duckduckgo-search not installed. Run: pip install duckduckgo-search"

        try:
            with DDGS() as ddgs:
                if search_type == "news":
                    raw = list(ddgs.news(query, max_results=limit))
                else:
                    raw = list(ddgs.text(query, max_results=limit))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ResearchTool search failed for '%s': %s", query, exc)
            return f"[research] Search failed: {exc}"

        if not raw:
            return f"[research] No results found for: {query}"

        lines: List[str] = [f"[SOURCE GUIDANCE]\n{self._guidance}\n", f"Query: {query}\n"]
        for i, item in enumerate(raw, 1):
            url = item.get("href") or item.get("url") or ""
            title = item.get("title") or ""
            body = item.get("body") or item.get("excerpt") or ""
            category = _categorise(url)
            lines.append(
                f"{i}. [{category.upper()}] {title}\n"
                f"   URL: {url}\n"
                f"   {body[:300]}"
            )

        return "\n".join(lines)


# ------------------------------------------------------------------
# Factory consumed by ToolLoader
# ------------------------------------------------------------------

def create_tool(config: Any) -> ResearchTool:
    guidance = _DEFAULT_GUIDANCE
    default_max = 10
    try:
        cfg = config.tools.research
        if cfg.source_guidance:
            guidance = cfg.source_guidance
        if cfg.max_results > 0:
            default_max = cfg.max_results
    except AttributeError:
        pass
    return ResearchTool(guidance, default_max)

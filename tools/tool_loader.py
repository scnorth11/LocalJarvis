"""ToolLoader — discovers and instantiates all registered tools.

Each tool sub-package must expose a ``create_tool(config)`` factory at its
package level.  The loader calls every factory, builds a ``{name: run}`` dict,
and returns it.  This dict is assigned to ``executor.tools`` so that the
existing ``registry.py`` ``tool_invoker`` can dispatch calls transparently.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

# Ordered list of tool sub-module paths.  Add new tools here.
_TOOL_MODULES = [
    "tools.writing.file_writer",
    "tools.file_search.file_searcher",
    "tools.utility.math_tool",
    "tools.utility.file_ops",
    "tools.research.ddg_search",
    "tools.calendar.calendar_tool",
    "tools.spotify.spotify_tool",
]


class ToolLoader:
    """Instantiate all tools from *config* and return a callable-per-name dict."""

    def load(self, config: Any) -> Dict[str, Callable[..., str]]:
        """Return ``{tool_name: tool.run}`` for every successfully loaded tool.

        Tools that fail to initialise (missing optional dependency, bad config,
        etc.) are skipped with a warning so the rest of the system keeps running.
        """
        tools: Dict[str, Callable[..., str]] = {}
        for module_path in _TOOL_MODULES:
            try:
                mod = importlib.import_module(module_path)
                factory = getattr(mod, "create_tool", None)
                if factory is None:
                    logger.warning("ToolLoader: %s has no create_tool(), skipping", module_path)
                    continue
                tool = factory(config)
                tools[tool.name] = tool.run
                logger.debug("ToolLoader: registered tool %r from %s", tool.name, module_path)
            except ImportError as exc:
                logger.warning(
                    "ToolLoader: skipping %s — missing dependency: %s", module_path, exc
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ToolLoader: failed to load %s: %s", module_path, exc)
        return tools

"""Base classes shared by all LocalJarvis tools."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List


class BaseTool(ABC):
    """Abstract base for every Jarvis tool.

    Subclasses must set *name* and *description* as class attributes and
    implement :meth:`run`.  The registry wires ``tool_instance.run`` as the
    callable that the Executor invokes.
    """

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a plain-text result string."""

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"


class PathSanitizer:
    """Resolve and validate file-system paths against an allow-list of roots.

    Design goals
    ------------
    * Prevent path-traversal attacks (``../``, symlinks that escape the root).
    * Ensure the resolved path sits inside at least one allowed root.
    * Never rely on string prefix matching — always use ``Path.is_relative_to``
      after resolving the full real path.

    Usage
    -----
    ::

        san = PathSanitizer(["/home/user/data", "/tmp/jarvis"])
        safe = san.resolve("notes/todo.txt")   # returns Path
        san.resolve("../../etc/passwd")        # raises PermissionError

    """

    def __init__(self, allowed_roots: List[str]) -> None:
        if not allowed_roots:
            raise ValueError("PathSanitizer requires at least one allowed root")
        # Resolve each root once so comparisons are always canonical.
        self._roots: List[Path] = [Path(r).resolve() for r in allowed_roots]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, path: str) -> Path:
        """Return the canonicalised, validated :class:`Path` for *path*.

        Raises :class:`PermissionError` if the resolved path is outside every
        allowed root.  The target file need not exist; the check applies to
        the path itself.
        """
        candidate = self._canonicalise(path)
        if not self._is_allowed(candidate):
            raise PermissionError(
                f"Path '{path}' resolves to '{candidate}' which is outside "
                f"the allowed directories: {[str(r) for r in self._roots]}"
            )
        return candidate

    def is_allowed(self, path: str) -> bool:
        """Return ``True`` iff *path* resolves inside an allowed root."""
        try:
            self.resolve(path)
            return True
        except PermissionError:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _canonicalise(path: str) -> Path:
        """Expand user home and environment variables, then resolve symlinks.

        ``Path.resolve()`` requires the path to exist to follow symlinks;
        for non-existent paths we use ``os.path.abspath`` which still
        collapses ``..`` components safely.
        """
        expanded = os.path.expandvars(os.path.expanduser(path))
        p = Path(expanded)
        if p.exists():
            return p.resolve()
        # For non-existent paths: resolve the nearest existing ancestor and
        # re-attach the remaining suffix so ``..`` is still collapsed.
        return Path(os.path.realpath(os.path.abspath(expanded)))

    def _is_allowed(self, candidate: Path) -> bool:
        return any(
            self._is_relative_to(candidate, root) for root in self._roots
        )

    @staticmethod
    def _is_relative_to(child: Path, parent: Path) -> bool:
        """Portable ``Path.is_relative_to`` (added in Python 3.9)."""
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

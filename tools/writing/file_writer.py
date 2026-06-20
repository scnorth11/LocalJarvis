"""WritingTool — create, read, append, and overwrite text files.

All paths are sandboxed via :class:`tools.base.PathSanitizer`.  The allowed
roots are read from ``config.tools.file.allowed_paths``; the project
``data_dir`` is always implicitly included so Jarvis can write to its own
data folder without extra configuration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

from tools.base import BaseTool, PathSanitizer

logger = logging.getLogger(__name__)

_VALID_OPS = {"create", "append", "overwrite", "read"}


class WritingTool(BaseTool):
    """Read and write plain-text files within sandboxed directories.

    Parameters (passed as ``**kwargs`` from the Executor step):

    op : str
        One of ``create``, ``append``, ``overwrite``, ``read``.
    path : str
        Target file path (absolute or relative; always validated).
    content : str, optional
        Text to write (ignored for ``read``).
    encoding : str, optional
        File encoding, default ``utf-8``.
    """

    name = "writing"
    description = "Create, read, append, or overwrite text files in allowed directories."

    def __init__(self, sanitizer: PathSanitizer) -> None:
        self._san = sanitizer

    # ------------------------------------------------------------------
    # BaseTool implementation
    # ------------------------------------------------------------------

    def run(self, *, op: str, path: str, content: str = "", encoding: str = "utf-8") -> str:
        op = op.lower().strip()
        if op not in _VALID_OPS:
            return f"[writing] Unknown op '{op}'. Valid ops: {sorted(_VALID_OPS)}"

        try:
            safe_path = self._san.resolve(path)
        except PermissionError as exc:
            return f"[writing] Access denied: {exc}"

        try:
            if op == "read":
                return self._read(safe_path, encoding)
            if op == "create":
                return self._create(safe_path, content, encoding)
            if op == "append":
                return self._append(safe_path, content, encoding)
            if op == "overwrite":
                return self._overwrite(safe_path, content, encoding)
        except OSError as exc:
            logger.warning("WritingTool %s failed on %s: %s", op, safe_path, exc)
            return f"[writing] OS error during '{op}': {exc}"

        return "[writing] Unexpected state"  # unreachable

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read(path: Path, encoding: str) -> str:
        if not path.exists():
            return f"[writing] File not found: {path}"
        return path.read_text(encoding=encoding)

    @staticmethod
    def _create(path: Path, content: str, encoding: str) -> str:
        if path.exists():
            return f"[writing] File already exists: {path}. Use 'overwrite' to replace it."
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return f"[writing] Created: {path} ({len(content)} chars)"

    @staticmethod
    def _append(path: Path, content: str, encoding: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding=encoding) as fh:
            fh.write(content)
        return f"[writing] Appended {len(content)} chars to: {path}"

    @staticmethod
    def _overwrite(path: Path, content: str, encoding: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return f"[writing] Overwritten: {path} ({len(content)} chars)"


# ------------------------------------------------------------------
# Factory consumed by ToolLoader
# ------------------------------------------------------------------

def create_tool(config: Any) -> WritingTool:
    """Build a :class:`WritingTool` from *config*."""
    allowed: List[str] = []
    try:
        allowed = list(config.tools.file.allowed_paths)
    except AttributeError:
        pass
    # Always include the project data_dir as an allowed root.
    try:
        data_dir = str(config.paths.data_dir)
        if data_dir not in allowed:
            allowed.append(data_dir)
    except AttributeError:
        pass
    if not allowed:
        allowed = ["data"]
    return WritingTool(PathSanitizer(allowed))

"""FileSearchTool — find files and search file content within sandboxed directories.

All paths are validated by :class:`tools.base.PathSanitizer`.  The allowed
roots are taken from ``config.tools.file.allowed_paths`` plus the project
``data_dir``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

from tools.base import BaseTool, PathSanitizer

logger = logging.getLogger(__name__)

_VALID_OPS = {"search_name", "search_content", "read", "list_dir"}
_DEFAULT_MAX = 20


class FileSearchTool(BaseTool):
    """Search for files by name pattern or content substring, read files, list dirs.

    Parameters (all passed as ``**kwargs`` from the Executor):

    op : str
        One of ``search_name``, ``search_content``, ``read``, ``list_dir``.
    path : str, optional
        Base directory for search / target file for read (default: first allowed root).
    query : str, optional
        Text to search for (used by ``search_content``).
    pattern : str, optional
        Glob pattern for ``search_name`` and ``list_dir`` (default ``*``).
    max_results : int, optional
        Cap on returned results (default 20).
    """

    name = "file_search"
    description = "Find files by name or content, read files, and list directories."

    def __init__(self, sanitizer: PathSanitizer, default_root: Path) -> None:
        self._san = sanitizer
        self._default_root = default_root

    # ------------------------------------------------------------------
    # BaseTool implementation
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        op: str,
        path: str = "",
        query: str = "",
        pattern: str = "*",
        max_results: int = _DEFAULT_MAX,
    ) -> str:
        op = op.lower().strip()
        if op not in _VALID_OPS:
            return f"[file_search] Unknown op '{op}'. Valid ops: {sorted(_VALID_OPS)}"

        base_path = Path(path) if path else self._default_root
        try:
            safe_base = self._san.resolve(str(base_path))
        except PermissionError as exc:
            return f"[file_search] Access denied: {exc}"

        try:
            if op == "read":
                return self._read(safe_base)
            if op == "list_dir":
                return self._list_dir(safe_base, pattern, max_results)
            if op == "search_name":
                return self._search_name(safe_base, pattern, max_results)
            if op == "search_content":
                return self._search_content(safe_base, query, max_results)
        except OSError as exc:
            logger.warning("FileSearchTool %s failed on %s: %s", op, safe_base, exc)
            return f"[file_search] OS error during '{op}': {exc}"

        return "[file_search] Unexpected state"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> str:
        if not path.exists():
            return f"[file_search] Not found: {path}"
        if path.is_dir():
            return f"[file_search] '{path}' is a directory. Use op='list_dir'."
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"[file_search] Cannot read '{path}': {exc}"

    @staticmethod
    def _list_dir(path: Path, pattern: str, max_results: int) -> str:
        if not path.is_dir():
            return f"[file_search] Not a directory: {path}"
        entries = sorted(path.glob(pattern))[:max_results]
        if not entries:
            return f"[file_search] No entries matching '{pattern}' in {path}"
        lines = [f"{'[dir] ' if e.is_dir() else '      '}{e.name}" for e in entries]
        return "\n".join(lines)

    @staticmethod
    def _search_name(base: Path, pattern: str, max_results: int) -> str:
        if not base.is_dir():
            return f"[file_search] Not a directory: {base}"
        matches = sorted(base.rglob(pattern))[:max_results]
        if not matches:
            return f"[file_search] No files matching '{pattern}' under {base}"
        return "\n".join(str(m) for m in matches)

    @staticmethod
    def _search_content(base: Path, query: str, max_results: int) -> str:
        if not query:
            return "[file_search] 'query' is required for search_content"
        if not base.is_dir():
            # Treat base as a single file to search.
            files = [base] if base.is_file() else []
        else:
            files = [f for f in base.rglob("*") if f.is_file()]

        hits: List[str] = []
        query_lower = query.lower()
        for fp in files:
            if len(hits) >= max_results:
                break
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if query_lower in line.lower():
                    hits.append(f"{fp}:{lineno}: {line.rstrip()}")
                    if len(hits) >= max_results:
                        break

        if not hits:
            return f"[file_search] No matches for '{query}'"
        return "\n".join(hits)


# ------------------------------------------------------------------
# Factory consumed by ToolLoader
# ------------------------------------------------------------------

def create_tool(config: Any) -> FileSearchTool:
    allowed: List[str] = []
    try:
        allowed = list(config.tools.file.allowed_paths)
    except AttributeError:
        pass
    try:
        data_dir = str(config.paths.data_dir)
        if data_dir not in allowed:
            allowed.append(data_dir)
    except AttributeError:
        pass
    if not allowed:
        allowed = ["data"]
    default_root = Path(allowed[0])
    return FileSearchTool(PathSanitizer(allowed), default_root)

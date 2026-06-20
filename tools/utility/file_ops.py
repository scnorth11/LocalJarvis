"""FileOpsTool — basic filesystem operations within sandboxed directories.

Supports copy, move, mkdir, delete, and list.  All source and destination
paths are validated by :class:`tools.base.PathSanitizer`.  Delete is
additionally restricted to paths inside the project ``data_dir`` as an extra
guard against accidental data loss.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, List, Optional

from tools.base import BaseTool, PathSanitizer

logger = logging.getLogger(__name__)

_VALID_OPS = {"copy", "move", "mkdir", "delete", "list"}


class FileOpsTool(BaseTool):
    """Perform basic filesystem operations within allowed directories.

    Parameters (all passed as ``**kwargs`` from the Executor):

    op : str
        One of ``copy``, ``move``, ``mkdir``, ``delete``, ``list``.
    src : str
        Source path (required for all ops).
    dst : str, optional
        Destination path (required for ``copy`` and ``move``).
    recursive : bool, optional
        When ``True``, copy/delete operates recursively on directories.
        Default ``False``.
    """

    name = "file_ops"
    description = "Copy, move, mkdir, delete, or list files in allowed directories."

    def __init__(self, sanitizer: PathSanitizer, data_root: Path) -> None:
        self._san = sanitizer
        self._data_root = data_root

    # ------------------------------------------------------------------
    # BaseTool implementation
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        op: str,
        src: str,
        dst: str = "",
        recursive: bool = False,
        **_: Any,
    ) -> str:
        op = op.lower().strip()
        if op not in _VALID_OPS:
            return f"[file_ops] Unknown op '{op}'. Valid ops: {sorted(_VALID_OPS)}"

        try:
            safe_src = self._san.resolve(src)
        except PermissionError as exc:
            return f"[file_ops] Access denied (src): {exc}"

        safe_dst: Optional[Path] = None
        if dst:
            try:
                safe_dst = self._san.resolve(dst)
            except PermissionError as exc:
                return f"[file_ops] Access denied (dst): {exc}"

        try:
            if op == "list":
                return self._list(safe_src)
            if op == "mkdir":
                return self._mkdir(safe_src)
            if op == "delete":
                return self._delete(safe_src, recursive)
            if op in ("copy", "move"):
                if safe_dst is None:
                    return f"[file_ops] 'dst' is required for op='{op}'"
                if op == "copy":
                    return self._copy(safe_src, safe_dst, recursive)
                return self._move(safe_src, safe_dst)
        except OSError as exc:
            logger.warning("FileOpsTool %s failed: %s", op, exc)
            return f"[file_ops] OS error during '{op}': {exc}"

        return "[file_ops] Unexpected state"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list(path: Path) -> str:
        if not path.exists():
            return f"[file_ops] Path does not exist: {path}"
        if path.is_file():
            return f"[file_ops] {path} is a file ({path.stat().st_size} bytes)"
        entries = sorted(path.iterdir())
        if not entries:
            return f"[file_ops] Empty directory: {path}"
        lines = [f"{'[dir] ' if e.is_dir() else '      '}{e.name}" for e in entries]
        return "\n".join(lines)

    @staticmethod
    def _mkdir(path: Path) -> str:
        path.mkdir(parents=True, exist_ok=True)
        return f"[file_ops] Directory created: {path}"

    def _delete(self, path: Path, recursive: bool) -> str:
        # Extra guard: delete is only permitted inside data_root.
        try:
            path.relative_to(self._data_root)
        except ValueError:
            return (
                f"[file_ops] Delete is restricted to the data directory "
                f"({self._data_root}).  '{path}' is outside that scope."
            )
        if not path.exists():
            return f"[file_ops] Path does not exist: {path}"
        if path.is_dir():
            if not recursive:
                return (
                    f"[file_ops] '{path}' is a directory. Set recursive=true to delete it."
                )
            shutil.rmtree(path)
            return f"[file_ops] Directory deleted: {path}"
        path.unlink()
        return f"[file_ops] File deleted: {path}"

    @staticmethod
    def _copy(src: Path, dst: Path, recursive: bool) -> str:
        if not src.exists():
            return f"[file_ops] Source not found: {src}"
        if src.is_dir():
            if not recursive:
                return f"[file_ops] '{src}' is a directory. Set recursive=true to copy it."
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return f"[file_ops] Directory copied: {src} → {dst}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return f"[file_ops] File copied: {src} → {dst}"

    @staticmethod
    def _move(src: Path, dst: Path) -> str:
        if not src.exists():
            return f"[file_ops] Source not found: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"[file_ops] Moved: {src} → {dst}"


# ------------------------------------------------------------------
# Factory consumed by ToolLoader
# ------------------------------------------------------------------

def create_tool(config: Any) -> FileOpsTool:
    allowed: List[str] = []
    try:
        allowed = list(config.tools.file.allowed_paths)
    except AttributeError:
        pass
    data_dir = "data"
    try:
        data_dir = str(config.paths.data_dir)
    except AttributeError:
        pass
    if data_dir not in allowed:
        allowed.append(data_dir)
    return FileOpsTool(PathSanitizer(allowed), Path(data_dir).resolve())

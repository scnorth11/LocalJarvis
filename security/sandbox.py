import shlex
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

ALLOWED_SHELL_COMMANDS = frozenset({"echo", "date", "uname"})
ALLOWED_HTTP_DOMAINS = frozenset({"example.com", "api.example.com"})
DEFAULT_HTTP_TIMEOUT = 10


def safe_shell(command: str) -> str:
    if not command or not isinstance(command, str):
        raise ValueError("command must be a non-empty string")
    parts = shlex.split(command)
    if not parts:
        raise ValueError("command must contain executable text")
    if parts[0] not in ALLOWED_SHELL_COMMANDS:
        raise PermissionError(f"Shell command '{parts[0]}' is not allowed")
    result = subprocess.run(
        parts,
        shell=False,
        capture_output=True,
        text=True,
        check=True,
        timeout=DEFAULT_HTTP_TIMEOUT,
    )
    return result.stdout.strip()


def _is_allowed_domain(hostname: str, allowed: Iterable[str]) -> bool:
    return any(hostname == allowed_host or hostname.endswith(f".{allowed_host}") for allowed_host in allowed)


def safe_http(url: str) -> str:
    if not url or not isinstance(url, str):
        raise ValueError("url must be a non-empty string")
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise PermissionError("Only https URLs are allowed")
    if not parsed.hostname or not _is_allowed_domain(parsed.hostname, ALLOWED_HTTP_DOMAINS):
        raise PermissionError(f"HTTP host '{parsed.hostname}' is not allowed")
    with urllib.request.urlopen(url, timeout=DEFAULT_HTTP_TIMEOUT) as response:
        return response.read().decode("utf-8")

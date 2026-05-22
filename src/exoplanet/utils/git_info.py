"""Captura información del repo git para reproducibilidad de cada corrida."""

from __future__ import annotations

import subprocess


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_git_commit() -> str:
    return _run(["git", "rev-parse", "HEAD"]) or "unknown"


def get_git_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"


def is_dirty() -> bool:
    """True si hay cambios sin commitear (working tree o index)."""
    out = _run(["git", "status", "--porcelain"])
    return bool(out)


def git_summary() -> dict[str, str | bool]:
    return {
        "commit": get_git_commit(),
        "branch": get_git_branch(),
        "dirty": is_dirty(),
    }

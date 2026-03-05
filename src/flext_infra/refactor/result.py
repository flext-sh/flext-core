"""Result model for flext_infra refactor operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _empty_str_list() -> list[str]:
    return []


@dataclass
class FlextInfraRefactorResult:
    """Result of applying refactor rules to a single file."""

    file_path: Path
    success: bool
    modified: bool
    error: str | None = None
    changes: list[str] = field(default_factory=_empty_str_list)
    refactored_code: str | None = None


__all__ = ["FlextInfraRefactorResult"]

"""Base rule type for flext_infra.refactor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import libcst as cst


class FlextInfraRefactorRule:
    """Base class for flext_infra refactor rules."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize rule metadata from rule config."""
        self.config = config
        self.rule_id = config.get("id", "unknown")
        self.name = config.get("name", self.rule_id)
        self.description = config.get("description", "")
        self.enabled = config.get("enabled", True)
        self.severity = config.get("severity", "warning")

    def apply(
        self, tree: cst.Module, _file_path: Path | None = None
    ) -> tuple[cst.Module, list[str]]:
        """Apply the rule to a CST module and return transformed tree plus changes."""
        return tree, []

    def matches_filter(self, filter_pattern: str) -> bool:
        """Return whether the rule matches a case-insensitive filter string."""
        pattern_lower = filter_pattern.lower()
        return (
            pattern_lower in (self.rule_id or "").lower()
            or pattern_lower in (self.name or "").lower()
            or pattern_lower in (self.description or "").lower()
        )

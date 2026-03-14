"""Base rule class for flext_infra.refactor.

Extracted to break circular import chain:
rule.py -> class_nesting.py -> class_reconstructor.py -> rule.py

This module has no dependencies on any rules/ submodule.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import libcst as cst
from pydantic import JsonValue

from flext_infra import c


class FlextInfraRefactorRule:
    """Base class for flext_infra refactor rules."""

    def __init__(self, config: Mapping[str, JsonValue]) -> None:
        """Initialize rule metadata from rule config."""
        self.config = dict(config.items())
        rule_id = self.config.get(c.Infra.ReportKeys.ID, c.Infra.Defaults.UNKNOWN)
        self.rule_id = str(rule_id)
        name_raw = self.config.get(c.Infra.Toml.NAME, self.rule_id)
        self.name = str(name_raw)
        description_raw = self.config.get("description", "")
        self.description = description_raw if isinstance(description_raw, str) else ""
        enabled_raw = self.config.get(c.Infra.ReportKeys.ENABLED, True)
        self.enabled = bool(enabled_raw)
        severity_raw = self.config.get("severity", c.Infra.Severity.WARNING)
        self.severity = str(severity_raw)

    def apply(
        self,
        tree: cst.Module,
        _file_path: Path | None = None,
    ) -> tuple[cst.Module, list[str]]:
        """Apply the rule to a CST module and return transformed tree plus changes."""
        return (tree, [])

    def matches_filter(self, filter_pattern: str) -> bool:
        """Return whether the rule matches a case-insensitive filter string."""
        pattern_lower = filter_pattern.lower()
        return (
            pattern_lower in (self.rule_id or "").lower()
            or pattern_lower in (self.name or "").lower()
            or pattern_lower in (self.description or "").lower()
        )


__all__ = ["FlextInfraRefactorRule"]

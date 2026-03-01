"""Census service for namespace violation counting and reporting.

Read-only service that counts and classifies namespace violations
across all workspace projects using FlextInfraNamespaceValidator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final, override

from flext_core import FlextService, r

from flext_infra.core.namespace_validator import FlextInfraNamespaceValidator
from flext_infra.discovery import FlextInfraDiscoveryService
from flext_infra.models import FlextInfraModels

__all__ = ["FlextInfraCodegenCensus"]

# Regex to parse violation strings: [NS-00X-NNN] path:line — message
_VIOLATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\[(?P<rule>NS-\d{3})-\d{3}\]\s+(?P<module>[^:]+):(?P<line>\d+)\s+\u2014\s+(?P<message>.+)",
)

_EXCLUDED_PROJECTS: Final[frozenset[str]] = frozenset({"flexcore"})


class FlextInfraCodegenCensus(FlextService[list[FlextInfraModels.CensusReport]]):
    """Read-only census service for namespace violation counting."""

    def __init__(self, workspace_root: Path) -> None:  # noqa: D107
        super().__init__()
        self._workspace_root = workspace_root

    @override
    def execute(self) -> r[list[FlextInfraModels.CensusReport]]:
        """Execute census across all workspace projects."""
        return r[list[FlextInfraModels.CensusReport]].ok(self.run())

    def run(self) -> list[FlextInfraModels.CensusReport]:
        """Run census on all projects in workspace.

        Returns:
            List of CensusReport models, one per scanned project.

        """
        discovery = FlextInfraDiscoveryService()
        projects_result = discovery.discover_projects(self._workspace_root)
        if not projects_result.is_success:
            return []

        reports: list[FlextInfraModels.CensusReport] = []
        for project in projects_result.unwrap():
            if project.name in _EXCLUDED_PROJECTS:
                continue
            if project.stack.startswith("go"):
                continue
            report = self._census_project(project)
            reports.append(report)
        return reports

    def _census_project(
        self,
        project: FlextInfraModels.ProjectInfo,
    ) -> FlextInfraModels.CensusReport:
        """Run census on a single project."""
        validator = FlextInfraNamespaceValidator()
        result = validator.validate(project.path)

        violations: list[FlextInfraModels.CensusViolation] = []
        if result.is_success:
            report = result.unwrap()
            for violation_str in report.violations:
                violation = self._parse_violation(violation_str)
                if violation is not None:
                    violations.append(violation)

        return FlextInfraModels.CensusReport(
            project=project.name,
            violations=violations,
            total=len(violations),
            fixable=sum(1 for v in violations if v.fixable),
        )

    @staticmethod
    def _parse_violation(
        violation_str: str,
    ) -> FlextInfraModels.CensusViolation | None:
        """Parse a violation string into a CensusViolation model."""
        match = _VIOLATION_PATTERN.match(violation_str)
        if match is None:
            return None

        rule = match.group("rule")
        # NS-000 = Rule 0 (module structure) = not auto-fixable
        # NS-001 = Rule 1 (constants centralization) = fixable
        # NS-002 = Rule 2 (types centralization) = fixable
        fixable = rule != "NS-000"

        return FlextInfraModels.CensusViolation(
            module=match.group("module"),
            rule=rule,
            line=int(match.group("line")),
            message=match.group("message"),
            fixable=fixable,
        )

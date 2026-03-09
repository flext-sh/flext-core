"""Census service for namespace violation counting and reporting.

Read-only service that counts and classifies namespace violations
across all workspace projects using FlextInfraNamespaceValidator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import override

from flext_core import r, s
from flext_infra import FlextInfraDiscoveryService, c, m, p
from flext_infra.core.namespace_validator import FlextInfraNamespaceValidator

__all__ = ["FlextInfraCodegenCensus"]


class FlextInfraCodegenCensus(s[list[m.Infra.Codegen.CensusReport]]):
    """Read-only census service for namespace violation counting."""

    def __init__(self, workspace_root: Path) -> None:
        """Initialize census service with workspace root."""
        super().__init__()
        self._workspace_root: Path = workspace_root

    @staticmethod
    def _is_fixable(*, rule: str, module: str, message: str) -> bool:
        _ = message
        if rule == "NS-000":
            return False
        if rule == "NS-001":
            return True
        if rule == "NS-002":
            return not module.endswith("typings.py")
        return False

    @staticmethod
    def _parse_violation(violation_str: str) -> m.Infra.Codegen.CensusViolation | None:
        """Parse a violation string into a CensusViolation model."""
        match = c.Infra.Codegen.VIOLATION_PATTERN.match(violation_str)
        if match is None:
            return None
        rule = match.group("rule")
        fixable = FlextInfraCodegenCensus._is_fixable(
            rule=rule,
            module=match.group("module"),
            message=match.group("message"),
        )
        return m.Infra.Codegen.CensusViolation(
            module=match.group("module"),
            rule=rule,
            line=int(match.group("line")),
            message=match.group("message"),
            fixable=fixable,
        )

    @override
    def execute(self) -> r[list[m.Infra.Codegen.CensusReport]]:
        """Execute census across all workspace projects."""
        return r[list[m.Infra.Codegen.CensusReport]].ok(self.run())

    def run(
        self,
        workspace_root: Path | None = None,
        *,
        output_format: str = "json",
    ) -> list[m.Infra.Codegen.CensusReport]:
        """Run census on all projects in workspace.

        Returns:
            List of CensusReport models, one per scanned project.

        """
        _ = output_format
        workspace = (
            workspace_root if workspace_root is not None else self._workspace_root
        )
        discovery = FlextInfraDiscoveryService()
        projects_result = discovery.discover_projects(workspace)
        if not projects_result.is_success:
            return []
        reports: list[m.Infra.Codegen.CensusReport] = []
        discovered: Sequence[p.ProjectInfo] = projects_result.unwrap()
        for project in discovered:
            if project.name in c.Infra.Codegen.EXCLUDED_PROJECTS:
                continue
            report = self._census_project(project)
            reports.append(report)
        return reports

    def _census_project(
        self,
        project: p.ProjectInfo,
    ) -> m.Infra.Codegen.CensusReport:
        """Run census on a single project."""
        validator = FlextInfraNamespaceValidator()
        result = validator.validate(project.path, scan_tests=False)
        violations: list[m.Infra.Codegen.CensusViolation] = []
        if result.is_success:
            report: m.Infra.Core.ValidationReport = result.unwrap()
            for violation_str in report.violations:
                violation = self._parse_violation(violation_str)
                if violation is not None:
                    violations.append(violation)
        return m.Infra.Codegen.CensusReport(
            project=project.name,
            violations=violations,
            total=len(violations),
            fixable=sum(1 for v in violations if v.fixable),
        )

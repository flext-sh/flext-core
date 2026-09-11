"""Project-layout quality gate (flext-0wuz, epic flext-hzox).

Reports layout-SSOT violations per project. Severity is config-driven
(``codegen.yaml layout.severity``): ``warning`` reports without failing the
pipeline; ``error`` fails on actionable (move/archive/gitignore) findings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, override

from flext_infra import c, config, m
from flext_infra.codegen.layout import FlextInfraCodegenLayout
from flext_infra.gates.base_gate import FlextInfraGate


class FlextInfraLayoutGate(FlextInfraGate):
    """Layout-SSOT conformance gate backed by the layout engine check mode."""

    gate_id: ClassVar[str] = "layout"
    gate_name: ClassVar[str] = "Project Layout"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["layout"][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["layout"][1]

    @override
    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Report layout violations for ``project_dir`` from the layout SSOT."""
        started = time.monotonic()
        spec = config.Infra.codegen.layout
        engine = FlextInfraCodegenLayout(repository_root=ctx.repository_root)
        report = engine.check_project(project_dir)
        warning = spec.severity == "warning"
        report_findings: tuple[m.Infra.LayoutFinding, ...] = report.findings
        issues = tuple(
            m.Infra.Issue(
                file=str(project_dir / finding.path),
                line=1,
                column=1,
                code=f"{self.gate_id}-{finding.rule}",
                message=finding.message,
                severity="WARNING" if warning or finding.rule == "review" else "ERROR",
            )
            for finding in report_findings
        )
        actionable: tuple[m.Infra.LayoutFinding, ...] = report.actionable
        blocking = tuple(finding for finding in actionable if not warning)
        passed = warning or not blocking
        return self._build_check_gate_execution(
            project_dir,
            passed=passed,
            issues=issues,
            raw_output="\n".join(issue.formatted for issue in issues),
            started=started,
            ctx=ctx,
        )


__all__: list[str] = ["FlextInfraLayoutGate"]

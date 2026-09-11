"""Static namespace-rule quality gate (NS-000..003)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, override

from flext_infra import c, m
from flext_infra.gates.base_gate import FlextInfraGate
from flext_infra.validate.namespace_validator import FlextInfraNamespaceValidator


class FlextInfraNamespaceGate(FlextInfraGate):
    """Rope-backed namespace rule gate."""

    gate_id: ClassVar[str] = "namespace"
    gate_name: ClassVar[str] = "Namespace Rules"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["namespace"][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["namespace"][1]

    @override
    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Run NS-000..003 validation scoped to ``project_dir``."""
        started = time.monotonic()
        validator = FlextInfraNamespaceValidator()
        report_result = validator.validate_project(project_dir)
        passed = report_result.success and report_result.value.passed
        errors: list[str] = []
        if report_result.failure:
            errors.append(report_result.error or "namespace validation failed")
        elif not passed:
            errors.extend(report_result.value.violations)
        return self._build_project_error_gate_result(
            project_dir, passed=passed, errors=errors, started=started, ctx=ctx
        )


__all__: list[str] = ["FlextInfraNamespaceGate"]

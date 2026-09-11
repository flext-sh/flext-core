"""Tier-whitelist / abstraction-boundary quality gate.

Replaces the legacy ``ban-direct-*.yml`` ast-grep rules with the
OWNERS-driven ``FlextInfraValidateTierWhitelist`` rope detector.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar, override

from flext_infra import c, m
from flext_infra.gates.base_gate import FlextInfraGate
from flext_infra.validate.tier_whitelist import FlextInfraValidateTierWhitelist


class FlextInfraTierWhitelistGate(FlextInfraGate):
    """Enforce the tier-whitelist abstraction boundary per project."""

    gate_id: ClassVar[str] = "tier-whitelist"
    gate_name: ClassVar[str] = "Tier Whitelist"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["tier-whitelist"][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["tier-whitelist"][1]

    @override
    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Run the tier-whitelist scan scoped to ``project_dir``."""
        started = time.monotonic()
        validator = FlextInfraValidateTierWhitelist(repository_root=project_dir)
        result = validator.execute()
        passed = result.success and result.value is True
        errors: list[str] = []
        if result.failure:
            errors.append(result.error or "tier-whitelist validation failed")
        elif not passed:
            errors.append(result.error or "tier-whitelist violations found")
        return self._build_project_error_gate_result(
            project_dir, passed=passed, errors=errors, started=started, ctx=ctx
        )


__all__: list[str] = ["FlextInfraTierWhitelistGate"]

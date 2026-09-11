"""Auto-fix service for namespace violations.

Orchestrates NS rule fixes, FLEXT migration, refactor service passes,
namespace enforcement, and lazy init propagation for each project.

Rule implementations live in ``_utilities_codegen_fixer_rules``.
Pipeline passes live in ``_utilities_codegen_fixer_passes``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, m
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.codegen._fixer_workspace import FlextInfraCodegenFixerWorkspaceMixin

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraCodegenFixer(
    FlextInfraProjectSelectionServiceBase[str], FlextInfraCodegenFixerWorkspaceMixin
):
    """Rope-oriented auto-fixer for namespace violations (Rules 1-5)."""

    dry_run: Annotated[
        bool, m.Field(description="Preview changes without modifying files")
    ] = False
    rules_only: Annotated[
        bool, m.Field(description="Only apply rule-based fixes, skip heuristic ones")
    ] = False

    @override
    def execute(self) -> p.Result[str]:
        """Execute auto-fix directly from the validated CLI service model."""
        dry_run = self.dry_run or not self.apply_changes
        try:
            results = self.fix_workspace()
        except c.EXC_OS_RUNTIME_TYPE as exc:
            return r[str].fail_op("auto-fix", exc)
        total_fixed = sum(len(result.violations_fixed) for result in results)
        total_skipped = sum(len(result.violations_skipped) for result in results)
        lines: t.MutableSequenceOf[str] = []
        if dry_run:
            lines.append("Dry-run mode: no files will be modified")
        lines.extend(
            (f"  {result.project}: fixed {len(result.violations_fixed)} violations")
            for result in results
            if result.violations_fixed
        )
        lines.append(
            f"Auto-fix: {total_fixed} fixed, {total_skipped} skipped"
            f" across {len(results)} projects"
        )
        return r[str].ok("\n".join(lines))


__all__: list[str] = ["FlextInfraCodegenFixer"]

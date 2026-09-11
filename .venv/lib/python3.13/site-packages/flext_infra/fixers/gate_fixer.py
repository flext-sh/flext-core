"""Gate-based fix adapter for enforcement rules backed by quality gates.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, u
from flext_infra.check.workspace_check_gates import FlextInfraGateRegistry
from flext_infra.fixers.base import FlextInfraFixerAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import p, t


class FlextInfraGateFixerAdapter(FlextInfraFixerAdapter):
    """Apply fixes by delegating to a registered ``FlextInfraGate``.

    This adapter handles ``fix_action.kind == "gate"`` by routing the rule
    to the gate registered under ``fix_action.target``. The gate must have
    ``can_fix=True`` and implement ``_build_fix_command``.
    """

    kind: ClassVar[str] = "gate"

    def __init__(self, repository_root: Path) -> None:
        """Bind the repository root used to instantiate gates."""
        super().__init__(repository_root)

    def _registry(self) -> FlextInfraGateRegistry:
        """Lazy import of the gate registry to avoid circular imports."""
        return FlextInfraGateRegistry.default()

    @override
    def can_fix(self, fix_action: me.EnforcementFixAction) -> bool:
        """Return whether this adapter handles ``fix_action``."""
        if fix_action.kind != self.kind:
            return False
        gate_cls = self._registry().get(fix_action.target)
        return gate_cls is not None and gate_cls.can_fix

    @override
    def fix_project(
        self,
        project_dir: Path,
        violations: t.SequenceOf[tuple[me.EnforcementRuleSpec, p.AttributeProbe]],
        ctx: m.Infra.FixEnforcementCommand,
    ) -> m.Infra.ProjectFixResult:
        """Apply gate fixes for the first violation group (all share target)."""
        if not violations:
            return m.Infra.ProjectFixResult(project=project_dir.name)
        rule, _probe = violations[0]
        fix_action = rule.fix_action
        if fix_action is None:
            return m.Infra.ProjectFixResult(project=project_dir.name)
        gate_cls = self._registry().get(fix_action.target)
        if gate_cls is None:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                failed=(
                    m.Infra.FailedFix(
                        rule_id=rule.id,
                        file_path=str(project_dir),
                        error=f"gate {fix_action.target} not registered",
                    ),
                ),
            )
        gate = gate_cls(self._repository_root)
        if not gate.can_fix:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                skipped=(
                    m.Infra.SkippedViolation(
                        rule_id=rule.id,
                        file_path=str(project_dir),
                        reason=f"gate {fix_action.target} does not support fix",
                    ),
                ),
            )
        reports_dir = project_dir / c.Infra.REPORTS_DIR_NAME / "fix-enforcement"
        if ctx.apply:
            reports_dir_result = u.Cli.ensure_dir(reports_dir)
            if reports_dir_result.failure:
                return m.Infra.ProjectFixResult(
                    project=project_dir.name,
                    failed=(
                        m.Infra.FailedFix(
                            rule_id=rule.id,
                            file_path=str(project_dir),
                            error=(
                                reports_dir_result.error
                                or "unable to create report dir"
                            ),
                        ),
                    ),
                )
            reports_dir = reports_dir_result.value
        gate_ctx = m.Infra.GateContext(
            workspace=self._repository_root,
            reports_dir=reports_dir,
            apply_fixes=ctx.apply,
            check_only=not ctx.apply,
            fail_fast=ctx.fail_fast,
        )
        try:
            execution = (
                gate.fix(project_dir, gate_ctx)
                if ctx.apply
                else gate.check(project_dir, gate_ctx)
            )
        except c.EXC_BROAD_RUNTIME as exc:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                failed=(
                    m.Infra.FailedFix(
                        rule_id=rule.id,
                        file_path=str(project_dir),
                        error=(
                            f"gate {fix_action.target} execution failed: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    ),
                ),
            )
        if not ctx.apply:
            return self._preview_from_check(
                project_dir=project_dir,
                rule=rule,
                target=fix_action.target,
                execution=execution,
            )
        fixed: list[m.Infra.FixedViolation] = []
        previewed: list[m.Infra.PreviewedViolation] = []
        skipped: list[m.Infra.SkippedViolation] = []
        failed: list[m.Infra.FailedFix] = []
        if execution.result.passed:
            message = f"gate {fix_action.target} fix applied"
            fixed_violation: m.Infra.FixedViolation = m.Infra.FixedViolation(
                rule_id=rule.id, file_path=str(project_dir), message=message
            )
            fixed = [fixed_violation]
        else:
            failed = [
                m.Infra.FailedFix(
                    rule_id=rule.id,
                    file_path=str(project_dir),
                    error=execution.raw_output or "gate fix failed",
                )
            ]
        return m.Infra.ProjectFixResult(
            project=project_dir.name,
            fixed=tuple(fixed),
            previewed=tuple(previewed),
            skipped=tuple(skipped),
            failed=tuple(failed),
        )

    def _preview_from_check(
        self,
        *,
        project_dir: Path,
        rule: me.EnforcementRuleSpec,
        target: str,
        execution: m.Infra.GateExecution,
    ) -> m.Infra.ProjectFixResult:
        """Build a dry-run result from the non-mutating gate check output."""
        matching = self._matching_issues(rule, execution.issues)
        if matching:
            return m.Infra.ProjectFixResult(
                project=project_dir.name,
                previewed=(
                    m.Infra.PreviewedViolation(
                        rule_id=rule.id,
                        file_path=str(project_dir),
                        message=(
                            f"would run gate {target} fix for "
                            f"{len(matching)} matching issue(s)"
                        ),
                    ),
                ),
            )
        if execution.result.passed:
            return m.Infra.ProjectFixResult(project=project_dir.name)
        details = execution.raw_output or "; ".join(
            issue.formatted for issue in execution.issues
        )
        return m.Infra.ProjectFixResult(
            project=project_dir.name,
            failed=(
                m.Infra.FailedFix(
                    rule_id=rule.id,
                    file_path=str(project_dir),
                    error=details or f"gate {target} check failed",
                ),
            ),
        )

    @staticmethod
    def _matching_issues(
        rule: me.EnforcementRuleSpec, issues: t.SequenceOf[m.Infra.Issue]
    ) -> tuple[m.Infra.Issue, ...]:
        """Return gate issues that correspond to the selected rule fix action."""
        fix_action = rule.fix_action
        if fix_action is None:
            return ()
        smell_tag = str(fix_action.params.get("smell_tag", ""))
        if not smell_tag:
            return tuple(issues)
        return tuple(
            issue
            for issue in issues
            if c.Infra.SMELLS_RULE_TAGS.get(issue.code, "") == smell_tag
        )


__all__: list[str] = ["FlextInfraGateFixerAdapter"]

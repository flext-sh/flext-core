"""Gate execution methods for workspace checker."""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import ClassVar

from flext_cli import cli
from flext_infra import c, m, p, r, t, u
from flext_infra.gates.abstraction_boundary import FlextInfraAbstractionBoundaryGate
from flext_infra.gates.bandit import FlextInfraBanditGate
from flext_infra.gates.base_gate import FlextInfraGate
from flext_infra.gates.canonical_alias import FlextInfraCanonicalAliasGate
from flext_infra.gates.codemod import FlextInfraCodemodGate
from flext_infra.gates.deferred_self_reference import (
    FlextInfraDeferredSelfReferenceGate,
)
from flext_infra.gates.direnv import FlextInfraDirenvGate
from flext_infra.gates.duplication import FlextInfraDuplicationGate
from flext_infra.gates.layout import FlextInfraLayoutGate
from flext_infra.gates.loc_cap import FlextInfraLocCapGate
from flext_infra.gates.markdown import FlextInfraMarkdownGate
from flext_infra.gates.mypy import FlextInfraMypyGate
from flext_infra.gates.namespace import FlextInfraNamespaceGate
from flext_infra.gates.pyrefly import FlextInfraPyreflyGate
from flext_infra.gates.pyright import FlextInfraPyrightGate
from flext_infra.gates.ruff_format import FlextInfraRuffFormatGate
from flext_infra.gates.ruff_lint import FlextInfraRuffLintGate
from flext_infra.gates.runtime_census import FlextInfraRuntimeCensusGate
from flext_infra.gates.silent_failure import FlextInfraSilentFailureGate
from flext_infra.gates.smells import FlextInfraSmellsGate
from flext_infra.gates.tier_whitelist import FlextInfraTierWhitelistGate


class FlextInfraGateRegistry:
    """Explicit gate registry mapping gate IDs to gate classes."""

    def __init__(self) -> None:
        """Build the gate-id to gate-class mapping used by check execution."""
        self._gates: dict[str, type[FlextInfraGate]] = {
            gate_cls.gate_id: gate_cls for gate_cls in self._gate_classes()
        }

    @staticmethod
    def _gate_classes() -> t.VariadicTuple[type[FlextInfraGate]]:
        """Return the runtime gate classes registered for workspace checks."""
        return (
            FlextInfraRuffLintGate,
            FlextInfraRuffFormatGate,
            FlextInfraPyreflyGate,
            FlextInfraMypyGate,
            FlextInfraPyrightGate,
            FlextInfraSilentFailureGate,
            FlextInfraDeferredSelfReferenceGate,
            FlextInfraBanditGate,
            FlextInfraMarkdownGate,
            FlextInfraLocCapGate,
            FlextInfraAbstractionBoundaryGate,
            FlextInfraCanonicalAliasGate,
            FlextInfraRuntimeCensusGate,
            FlextInfraNamespaceGate,
            FlextInfraLayoutGate,
            FlextInfraTierWhitelistGate,
            FlextInfraSmellsGate,
            FlextInfraCodemodGate,
            FlextInfraDirenvGate,
            FlextInfraDuplicationGate,
        )

    def get(self, gate_id: str) -> type[FlextInfraGate] | None:
        """Return the registered gate class for one gate id, when present."""
        return self._gates.get(gate_id)

    def create(self, gate_id: str, repository_root: Path) -> FlextInfraGate | None:
        """Instantiate one registered gate for ``repository_root`` when available."""
        gate_cls = self._gates.get(gate_id)
        return gate_cls(repository_root) if gate_cls else None

    @classmethod
    def default(cls) -> FlextInfraGateRegistry:
        """Return the default registry instance for workspace checks."""
        return cls()


class FlextInfraWorkspaceCheckGatesMixin:
    """Gate execution, project loop, and individual gate runner methods."""

    _repository_root: Path
    _registry: FlextInfraGateRegistry
    _default_reports_dir: Path
    _gate_logger: ClassVar[p.Logger] = u.fetch_logger(__name__)

    def _isolate_context(
        self, ctx: m.Infra.GateContext, target: m.Infra.CheckProjectTarget
    ) -> m.Infra.GateContext:
        """Create a fresh GateContext scoped to a single project."""
        return m.Infra.GateContext(
            workspace=ctx.repository_root,
            reports_dir=ctx.reports_dir / target.name,
            apply_fixes=ctx.apply_fixes,
            check_only=ctx.check_only,
            fail_fast=ctx.fail_fast,
            ruff_args=ctx.ruff_args,
            pyright_args=ctx.pyright_args,
        )

    def _run_single_project(
        self,
        target: m.Infra.CheckProjectTarget,
        index: int,
        total: int,
        resolved_gates: t.StrSequence,
        ctx: m.Infra.GateContext,
    ) -> m.Infra.ProjectResult | None:
        """Check one project, returning None when the project should be skipped."""
        project_dir = target.path
        pyproject_path = project_dir / c.Infra.PYPROJECT_FILENAME
        if not project_dir.is_dir() or not pyproject_path.exists():
            u.Cli.progress(index, total, target.name, c.Infra.SeverityLevel.SKIP)
            return None
        u.Cli.progress(index, total, target.name, c.Infra.VERB_CHECK)
        project_ctx = self._isolate_context(ctx, target)
        _ = u.Cli.ensure_dir(project_ctx.reports_dir)
        start = time.monotonic()
        project_result = self._check_project_with_ctx(
            project_dir, resolved_gates, project_ctx
        )
        elapsed = time.monotonic() - start
        u.Cli.status(
            c.Infra.VERB_CHECK,
            target.name,
            result=project_result.passed,
            elapsed=elapsed,
        )
        return project_result

    def _run_project_loop(
        self,
        projects: t.SequenceOf[m.Infra.CheckProjectTarget],
        resolved_gates: t.StrSequence,
        ctx: m.Infra.GateContext,
        *,
        fail_fast: bool,
    ) -> m.Infra.LoopOutcome:
        """Execute gate checks across projects, collecting results and timing."""
        results: t.MutableSequenceOf[m.Infra.ProjectResult] = []
        total = len(projects)
        failed = 0
        skipped = 0
        loop_start = time.monotonic()
        for index, target in enumerate(projects, 1):
            project_result = self._run_single_project(
                target, index, total, resolved_gates, ctx
            )
            if project_result is None:
                skipped += 1
                continue
            results.append(project_result)
            project_passed: bool = project_result.passed
            if not project_passed:
                failed += 1
                if fail_fast:
                    break
        return m.Infra.LoopOutcome(
            results=tuple(results),
            failed=failed,
            skipped=skipped,
            total_elapsed=time.monotonic() - loop_start,
        )

    def _gate_ctx(self, reports_dir: Path | None = None) -> m.Infra.GateContext:
        """Gate ctx."""
        return m.Infra.GateContext(
            workspace=self._repository_root,
            reports_dir=reports_dir or self._default_reports_dir,
        )

    def _run_gate(
        self,
        gate_id: str,
        project_dir: Path,
        reports_dir: Path | None = None,
        *,
        ctx: m.Infra.GateContext | None = None,
    ) -> m.Infra.GateExecution:
        """Run gate."""
        gate = self._registry.create(gate_id, self._repository_root)
        if gate is None:
            return m.Infra.GateExecution(
                result=m.Infra.GateResult(
                    gate=gate_id,
                    project=project_dir.name,
                    passed=False,
                    errors=[f"{gate_id} gate not registered"],
                    duration=0.0,
                ),
                issues=(),
                raw_output=f"{gate_id} gate not registered",
            )
        return gate.check(project_dir, ctx or self._gate_ctx(reports_dir))

    def _check_project_with_ctx(
        self, project_dir: Path, gates: t.StrSequence, ctx: m.Infra.GateContext
    ) -> m.Infra.ProjectResult:
        """Run gates for one project as independent DAG stages."""
        project_name = project_dir.name
        result = m.Infra.ProjectResult(project=project_name)

        stages: t.MutableSequenceOf[m.Cli.PipelineStageSpec] = []
        for gate_id in gates:
            gate_instance = self._registry.create(gate_id, self._repository_root)
            if gate_instance is None:
                continue
            stages.append(
                cli.stage(
                    gate_id,
                    handler=self._make_gate_handler(
                        gate_instance, project_dir, ctx, result.gates
                    ),
                )
            )

        if not stages:
            return result

        cli.pipeline(
            stages, context=cli.stage_context(project_dir), logger=self._gate_logger
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline stage helpers
    # ------------------------------------------------------------------

    def _make_gate_handler(
        self,
        gate_instance: FlextInfraGate,
        project_dir: Path,
        ctx: m.Infra.GateContext,
        gates_sink: MutableMapping[str, m.Infra.GateExecution],
    ) -> t.Cli.PipelineHandler:
        """Build a pipeline stage handler that executes a single gate.

        The handler writes GateExecution into *gates_sink* as a side-effect
        (same pattern as _CodegenPipelineState in the codegen pipeline).
        """
        gate_id = gate_instance.gate_id
        project_name = project_dir.name

        def _handler(
            _pipeline_ctx: m.Cli.PipelineStageContext,
        ) -> p.Result[m.Cli.PipelineStageResult]:
            """Run the gate and record its execution in the sink."""
            gate_ctx = m.Infra.GateContext(
                workspace=ctx.repository_root,
                reports_dir=ctx.reports_dir,
                apply_fixes=ctx.apply_fixes,
                check_only=ctx.check_only,
                fail_fast=ctx.fail_fast,
                ruff_args=ctx.ruff_args,
                pyright_args=ctx.pyright_args,
                gate_mode="warn"
                if gate_id in c.Infra.ENFORCEMENT_ADVISORY_GATES
                else "error",
            )
            execution = self._execute_gate(gate_instance, project_dir, gate_ctx)
            gates_sink[gate_id] = execution
            self._gate_logger.debug(
                "gate_executed",
                project=project_name,
                gate=gate_id,
                passed=execution.result.passed,
            )
            u.Cli.gate_result(
                gate_id,
                execution.error_count,
                passed=execution.result.passed,
                elapsed=execution.result.duration,
            )
            if not execution.result.passed:
                inline_errors = execution.result.errors[
                    : c.Infra.GATE_ERROR_OUTPUT_LIMIT
                ]
                for error in inline_errors:
                    u.Cli.error(error)
                remaining = len(execution.result.errors) - len(inline_errors)
                if remaining > 0:
                    u.Cli.error(
                        f"... {remaining} additional diagnostics in the check report"
                    )
                if not inline_errors and execution.raw_output.strip():
                    u.Cli.error(execution.raw_output.strip())
            status: t.Cli.PipelineStageStatus = (
                c.Cli.PipelineStageStatus.OK
                if execution.result.passed
                else c.Cli.PipelineStageStatus.FAILED
            )
            return r[m.Cli.PipelineStageResult].ok(
                cli.stage_result(
                    gate_id, status=status, output={"errors": execution.error_count}
                )
            )

        return _handler

    @staticmethod
    def _execute_gate(
        gate_instance: FlextInfraGate, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Run fix-then-check or check-only for a single gate instance."""
        if ctx.apply_fixes and (not ctx.check_only) and gate_instance.can_fix:
            fix_execution = gate_instance.fix(project_dir, ctx)
            if not fix_execution.result.passed:
                return fix_execution
        return gate_instance.check(project_dir, ctx)


__all__: list[str] = ["FlextInfraGateRegistry", "FlextInfraWorkspaceCheckGatesMixin"]

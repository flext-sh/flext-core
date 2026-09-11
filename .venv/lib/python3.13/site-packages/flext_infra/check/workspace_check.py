"""FLEXT infrastructure workspace checker."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import override

from flext_infra import c, config, m, p, r, s, t, u
from flext_infra.check._workspace_check_reports import (
    FlextInfraWorkspaceCheckReportsMixin,
)
from flext_infra.check.workspace_check_gates import (
    FlextInfraGateRegistry,
    FlextInfraWorkspaceCheckGatesMixin,
)


class FlextInfraWorkspaceChecker(
    s[bool], FlextInfraWorkspaceCheckGatesMixin, FlextInfraWorkspaceCheckReportsMixin
):
    """Run workspace quality gates and generate reports."""

    _repository_root: Path
    _registry: FlextInfraGateRegistry
    _default_reports_dir: Path

    def __init__(
        self, repository_root: Path | None = None, *, workspace: Path | None = None
    ) -> None:
        """Initialize workspace checker services and paths."""
        resolved_workspace = u.Infra.resolve_repository_root_or_cwd(
            repository_root or workspace
        )
        super().__init__(repository_root=resolved_workspace)
        self._repository_root = self.repository_root
        self._registry = FlextInfraGateRegistry.default()
        report_dir = u.Cli.resolve_report_dir(
            self._repository_root, c.Infra.PROJECT, c.Infra.VERB_CHECK
        )
        dir_result = u.Cli.ensure_dir(report_dir)
        if dir_result.failure:
            self._default_reports_dir = (
                self._repository_root / c.Infra.REPORTS_DIR_NAME / c.Infra.VERB_CHECK
            )
        else:
            self._default_reports_dir = report_dir

    @staticmethod
    def parse_tool_args(raw: str | None) -> t.StrSequence:
        """Parse extra gate arguments passed as a shell-style string."""
        if raw is None:
            return list[str]()
        return [item for item in shlex.split(raw) if item]

    @staticmethod
    def resolve_gates(gates: t.StrSequence) -> p.Result[list[str]]:
        """Resolve, validate and deduplicate requested gate names."""
        resolved: list[str] = []
        for gate in gates:
            name = gate.strip()
            if not name:
                continue
            if name not in c.Infra.ALLOWED_GATES:
                return r[list[str]].fail(f"ERROR: unknown gate '{gate}'")
            if name not in resolved:
                resolved.append(name)
        return r[list[str]].ok(list(resolved))

    @staticmethod
    def apply_ci_gate_rules(gates: t.StrSequence) -> list[str]:
        """Scope *gates* to the CI ternary owner set (RULING 2)."""
        ci = config.Infra.codegen.make.ci
        raw = u.Cli.env_read(ci.variable).unwrap().strip()
        owned: frozenset[str]
        if raw == ci.value:
            owned = frozenset(ci.check_gates)
        elif raw == ci.local_value:
            owned = frozenset(ci.local_check_gates)
        else:
            return [gate for gate in gates if gate]
        scoped = [gate for gate in gates if gate and gate in owned]
        FlextInfraWorkspaceChecker._gate_logger.info(
            "ci_run_check_gates",
            gates=scoped,
            reason=f"{ci.variable}={raw} scopes check gates to its owner set",
        )
        return scoped

    @override
    def execute(self) -> p.Result[bool]:
        """Execute."""
        return r[bool].fail("Use execute_command() directly")

    @classmethod
    def execute_payload(cls, params: m.Infra.RunCommand) -> p.Result[bool]:
        """Execute quality gates from the canonical check command payload."""
        checker = cls(repository_root=params.workspace_path)
        project_targets_result = cls._resolve_project_targets(params)
        if project_targets_result.failure:
            return r[bool].fail(
                project_targets_result.error or "project resolution failed"
            )
        project_targets = project_targets_result.value
        requested_gates = [gate for gate in params.gates if gate]
        gates = cls.apply_ci_gate_rules(params.gates)
        if not gates:
            if requested_gates:
                # A caller that named its gates (``make fix APPLY=Y`` asks for
                # the fixable set) and whose selection the CI token does not
                # own ran them in the token's complementary stage instead:
                # pre-commit (CI=Y) owns markdown/smells fixing, pre-push
                # (CI=N) owns the whole-program type checkers. The verb is a
                # documented no-op here, never a failure.
                FlextInfraWorkspaceChecker._gate_logger.info(
                    "ci_gate_noop",
                    gates=requested_gates,
                    reason=(
                        "requested gates are owned by the complementary CI "
                        "stage; nothing to run under this token"
                    ),
                )
                return r[bool].ok(True)
            return r[bool].fail(
                "no check gates remain after CI token filtering "
                f"({config.Infra.codegen.make.ci.variable}="
                f"{config.Infra.codegen.make.ci.value})"
            )
        gate_ctx = m.Infra.GateContext(
            workspace=params.workspace_path,
            reports_dir=params.reports_dir_path,
            apply_fixes=params.fix,
            check_only=params.check_only,
            ruff_args=tuple(cls.parse_tool_args(params.ruff_args)),
            pyright_args=tuple(cls.parse_tool_args(params.pyright_args)),
        )
        run_result = checker.run_projects(
            projects=project_targets,
            gates=gates,
            reports_dir=params.reports_dir_path,
            fail_fast=params.fail_fast,
            ctx=gate_ctx,
        )
        if run_result.failure:
            return r[bool].fail(run_result.error or "check failed")
        failed_projects = [
            project for project in run_result.value if not project.passed
        ]
        if failed_projects:
            failed_names = ", ".join(project.project for project in failed_projects)
            return r[bool].fail(f"quality gates failed for: {failed_names}")
        return r[bool].ok(True)

    @staticmethod
    def _resolve_project_targets(
        params: m.Infra.RunCommand,
    ) -> p.Result[t.SequenceOf[m.Infra.CheckProjectTarget]]:
        """Resolve explicit projects or discover the workspace project set."""
        requested = params.project_names
        if requested:
            return r[t.SequenceOf[m.Infra.CheckProjectTarget]].ok(
                tuple(
                    m.Infra.CheckProjectTarget.from_workspace_name(
                        params.workspace_path, project_name
                    )
                    for project_name in requested
                )
            )
        discovered = u.Infra.resolve_projects(params.workspace_path, ())
        if discovered.failure:
            return r[t.SequenceOf[m.Infra.CheckProjectTarget]].fail(
                discovered.error or "project discovery failed"
            )
        project_targets = tuple(
            m.Infra.CheckProjectTarget(name=project.name, path=project.path)
            for project in discovered.value
        )
        if not project_targets:
            return r[t.SequenceOf[m.Infra.CheckProjectTarget]].fail(
                "no projects discovered"
            )
        return r[t.SequenceOf[m.Infra.CheckProjectTarget]].ok(project_targets)

    def format(self, project_dir: Path) -> p.Result[m.Infra.GateResult]:
        """Run format checks for one project."""
        return r[m.Infra.GateResult].ok(
            self._run_gate(c.Infra.FORMAT, project_dir).result
        )

    def lint(self, project_dir: Path) -> p.Result[m.Infra.GateResult]:
        """Run lint checks for one project."""
        return r[m.Infra.GateResult].ok(
            self._run_gate(c.Infra.LINT, project_dir).result
        )

    def run_project(
        self, project: str, gates: t.StrSequence
    ) -> p.Result[t.SequenceOf[m.Infra.ProjectResult]]:
        """Run selected gates for one project."""
        return self.run_projects([project], list(gates))

    def run_projects(
        self,
        projects: t.StrSequence | t.SequenceOf[m.Infra.CheckProjectTarget],
        gates: t.StrSequence,
        *,
        reports_dir: Path | None = None,
        fail_fast: bool = False,
        ctx: m.Infra.GateContext | None = None,
    ) -> p.Result[t.SequenceOf[m.Infra.ProjectResult]]:
        """Run selected gates for multiple projects."""
        resolved_gates_result = self.resolve_gates(gates)
        if resolved_gates_result.failure:
            return r[t.SequenceOf[m.Infra.ProjectResult]].fail(
                resolved_gates_result.error or "invalid gates"
            )
        resolved_gates = resolved_gates_result.value
        report_base = reports_dir or self._default_reports_dir
        dir_ensure = u.Cli.ensure_dir(report_base)
        if dir_ensure.failure:
            return r[t.SequenceOf[m.Infra.ProjectResult]].fail(
                dir_ensure.error or "failed to create report directory"
            )
        effective_ctx = ctx or m.Infra.GateContext(
            workspace=self._repository_root, reports_dir=report_base
        )
        outcome = self._run_project_loop(
            self._project_targets(projects),
            resolved_gates,
            effective_ctx,
            fail_fast=fail_fast,
        )
        return self._write_reports_and_summary(resolved_gates, report_base, outcome)

    def _project_targets(
        self, projects: t.StrSequence | t.SequenceOf[m.Infra.CheckProjectTarget]
    ) -> t.SequenceOf[m.Infra.CheckProjectTarget]:
        """Return typed project targets from public names or internal selections."""
        targets: list[m.Infra.CheckProjectTarget] = []
        for project in projects:
            if isinstance(project, m.Infra.CheckProjectTarget):
                targets.append(project)
                continue
            targets.append(
                m.Infra.CheckProjectTarget.from_workspace_name(
                    self._repository_root, project
                )
            )
        return tuple(targets)


__all__: list[str] = ["FlextInfraWorkspaceChecker"]

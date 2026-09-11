"""Automated namespace enforcement orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_cli import cli
from flext_core import r
from flext_infra import m, u
from flext_infra.refactor._namespace_enforcer_project import (
    FlextInfraNamespaceEnforcerProjectMixin,
)
from flext_infra.refactor.namespace_enforcer_phases import (
    FlextInfraNamespaceEnforcerPhasesMixin,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraNamespaceEnforcer(
    FlextInfraNamespaceEnforcerPhasesMixin, FlextInfraNamespaceEnforcerProjectMixin
):
    """Orchestrate namespace enforcement across a workspace."""

    def __init__(self, *, repository_root: Path) -> None:
        """Initialize with the repository root path."""
        super().__init__()
        self._repository_root = repository_root.resolve()
        self._rope_project: t.Infra.RopeProject = u.Infra.init_rope_project(
            self._repository_root
        )

    @override
    def enforce(
        self,
        *,
        apply: bool = False,
        project_names: t.StrSequence | None = None,
        gates: t.StrSequence | None = None,
    ) -> m.Infra.WorkspaceEnforcementReport:
        """Run namespace enforcement across projects in the workspace.

        Args:
            apply: If True, auto-fix detected violations.
            project_names: If provided, only enforce these projects.
            gates: If provided, only run these enforcement gates.

        """
        project_roots = self._resolve_project_roots(project_names=project_names)
        project_reports: list[m.Infra.ProjectEnforcementReport] = []
        for project_root in project_roots:
            report = self._enforce_project(
                project_root=project_root,
                project_name=project_root.name,
                apply=apply,
                gates=gates,
            )
            project_reports.append(report)
        return m.Infra.WorkspaceEnforcementReport.from_projects(
            workspace=str(self._repository_root), projects=project_reports
        )

    @override
    def _resolve_project_roots(
        self, *, project_names: t.StrSequence | None = None
    ) -> t.SequenceOf[Path]:
        """Discover and optionally filter project roots."""
        project_roots = u.Infra.discover_project_roots(
            repository_root=self._repository_root
        )
        project_roots = [
            project_root
            for project_root in project_roots
            if u.Infra.namespace_enabled(project_root)
        ]
        if project_names:
            name_set = set(project_names)
            project_roots = [r for r in project_roots if r.name in name_set]
        return project_roots

    @override
    def _detect_and_apply[V](
        self,
        *,
        py_files: t.SequenceOf[Path],
        detect_fn: Callable[[Path], t.SequenceOf[V]],
        rewrite_fn: Callable[[t.MutableSequenceOf[V]], None] | None,
        apply: bool,
    ) -> t.MutableSequenceOf[V]:
        """Run detect -> optional apply -> re-detect cycle for a violation type.

        Re-detection only runs when apply=True AND a real rewrite_fn is provided.
        """
        violations: t.MutableSequenceOf[V] = []
        for py_file in py_files:
            violations.extend(detect_fn(py_file))
        if not (apply and violations and rewrite_fn is not None):
            return violations
        rewrite_fn(violations)
        self._rope_project.validate(self._rope_project.root)
        post_violations: t.MutableSequenceOf[V] = []
        for py_file in py_files:
            post_violations.extend(detect_fn(py_file))
        return post_violations

    @staticmethod
    def render_text(report: m.Infra.WorkspaceEnforcementReport) -> str:
        """Render a workspace enforcement report as plain text."""
        lines = [
            "Namespace Enforcement Report",
            f"Workspace: {report.workspace}",
            f"Projects: {len(report.projects)}",
            f"Violations: {'YES' if report.has_violations else 'NO'}",
            f"Missing facades: {report.total_facades_missing}",
            f"Loose objects: {report.total_loose_objects}",
            f"Import violations: {report.total_import_violations}",
            f"Namespace source violations: {report.total_namespace_source_violations}",
            f"Internal import violations: {report.total_internal_import_violations}",
            f"Private import bypass violations: {report.total_private_import_bypass_violations}",
            f"Manual protocol violations: {report.total_manual_protocol_violations}",
            f"Cyclic imports: {report.total_cyclic_imports}",
            f"Runtime alias violations: {report.total_runtime_alias_violations}",
            f"Future violations: {report.total_future_violations}",
            f"Manual typing violations: {report.total_manual_typing_violations}",
            f"Compatibility alias violations: {report.total_compatibility_alias_violations}",
            f"Foreign canonical alias violations: {report.total_foreign_canonical_alias_violations}",
            f"Class placement violations: {report.total_class_placement_violations}",
            f"Bare except violations: {report.total_bare_except_violations}",
            f"Print violations: {report.total_print_violations}",
            f"Breakpoint violations: {report.total_breakpoint_violations}",
            f"Open-encoding violations: {report.total_open_encoding_violations}",
            f"Dict annotation violations: {report.total_dict_annotation_violations}",
            f"typing.Dict attr violations: {report.total_typing_dict_attr_violations}",
            f"typing.Dict import violations: {report.total_typing_dict_import_violations}",
            f"Hardcoded-version violations: {report.total_hardcoded_version_violations}",
            f"Parse failures: {report.total_parse_failures}",
            f"Files scanned: {report.total_files_scanned}",
        ]
        return "\n".join(lines)

    @classmethod
    def execute_command(
        cls, params: m.Infra.RefactorNamespaceEnforceInput
    ) -> p.Result[m.Infra.WorkspaceEnforcementReport]:
        """Execute namespace enforcement directly from the canonical payload."""
        enforcer = cls(repository_root=params.workspace_path)
        report = enforcer.enforce(
            apply=params.apply, project_names=params.project_names, gates=params.gates
        )
        cli.display_text(cls.render_text(report))
        has_violations: bool = report.has_violations
        if has_violations:
            return r[m.Infra.WorkspaceEnforcementReport].fail(
                "Namespace violations found"
            )
        return r[m.Infra.WorkspaceEnforcementReport].ok(report)


__all__: list[str] = ["FlextInfraNamespaceEnforcer"]

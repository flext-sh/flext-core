"""Pipeline pass helpers for the codegen fixer service."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from flext_infra import m, u
from flext_infra.codegen._fixer_refactor import FlextInfraCodegenFixerRefactorMixin
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit

_log = u.fetch_logger(__name__)


class FlextInfraCodegenFixerPassesMixin(FlextInfraCodegenFixerRefactorMixin):
    """Private pipeline passes for codegen fixer composition."""

    @staticmethod
    def _run_namespace_enforcement(
        ctx: m.Infra.FixContext,
        project_path: Path,
        enforce: Callable[[str], m.Infra.WorkspaceEnforcementReport],
    ) -> None:
        """Run namespace enforcement and record any unresolved violations."""
        enforcement = enforce(project_path.name)
        violating_projects = tuple(
            project_report
            for project_report in enforcement.projects
            if project_report.has_violations
        )
        if not violating_projects:
            return
        _log.warning(
            "namespace_enforcement_failed",
            project=project_path.name,
            error="violations remain after namespace enforcement",
        )
        ctx.violations_skipped.extend(
            m.Infra.CensusViolation(
                module=project_report.project,
                rule="NAMESPACE",
                line=0,
                message="violations remain after namespace enforcement",
                fixable=False,
            )
            for project_report in violating_projects
        )

    @staticmethod
    def _run_lazy_init_regeneration(
        ctx: m.Infra.FixContext, project_path: Path
    ) -> None:
        """Regenerate lazy ``__init__.py`` files and record skip on errors."""
        lazy_generator = FlextInfraCodegenLazyInit(repository_root=project_path)
        lazy_errors = lazy_generator.generate_inits(check_only=False)
        ctx.files_modified |= set(lazy_generator.modified_files)
        if lazy_errors > 0:
            ctx.skip(
                module=project_path.name,
                rule="LAZY-INIT",
                line=0,
                message=f"lazy propagation finished with {lazy_errors} errors",
            )


__all__: list[str] = ["FlextInfraCodegenFixerPassesMixin"]

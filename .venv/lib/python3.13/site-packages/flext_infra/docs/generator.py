"""Documentation generator service driven by code, exports, and docstrings."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c, m, u
from flext_infra.docs.base import FlextInfraDocServiceBase

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraDocGenerator(FlextInfraDocServiceBase):
    """Generate managed docs artifacts from package exports and docstrings."""

    def generate(
        self, request: m.Infra.DocsGenerateRequest
    ) -> p.Result[t.SequenceOf[m.Infra.DocsPhaseReport]]:
        """Generate docs across the repository root and governed FLEXT projects."""
        return self.run_scoped_docs(
            request.repository_root,
            projects=request.projects,
            output_dir=request.output_dir,
            handler=lambda scope: self._generate_scope(scope, request=request),
        )

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the configured docs generation flow."""
        return self._propagate_phase_outcome(
            "generate",
            self.generate(
                m.Infra.DocsGenerateRequest(
                    repository_root=self.repository_root,
                    projects=self.selected_projects,
                    output_dir=self.output_dir,
                    apply=self.apply_changes,
                )
            ),
            failure_predicate=lambda report: not report.passed,
        )

    def _generate_scope(
        self, scope: m.Infra.DocScope, *, request: m.Infra.DocsGenerateRequest
    ) -> m.Infra.DocsPhaseReport:
        """Generate one scope via the docs generator utilities and log the result."""
        repository_root = request.repository_root.resolve()
        collocated_workspace_project = (
            scope.name != c.Infra.RK_ROOT
            and scope.path.resolve() == repository_root
            and bool(u.Infra.workspace_project_paths(repository_root))
        )
        report: m.Infra.DocsPhaseReport = (
            m.Infra.DocsPhaseReport(
                phase="generate",
                scope=scope.name,
                source="workspace-aggregate-ssot",
                result=c.Infra.ResultStatus.OK,
                reason="aggregate-root-owner",
                passed=True,
            )
            if collocated_workspace_project
            else u.Infra.docs_generate_scope(
                scope,
                apply=request.apply,
                repository_root=request.repository_root,
                projects=request.projects,
            )
        )
        self.logger.info(
            "docs_generate_scope_completed",
            project=scope.name,
            phase="generate",
            result=report.result,
            reason=report.reason,
        )
        return report


__all__: list[str] = ["FlextInfraDocGenerator"]

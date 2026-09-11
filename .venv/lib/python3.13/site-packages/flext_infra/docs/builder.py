"""Documentation builder service."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c, u
from flext_infra.docs.base import FlextInfraDocServiceBase

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, p, t


class FlextInfraDocBuilder(FlextInfraDocServiceBase):
    """Build MkDocs sites for governed FLEXT scopes."""

    _runner: p.Cli.CommandRunner = u.PrivateAttr(default_factory=u.Cli)

    def build(
        self,
        repository_root: Path,
        *,
        projects: t.StrSequence | None = None,
        output_dir: Path | str | None = None,
    ) -> p.Result[t.SequenceOf[m.Infra.DocsPhaseReport]]:
        """Build MkDocs sites across project scopes."""
        return self.run_scoped_docs(
            repository_root,
            projects=projects,
            output_dir=output_dir,
            handler=self._build_scope,
        )

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the configured docs build flow."""
        return self._propagate_phase_outcome(
            "build",
            self.build(
                repository_root=self.repository_root,
                projects=self.selected_projects,
                output_dir=self.output_dir,
            ),
            failure_predicate=lambda report: report.result == c.Infra.ResultStatus.FAIL,
        )

    def _build_scope(self, scope: m.Infra.DocScope) -> m.Infra.DocsPhaseReport:
        """Build one scope via the docs build utilities and persist its reports."""
        report: m.Infra.DocsPhaseReport = u.Infra.docs_run_mkdocs(
            scope, runner=self._runner
        )
        u.Infra.docs_write_build_reports(scope, report)
        self.logger.info(
            "docs_build_scope_completed",
            project=scope.name,
            phase=c.Infra.DIR_BUILD,
            result=report.result,
            reason=report.reason,
        )
        return report


__all__: list[str] = ["FlextInfraDocBuilder"]

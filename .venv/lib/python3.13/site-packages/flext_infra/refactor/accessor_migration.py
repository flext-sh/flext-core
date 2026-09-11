"""Accessor migration orchestration for get_/set_/is_ modernization."""

from __future__ import annotations

from typing import Annotated, override

from flext_cli import cli
from flext_infra import c, m, p, r, t, u
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.refactor._accessor_report import FlextInfraAccessorMigrationReportMixin
from flext_infra.refactor._accessor_rewrite import (
    FlextInfraAccessorMigrationRewriteMixin,
)


class FlextInfraAccessorMigrationOrchestrator(
    FlextInfraProjectSelectionServiceBase[m.Infra.AccessorMigrationReport],
    FlextInfraAccessorMigrationRewriteMixin,
    FlextInfraAccessorMigrationReportMixin,
):
    """Dry-run/apply orchestrator for public accessor migration candidates."""

    preview_limit: Annotated[
        int,
        m.Field(description="Maximum number of file previews to include in the report"),
    ] = 10
    gates: Annotated[
        str,
        m.Field(description="Comma-separated lint gates for preview/apply validation"),
    ] = c.Infra.SAFE_EXECUTION_DEFAULT_GATES

    @property
    @override
    def gate_names(self) -> t.StrSequence:
        """Normalized lint gate names."""
        return u.Infra.normalize_cli_values(self.gates)

    @property
    @override
    def lint_tool_names(self) -> t.StrSequence:
        """Selected lint tool names resolved from gate names."""
        return u.Infra.selected_lint_tool_names(self.gate_names)

    @override
    def execute(self) -> p.Result[m.Infra.AccessorMigrationReport]:
        """Execute."""
        selected_projects: t.StrSequence = (
            self.project_names if self.project_names is not None else ()
        )
        resolved = u.Infra.resolve_projects(self.repository_root, selected_projects)
        if resolved.failure:
            return r[m.Infra.AccessorMigrationReport].fail(
                resolved.error or "project resolution failed"
            )
        iter_result = u.Infra.iter_python_files(
            m.Infra.SourceScanRequest(
                project_roots=tuple(project.path for project in resolved.value)
            )
        )
        if iter_result.failure:
            return r[m.Infra.AccessorMigrationReport].fail(
                iter_result.error or "python file iteration failed"
            )
        previews: t.MutableSequenceOf[m.Infra.AccessorMigrationFile] = []
        files_with_changes = 0
        automated_change_count = 0
        warning_count = 0
        lint_before_totals: dict[str, int] = {}
        lint_after_totals: dict[str, int] = {}
        new_lint_error_totals: dict[str, int] = {}
        with u.Infra.open_project(self.repository_root) as rope_project:
            for py_file in iter_result.value:
                read = u.Cli.files_read_text(py_file)
                if read.failure:
                    return r[m.Infra.AccessorMigrationReport].fail(
                        read.error or f"failed to read {py_file}"
                    )
                source = read.value
                updated_source, automated_changes = self._apply_automated_rewrites(
                    rope_project, py_file, source
                )
                warnings = list(self._collect_manual_warnings(py_file, source))
                file_report = self._process_file(
                    py_file,
                    source=source,
                    updated_source=updated_source,
                    automated_changes=automated_changes,
                    warnings=warnings,
                    include_preview=(bool(automated_changes or warnings))
                    and len(previews) < self.preview_limit,
                )
                automated_change_count += len(file_report.automated_changes)
                warning_count += len(file_report.warnings)
                if file_report.automated_changes:
                    files_with_changes += 1
                if (file_report.automated_changes or file_report.warnings) and len(
                    previews
                ) < self.preview_limit:
                    previews.append(file_report)
                self._accumulate_lint_totals(
                    lint_before_totals, file_report.lint_before
                )
                self._accumulate_lint_totals(lint_after_totals, file_report.lint_after)
                self._accumulate_lint_totals(
                    new_lint_error_totals, file_report.new_lint_errors
                )
        return r[m.Infra.AccessorMigrationReport].ok(
            m.Infra.AccessorMigrationReport(
                workspace=str(self.repository_root),
                dry_run=self.dry_run,
                files_scanned=len(iter_result.value),
                files_with_changes=files_with_changes,
                automated_change_count=automated_change_count,
                warning_count=warning_count,
                lint_tools=tuple(self.lint_tool_names),
                lint_before_totals=lint_before_totals,
                lint_after_totals=lint_after_totals,
                new_lint_error_totals=new_lint_error_totals,
                files=tuple(previews),
            )
        )

    @classmethod
    def execute_payload(
        cls, params: m.Infra.AccessorMigrationInput
    ) -> p.Result[m.Infra.AccessorMigrationReport]:
        """Execute accessor migration from the validated command service."""
        result = cls(
            repository_root=params.workspace_path,
            selected_projects=params.projects,
            apply_changes=params.apply,
            target_module=params.module,
            target_namespace=params.namespace,
            preview_limit=params.preview_limit,
            gates=",".join(params.gates),
        ).execute()
        if result.failure:
            return r[m.Infra.AccessorMigrationReport].fail(
                result.error or "accessor migration execution failed"
            )
        cli.display_text(cls.render_text(result.value))
        return r[m.Infra.AccessorMigrationReport].ok(result.value)


__all__: list[str] = ["FlextInfraAccessorMigrationOrchestrator"]

"""Accessor per-file lint processing + report rendering — extracted concern."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

from flext_infra import m, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraAccessorMigrationReportMixin:
    """Per-file lint snapshot/write and CLI report rendering.

    Composed into FlextInfraAccessorMigrationOrchestrator via inheritance; the
    facade provides ``dry_run`` / ``repository_root`` / the gate-name properties
    through FLEXT (declared below for static resolution).
    """

    if TYPE_CHECKING:
        dry_run: bool
        repository_root: Path

        @property
        def gate_names(self) -> t.StrSequence: ...

        @property
        def lint_tool_names(self) -> t.StrSequence: ...

    @staticmethod
    def _accumulate_lint_totals(
        totals: dict[str, int], snapshot: t.Infra.LintSnapshot
    ) -> None:
        """Accumulate lint totals."""
        for tool, lines in snapshot.items():
            totals[tool] = totals.get(tool, 0) + len(tuple(lines))

    def _process_file(
        self,
        py_file: Path,
        *,
        source: str,
        updated_source: str,
        automated_changes: t.SequenceOf[m.Infra.AccessorMigrationChange],
        warnings: t.MutableSequenceOf[m.Infra.AccessorMigrationChange],
        include_preview: bool,
    ) -> m.Infra.AccessorMigrationFile:
        """Process file."""
        lint_before: dict[str, t.StrSequence] = {}
        lint_after: dict[str, t.StrSequence] = {}
        new_lint_errors: dict[str, t.StrSequence] = {}
        before: t.Infra.LintSnapshot = {}
        after: t.Infra.LintSnapshot = {}
        if automated_changes:
            if self.dry_run and include_preview:
                before, after = u.Infra.preview_source_lint(
                    py_file,
                    self.repository_root,
                    updated_source=updated_source,
                    gates=self.gate_names,
                )
            elif not self.dry_run:
                before = (
                    u.Infra.lint_snapshot(
                        py_file, self.repository_root, gates=self.gate_names
                    )
                    if include_preview
                    else {}
                )
                ok, report = u.Infra.protected_source_write(
                    py_file,
                    request=m.Infra.ProtectedSourceWriteRequest(
                        workspace=self.repository_root,
                        updated_source=updated_source,
                        gates=self.gate_names,
                    ),
                )
                if not ok:
                    warnings.append(
                        m.Infra.AccessorMigrationChange(
                            file=str(py_file),
                            line=0,
                            original_name="protected_write",
                            replacement_name="",
                            automated=False,
                            reason=" ; ".join(report[:3]) or "protected write failed",
                        )
                    )
                after = (
                    u.Infra.lint_snapshot(
                        py_file, self.repository_root, gates=self.gate_names
                    )
                    if include_preview
                    else {}
                )
            else:
                before = {}
                after = {}
            if include_preview:
                lint_before = self._freeze_lints(before)
                lint_after = self._freeze_lints(after)
                new_lint_errors = self._freeze_lints(
                    u.Infra.lint_new_errors(before, after)
                )
        return m.Infra.AccessorMigrationFile(
            file=str(py_file),
            lint_tools=tuple(self.lint_tool_names)
            if automated_changes and include_preview
            else (),
            automated_changes=tuple(automated_changes),
            warnings=tuple(warnings),
            diff=self._diff(py_file, source, updated_source)
            if automated_changes and include_preview
            else "",
            lint_before=lint_before,
            lint_after=lint_after,
            new_lint_errors=new_lint_errors,
        )

    @staticmethod
    def _freeze_lints(snapshot: t.Infra.LintSnapshot) -> dict[str, t.StrSequence]:
        """Freeze lints."""
        return {tool: tuple(lines) for tool, lines in snapshot.items()}

    @staticmethod
    def _diff(py_file: Path, before: str, after: str) -> str:
        """Diff."""
        diff_lines = list(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{py_file}",
                tofile=f"b/{py_file}",
                n=3,
            )
        )
        return "".join(diff_lines[:80])

    @staticmethod
    def render_text(report: m.Infra.AccessorMigrationReport) -> str:
        """Render an accessor migration report as CLI text."""
        lines: t.MutableSequenceOf[str] = [
            "Accessor Migration",
            f"workspace: {report.workspace}",
            f"mode: {'dry-run' if report.dry_run else 'apply'}",
            f"files_scanned: {report.files_scanned}",
            f"files_with_changes: {report.files_with_changes}",
            f"automated_changes: {report.automated_change_count}",
            f"warnings: {report.warning_count}",
            f"lint_tools: {', '.join(report.lint_tools)}",
        ]
        for tool in report.lint_tools:
            lines.append(
                f"lint-totals:{tool} before={report.lint_before_totals.get(tool, 0)} after={report.lint_after_totals.get(tool, 0)} new={report.new_lint_error_totals.get(tool, 0)}"
            )
        for file_report in report.files:
            lines.append(f"\n{file_report.file}")
            for change in file_report.automated_changes:
                lines.append(
                    f"  auto:{change.line} {change.original_name} -> {change.replacement_name}"
                )
            for warning in file_report.warnings:
                target = (
                    f" -> {warning.replacement_name}"
                    if warning.replacement_name
                    else ""
                )
                lines.append(f"  warn:{warning.line} {warning.original_name}{target}")
                lines.append(f"    {warning.reason}")
            for tool in file_report.lint_tools:
                issues = tuple(file_report.lint_after.get(tool, ()))
                lines.append(f"  lint-after:{tool}")
                if not issues:
                    lines.append("    ok")
                    continue
                lines.extend(f"    {issue}" for issue in issues[:4])
            for tool in file_report.lint_tools:
                issues = tuple(file_report.new_lint_errors.get(tool, ()))
                if not issues:
                    continue
                lines.append(f"  new-lint:{tool}")
                lines.extend(f"    {issue}" for issue in issues[:4])
            if file_report.diff:
                lines.append("  diff:")
                lines.extend(
                    f"    {line}"
                    for line in file_report.diff.rstrip().splitlines()[:40]
                )
        return "\n".join(lines)


__all__: list[str] = ["FlextInfraAccessorMigrationReportMixin"]

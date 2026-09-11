"""Apply orchestration for the project-layout engine (flext-0wuz).

Executes the typed findings produced by the planning mixin, composing the
file/archive primitives and the gitignore owner through FLEXT.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import m, p, r, t
from flext_infra.codegen._layout_files import FlextInfraCodegenLayoutFilesMixin
from flext_infra.codegen._layout_gitignore import FlextInfraCodegenLayoutGitignoreMixin


class FlextInfraCodegenLayoutApplyMixin(
    FlextInfraCodegenLayoutFilesMixin, FlextInfraCodegenLayoutGitignoreMixin
):
    """Execute layout findings idempotently against one project directory."""

    def apply_project(
        self, project_dir: Path, report: m.Infra.LayoutProjectReport
    ) -> p.Result[m.Infra.LayoutProjectReport]:
        """Execute every planned finding and return the updated report."""
        findings: list[m.Infra.LayoutFinding] = []
        gitignore_status: t.Infra.LayoutStatus = "noop"
        actionable: tuple[m.Infra.LayoutFinding, ...] = report.actionable
        gitignore_patterns = tuple(
            finding.target for finding in actionable if finding.rule == "gitignore"
        )
        if gitignore_patterns:
            applied = self._apply_gitignore(project_dir, gitignore_patterns)
            if applied.failure:
                return r[m.Infra.LayoutProjectReport].fail(
                    applied.error or "gitignore update failed"
                )
            gitignore_status = applied.value
        for finding in report.findings:
            if finding.rule == "review":
                findings.append(finding)
                continue
            if finding.rule == "gitignore":
                findings.append(finding.model_copy(update={"status": gitignore_status}))
                continue
            executed = self._execute_path_finding(project_dir, finding)
            if executed.failure:
                return r[m.Infra.LayoutProjectReport].fail(
                    executed.error or f"layout action failed: {finding.path}"
                )
            findings.append(executed.value)
        return r[m.Infra.LayoutProjectReport].ok(
            report.model_copy(update={"findings": tuple(findings)})
        )

    def _execute_path_finding(
        self, project_dir: Path, finding: m.Infra.LayoutFinding
    ) -> p.Result[m.Infra.LayoutFinding]:
        """Execute one move/archive finding; missing sources are no-ops."""
        source = project_dir / finding.path
        if not source.exists():
            return r[m.Infra.LayoutFinding].ok(
                finding.model_copy(update={"status": "noop"})
            )
        if finding.rule == "move":
            return self._apply_move(project_dir, finding, source)
        return self._apply_archive(project_dir, finding, source)

    def _apply_move(
        self, project_dir: Path, finding: m.Infra.LayoutFinding, source: Path
    ) -> p.Result[m.Infra.LayoutFinding]:
        """Move one file/dir to its canonical target, merging dir collisions."""
        target = project_dir / finding.target
        if source.is_dir() and target.exists():
            merge = self._merge_directory(project_dir, source, target, finding.path)
            if merge.failure:
                return r[m.Infra.LayoutFinding].fail(
                    merge.error or f"docs merge failed: {finding.path}"
                )
            if source.exists():
                return r[m.Infra.LayoutFinding].ok(
                    finding.model_copy(
                        update={
                            "status": "skipped",
                            "message": f"{finding.message} (merge incomplete: review)",
                        }
                    )
                )
            return r[m.Infra.LayoutFinding].ok(
                finding.model_copy(update={"status": "applied"})
            )
        moved = self._move_entry(project_dir, source, target, finding.path)
        if moved.failure:
            return r[m.Infra.LayoutFinding].fail(
                moved.error or f"move failed: {finding.path}"
            )
        status: t.Infra.LayoutStatus = "applied"
        return r[m.Infra.LayoutFinding].ok(
            finding.model_copy(update={"status": status, "message": moved.value})
        )

    def _apply_archive(
        self, project_dir: Path, finding: m.Infra.LayoutFinding, source: Path
    ) -> p.Result[m.Infra.LayoutFinding]:
        """Archive one entry into the archive root, preserving content."""
        archived = self._archive_path(project_dir, source, finding.path, finding)
        if archived.failure:
            return r[m.Infra.LayoutFinding].fail(
                archived.error or f"archive failed: {finding.path}"
            )
        status, message = archived.value
        return r[m.Infra.LayoutFinding].ok(
            finding.model_copy(update={"status": status, "message": message})
        )

    def _merge_directory(
        self, project_dir: Path, source: Path, target: Path, source_rel: str
    ) -> p.Result[bool]:
        """Merge a docs dir into an existing target dir file-by-file."""
        files = sorted(path for path in source.rglob("*") if path.is_file())
        for file_path in files:
            rel = file_path.relative_to(source).as_posix()
            moved = self._move_entry(
                project_dir, file_path, target / rel, f"{source_rel}/{rel}"
            )
            if moved.failure:
                return r[bool].fail(moved.error or f"merge move failed: {rel}")
        self._prune_empty_dirs(source)
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraCodegenLayoutApplyMixin"]

"""Automated migration of loose constants/typings into MRO facade classes."""

from __future__ import annotations

from pathlib import Path

from flext_infra import c, m

from .mro_migrator import FlextInfraRefactorMROMigrationTransformer
from .mro_resolver import (
    FlextInfraRefactorMROImportRewriter,
    FlextInfraRefactorMROMigrationScanner,
)
from .validation import FlextInfraRefactorMROMigrationValidator


class FlextInfraRefactorMigrateToClassMRO:
    """Orchestrate scan, migration, rewrite, and validation phases."""

    def __init__(self, *, workspace_root: Path) -> None:
        """Create migration service bound to a workspace root."""
        self._workspace_root = workspace_root.resolve()

    def run(
        self,
        *,
        target: str,
        apply: bool,
    ) -> m.Infra.Refactor.MROMigrationReport:
        """Run scan, transform, rewrite, and validation phases."""
        normalized_target = self._normalize_target(target=target)
        scan_results, files_scanned = (
            FlextInfraRefactorMROMigrationScanner.scan_workspace(
                workspace_root=self._workspace_root,
                target=normalized_target,
            )
        )
        warnings: list[str] = []
        errors: list[str] = []
        stash_ref = ""
        moved_index: dict[str, dict[str, str]] = {}
        module_facade_aliases: dict[str, str] = {}
        migrations: list[m.Infra.Refactor.MROFileMigration] = []
        for scan_result in scan_results:
            try:
                updated_source, migration, symbol_alias_map = (
                    FlextInfraRefactorMROMigrationTransformer.migrate_file(
                        scan_result=scan_result,
                    )
                )
            except Exception as exc:
                errors.append(f"{scan_result.file}: {exc}")
                continue
            if len(migration.moved_symbols) == 0:
                continue
            migrations.append(migration)
            moved_index[scan_result.module] = symbol_alias_map
            module_facade_aliases[scan_result.module] = scan_result.facade_alias
            if apply:
                Path(scan_result.file).write_text(
                    updated_source,
                    encoding=c.Infra.Encoding.DEFAULT,
                )
        rewrite_results = FlextInfraRefactorMROImportRewriter.rewrite_workspace(
            workspace_root=self._workspace_root,
            moved_index=moved_index,
            module_facade_aliases=module_facade_aliases,
            apply=apply,
        )
        rewrites = tuple(rewrite_results)
        remaining_violations, mro_failures = (
            FlextInfraRefactorMROMigrationValidator.validate(
                workspace_root=self._workspace_root,
                target=normalized_target,
            )
        )
        return m.Infra.Refactor.MROMigrationReport(
            workspace=str(self._workspace_root),
            target=normalized_target,
            dry_run=not apply,
            files_scanned=files_scanned,
            files_with_candidates=len(scan_results),
            migrations=tuple(migrations),
            rewrites=rewrites,
            remaining_violations=remaining_violations,
            mro_failures=mro_failures,
            stash_ref=stash_ref,
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    @staticmethod
    def render_text(report: m.Infra.Refactor.MROMigrationReport) -> str:
        """Render migration report in CLI-friendly plain text."""
        lines = [
            f"Workspace: {report.workspace}",
            f"Target: {report.target}",
            f"Mode: {('dry-run' if report.dry_run else 'apply')}",
            f"Files scanned: {report.files_scanned}",
            f"Files with candidates: {report.files_with_candidates}",
            f"Migrations: {len(report.migrations)}",
            f"Rewrites: {len(report.rewrites)}",
            f"Remaining violations: {report.remaining_violations}",
            f"MRO failures: {report.mro_failures}",
        ]
        if report.stash_ref:
            lines.append(f"Rollback stash: {report.stash_ref}")
        if len(report.warnings) > 0:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in report.warnings)
        if len(report.errors) > 0:
            lines.append("Errors:")
            lines.extend(f"- {error}" for error in report.errors)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _normalize_target(*, target: str) -> str:
        value = target.strip().lower()
        if value in c.Infra.Refactor.MRO_TARGETS:
            return value
        msg = f"unsupported target: {target}"
        raise ValueError(msg)


__all__ = ["FlextInfraRefactorMigrateToClassMRO"]

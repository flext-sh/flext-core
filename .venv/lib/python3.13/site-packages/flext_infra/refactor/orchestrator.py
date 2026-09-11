"""Refactor orchestration layer for file/project/workspace execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_infra import c, m, u
from flext_infra.refactor._orchestrator_dispatch import (
    FlextInfraRefactorOrchestratorDispatchMixin,
)
from flext_infra.refactor._orchestrator_scope import (
    FlextInfraRefactorOrchestratorScopeMixin,
)
from flext_infra.refactor.file_executor import (
    FlextInfraClassNestingPostCheckGate,
    FlextInfraRefactorFileExecutor,
)
from flext_infra.refactor.safety import FlextInfraRefactorSafetyManager
from flext_infra.refactor.text_executor import FlextInfraRefactorTextExecutor

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t
    from flext_infra.refactor.loader import FlextInfraRefactorRuleLoader

_log = u.fetch_logger(__name__)


class FlextInfraRefactorOrchestrator(
    FlextInfraRefactorTextExecutor,
    FlextInfraRefactorFileExecutor,
    FlextInfraRefactorOrchestratorDispatchMixin,
    FlextInfraRefactorOrchestratorScopeMixin,
):
    """Coordinate refactor execution using loaded rule selections and safety flow."""

    def __init__(
        self,
        loader: FlextInfraRefactorRuleLoader,
        *,
        safety_manager: FlextInfraRefactorSafetyManager | None = None,
    ) -> None:
        """Initialize the orchestrator with a loader and optional safety service."""
        self.loader = loader
        self.safety_manager = safety_manager or FlextInfraRefactorSafetyManager()
        self._class_nesting_config: t.JsonMapping | None = None
        self._class_nesting_policy_by_family: (
            t.MappingKV[str, m.Infra.ClassNestingPolicy] | None
        ) = None
        self._class_nesting_gate: FlextInfraClassNestingPostCheckGate | None = None

    @override
    def refactor_file(
        self,
        file_path: Path,
        *,
        dry_run: bool = False,
        gates: t.StrSequence | None = None,
    ) -> m.Infra.Result:
        """Refactor one file using the loader's current rule selections."""
        try:
            if file_path.suffix != c.Infra.EXT_PYTHON:
                return self._skip_result(file_path)
            return self._refactor_python_file(file_path, dry_run=dry_run, gates=gates)
        except c.EXC_BROAD_IO_TYPE as exc:
            return self._error_result(file_path, str(exc))

    def _refactor_python_file(
        self, file_path: Path, *, dry_run: bool, gates: t.StrSequence | None
    ) -> m.Infra.Result:
        """Refactor one Python source file after caller-level exception handling."""
        repository_root = u.Infra.project_root(file_path) or file_path.parent
        read = u.Cli.files_read_text(file_path)
        if read.failure:
            return self._error_result(
                file_path, read.error or f"failed to read {file_path}"
            )
        original = read.value
        current, all_changes = original, list[str]()
        current, error_result = self._apply_file_rules(
            file_path, repository_root, current, all_changes
        )
        if error_result is not None:
            return error_result
        current = self._apply_text_rules(file_path, current, all_changes)
        modified = current != original
        if not dry_run and modified:
            error_result = self._write_refactored_source(
                file_path=file_path,
                repository_root=repository_root,
                current=current,
                original=original,
                all_changes=all_changes,
                gates=gates,
            )
        return error_result or m.Infra.Result(
            file_path=file_path,
            success=True,
            modified=modified,
            changes=all_changes,
            refactored_code=current,
        )

    def _apply_file_rules(
        self,
        file_path: Path,
        repository_root: Path,
        current: str,
        all_changes: t.MutableSequenceOf[str],
    ) -> tuple[str, m.Infra.Result | None]:
        """Apply Rope-backed file rules and collect changes."""
        if not self.loader.file_rules:
            return current, None
        updated_source = current
        with u.Infra.open_project(repository_root) as rope_project:
            resource = u.Infra.get_resource_from_path(rope_project, file_path)
            if resource is None:
                return (
                    updated_source,
                    self._error_result(
                        file_path, f"Could not resolve rope resource for {file_path}"
                    ),
                )
            for kind, settings in self.loader.file_rules:
                file_rule_result = self._apply_file_rule_selection(
                    kind, settings, rope_project, resource, dry_run=True
                )
                if not file_rule_result.success:
                    return (
                        updated_source,
                        m.Infra.Result(
                            file_path=file_path,
                            success=False,
                            modified=False,
                            error=file_rule_result.error,
                            changes=file_rule_result.changes,
                            refactored_code=None,
                        ),
                    )
                if file_rule_result.modified and file_rule_result.refactored_code:
                    updated_source = file_rule_result.refactored_code
                all_changes.extend(file_rule_result.changes)
        return updated_source, None

    def _apply_text_rules(
        self, file_path: Path, current: str, all_changes: t.MutableSequenceOf[str]
    ) -> str:
        """Apply enabled text rules and collect changes."""
        updated_source = current
        for kind, settings in self.loader.rules:
            if not bool(settings.get(c.Infra.RK_ENABLED, True)):
                continue
            updated_source, changes = self._apply_text_rule_selection(
                kind, settings, updated_source, file_path
            )
            all_changes.extend(changes)
        return updated_source

    def _write_refactored_source(
        self,
        *,
        file_path: Path,
        repository_root: Path,
        current: str,
        original: str,
        all_changes: t.MutableSequenceOf[str],
        gates: t.StrSequence | None,
    ) -> m.Infra.Result | None:
        """Write transformed source through protected validation."""
        ok, report = u.Infra.protected_source_write(
            file_path,
            request=m.Infra.ProtectedSourceWriteRequest(
                workspace=repository_root,
                updated_source=current,
                keep_backup=True,
                gates=gates,
            ),
        )
        all_changes.extend(report)
        if ok:
            return None
        return m.Infra.Result(
            file_path=file_path,
            success=False,
            modified=False,
            error="Protected refactor validation failed",
            changes=all_changes,
            refactored_code=original,
        )

    @override
    def refactor_files(
        self,
        file_paths: t.SequenceOf[Path],
        *,
        dry_run: bool = False,
        gates: t.StrSequence | None = None,
    ) -> t.SequenceOf[m.Infra.Result]:
        """Refactor many files and collect individual results."""
        results: t.MutableSequenceOf[m.Infra.Result] = []
        for file_path in file_paths:
            result = self.refactor_file(file_path, dry_run=dry_run, gates=gates)
            results.append(result)
            if result.success and result.modified:
                u.Cli.info(
                    f"{'[DRY-RUN] ' if dry_run else ''}Modified: {file_path.name}"
                )
                for change in result.changes:
                    u.Cli.info(f"  - {change}")
            elif result.success:
                _log.debug("refactor_noop", file=str(result.file_path))
                self._refactor_debug(f"Unchanged: {file_path.name}")
            else:
                u.Cli.error(f"Failed: {file_path.name} - {result.error}")
        return results


__all__: list[str] = ["FlextInfraRefactorOrchestrator"]

"""Enforcement phase methods for namespace enforcer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import m, u
from flext_infra.detectors.facade_scanner import FlextInfraScanner

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from flext_infra import t


class FlextInfraNamespaceEnforcerPhasesMixin:
    """Project enforcement methods for namespace enforcer."""

    _repository_root: Path
    _rope_project: t.Infra.RopeProject

    def _resolve_project_roots(
        self, *, project_names: t.StrSequence | None = None
    ) -> t.SequenceOf[Path]:
        """Resolve project roots."""
        msg = "_resolve_project_roots must be provided by the concrete enforcer"
        raise NotImplementedError(msg)

    def _detect_and_apply[V](
        self,
        *,
        py_files: t.SequenceOf[Path],
        detect_fn: Callable[[Path], t.SequenceOf[V]],
        rewrite_fn: Callable[[t.MutableSequenceOf[V]], None] | None,
        apply: bool,
    ) -> t.MutableSequenceOf[V]:
        """Detect and apply."""
        _ = self, py_files, detect_fn, rewrite_fn, apply
        msg = "_detect_and_apply must be provided by the concrete enforcer"
        raise NotImplementedError(msg)

    def enforce(
        self,
        *,
        apply: bool = False,
        project_names: t.StrSequence | None = None,
        gates: t.StrSequence | None = None,
    ) -> m.Infra.WorkspaceEnforcementReport:
        """Enforce namespace rules across the workspace."""
        _ = apply, project_names, gates
        msg = "enforce must be provided by the concrete enforcer"
        raise NotImplementedError(msg)

    @staticmethod
    def _scan_facades(
        *,
        project: tuple[Path, str],
        rope_project: t.Infra.RopeProject,
        apply: bool,
        repository_root: Path,
    ) -> t.SequenceOf[m.Infra.FacadeStatus]:
        """Scan facades and optionally create missing ones."""
        project_root, project_name = project
        facade_statuses = FlextInfraScanner.scan_project(
            project_root=project_root, rope_project=rope_project
        )
        if not apply:
            return facade_statuses
        u.Infra.ensure_missing_facades(
            project_root=project_root,
            project_name=project_name,
            facade_statuses=facade_statuses,
            repository_root=repository_root,
        )
        return FlextInfraScanner.scan_project(
            project_root=project_root, rope_project=rope_project
        )

    @staticmethod
    def _collect_py_files(*, project_root: Path) -> t.SequenceOf[Path]:
        """Collect Python files for scanning."""
        py_files_result = u.Infra.iter_python_files(
            m.Infra.SourceScanRequest(project_roots=(project_root,))
        )
        if py_files_result.failure:
            msg = py_files_result.error or (
                f"failed to collect Python files for {project_root}"
            )
            raise RuntimeError(msg)
        files: t.SequenceOf[Path] = py_files_result.value
        return files


__all__: list[str] = ["FlextInfraNamespaceEnforcerPhasesMixin"]

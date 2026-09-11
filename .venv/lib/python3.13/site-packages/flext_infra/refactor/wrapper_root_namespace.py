"""Wrapper-root namespace refactor command for production script aliases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, override

from flext_infra import c, m, p, r, t, u
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.refactor._wrapper_rewrite import (
    FlextInfraWrapperRootNamespaceRewriteMixin,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class FlextInfraWrapperRootNamespaceRefactor(
    FlextInfraProjectSelectionServiceBase[t.JsonPayload],
    FlextInfraWrapperRootNamespaceRewriteMixin,
):
    """Refactor wrapper alias imports and deprecated ``*.Core.Tests`` paths."""

    _WRAPPER_PACKAGES: ClassVar[t.StrSequence] = tuple(
        segment
        for segment in c.Infra.ROOT_WRAPPER_SEGMENTS
        if segment != c.Infra.DEFAULT_SRC_DIR
    )

    include_init: Annotated[
        bool,
        m.Field(
            alias="include-init",
            description="Also process __init__.py files (default: false)",
        ),
    ] = False

    @override
    def execute(self) -> p.Result[t.JsonPayload]:
        """Discover wrapper files, rewrite ``Core.Tests`` chains, persist results."""
        scan = self._scan_workspace()
        if scan.failure:
            return r[t.JsonPayload].fail(scan.error or "wrapper scan failed")
        py_files, project_runtime_aliases, wrapper_submodules = scan.value
        accumulator = m.Infra.WrapperRewriteAccumulator()
        metadata_aliases = u.runtime_alias_names(c.Infra.PKG_INFRA_UNDERSCORE)
        for file_path in py_files:
            self._process_wrapper_file(
                file_path,
                accumulator=accumulator,
                project_runtime_aliases=project_runtime_aliases,
                wrapper_submodules=wrapper_submodules,
                metadata_runtime_aliases=metadata_aliases,
            )
        write_failure = self._persist_updates(accumulator.updates)
        if write_failure is not None:
            return r[t.JsonPayload].fail(write_failure)
        effective_dry_run: bool = self.effective_dry_run
        if not effective_dry_run and accumulator.wrapper_candidates:
            for wrapper in self._WRAPPER_PACKAGES:
                u.Infra.rewrite_import_violations(
                    py_files=accumulator.wrapper_candidates, project_package=wrapper
                )
        report_payload = self._build_report_payload(len(py_files), accumulator)
        if self.check_only and accumulator.changed_files:
            return r[t.JsonPayload].fail(
                "pending wrapper-root namespace rewrites detected"
            )
        return r[t.JsonPayload].ok(report_payload)

    def _scan_workspace(
        self,
    ) -> p.Result[tuple[t.SequenceOf[Path], dict[str, frozenset[str]], frozenset[str]]]:
        """Resolve project paths and discover Python files + runtime alias map."""
        selected_projects: t.StrSequence = (
            self.project_names if self.project_names is not None else ()
        )
        resolved = u.Infra.resolve_projects(self.repository_root, selected_projects)
        if resolved.failure:
            return r[
                tuple[t.SequenceOf[Path], dict[str, frozenset[str]], frozenset[str]]
            ].fail(resolved.error or "project resolution failed")
        iter_result = u.Infra.iter_python_files(
            m.Infra.SourceScanRequest(
                project_roots=tuple(project.path for project in resolved.value)
            )
        )
        if iter_result.failure:
            return r[
                tuple[t.SequenceOf[Path], dict[str, frozenset[str]], frozenset[str]]
            ].fail(iter_result.error or "python file iteration failed")
        project_runtime_aliases = {
            project.path.name: frozenset(layout.runtime_aliases)
            for project in resolved.value
            if (layout := u.Infra.layout(project.path)) is not None
        }
        return r[
            tuple[t.SequenceOf[Path], dict[str, frozenset[str]], frozenset[str]]
        ].ok((
            iter_result.value,
            project_runtime_aliases,
            u.facade_module_names(c.Infra.PKG_INFRA_UNDERSCORE),
        ))

    def _persist_updates(self, updates: Mapping[Path, str]) -> str | None:
        """Write batched updates via the protected pipeline; ``None`` on success."""
        if not updates:
            return None
        ok, report = u.Infra.protected_source_writes(
            dict(updates),
            request=m.Infra.ProtectedSourceWritesRequest(
                workspace=self.repository_root, skip_pytest=True
            ),
        )
        if ok:
            return None
        return " ; ".join(report[:5]) or "protected write failed"

    def _build_report_payload(
        self, files_scanned: int, accumulator: m.Infra.WrapperRewriteAccumulator
    ) -> t.MutableJsonMapping:
        """Build the canonical JSON payload from the accumulated wrapper run state."""
        mode_value = (
            "check"
            if self.check_only
            else "dry-run"
            if self.effective_dry_run
            else "apply"
        )
        per_project_changes_payload: t.JsonDict = dict(
            accumulator.per_project_changes.items()
        )
        per_project_replacements_payload: t.JsonDict = dict(
            accumulator.per_project_replacements.items()
        )
        changed_files_preview: t.JsonValueList = list(accumulator.changed_files[:200])
        report_payload: t.MutableJsonMapping = {
            "files_scanned": files_scanned,
            "files_changed": len(accumulator.changed_files),
            "replacements": accumulator.total_replacements,
            "core_tests_replacements": accumulator.total_core_replacements,
            "import_rewrite_candidates": accumulator.import_rewrite_candidates,
            "per_project_changes": per_project_changes_payload,
            "per_project_replacements": per_project_replacements_payload,
            "changed_files_preview": changed_files_preview,
            "workspace": str(self.repository_root),
            "mode": mode_value,
        }
        return report_payload


__all__: list[str] = ["FlextInfraWrapperRootNamespaceRefactor"]

"""Docs scope construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import r
from flext_infra._utilities._docs_scope_selection import (
    FlextInfraUtilitiesDocsScopeSelectionMixin,
)
from flext_infra._utilities.base import FlextInfraUtilitiesBase
from flext_infra._utilities.docs_scope import FlextInfraUtilitiesDocsScope
from flext_infra._utilities.pyproject import FlextInfraUtilitiesPyproject
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra.protocols import p


class FlextInfraUtilitiesDocsScopeBuildMixin(
    FlextInfraUtilitiesDocsScopeSelectionMixin
):
    """Build canonical DocScope models for docs commands."""

    @staticmethod
    def _selected_project_names(
        repository_root: Path, projects: t.StrSequence | None
    ) -> list[str]:
        """Return normalized project filters for docs-scoped operations."""
        _ = repository_root
        return list(FlextInfraUtilitiesBase.normalize_sequence_values(projects) or ())

    @staticmethod
    def build_scopes(
        repository_root: Path, projects: t.StrSequence | None, output_dir: Path | str
    ) -> p.Result[t.SequenceOf[m.Infra.DocScope]]:
        """Build DocScope objects for repository root and selected projects."""
        try:
            scopes = FlextInfraUtilitiesDocsScopeBuildMixin._build_scopes_unchecked(
                repository_root, projects, output_dir
            )
        except c.EXC_OS_TYPE_VALUE as exc:
            return r[t.SequenceOf[m.Infra.DocScope]].fail_op("scope resolution", exc)
        return r[t.SequenceOf[m.Infra.DocScope]].ok(scopes)

    @staticmethod
    def _build_scopes_unchecked(
        repository_root: Path, projects: t.StrSequence | None, output_dir: Path | str
    ) -> t.SequenceOf[m.Infra.DocScope]:
        """Build docs scopes without exception wrapping."""
        resolved_root = repository_root.resolve()
        project_state = FlextInfraUtilitiesDocsScope.project_state(resolved_root)
        enabled = project_state.docs_meta.get("enabled", True)
        is_enabled = enabled if isinstance(enabled, bool) else True
        discovered = FlextInfraUtilitiesDocsScopeBuildMixin._discover_projects(
            resolved_root
        )
        has_declared_members = bool(
            FlextInfraUtilitiesPyproject.workspace_project_paths(resolved_root)
        )
        has_child_projects = any(
            project.path.resolve() != resolved_root for project in discovered
        )
        if (
            (resolved_root / c.Infra.PYPROJECT_FILENAME).is_file()
            and not has_declared_members
            and not has_child_projects
            and is_enabled
        ):
            return (
                FlextInfraUtilitiesDocsScopeBuildMixin._governed_scope(
                    resolved_root, output_dir
                ),
            )
        return FlextInfraUtilitiesDocsScopeBuildMixin._workspace_scopes(
            resolved_root, projects, output_dir, discovered
        )

    @staticmethod
    def _workspace_scopes(
        repository_root: Path,
        projects: t.StrSequence | None,
        output_dir: Path | str,
        discovered: t.SequenceOf[m.Infra.ProjectInfo],
    ) -> t.SequenceOf[m.Infra.DocScope]:
        """Build docs scopes for a repository root plus child projects."""
        scopes: list[m.Infra.DocScope] = [
            m.Infra.DocScope(
                name=c.Infra.RK_ROOT,
                path=repository_root,
                report_dir=(repository_root / output_dir).resolve(),
                project_class="root",
                package_name="",
            )
        ]
        selected_names = FlextInfraUtilitiesDocsScopeBuildMixin._selected_project_names(
            repository_root, projects
        )
        if selected_names:
            scopes.extend(
                FlextInfraUtilitiesDocsScopeBuildMixin._selected_project_scopes(
                    repository_root, discovered, selected_names, output_dir
                )
            )
            return tuple(scopes)
        scopes.extend(
            FlextInfraUtilitiesDocsScopeBuildMixin._doc_scope(
                project=project, output_dir=output_dir
            )
            for project in discovered
        )
        return tuple(scopes)

    @staticmethod
    def _discover_projects(repository_root: Path) -> t.SequenceOf[m.Infra.ProjectInfo]:
        """Discover workspace projects or raise a typed value error."""
        discovered_result = FlextInfraUtilitiesDocsScope.discover_projects(
            repository_root
        )
        if discovered_result.failure:
            msg = discovered_result.error or "project discovery failed"
            raise ValueError(msg)
        return discovered_result.value


__all__: list[str] = ["FlextInfraUtilitiesDocsScopeBuildMixin"]

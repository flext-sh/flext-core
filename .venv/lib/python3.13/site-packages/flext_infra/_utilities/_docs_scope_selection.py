"""Docs scope selection helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra._utilities.docs_scope import FlextInfraUtilitiesDocsScope
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra.typings import t


class FlextInfraUtilitiesDocsScopeSelectionMixin:
    """Resolve selected docs project names into DocScope models."""

    @staticmethod
    def _selected_project_scopes(
        repository_root: Path,
        discovered: t.SequenceOf[m.Infra.ProjectInfo],
        selected_names: t.StrSequence,
        output_dir: Path | str,
    ) -> t.SequenceOf[m.Infra.DocScope]:
        """Build docs scopes for selected project names."""
        project_by_name = FlextInfraUtilitiesDocsScopeSelectionMixin._project_by_name(
            discovered
        )
        scopes: list[m.Infra.DocScope] = []
        for name in selected_names:
            scope = FlextInfraUtilitiesDocsScopeSelectionMixin._selected_scope(
                repository_root, name, project_by_name, output_dir
            )
            if scope is not None:
                scopes.append(scope)
        return tuple(scopes)

    @staticmethod
    def _selected_scope(
        repository_root: Path,
        name: str,
        project_by_name: dict[str, m.Infra.ProjectInfo],
        output_dir: Path | str,
    ) -> m.Infra.DocScope | None:
        """Build one selected scope from discovery or a local path."""
        selected = project_by_name.get(name)
        if selected is not None:
            return FlextInfraUtilitiesDocsScopeSelectionMixin._doc_scope(
                project=selected, output_dir=output_dir
            )
        return FlextInfraUtilitiesDocsScopeSelectionMixin._optional_path_scope(
            repository_root, name, output_dir
        )

    @staticmethod
    def _project_by_name(
        discovered: t.SequenceOf[m.Infra.ProjectInfo],
    ) -> dict[str, m.Infra.ProjectInfo]:
        """Index discovered projects by canonical and directory names."""
        project_by_name: dict[str, m.Infra.ProjectInfo] = {}
        for project in discovered:
            project_by_name.setdefault(project.name, project)
            project_by_name.setdefault(project.path.name, project)
        return project_by_name

    @staticmethod
    def _optional_path_scope(
        repository_root: Path, name: str, output_dir: Path | str
    ) -> m.Infra.DocScope | None:
        """Build a selected path scope when it is a local pyproject project."""
        project_root = (repository_root / name).resolve()
        if not (
            project_root.is_dir()
            and (project_root / c.Infra.PYPROJECT_FILENAME).is_file()
        ):
            return None
        return FlextInfraUtilitiesDocsScopeSelectionMixin._governed_scope(
            project_root, output_dir
        )

    @staticmethod
    def _doc_scope(
        *, project: m.Infra.ProjectInfo, output_dir: Path | str
    ) -> m.Infra.DocScope:
        """Build one canonical docs scope model."""
        resolved = project.path.resolve()
        return m.Infra.DocScope(
            name=project.name,
            path=resolved,
            report_dir=(resolved / output_dir).resolve(),
            project_class=project.project_class,
            package_name=project.package_name,
        )

    @staticmethod
    def _governed_scope(project_root: Path, output_dir: Path | str) -> m.Infra.DocScope:
        """Build docs scope for a governed project root."""
        payload = FlextInfraUtilitiesDocsScope.project_payload(project_root)
        docs_meta = FlextInfraUtilitiesDocsScope.docs_meta_from_payload(payload)
        project_name = FlextInfraUtilitiesDocsScope.project_name_from_payload(
            project_root, payload
        )
        return m.Infra.DocScope(
            name=project_name,
            path=project_root,
            report_dir=(project_root / output_dir).resolve(),
            project_class=FlextInfraUtilitiesDocsScope.classify_project_from_meta(
                project_name, docs_meta
            ),
            package_name=FlextInfraUtilitiesDocsScope.package_name_from_payload(
                project_root, payload, docs_meta
            ),
        )


__all__: list[str] = ["FlextInfraUtilitiesDocsScopeSelectionMixin"]

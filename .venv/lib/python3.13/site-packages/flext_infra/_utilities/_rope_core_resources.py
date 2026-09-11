"""Rope resource path helpers for FlextInfraUtilitiesRopeCore."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra._constants.namespace import FlextInfraConstantsNamespace
from flext_infra._constants.validate import FlextInfraConstantsSharedInfra
from flext_infra._utilities.rope_runtime import FlextInfraUtilitiesRopeRuntime

if TYPE_CHECKING:
    from flext_infra.typings import t


class FlextInfraUtilitiesRopeCoreResourcesMixin:
    """Filesystem-to-Rope resource helpers."""

    @staticmethod
    def get_resource_from_path(
        rope_project: t.Infra.RopeProject, file_path: Path
    ) -> t.Infra.RopeResource | None:
        """Return rope File for a filesystem Path, or None if outside project."""
        try:
            root_real_path = getattr(
                getattr(rope_project, "root", None), "real_path", None
            )
            if not isinstance(root_real_path, str):
                return None
            relative_path = str(file_path.resolve().relative_to(Path(root_real_path)))
            resource = rope_project.get_resource(relative_path)
            return (
                resource
                if FlextInfraUtilitiesRopeRuntime.is_resource(resource)
                else None
            )
        except (*FlextInfraUtilitiesRopeRuntime.rope_runtime_errors(), ValueError):
            return None

    @staticmethod
    def fetch_python_resource(
        rope_project: t.Infra.RopeProject,
        file_path: Path,
        *,
        skip_protected: bool = False,
        skip_settings: bool = False,
        skip_alias_modules: bool = False,
        skip_init_py: bool = False,
    ) -> t.Infra.RopeResource | None:
        """Resolve a Python source as a Rope resource, or None when skipped."""
        if not FlextInfraUtilitiesRopeCoreResourcesMixin._python_resource_allowed(
            file_path,
            skip_protected=skip_protected,
            skip_settings=skip_settings,
            skip_alias_modules=skip_alias_modules,
            skip_init_py=skip_init_py,
        ):
            return None
        return FlextInfraUtilitiesRopeCoreResourcesMixin.get_resource_from_path(
            rope_project, file_path
        )

    @staticmethod
    def _python_resource_allowed(
        file_path: Path,
        *,
        skip_protected: bool,
        skip_settings: bool,
        skip_alias_modules: bool,
        skip_init_py: bool,
    ) -> bool:
        """Return whether a path should be exposed as a Python Rope resource."""
        return (
            file_path.suffix == FlextInfraConstantsSharedInfra.EXT_PYTHON
            and not (
                skip_init_py
                and file_path.name == FlextInfraConstantsSharedInfra.INIT_PY
            )
            and not (
                skip_protected
                and file_path.name
                in FlextInfraConstantsNamespace.NAMESPACE_PROTECTED_FILES
            )
            and not (
                skip_settings
                and file_path.name
                in FlextInfraConstantsNamespace.NAMESPACE_SETTINGS_FILE_NAMES
            )
            and not (
                skip_alias_modules
                and file_path.stem
                in FlextInfraConstantsNamespace.NAMESPACE_CANONICAL_ALIAS_MODULE_STEMS
            )
        )

    @staticmethod
    def python_resources(
        rope_project: t.Infra.RopeProject,
    ) -> t.SequenceOf[t.Infra.RopeResource]:
        """Return Rope's already-filtered Python resources without a path roundtrip."""
        return tuple(
            sorted(
                (
                    resource
                    for resource in rope_project.get_python_files()
                    if FlextInfraUtilitiesRopeRuntime.is_resource(resource)
                ),
                key=operator.attrgetter("path"),
            )
        )

    @staticmethod
    def python_file_paths(rope_project: t.Infra.RopeProject) -> t.SequenceOf[Path]:
        """Return stable Python file paths for one Rope project."""
        resources = FlextInfraUtilitiesRopeCoreResourcesMixin.python_resources(
            rope_project
        )
        return tuple(
            sorted(
                (
                    file_path
                    for resource in resources
                    if (
                        file_path
                        := FlextInfraUtilitiesRopeCoreResourcesMixin.resource_file_path(
                            rope_project, resource
                        )
                    )
                    is not None
                ),
                key=Path.as_posix,
            )
        )

    @staticmethod
    def resource_file_path(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> Path | None:
        """Resolve one Rope resource back to an absolute filesystem path."""
        root_real_path = getattr(getattr(rope_project, "root", None), "real_path", None)
        if not isinstance(root_real_path, str):
            return None
        return Path(root_real_path, resource.path).resolve()


__all__: list[str] = ["FlextInfraUtilitiesRopeCoreResourcesMixin"]

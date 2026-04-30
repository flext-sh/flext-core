"""FlextUtilitiesProjectMetadata — Tier 4 SSOT utilities (flat on ``u.*``).

Static helpers for deriving project metadata and reading standardized
pyproject.toml tables. Every reader/writer across the monorepo routes
through this class via ``u.*`` (``u.derive_class_stem``,
``u.read_project_metadata``, etc.). Project-name derivation
(``pascalize``, ``derive_class_stem``) implementation is owned by
Tier 3 ``FlextModelsProjectMetadata``; this class re-exposes the
same callables on ``u.*`` via MRO without duplicating the logic.

Architecture: Tier 4 — depends on Tier 0 (_constants) and Tier 3 (_models).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import tomllib
from collections.abc import Mapping
from functools import cache
from importlib.metadata import PackageNotFoundError, metadata, packages_distributions
from pathlib import Path
from types import MappingProxyType

from flext_core import FlextTypes as t
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm


class FlextUtilitiesProjectMetadata(mpm):
    """SSOT utilities for project-metadata derivation and pyproject reads.

    Inherits from Tier 3 ``FlextModelsProjectMetadata`` so name-derivation
    static methods (``pascalize``, ``derive_class_stem``) flow through the
    real Python MRO — callable as ``u.pascalize`` / ``u.derive_class_stem``
    via ``FlextUtilities`` composition without local wrappers.
    """

    @staticmethod
    @cache
    def load_pyproject_toml(
        root: Path,
    ) -> t.JsonMapping:
        """Load and return the parsed pyproject.toml under ``root``.

        Result is cached by ``root`` (``Path`` is hashable). The cache is
        process-global so every workspace reader — typed config in
        ``read_tool_flext_config``/``read_project_metadata`` and the
        flext-infra ``pyproject_payload`` validation layer — shares the
        same disk read; no parallel file I/O.
        """
        pyproject = root / cpm.PYPROJECT_FILENAME
        if not pyproject.is_file():
            msg = f"{cpm.PYPROJECT_FILENAME} not found under {root}"
            raise FileNotFoundError(msg)
        with pyproject.open("rb") as stream:
            validated_payload: t.JsonMapping = t.json_mapping_adapter().validate_python(
                tomllib.load(stream)
            )
            return validated_payload

    @staticmethod
    def read_project_metadata(root: Path) -> mpm.ProjectMetadata:
        """Read canonical project metadata from a project's pyproject.toml."""
        data = FlextUtilitiesProjectMetadata.load_pyproject_toml(root)
        project_raw = data.get("project")
        project: dict[str, t.JsonValue] = (
            dict(project_raw) if isinstance(project_raw, Mapping) else {}
        )
        if "name" not in project:
            msg = f"{root}: missing [project].name in pyproject.toml"
            raise ValueError(msg)
        if "version" not in project:
            msg = f"{root}: missing [project].version in pyproject.toml"
            raise ValueError(msg)
        return mpm.PyprojectProject.model_validate(project).to_metadata(root)

    @staticmethod
    def derive_project_constants(root: Path) -> mpm.ProjectConstants:
        """Derive package constants from a project's pyproject.toml."""
        metadata = FlextUtilitiesProjectMetadata.read_project_metadata(root)
        return FlextUtilitiesProjectMetadata.read_project_constants(metadata.name, root)

    @staticmethod
    def _distribution_name(package_name: str) -> str:
        """Resolve the installed distribution for an import package name."""
        root_name = package_name.split(".", 1)[0]
        for candidate in packages_distributions().get(root_name, ()):
            return candidate
        candidate = root_name.replace("_", "-")
        try:
            metadata(candidate)
        except PackageNotFoundError as exc:
            msg = f"installed distribution for package {root_name!r} was not found"
            raise RuntimeError(msg) from exc
        return candidate

    @staticmethod
    def _package_name(distribution_name: str) -> str:
        """Return the import package name for an installed distribution."""
        return distribution_name.replace("-", "_")

    @staticmethod
    def read_project_constants(
        package_name: str,
        root: Path | None = None,
    ) -> mpm.ProjectConstants:
        """Read package constants from installed metadata and generated lazy exports."""
        distribution_name = FlextUtilitiesProjectMetadata._distribution_name(
            FlextUtilitiesProjectMetadata._package_name(package_name)
        )
        import_name = FlextUtilitiesProjectMetadata._package_name(distribution_name)
        package = importlib.import_module(import_name)
        package_version = importlib.import_module(f"{import_name}.__version__")
        package_file = package.__file__
        if package_file is None:
            msg = f"package {import_name!r} has no __file__"
            raise RuntimeError(msg)
        package_root = root or Path(package_file).resolve().parents[2]
        return mpm.ProjectConstants(
            PACKAGE_NAME=package_version.__title__,
            PACKAGE_VERSION=package_version.__version__,
            PACKAGE_LICENSE=package_version.__license__,
            PACKAGE_URL=package_version.__url__,
            PACKAGE_AUTHORS=tuple(
                author.strip()
                for author in package_version.__author__.split(",")
                if author.strip()
            ),
            PACKAGE_ROOT=package_root,
            PYTHON_PACKAGE_NAME=FlextUtilitiesProjectMetadata._package_name(
                distribution_name
            ),
            CLASS_STEM=mpm.derive_class_stem(distribution_name),
        )

    @staticmethod
    def read_tool_flext_config(root: Path) -> mpm.ProjectToolFlext:
        """Read ``[tool.flext.*]`` tables from pyproject.toml."""
        data = FlextUtilitiesProjectMetadata.load_pyproject_toml(root)
        tool_raw = data.get("tool")
        tool: t.JsonMapping = (
            dict(tool_raw) if isinstance(tool_raw, Mapping) else MappingProxyType({})
        )
        tool_flext_raw = tool.get("flext")
        tool_flext: t.JsonMapping = (
            dict(tool_flext_raw)
            if isinstance(tool_flext_raw, Mapping)
            else MappingProxyType({})
        )
        return mpm.ProjectToolFlext.model_validate(tool_flext)

    @staticmethod
    def compose_namespace_config(root: Path) -> mpm.ProjectNamespaceConfig:
        """Build the effective namespace config for a project."""
        meta = FlextUtilitiesProjectMetadata.read_project_metadata(root)
        cfg = FlextUtilitiesProjectMetadata.read_tool_flext_config(root)
        return mpm.ProjectNamespaceConfig(
            project_name=meta.name,
            enabled=cfg.namespace.enabled,
            scan_dirs=cfg.namespace.scan_dirs,
            include_dynamic_dirs=cfg.namespace.include_dynamic_dirs,
            alias_parent_sources={
                **dict(cpm.UNIVERSAL_ALIAS_PARENT_SOURCES),
                **dict(cfg.namespace.alias_parent_sources),
            },
        )

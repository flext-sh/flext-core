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
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType, ModuleType

from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core._typings.base import FlextTypingBase as tb

from .project_metadata_part_02 import (
    FlextUtilitiesProjectMetadata as FlextUtilitiesProjectMetadataPart02,
)


class FlextUtilitiesProjectMetadata(FlextUtilitiesProjectMetadataPart02):
    @staticmethod
    def _discover_eager_alias_parent_sources(
        import_name: str,
        package: ModuleType,
    ) -> dict[str, str]:
        """Discover canonical one-letter aliases exported directly by package module."""
        eager_parent_sources: dict[str, str] = {}
        published_aliases = getattr(package, "__all__", ())
        if not isinstance(published_aliases, (tuple, list, frozenset, set)):
            return eager_parent_sources
        for alias in published_aliases:
            if not isinstance(alias, str) or len(alias) != 1 or not alias.islower():
                continue
            exported = vars(package).get(alias)
            module_name = getattr(exported, "__module__", "")
            if not isinstance(module_name, str) or not module_name:
                continue
            parent_source = module_name.split(".", 1)[0]
            if parent_source == import_name:
                continue
            if parent_source.startswith("_"):
                continue
            eager_parent_sources[alias] = parent_source
        return eager_parent_sources

    @staticmethod
    def _scan_directories(package_root: Path) -> tb.StrSequence:
        """Derive workspace scan directories from existing project folders."""
        candidates = ("docs", "examples", "scripts", "src", "tests")
        scan_dirs = tuple(name for name in candidates if (package_root / name).is_dir())
        if not scan_dirs:
            msg = f"no scan directories found under {package_root}"
            raise RuntimeError(msg)
        return scan_dirs

    @staticmethod
    @cache
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
        alias_metadata = FlextUtilitiesProjectMetadata.read_lazy_alias_metadata(
            distribution_name
        )
        alias_to_suffix = {
            item.alias: item.suffix for item in alias_metadata if item.suffix
        }
        facade_aliases = frozenset(item.alias for item in alias_metadata if item.facade)
        facade_modules = frozenset(
            Path(item.module_path.replace(".", "/")).name
            for item in alias_metadata
            if item.facade
        )
        eager_parent_sources = FlextUtilitiesProjectMetadata._discover_eager_alias_parent_sources(
            import_name, package
        )
        parent_sources = {
            item.alias: item.parent_source
            for item in alias_metadata
            if item.parent_source != import_name
        }
        parent_sources.update(
            {
                alias: parent_source
                for alias, parent_source in eager_parent_sources.items()
                if alias not in parent_sources
            }
        )
        scan_dirs = FlextUtilitiesProjectMetadata._scan_directories(package_root)
        tier_facade_prefix = {
            directory: (
                FlextUtilitiesProjectMetadata.derive_class_stem(distribution_name)
                if directory == "src"
                else f"{FlextUtilitiesProjectMetadata.pascalize(directory)}"
                f"{FlextUtilitiesProjectMetadata.derive_class_stem(distribution_name)}"
            )
            for directory in scan_dirs
        }
        tier_sub_namespace = {
            directory: (
                ""
                if directory == "src"
                else FlextUtilitiesProjectMetadata.pascalize(directory)
            )
            for directory in scan_dirs
        }
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
            CLASS_STEM=FlextUtilitiesProjectMetadata.derive_class_stem(
                distribution_name
            ),
            ALIAS_TO_SUFFIX=MappingProxyType(alias_to_suffix),
            RUNTIME_ALIAS_NAMES=frozenset(
                {item.alias for item in alias_metadata} | set(parent_sources),
            ),
            FACADE_ALIAS_NAMES=facade_aliases,
            FACADE_MODULE_NAMES=facade_modules,
            UNIVERSAL_ALIAS_PARENT_SOURCES=MappingProxyType(parent_sources),
            TIER_FACADE_PREFIX=MappingProxyType(tier_facade_prefix),
            SCAN_DIRECTORIES=scan_dirs,
            TIER_SUB_NAMESPACE=MappingProxyType(tier_sub_namespace),
            PYPROJECT_FILENAME=cpm.PYPROJECT_FILENAME,
        )

    @staticmethod
    def read_tool_flext_config(root: Path) -> mpm.ProjectToolFlext:
        """Read ``[tool.flext.*]`` tables from pyproject.toml."""
        data = FlextUtilitiesProjectMetadata.load_pyproject_toml(root)
        tool_raw = data.get("tool")
        tool: tb.JsonMapping = (
            dict(tool_raw) if isinstance(tool_raw, Mapping) else MappingProxyType({})
        )
        tool_flext_raw = tool.get("flext")
        tool_flext: tb.JsonMapping = (
            dict(tool_flext_raw)
            if isinstance(tool_flext_raw, Mapping)
            else MappingProxyType({})
        )
        project_tool_flext: mpm.ProjectToolFlext = mpm.ProjectToolFlext.model_validate(
            tool_flext
        )
        return project_tool_flext

    @staticmethod
    def compose_namespace_config(root: Path) -> mpm.ProjectNamespaceConfig:
        """Build the effective namespace config for a project."""
        meta = FlextUtilitiesProjectMetadata.read_project_metadata(root)
        cfg = FlextUtilitiesProjectMetadata.read_tool_flext_config(root)
        constants = FlextUtilitiesProjectMetadata.read_project_constants(meta.name)
        sources = dict(cfg.namespace.alias_parent_sources)
        unknown = set(sources) - set(constants.RUNTIME_ALIAS_NAMES)
        if unknown:
            msg = f"unknown alias(es): {sorted(unknown)}"
            raise ValueError(msg)
        for alias, canonical in constants.UNIVERSAL_ALIAS_PARENT_SOURCES.items():
            if alias in sources and sources[alias] != canonical:
                msg = (
                    f"cannot override universal alias {alias!r}: "
                    f"must remain {canonical!r}"
                )
                raise ValueError(msg)
        return mpm.ProjectNamespaceConfig(
            project_name=meta.name,
            enabled=cfg.namespace.enabled,
            scan_dirs=cfg.namespace.scan_dirs or constants.SCAN_DIRECTORIES,
            include_dynamic_dirs=cfg.namespace.include_dynamic_dirs,
            alias_parent_sources=MappingProxyType({
                **dict(constants.UNIVERSAL_ALIAS_PARENT_SOURCES),
                **sources,
            }),
        )

    @staticmethod
    def derive_project_constants(root: Path) -> mpm.ProjectConstants:
        """Derive package constants directly from a project's pyproject.toml."""
        return FlextUtilitiesProjectMetadata.read_project_constants(
            FlextUtilitiesProjectMetadata.read_project_metadata(root).name,
            root=root,
        )


__all__: list[str] = ["FlextUtilitiesProjectMetadata"]

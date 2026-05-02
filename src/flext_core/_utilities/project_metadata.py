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

import ast
import importlib
import tomllib
from collections.abc import Mapping
from functools import cache
from importlib.metadata import PackageNotFoundError, metadata, packages_distributions
from importlib.util import find_spec
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import override

from flext_core import FlextTypes as t
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core.lazy import normalize_lazy_imports


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
        """Derive package constants directly from a project's pyproject.toml."""
        return FlextUtilitiesProjectMetadata.read_project_constants(
            FlextUtilitiesProjectMetadata.read_project_metadata(root).name,
            root=root,
        )

    @staticmethod
    @cache
    def _distribution_name(package_name: str) -> str:
        """Resolve the installed distribution for an import package name."""
        root_name = package_name.split(".", 1)[0]
        candidate = root_name.replace("_", "-")
        try:
            metadata(candidate)
            return candidate
        except PackageNotFoundError:
            for distribution_name in packages_distributions().get(root_name, ()):
                return distribution_name
        msg = f"installed distribution for package {root_name!r} was not found"
        raise RuntimeError(msg)

    @staticmethod
    def _package_name(distribution_name: str) -> str:
        """Return the import package name for an installed distribution."""
        return distribution_name.replace("-", "_")

    @staticmethod
    def _module_class_names(module_path: str) -> tuple[str, ...]:
        """Read class names from a module source without importing it."""
        spec = find_spec(module_path)
        if spec is None or spec.origin is None:
            msg = f"module spec for {module_path!r} was not found"
            raise RuntimeError(msg)
        source_path = Path(spec.origin)
        if not source_path.is_file():
            msg = f"module source for {module_path!r} was not found"
            raise RuntimeError(msg)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        return tuple(
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        )

    @staticmethod
    def _class_stem_from_lazy(
        package_name: str,
        package: ModuleType,
    ) -> str:
        """Derive class stem from generated package lazy exports."""
        lazy_map = FlextUtilitiesProjectMetadata._normalized_lazy_imports(
            package_name,
            package,
        )
        constants_module = f"{package_name}.constants"
        for export_name, entry in lazy_map.items():
            module_path = entry if isinstance(entry, str) else entry[0]
            if module_path == constants_module and export_name.endswith("Constants"):
                return export_name.removesuffix("Constants")
        for class_name in FlextUtilitiesProjectMetadata._module_class_names(
            constants_module
        ):
            if class_name.endswith("Constants"):
                return class_name.removesuffix("Constants")
        msg = f"constants class stem for package {package_name!r} was not found"
        raise RuntimeError(msg)

    @staticmethod
    @override
    def derive_class_stem(project_name: str) -> str:
        """Derive the installed package class stem from generated lazy exports."""
        if not project_name:
            msg = "empty project name"
            raise ValueError(msg)
        distribution_name = FlextUtilitiesProjectMetadata._distribution_name(
            FlextUtilitiesProjectMetadata._package_name(project_name)
        )
        import_name = FlextUtilitiesProjectMetadata._package_name(distribution_name)
        package = importlib.import_module(import_name)
        return FlextUtilitiesProjectMetadata._class_stem_from_lazy(
            import_name,
            package,
        )

    @staticmethod
    def _normalized_lazy_imports(
        package_name: str,
        package: ModuleType,
    ) -> dict[str, str | tuple[str, str]]:
        """Read generated lazy imports through the shared lazy normalizer."""
        raw = vars(package).get("_LAZY_IMPORTS")
        normalized = normalize_lazy_imports(package_name, raw)
        if not normalized:
            msg = f"package {package_name!r} has no generated lazy imports"
            raise RuntimeError(msg)
        return normalized

    @staticmethod
    def read_lazy_alias_metadata(package_name: str) -> tuple[mpm.LazyAliasMetadata, ...]:
        """Return alias metadata derived from installed generated lazy exports."""
        distribution_name = FlextUtilitiesProjectMetadata._distribution_name(
            FlextUtilitiesProjectMetadata._package_name(package_name)
        )
        import_name = FlextUtilitiesProjectMetadata._package_name(distribution_name)
        package = importlib.import_module(import_name)
        lazy_map = FlextUtilitiesProjectMetadata._normalized_lazy_imports(
            import_name,
            package,
        )
        class_stem = FlextUtilitiesProjectMetadata._class_stem_from_lazy(
            import_name,
            package,
        )
        result: list[mpm.LazyAliasMetadata] = []
        for export_name, entry in lazy_map.items():
            if len(export_name) != 1 or not export_name.islower():
                continue
            module_path = entry if isinstance(entry, str) else entry[0]
            parent_source = module_path.split(".", 1)[0]
            target_name = export_name if isinstance(entry, str) else entry[1]
            suffix = target_name.removeprefix(class_stem)
            result.append(
                mpm.LazyAliasMetadata(
                    alias=export_name,
                    module_path=module_path,
                    parent_source=parent_source,
                    suffix=suffix,
                    facade=parent_source == import_name,
                )
            )
        if not result:
            msg = f"package {import_name!r} exposes no runtime aliases"
            raise RuntimeError(msg)
        return tuple(sorted(result, key=lambda item: item.alias))

    @staticmethod
    def _scan_directories(package_root: Path) -> tuple[str, ...]:
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
        facade_aliases = frozenset(
            item.alias for item in alias_metadata if item.facade
        )
        facade_modules = frozenset(
            Path(item.module_path.replace(".", "/")).name
            for item in alias_metadata
            if item.facade
        )
        parent_sources = {
            item.alias: item.parent_source
            for item in alias_metadata
            if item.parent_source != import_name
        }
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
            RUNTIME_ALIAS_NAMES=frozenset(item.alias for item in alias_metadata),
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
            alias_parent_sources=cfg.namespace.alias_parent_sources,
        )

"""FlextUtilitiesProjectMetadata — Tier 4 SSOT utilities (flat on ``u.*``).

Static helpers for deriving project metadata and reading standardized
pyproject.toml tables. Every reader/writer across the monorepo routes
through this class via ``u.*`` (``u.derive_class_stem``,
``u.read_project_metadata``, etc.). Never hand-roll name derivation or
pyproject parsing — duplication is forbidden.

Architecture: Tier 4 — depends on Tier 0 (_constants) and Tier 3 (_models).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import tomllib
from collections.abc import Mapping
from functools import cache
from importlib.metadata import PackageNotFoundError, metadata, packages_distributions
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

from flext_core import FlextTypes as t
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core.lazy import normalize_lazy_imports


class FlextUtilitiesProjectMetadata:
    """SSOT utilities for project-metadata derivation and pyproject reads.

    Inherited into ``FlextUtilities`` via MRO so each static method is
    callable as ``u.pascalize`` / ``u.derive_class_stem`` /
    ``u.read_project_metadata`` / etc. (flat access).
    """

    PYPROJECT_FILENAME: ClassVar[str] = cpm.PYPROJECT_FILENAME

    @staticmethod
    def pascalize(slug: str) -> str:
        """Simple kebab/snake → PascalCase (no project-name override lookup).

        Use this when the input is a Python package segment or an
        arbitrary identifier — NOT a full kebab-case project name.
        For project names use ``derive_class_stem`` which additionally
        applies ``SPECIAL_NAME_OVERRIDES`` (flext → FlextRoot etc.).
        """
        parts = slug.replace("-", "_").split("_")
        return "".join(part[:1].upper() + part[1:] for part in parts if part)

    @staticmethod
    def derive_class_stem(project_name: str) -> str:
        """Return the canonical PascalCase class stem for a project name.

        Accepts either kebab-case (``flext-core``) or snake-case
        (``flext_core``) — inputs are normalized before the
        ``SPECIAL_NAME_OVERRIDES`` lookup so that the SSOT is robust to
        both Python package names and pyproject project names.
        """
        if not project_name:
            msg = "empty project name"
            raise ValueError(msg)
        if "-" not in project_name and "_" not in project_name:
            return FlextUtilitiesProjectMetadata.pascalize(project_name)
        distribution_name = FlextUtilitiesProjectMetadata._distribution_name(
            FlextUtilitiesProjectMetadata._package_name(project_name)
        )
        import_name = FlextUtilitiesProjectMetadata._package_name(distribution_name)
        package = importlib.import_module(import_name)
        lazy_imports = normalize_lazy_imports(
            package.__name__,
            getattr(package, "_LAZY_IMPORTS"),
        )
        entry = lazy_imports["c"]
        module_path = entry if isinstance(entry, str) else entry[0]
        return FlextUtilitiesProjectMetadata._class_prefix_from_module(
            module_path,
            "Constants",
        )

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
        pyproject = root / FlextUtilitiesProjectMetadata.PYPROJECT_FILENAME
        if not pyproject.is_file():
            msg = (
                f"{FlextUtilitiesProjectMetadata.PYPROJECT_FILENAME} "
                f"not found under {root}"
            )
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
    @cache
    def _module_class_names(module_path: str) -> tuple[str, ...]:
        """Read class names from a Python module without importing it."""
        spec = importlib.util.find_spec(module_path)
        if spec is None or spec.origin is None:
            msg = f"module spec for {module_path!r} was not found"
            raise RuntimeError(msg)
        source = Path(spec.origin).read_text(encoding="utf-8")
        tree = ast.parse(source)
        return tuple(node.name for node in tree.body if isinstance(node, ast.ClassDef))

    @staticmethod
    def _class_prefix_from_module(module_path: str, suffix: str) -> str:
        """Derive a class prefix from a module class ending with ``suffix``."""
        for class_name in FlextUtilitiesProjectMetadata._module_class_names(
            module_path
        ):
            if class_name.endswith(suffix):
                return class_name.removesuffix(suffix)
        msg = f"module {module_path!r} exposes no class ending with {suffix!r}"
        raise RuntimeError(msg)

    @staticmethod
    def _alias_suffix(alias: str, module_path: str, class_stem: str) -> str:
        """Derive an alias suffix from the generated runtime alias target."""
        _ = alias
        for class_name in FlextUtilitiesProjectMetadata._module_class_names(
            module_path
        ):
            for prefix in (class_stem, "Flext"):
                if not class_name.startswith(prefix):
                    continue
                suffix = class_name.removeprefix(prefix).removesuffix("Base")
                if suffix:
                    return suffix
        msg = f"cannot derive alias suffix for {alias!r} from {module_path!r}"
        raise RuntimeError(msg)

    @staticmethod
    @cache
    def read_lazy_alias_metadata(
        package_name: str,
    ) -> tuple[mpm.LazyAliasMetadata, ...]:
        """Read normalized alias metadata from an installed package ``_LAZY_IMPORTS``."""
        import_name = FlextUtilitiesProjectMetadata._package_name(package_name)
        package = importlib.import_module(import_name)
        distribution_name = FlextUtilitiesProjectMetadata._distribution_name(
            import_name
        )
        class_stem = FlextUtilitiesProjectMetadata.derive_class_stem(distribution_name)
        lazy_imports = normalize_lazy_imports(
            package.__name__,
            getattr(package, "_LAZY_IMPORTS"),
        )
        rows: list[mpm.LazyAliasMetadata] = []
        facade_suffixes = {"Constants", "Models", "Protocols", "Types", "Utilities"}
        for alias, entry in lazy_imports.items():
            if len(alias) != 1 or not alias.islower():
                continue
            module_path = entry if isinstance(entry, str) else entry[0]
            parent_source = module_path.split(".", 1)[0]
            suffix = (
                next(
                    row.suffix
                    for row in FlextUtilitiesProjectMetadata.read_lazy_alias_metadata(
                        parent_source
                    )
                    if row.alias == alias
                )
                if module_path == parent_source and parent_source != package.__name__
                else FlextUtilitiesProjectMetadata._alias_suffix(
                    alias,
                    module_path,
                    class_stem,
                )
            )
            rows.append(
                mpm.LazyAliasMetadata(
                    alias=alias,
                    module=module_path,
                    parent_source=parent_source,
                    suffix=suffix,
                    facade=suffix in facade_suffixes
                    and parent_source == package.__name__,
                )
            )
        return tuple(rows)

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
        aliases = FlextUtilitiesProjectMetadata.read_lazy_alias_metadata(
            distribution_name
        )
        package_file = package.__file__
        if package_file is None:
            msg = f"package {import_name!r} has no __file__"
            raise RuntimeError(msg)
        package_root = root or Path(package_file).resolve().parents[2]
        scan_dirs = tuple(
            name
            for name in ("src", "tests", "examples", "scripts", "docs")
            if (package_root / name).exists()
        )
        tier_prefix = MappingProxyType({
            name: (
                f"{FlextUtilitiesProjectMetadata.pascalize(name)}{FlextUtilitiesProjectMetadata.derive_class_stem(distribution_name)}"
                if name != "src"
                else FlextUtilitiesProjectMetadata.derive_class_stem(distribution_name)
            )
            for name in scan_dirs
        })
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
            ALIAS_TO_SUFFIX=MappingProxyType({
                row.alias: row.suffix for row in aliases
            }),
            RUNTIME_ALIAS_NAMES=frozenset(row.alias for row in aliases),
            FACADE_ALIAS_NAMES=frozenset(row.alias for row in aliases if row.facade),
            FACADE_MODULE_NAMES=frozenset(
                row.module.rsplit(".", 1)[-1] for row in aliases if row.facade
            ),
            UNIVERSAL_ALIAS_PARENT_SOURCES=MappingProxyType({
                row.alias: row.parent_source
                for row in aliases
                if not row.facade and row.parent_source != import_name
            }),
            TIER_FACADE_PREFIX=tier_prefix,
            SCAN_DIRECTORIES=scan_dirs,
            TIER_SUB_NAMESPACE=MappingProxyType({
                name: tier_prefix[name].removesuffix(
                    FlextUtilitiesProjectMetadata.derive_class_stem(distribution_name)
                )
                for name in scan_dirs
            }),
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
        constants = FlextUtilitiesProjectMetadata.read_project_constants(meta.name)
        return mpm.ProjectNamespaceConfig(
            project_name=meta.name,
            enabled=cfg.namespace.enabled,
            scan_dirs=cfg.namespace.scan_dirs,
            include_dynamic_dirs=cfg.namespace.include_dynamic_dirs,
            alias_parent_sources={
                **dict(constants.UNIVERSAL_ALIAS_PARENT_SOURCES),
                **dict(cfg.namespace.alias_parent_sources),
            },
        )

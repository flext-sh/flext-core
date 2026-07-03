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
from types import ModuleType

from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm
from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.typeadapters import FlextTypesTypeAdapters as ta
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
    ) -> tb.JsonMapping:
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
            # No PEP 526 local annotation: beartype.claw resolves local-variable
            # alias hints in the wrong namespace (JsonValue unresolvable here).
            parsed = tomllib.load(stream)
            validated: tb.JsonMapping = ta.json_mapping_adapter().validate_python(
                parsed
            )
            return validated

    @staticmethod
    def read_project_metadata(root: Path) -> mpm.ProjectMetadata:
        """Read canonical project metadata from a project's pyproject.toml."""
        data = FlextUtilitiesProjectMetadata.load_pyproject_toml(root)
        project_raw = data.get("project")
        project: tb.MutableJsonMapping = (
            dict(project_raw) if isinstance(project_raw, Mapping) else {}
        )
        if "name" not in project:
            msg = f"{root}: missing [project].name in pyproject.toml"
            raise ValueError(msg)
        if "version" not in project:
            msg = f"{root}: missing [project].version in pyproject.toml"
            raise ValueError(msg)
        pyproject_project: mpm.PyprojectProject = mpm.PyprojectProject.model_validate(
            project,
        )
        return pyproject_project.to_metadata(root)

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
    @cache
    def _module_runtime(module_path: str) -> ModuleType:
        """Import and cache one module for runtime metadata introspection."""
        try:
            return importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError) as exc:
            msg = f"module {module_path!r} could not be imported"
            raise RuntimeError(msg) from exc

    @staticmethod
    @cache
    def _module_class_names(module_path: str) -> tb.StrSequence:
        """Return locally declared class names from one runtime module."""
        module = FlextUtilitiesProjectMetadata._module_runtime(module_path)
        return tuple(
            name
            for name, value in vars(module).items()
            if isinstance(value, type) and value.__module__ == module.__name__
        )

    @staticmethod
    def _normalized_lazy_imports(
        package_name: str,
        package: ModuleType,
    ) -> tb.LazyImportDict:
        """Read generated lazy imports through the shared lazy normalizer."""
        raw = vars(package).get("_LAZY_IMPORTS")
        normalized: tb.LazyImportDict = normalize_lazy_imports(package_name, raw)
        if not normalized:
            msg = f"package {package_name!r} has no generated lazy imports"
            raise RuntimeError(msg)
        return normalized

    @staticmethod
    def _alias_target_name(
        module_path: str,
        export_name: str,
        class_stem: str,
    ) -> str:
        """Return the facade class exported by a runtime alias module."""
        module = FlextUtilitiesProjectMetadata._module_runtime(module_path)
        exported = vars(module).get(export_name)
        if isinstance(exported, type) and exported.__module__ == module.__name__:
            return exported.__name__
        candidates = tuple(
            class_name
            for class_name in FlextUtilitiesProjectMetadata._module_class_names(
                module_path,
            )
            if class_name.startswith(class_stem)
        )
        if len(candidates) != 1:
            msg = (
                f"expected one {class_stem!r} facade class in {module_path!r}; "
                f"found {candidates!r}"
            )
            raise RuntimeError(msg)
        return candidates[0]


__all__: list[str] = ["FlextUtilitiesProjectMetadata"]

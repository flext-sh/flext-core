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
from typing import TYPE_CHECKING

from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm

from .project_metadata_part_01 import (
    FlextUtilitiesProjectMetadata as FlextUtilitiesProjectMetadataPart01,
)

if TYPE_CHECKING:
    from types import ModuleType

    from flext_core._typings.base import FlextTypingBase as tb


class FlextUtilitiesProjectMetadata(FlextUtilitiesProjectMetadataPart01):
    @staticmethod
    def _alias_suffix_from_entry(
        export_name: str,
        entry: tb.LazyImportEntry,
        import_name: str,
        class_stem: str,
    ) -> str:
        """Resolve alias suffix against the package that owns the target."""
        if not isinstance(entry, str):
            return entry[1].removeprefix(class_stem)
        module_path = entry
        parent_source = module_path.split(".", 1)[0]
        parent_package = importlib.import_module(parent_source)
        parent_lazy_map = FlextUtilitiesProjectMetadata._normalized_lazy_imports(
            parent_source,
            parent_package,
        )
        if parent_source == import_name:
            local_targets = tuple(
                sibling_name
                for sibling_name, sibling_entry in parent_lazy_map.items()
                if sibling_name.startswith(class_stem)
                and (
                    sibling_entry
                    if isinstance(sibling_entry, str)
                    else sibling_entry[0]
                )
                == module_path
            )
            if len(local_targets) == 1:
                return local_targets[0].removeprefix(class_stem)
            return FlextUtilitiesProjectMetadata._alias_target_name(
                module_path,
                export_name,
                class_stem,
            ).removeprefix(class_stem)
        parent_entry = parent_lazy_map.get(export_name)
        if parent_entry is None:
            msg = f"alias {export_name!r} not found in parent package {parent_source!r}"
            raise RuntimeError(msg)
        parent_class_stem = FlextUtilitiesProjectMetadata._class_stem_from_lazy(
            parent_source,
            parent_package,
        )
        return FlextUtilitiesProjectMetadata._alias_suffix_from_entry(
            export_name,
            parent_entry,
            parent_source,
            parent_class_stem,
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
            constants_module,
        ):
            if class_name.endswith("Constants"):
                return class_name.removesuffix("Constants")
        msg = f"constants class stem for package {package_name!r} was not found"
        raise RuntimeError(msg)

    @staticmethod
    def read_lazy_alias_metadata(
        package_name: str,
    ) -> tuple[mpm.LazyAliasMetadata, ...]:
        """Return alias metadata derived from installed generated lazy exports."""
        distribution_name = FlextUtilitiesProjectMetadata._distribution_name(
            FlextUtilitiesProjectMetadata._package_name(package_name),
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
            suffix = FlextUtilitiesProjectMetadata._alias_suffix_from_entry(
                export_name,
                entry,
                import_name,
                class_stem,
            )
            result.append(
                mpm.LazyAliasMetadata(
                    alias=export_name,
                    module_path=module_path,
                    parent_source=parent_source,
                    suffix=suffix,
                    facade=parent_source == import_name,
                ),
            )
        if not result:
            msg = f"package {import_name!r} exposes no runtime aliases"
            raise RuntimeError(msg)
        return tuple(sorted(result, key=lambda item: item.alias))


__all__: list[str] = ["FlextUtilitiesProjectMetadata"]

"""Project metadata namespace configuration helpers."""

from __future__ import annotations

import typing as _typing
from types import MappingProxyType

from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm

from .project_metadata_part_03 import (
    FlextUtilitiesProjectMetadata as FlextUtilitiesProjectMetadataPart03,
)

if _typing.TYPE_CHECKING:
    from pathlib import Path

    from flext_core._typings.base import FlextTypingBase as tb


class FlextUtilitiesProjectMetadata(FlextUtilitiesProjectMetadataPart03):
    @staticmethod
    def read_tool_flext_config(root: Path) -> mpm.ProjectToolFlext:
        """Read ``[tool.flext.*]`` tables from pyproject.toml."""
        data = FlextUtilitiesProjectMetadata.load_pyproject_toml(root)
        tool_raw = data.get("tool", {})
        # No PEP 526 local annotations: beartype.claw resolves local-variable
        # alias hints in the wrong namespace (JsonValue unresolvable here).
        tool = _typing.cast("tb.JsonMapping", tool_raw)
        tool_flext = _typing.cast("tb.JsonMapping", tool.get("flext", {}))
        return mpm.ProjectToolFlext.model_validate(dict(tool_flext))

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

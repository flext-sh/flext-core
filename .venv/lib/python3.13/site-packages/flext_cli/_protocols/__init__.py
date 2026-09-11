# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Protocols package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _base_parts as _base_parts
    from .base import FlextCliProtocolsBase
    from .config import FlextCliProtocolsConfig
    from .domain import FlextCliProtocolsDomain
    from .framework import FlextCliProtocolsFramework
    from .pipeline import FlextCliProtocolsPipeline
    from .xlsx import FlextCliProtocolsXlsx
    from .xlsx_archive import FlextCliProtocolsXlsxArchive
    from .xlsx_rules import FlextCliProtocolsXlsxRules
    from .xlsx_snapshot import FlextCliProtocolsXlsxSnapshot
    from .xlsx_snapshot_structure import FlextCliProtocolsXlsxSnapshotStructure
    from .xlsx_workbook import FlextCliProtocolsXlsxWorkbook
__all__: tuple[str, ...] = (
    "FlextCliProtocolsBase",
    "FlextCliProtocolsConfig",
    "FlextCliProtocolsDomain",
    "FlextCliProtocolsFramework",
    "FlextCliProtocolsPipeline",
    "FlextCliProtocolsXlsx",
    "FlextCliProtocolsXlsxArchive",
    "FlextCliProtocolsXlsxRules",
    "FlextCliProtocolsXlsxSnapshot",
    "FlextCliProtocolsXlsxSnapshotStructure",
    "FlextCliProtocolsXlsxWorkbook",
    "_base_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._base_parts": ("_base_parts",),
            ".base": ("FlextCliProtocolsBase",),
            ".config": ("FlextCliProtocolsConfig",),
            ".domain": ("FlextCliProtocolsDomain",),
            ".framework": ("FlextCliProtocolsFramework",),
            ".pipeline": ("FlextCliProtocolsPipeline",),
            ".xlsx": ("FlextCliProtocolsXlsx",),
            ".xlsx_archive": ("FlextCliProtocolsXlsxArchive",),
            ".xlsx_rules": ("FlextCliProtocolsXlsxRules",),
            ".xlsx_snapshot": ("FlextCliProtocolsXlsxSnapshot",),
            ".xlsx_snapshot_structure": ("FlextCliProtocolsXlsxSnapshotStructure",),
            ".xlsx_workbook": ("FlextCliProtocolsXlsxWorkbook",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)

# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Models. Xlsx package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .xlsx_archive import FlextCliModelsXlsxArchive
    from .xlsx_cells import FlextCliModelsXlsxCells
    from .xlsx_layout import FlextCliModelsXlsxLayout
    from .xlsx_recalc import FlextCliModelsXlsxRecalc
    from .xlsx_rules import FlextCliModelsXlsxRules
    from .xlsx_snapshot import FlextCliModelsXlsxSnapshot
    from .xlsx_style_catalog import FlextCliModelsXlsxStyleCatalog
    from .xlsx_style_fills import FlextCliModelsXlsxStyleFills
    from .xlsx_style_primitives import FlextCliModelsXlsxStylePrimitives
    from .xlsx_styles import FlextCliModelsXlsxStyles
    from .xlsx_tables import FlextCliModelsXlsxTables
    from .xlsx_validation import FlextCliModelsXlsxValidation
    from .xlsx_workbook import FlextCliModelsXlsxWorkbook
__all__: tuple[str, ...] = (
    "FlextCliModelsXlsxArchive",
    "FlextCliModelsXlsxCells",
    "FlextCliModelsXlsxLayout",
    "FlextCliModelsXlsxRecalc",
    "FlextCliModelsXlsxRules",
    "FlextCliModelsXlsxSnapshot",
    "FlextCliModelsXlsxStyleCatalog",
    "FlextCliModelsXlsxStyleFills",
    "FlextCliModelsXlsxStylePrimitives",
    "FlextCliModelsXlsxStyles",
    "FlextCliModelsXlsxTables",
    "FlextCliModelsXlsxValidation",
    "FlextCliModelsXlsxWorkbook",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".xlsx_archive": ("FlextCliModelsXlsxArchive",),
            ".xlsx_cells": ("FlextCliModelsXlsxCells",),
            ".xlsx_layout": ("FlextCliModelsXlsxLayout",),
            ".xlsx_recalc": ("FlextCliModelsXlsxRecalc",),
            ".xlsx_rules": ("FlextCliModelsXlsxRules",),
            ".xlsx_snapshot": ("FlextCliModelsXlsxSnapshot",),
            ".xlsx_style_catalog": ("FlextCliModelsXlsxStyleCatalog",),
            ".xlsx_style_fills": ("FlextCliModelsXlsxStyleFills",),
            ".xlsx_style_primitives": ("FlextCliModelsXlsxStylePrimitives",),
            ".xlsx_styles": ("FlextCliModelsXlsxStyles",),
            ".xlsx_tables": ("FlextCliModelsXlsxTables",),
            ".xlsx_validation": ("FlextCliModelsXlsxValidation",),
            ".xlsx_workbook": ("FlextCliModelsXlsxWorkbook",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)

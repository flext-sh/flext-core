# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _base as _base
    from . import _xlsx as _xlsx
    from ._xlsx.xlsx_archive import FlextCliModelsXlsxArchive
    from ._xlsx.xlsx_cells import FlextCliModelsXlsxCells
    from ._xlsx.xlsx_layout import FlextCliModelsXlsxLayout
    from ._xlsx.xlsx_recalc import FlextCliModelsXlsxRecalc
    from ._xlsx.xlsx_rules import FlextCliModelsXlsxRules
    from ._xlsx.xlsx_snapshot import FlextCliModelsXlsxSnapshot
    from ._xlsx.xlsx_style_catalog import FlextCliModelsXlsxStyleCatalog
    from ._xlsx.xlsx_style_fills import FlextCliModelsXlsxStyleFills
    from ._xlsx.xlsx_style_primitives import FlextCliModelsXlsxStylePrimitives
    from ._xlsx.xlsx_styles import FlextCliModelsXlsxStyles
    from ._xlsx.xlsx_tables import FlextCliModelsXlsxTables
    from ._xlsx.xlsx_validation import FlextCliModelsXlsxValidation
    from ._xlsx.xlsx_workbook import FlextCliModelsXlsxWorkbook
    from .base import FlextCliModelsBase
    from .config import FlextCliConfigModels
    from .docx import FlextCliModelsDocx
    from .docx_document import FlextCliModelsDocxDocument
    from .docx_styles import FlextCliModelsDocxStyles
    from .pipeline import FlextCliModelsPipeline
    from .pptx import FlextCliModelsPptx
    from .pptx_presentation import FlextCliModelsPptxPresentation
    from .rules import FlextCliModelsRules
    from .template import FlextCliModelsTemplate
    from .xlsx import FlextCliModelsXlsx
__all__: tuple[str, ...] = (
    "FlextCliConfigModels",
    "FlextCliModelsBase",
    "FlextCliModelsDocx",
    "FlextCliModelsDocxDocument",
    "FlextCliModelsDocxStyles",
    "FlextCliModelsPipeline",
    "FlextCliModelsPptx",
    "FlextCliModelsPptxPresentation",
    "FlextCliModelsRules",
    "FlextCliModelsTemplate",
    "FlextCliModelsXlsx",
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
    "_base",
    "_xlsx",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._base": ("_base",),
            "._xlsx": ("_xlsx",),
            "._xlsx.xlsx_archive": ("FlextCliModelsXlsxArchive",),
            "._xlsx.xlsx_cells": ("FlextCliModelsXlsxCells",),
            "._xlsx.xlsx_layout": ("FlextCliModelsXlsxLayout",),
            "._xlsx.xlsx_recalc": ("FlextCliModelsXlsxRecalc",),
            "._xlsx.xlsx_rules": ("FlextCliModelsXlsxRules",),
            "._xlsx.xlsx_snapshot": ("FlextCliModelsXlsxSnapshot",),
            "._xlsx.xlsx_style_catalog": ("FlextCliModelsXlsxStyleCatalog",),
            "._xlsx.xlsx_style_fills": ("FlextCliModelsXlsxStyleFills",),
            "._xlsx.xlsx_style_primitives": ("FlextCliModelsXlsxStylePrimitives",),
            "._xlsx.xlsx_styles": ("FlextCliModelsXlsxStyles",),
            "._xlsx.xlsx_tables": ("FlextCliModelsXlsxTables",),
            "._xlsx.xlsx_validation": ("FlextCliModelsXlsxValidation",),
            "._xlsx.xlsx_workbook": ("FlextCliModelsXlsxWorkbook",),
            ".base": ("FlextCliModelsBase",),
            ".config": ("FlextCliConfigModels",),
            ".docx": ("FlextCliModelsDocx",),
            ".docx_document": ("FlextCliModelsDocxDocument",),
            ".docx_styles": ("FlextCliModelsDocxStyles",),
            ".pipeline": ("FlextCliModelsPipeline",),
            ".pptx": ("FlextCliModelsPptx",),
            ".pptx_presentation": ("FlextCliModelsPptxPresentation",),
            ".rules": ("FlextCliModelsRules",),
            ".template": ("FlextCliModelsTemplate",),
            ".xlsx": ("FlextCliModelsXlsx",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)

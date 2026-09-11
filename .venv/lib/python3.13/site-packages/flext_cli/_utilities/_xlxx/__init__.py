# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Cli. Utilities. Xlxx package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .xlsx_addresses import FlextCliUtilitiesXlsxAddresses
    from .xlsx_archive import FlextCliUtilitiesXlsxArchive
    from .xlsx_archive_checks import FlextCliUtilitiesXlsxArchiveChecks
    from .xlsx_cells import FlextCliUtilitiesXlsxCells
    from .xlsx_conditional import FlextCliUtilitiesXlsxConditional
    from .xlsx_defined_name_values import FlextCliUtilitiesXlsxDefinedNameValues
    from .xlsx_formula_codec import FlextCliUtilitiesXlsxFormulaCodec
    from .xlsx_layout import FlextCliUtilitiesXlsxLayout
    from .xlsx_protection import FlextCliUtilitiesXlsxProtection
    from .xlsx_recalc import FlextCliUtilitiesXlsxRecalc
    from .xlsx_recalc_evidence import FlextCliUtilitiesXlsxRecalcEvidence
    from .xlsx_renderer import FlextCliUtilitiesXlsxRenderer
    from .xlsx_rules import FlextCliUtilitiesXlsxRules
    from .xlsx_snapshot import FlextCliUtilitiesXlsxSnapshot
    from .xlsx_snapshot_sheet import FlextCliUtilitiesXlsxSnapshotSheet
    from .xlsx_snapshot_structure import FlextCliUtilitiesXlsxSnapshotStructure
    from .xlsx_snapshot_values import FlextCliUtilitiesXlsxSnapshotValues
    from .xlsx_style_builders import FlextCliUtilitiesXlsxStyleBuilders
    from .xlsx_style_catalog import FlextCliUtilitiesXlsxStyleCatalog
    from .xlsx_style_codec import FlextCliUtilitiesXlsxStyleCodec
    from .xlsx_style_readers import FlextCliUtilitiesXlsxStyleReaders
    from .xlsx_tables import FlextCliUtilitiesXlsxTables
    from .xlsx_validations import FlextCliUtilitiesXlsxValidations
    from .xlsx_workbook_io import FlextCliUtilitiesXlsxWorkbookIo
    from .xlsx_workbook_plan import FlextCliUtilitiesXlsxWorkbookPlan
__all__: tuple[str, ...] = (
    "FlextCliUtilitiesXlsxAddresses",
    "FlextCliUtilitiesXlsxArchive",
    "FlextCliUtilitiesXlsxArchiveChecks",
    "FlextCliUtilitiesXlsxCells",
    "FlextCliUtilitiesXlsxConditional",
    "FlextCliUtilitiesXlsxDefinedNameValues",
    "FlextCliUtilitiesXlsxFormulaCodec",
    "FlextCliUtilitiesXlsxLayout",
    "FlextCliUtilitiesXlsxProtection",
    "FlextCliUtilitiesXlsxRecalc",
    "FlextCliUtilitiesXlsxRecalcEvidence",
    "FlextCliUtilitiesXlsxRenderer",
    "FlextCliUtilitiesXlsxRules",
    "FlextCliUtilitiesXlsxSnapshot",
    "FlextCliUtilitiesXlsxSnapshotSheet",
    "FlextCliUtilitiesXlsxSnapshotStructure",
    "FlextCliUtilitiesXlsxSnapshotValues",
    "FlextCliUtilitiesXlsxStyleBuilders",
    "FlextCliUtilitiesXlsxStyleCatalog",
    "FlextCliUtilitiesXlsxStyleCodec",
    "FlextCliUtilitiesXlsxStyleReaders",
    "FlextCliUtilitiesXlsxTables",
    "FlextCliUtilitiesXlsxValidations",
    "FlextCliUtilitiesXlsxWorkbookIo",
    "FlextCliUtilitiesXlsxWorkbookPlan",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".xlsx_addresses": ("FlextCliUtilitiesXlsxAddresses",),
            ".xlsx_archive": ("FlextCliUtilitiesXlsxArchive",),
            ".xlsx_archive_checks": ("FlextCliUtilitiesXlsxArchiveChecks",),
            ".xlsx_cells": ("FlextCliUtilitiesXlsxCells",),
            ".xlsx_conditional": ("FlextCliUtilitiesXlsxConditional",),
            ".xlsx_defined_name_values": ("FlextCliUtilitiesXlsxDefinedNameValues",),
            ".xlsx_formula_codec": ("FlextCliUtilitiesXlsxFormulaCodec",),
            ".xlsx_layout": ("FlextCliUtilitiesXlsxLayout",),
            ".xlsx_protection": ("FlextCliUtilitiesXlsxProtection",),
            ".xlsx_recalc": ("FlextCliUtilitiesXlsxRecalc",),
            ".xlsx_recalc_evidence": ("FlextCliUtilitiesXlsxRecalcEvidence",),
            ".xlsx_renderer": ("FlextCliUtilitiesXlsxRenderer",),
            ".xlsx_rules": ("FlextCliUtilitiesXlsxRules",),
            ".xlsx_snapshot": ("FlextCliUtilitiesXlsxSnapshot",),
            ".xlsx_snapshot_sheet": ("FlextCliUtilitiesXlsxSnapshotSheet",),
            ".xlsx_snapshot_structure": ("FlextCliUtilitiesXlsxSnapshotStructure",),
            ".xlsx_snapshot_values": ("FlextCliUtilitiesXlsxSnapshotValues",),
            ".xlsx_style_builders": ("FlextCliUtilitiesXlsxStyleBuilders",),
            ".xlsx_style_catalog": ("FlextCliUtilitiesXlsxStyleCatalog",),
            ".xlsx_style_codec": ("FlextCliUtilitiesXlsxStyleCodec",),
            ".xlsx_style_readers": ("FlextCliUtilitiesXlsxStyleReaders",),
            ".xlsx_tables": ("FlextCliUtilitiesXlsxTables",),
            ".xlsx_validations": ("FlextCliUtilitiesXlsxValidations",),
            ".xlsx_workbook_io": ("FlextCliUtilitiesXlsxWorkbookIo",),
            ".xlsx_workbook_plan": ("FlextCliUtilitiesXlsxWorkbookPlan",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)

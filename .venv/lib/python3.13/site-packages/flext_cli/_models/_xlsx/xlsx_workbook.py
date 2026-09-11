"""Aggregate plans and byte-boundary results for generic XLSX rendering."""

from __future__ import annotations

from typing import Annotated

from flext_core import m

from .xlsx_cells import FlextCliModelsXlsxCells
from .xlsx_layout import FlextCliModelsXlsxLayout
from .xlsx_rules import FlextCliModelsXlsxRules
from .xlsx_styles import FlextCliModelsXlsxStyles
from .xlsx_tables import FlextCliModelsXlsxTables


class FlextCliModelsXlsxWorkbook:
    """Immutable sheet, workbook, render request, and render result models."""

    # NOTE (multi-agent, mro-j2yt.1): consumers exchange only validated plans
    # and bytes; workbook implementation objects never cross this boundary.
    class XlsxSheetPlan(m.FrozenModel):
        name: Annotated[str, m.Field(min_length=1, description="Worksheet name.")]
        cells: tuple[FlextCliModelsXlsxCells.XlsxCellPlan, ...] = m.Field(
            default=(), strict=False, description="Concrete cell writes."
        )
        tables: tuple[FlextCliModelsXlsxTables.XlsxTablePlan, ...] = m.Field(
            default=(), strict=False, description="Concrete worksheet tables."
        )
        layout: FlextCliModelsXlsxLayout.XlsxSheetLayoutPlan = m.Field(
            description="Worksheet layout operations."
        )
        rules: FlextCliModelsXlsxRules.XlsxSheetRulesPlan = m.Field(
            description="Worksheet validation and protection operations."
        )

    class XlsxWorkbookPlan(m.FrozenModel):
        sheets: tuple[FlextCliModelsXlsxWorkbook.XlsxSheetPlan, ...] = m.Field(
            min_length=1, strict=False, description="Exact worksheet order."
        )
        defined_names: tuple[FlextCliModelsXlsxTables.XlsxDefinedNamePlan, ...] = (
            m.Field(default=(), strict=False, description="Workbook defined names.")
        )
        named_styles: tuple[FlextCliModelsXlsxStyles.XlsxNamedStyleSpec, ...] = m.Field(
            default=(), strict=False, description="Visual styles to register."
        )
        full_calculation_on_load: bool = m.Field(
            default=True, description="Require complete formula recalculation."
        )

    class XlsxRenderRequest(m.FrozenModel):
        template: (
            Annotated[
                bytes, m.Field(min_length=1, description="Formatting template bytes.")
            ]
            | None
        ) = m.Field(default=None, description="Optional source workbook.")
        plan: FlextCliModelsXlsxWorkbook.XlsxWorkbookPlan = m.Field(
            description="Validated workbook plan."
        )

    class XlsxRenderResult(m.FrozenModel):
        content: Annotated[
            bytes, m.Field(min_length=1, description="Rendered workbook bytes.")
        ]
        plan: FlextCliModelsXlsxWorkbook.XlsxWorkbookPlan = m.Field(
            description="Exact source plan."
        )


__all__: tuple[str, ...] = ("FlextCliModelsXlsxWorkbook",)

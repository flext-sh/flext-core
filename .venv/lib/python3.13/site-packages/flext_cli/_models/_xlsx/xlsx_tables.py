"""Typed table and defined-name declarations for XLSX rendering."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m

from .xlsx_cells import FlextCliModelsXlsxCells


class FlextCliModelsXlsxTables:
    """Immutable workbook table and defined-name plans."""

    # NOTE (multi-agent, mro-j2yt.1): table and name cardinality is supplied
    # by consumers; this generic boundary owns no document policy.
    class XlsxTablePlan(m.FrozenModel):
        name: Annotated[str, m.Field(min_length=1, description="Table name.")]
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Concrete table range."
        )
        style: Annotated[str, m.Field(min_length=1, description="Table style.")]
        show_first_column: bool = m.Field(
            default=False, description="Emphasize first column."
        )
        show_last_column: bool = m.Field(
            default=False, description="Emphasize last column."
        )
        show_row_stripes: bool = m.Field(
            default=True, description="Show alternating row stripes."
        )
        show_column_stripes: bool = m.Field(
            default=False, description="Show alternating column stripes."
        )

    class XlsxRangeDefinedNamePlan(m.FrozenModel):
        kind: Literal["range"] = m.Field(default="range", description="Name kind.")
        name: Annotated[str, m.Field(min_length=1, description="Defined name.")]
        sheet: Annotated[
            str, m.Field(min_length=1, description="Referenced worksheet.")
        ]
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Referenced cell range."
        )

    class XlsxFormulaDefinedNamePlan(m.FrozenModel):
        kind: Literal["formula"] = m.Field(default="formula", description="Name kind.")
        name: Annotated[str, m.Field(min_length=1, description="Defined name.")]
        expression: Annotated[
            str, m.Field(min_length=1, description="Defined-name expression.")
        ]

    type XlsxDefinedNamePlan = Annotated[
        XlsxRangeDefinedNamePlan | XlsxFormulaDefinedNamePlan,
        m.Field(discriminator="kind"),
    ]


__all__: tuple[str, ...] = ("FlextCliModelsXlsxTables",)

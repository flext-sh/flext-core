"""Typed cell declarations for generic XLSX rendering."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Annotated, Literal

from flext_cli import t
from flext_core import m


class FlextCliModelsXlsxCells:
    """Immutable cell coordinates, values, and write plans."""

    # NOTE (multi-agent, mro-j2yt.1): cells carry typed values from the
    # validated ingress to the external workbook egress without mappings.
    class XlsxCellAddress(m.FrozenModel):
        row: Annotated[int, m.Field(ge=1, description="One-based row index.")]
        column: Annotated[int, m.Field(ge=1, description="One-based column index.")]

    class XlsxCellRange(m.FrozenModel):
        first: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="Top-left range address."
        )
        last: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="Bottom-right range address."
        )

    class XlsxParseRangeRequest(m.FrozenModel):
        reference: Annotated[
            str,
            m.Field(min_length=1, description="Concrete A1 cell or range reference."),
        ]

    # mro-j2yt.1 (xlsx_reference_api): typed public reference formatting contract.
    class XlsxFormatReferenceRequest(m.FrozenModel):
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Validated cell range to format."
        )
        sheet: (
            Annotated[
                str, m.Field(min_length=1, description="Optional worksheet qualifier.")
            ]
            | None
        ) = None
        absolute: bool = m.Field(description="Render absolute row and column markers.")
        collapse_single_cell: bool = m.Field(
            description="Render equal bounds as one cell instead of a range."
        )

    class XlsxReference(m.FrozenModel):
        reference: Annotated[
            str, m.Field(min_length=1, description="Formatted Excel reference.")
        ]

    class XlsxBlankValue(m.FrozenModel):
        kind: Literal["blank"] = m.Field(default="blank", description="Value kind.")

    class XlsxTextValue(m.FrozenModel):
        kind: Literal["text"] = m.Field(default="text", description="Value kind.")
        value: str = m.Field(description="Cell text.")

    class XlsxIntegerValue(m.FrozenModel):
        kind: Literal["integer"] = m.Field(default="integer", description="Value kind.")
        value: int = m.Field(description="Cell integer.")

    class XlsxDecimalValue(m.FrozenModel):
        kind: Literal["decimal"] = m.Field(default="decimal", description="Value kind.")
        value: Annotated[
            Decimal, m.Field(strict=False, description="Cell decimal value.")
        ]

    class XlsxBooleanValue(m.FrozenModel):
        kind: Literal["boolean"] = m.Field(default="boolean", description="Value kind.")
        value: bool = m.Field(description="Cell boolean.")

    class XlsxDateValue(m.FrozenModel):
        kind: Literal["date"] = m.Field(default="date", description="Value kind.")
        value: Annotated[
            dt.date, m.Field(strict=False, description="Cell calendar date.")
        ]

    class XlsxDateTimeValue(m.FrozenModel):
        kind: Literal["datetime"] = m.Field(
            default="datetime", description="Value kind."
        )
        value: Annotated[
            t.NaiveDatetime,
            m.Field(strict=False, description="Timezone-naive cell date and time."),
        ]

    class XlsxFormulaValue(m.FrozenModel):
        kind: Literal["formula"] = m.Field(default="formula", description="Value kind.")
        value: Annotated[
            str, m.Field(min_length=2, pattern=r"^=", description="Excel formula.")
        ]

    type XlsxCellValue = Annotated[
        XlsxBlankValue
        | XlsxTextValue
        | XlsxIntegerValue
        | XlsxDecimalValue
        | XlsxBooleanValue
        | XlsxDateValue
        | XlsxDateTimeValue
        | XlsxFormulaValue,
        m.Field(discriminator="kind"),
    ]

    class XlsxCellPlan(m.FrozenModel):
        at: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="Destination address."
        )
        value: FlextCliModelsXlsxCells.XlsxCellValue = m.Field(
            description="Typed destination value."
        )
        style: Annotated[str, m.Field(min_length=1, description="Named visual style.")]


__all__: tuple[str, ...] = ("FlextCliModelsXlsxCells",)

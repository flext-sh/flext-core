"""Typed data-validation declarations for XLSX rendering."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m

from .xlsx_cells import FlextCliModelsXlsxCells


class FlextCliModelsXlsxValidation:
    """Immutable validation variants with explicit positive UI semantics."""

    # NOTE (multi-agent, mro-j2yt.1): show_dropdown is translated at the
    # openpyxl edge where its external meaning is inverted.
    class XlsxValidationMessages(m.FrozenModel):
        allow_blank: bool = m.Field(default=True, description="Allow blank cells.")
        show_dropdown: bool = m.Field(default=True, description="Show list selector.")
        show_input: bool = m.Field(default=True, description="Show input guidance.")
        input_title: str | None = m.Field(default=None, description="Input title.")
        input_message: str | None = m.Field(default=None, description="Input message.")
        show_error: bool = m.Field(default=True, description="Show validation error.")
        error_style: Literal["stop", "warning", "information"] | None = m.Field(
            default=None, description="Error behavior."
        )
        error_title: str | None = m.Field(default=None, description="Error title.")
        error_message: str | None = m.Field(default=None, description="Error message.")

    class XlsxInlineListSource(m.FrozenModel):
        kind: Literal["values"] = m.Field(default="values", description="Source kind.")
        values: tuple[str, ...] = m.Field(
            min_length=1, strict=False, description="Allowed literal values."
        )

    class XlsxFormulaListSource(m.FrozenModel):
        kind: Literal["formula"] = m.Field(
            default="formula", description="Source kind."
        )
        expression: Annotated[
            str, m.Field(min_length=1, description="List formula or name.")
        ]

    type XlsxListSource = Annotated[
        XlsxInlineListSource | XlsxFormulaListSource, m.Field(discriminator="kind")
    ]

    class XlsxUnaryComparison(m.FrozenModel):
        mode: Literal[
            "equal",
            "not_equal",
            "less_than",
            "less_or_equal",
            "greater_than",
            "greater_or_equal",
        ] = m.Field(description="Unary comparison operator.")
        expression: Annotated[
            str, m.Field(min_length=1, description="Comparison operand.")
        ]

    class XlsxRangeComparison(m.FrozenModel):
        mode: Literal["between", "not_between"] = m.Field(
            description="Range comparison operator."
        )
        minimum: Annotated[
            str, m.Field(min_length=1, description="Lower comparison operand.")
        ]
        maximum: Annotated[
            str, m.Field(min_length=1, description="Upper comparison operand.")
        ]

    type XlsxComparison = Annotated[
        XlsxUnaryComparison | XlsxRangeComparison, m.Field(discriminator="mode")
    ]

    class XlsxListValidationPlan(m.FrozenModel):
        kind: Literal["list"] = m.Field(default="list", description="Rule kind.")
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Validated cell range."
        )
        source: FlextCliModelsXlsxValidation.XlsxListSource = m.Field(
            description="Allowed-value source."
        )
        messages: FlextCliModelsXlsxValidation.XlsxValidationMessages = m.Field(
            description="Validation UI behavior."
        )

    class XlsxComparisonValidationPlan(m.FrozenModel):
        kind: Literal["whole", "decimal", "date", "time", "text_length"] = m.Field(
            description="Rule kind."
        )
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Validated cell range."
        )
        comparison: FlextCliModelsXlsxValidation.XlsxComparison = m.Field(
            description="Typed comparison."
        )
        messages: FlextCliModelsXlsxValidation.XlsxValidationMessages = m.Field(
            description="Validation UI behavior."
        )

    class XlsxCustomValidationPlan(m.FrozenModel):
        kind: Literal["custom"] = m.Field(default="custom", description="Rule kind.")
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Validated cell range."
        )
        expression: Annotated[
            str, m.Field(min_length=1, description="Custom validation formula.")
        ]
        messages: FlextCliModelsXlsxValidation.XlsxValidationMessages = m.Field(
            description="Validation UI behavior."
        )

    type XlsxDataValidationPlan = Annotated[
        XlsxListValidationPlan
        | XlsxComparisonValidationPlan
        | XlsxCustomValidationPlan,
        m.Field(discriminator="kind"),
    ]


__all__: tuple[str, ...] = ("FlextCliModelsXlsxValidation",)

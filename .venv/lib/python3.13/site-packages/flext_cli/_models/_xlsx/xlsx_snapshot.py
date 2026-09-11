"""Typed semantic snapshot declarations for generic XLSX workbooks."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m

from .xlsx_cells import FlextCliModelsXlsxCells


class FlextCliModelsXlsxSnapshot:
    """Immutable workbook evidence for semantic parity comparisons."""

    # NOTE (multi-agent, mro-j2yt.1): snapshots retain typed cell values and
    # ordered workbook structure without exposing vendor objects or mappings.
    class XlsxSnapshotRequest(m.FrozenModel):
        source: Annotated[
            bytes, m.Field(min_length=1, description="Source workbook bytes.")
        ]
        data_only: bool = m.Field(
            default=False, description="Read cached values instead of formulas."
        )

    class XlsxCellSnapshot(m.FrozenModel):
        coordinate: Annotated[
            str, m.Field(min_length=2, description="A1 cell coordinate.")
        ]
        position: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="One-based row and column position."
        )
        value: FlextCliModelsXlsxCells.XlsxCellValue = m.Field(
            description="Typed value from the selected workbook view."
        )
        formula: (
            Annotated[
                str,
                m.Field(
                    min_length=2,
                    pattern=r"^=",
                    description="Original formula when this is a formula cell.",
                ),
            ]
            | None
        ) = None
        style_name: (
            Annotated[str, m.Field(min_length=1, description="Applied named style.")]
            | None
        ) = m.Field(
            default=None,
            description="Named style when the external workbook registers its label.",
        )
        style_id: Annotated[
            int, m.Field(ge=0, description="Source workbook style identifier.")
        ]
        number_format: Annotated[
            str, m.Field(min_length=1, description="Applied number format.")
        ]
        locked: bool = m.Field(description="Cell protection locked state.")
        hidden: bool = m.Field(description="Cell protection hidden state.")

    class XlsxTableSnapshot(m.FrozenModel):
        name: Annotated[str, m.Field(min_length=1, description="Table name.")]
        reference: Annotated[
            str, m.Field(min_length=2, description="Table range reference.")
        ]
        style_name: str | None = m.Field(
            default=None, description="Optional table style name."
        )

    class XlsxDefinedNameSnapshot(m.FrozenModel):
        name: Annotated[str, m.Field(min_length=1, description="Defined name.")]
        expression: Annotated[
            str, m.Field(min_length=1, description="Defined-name expression.")
        ]
        kind: Annotated[
            str, m.Field(min_length=1, description="Defined-name value kind.")
        ]
        sheet_position: (
            Annotated[
                int, m.Field(ge=0, description="Zero-based local sheet position.")
            ]
            | None
        ) = None
        hidden: bool | None = m.Field(
            default=None, description="Defined-name hidden state when declared."
        )

    class XlsxDefinedNameValuesRequest(m.FrozenModel):
        source: Annotated[
            bytes, m.Field(min_length=1, description="Source workbook bytes.")
        ]
        name: Annotated[
            str, m.Field(min_length=1, description="Defined name to resolve.")
        ]

    class XlsxDefinedNameCell(m.FrozenModel):
        sheet: Annotated[
            str, m.Field(min_length=1, description="Owning worksheet name.")
        ]
        coordinate: Annotated[
            str, m.Field(min_length=2, description="A1 cell coordinate.")
        ]
        value: FlextCliModelsXlsxCells.XlsxCellValue = m.Field(
            description="Typed cached value from the data-only workbook view."
        )

    class XlsxDefinedNameValuesResult(m.FrozenModel):
        name: Annotated[
            str, m.Field(min_length=1, description="Resolved defined name.")
        ]
        cells: tuple[FlextCliModelsXlsxSnapshot.XlsxDefinedNameCell, ...] = m.Field(
            min_length=1,
            strict=False,
            description="Ordered cached cell values for the defined-name extent.",
        )

    class XlsxRowDimensionSnapshot(m.FrozenModel):
        position: Annotated[int, m.Field(ge=1, description="One-based row index.")]
        size: Annotated[float, m.Field(gt=0, description="Explicit row height.")] | None
        hidden: bool = m.Field(description="Hidden row state.")
        outline_level: (
            Annotated[int, m.Field(ge=0, le=8, description="Row outline level.")] | None
        )

    class XlsxColumnDimensionSnapshot(m.FrozenModel):
        name: Annotated[str, m.Field(min_length=1, description="Column dimension key.")]
        first: Annotated[int, m.Field(ge=1, description="First column index.")]
        last: Annotated[int, m.Field(ge=1, description="Last column index.")]
        size: (
            Annotated[float, m.Field(gt=0, description="Explicit column width.")] | None
        )
        hidden: bool = m.Field(description="Hidden column state.")
        outline_level: (
            Annotated[int, m.Field(ge=0, le=8, description="Column outline level.")]
            | None
        )

    class XlsxSheetProtectionSnapshot(m.FrozenModel):
        enabled: bool = m.Field(description="Whether worksheet protection is active.")
        legacy_password_hash: (
            Annotated[
                str, m.Field(min_length=1, description="Stored legacy protection hash.")
            ]
            | None
        ) = None

    class XlsxSheetSnapshot(m.FrozenModel):
        name: Annotated[str, m.Field(min_length=1, description="Worksheet name.")]
        position: Annotated[
            int, m.Field(ge=1, description="One-based worksheet position.")
        ]
        state: Literal["visible", "hidden", "veryHidden"] = m.Field(
            description="Worksheet visibility state."
        )
        max_row: Annotated[int, m.Field(ge=1, description="Maximum occupied row.")]
        max_column: Annotated[
            int, m.Field(ge=1, description="Maximum occupied column.")
        ]
        cells: tuple[FlextCliModelsXlsxSnapshot.XlsxCellSnapshot, ...] = m.Field(
            default=(), strict=False, description="Ordered semantic cells."
        )
        tables: tuple[FlextCliModelsXlsxSnapshot.XlsxTableSnapshot, ...] = m.Field(
            default=(), strict=False, description="Ordered worksheet tables."
        )
        row_dimensions: tuple[
            FlextCliModelsXlsxSnapshot.XlsxRowDimensionSnapshot, ...
        ] = m.Field(default=(), strict=False, description="Explicit row dimensions.")
        column_dimensions: tuple[
            FlextCliModelsXlsxSnapshot.XlsxColumnDimensionSnapshot, ...
        ] = m.Field(default=(), strict=False, description="Explicit column dimensions.")
        merged_ranges: tuple[str, ...] = m.Field(
            default=(), strict=False, description="Ordered merged ranges."
        )
        freeze_pane: str | None = m.Field(
            default=None, description="Optional frozen-pane coordinate."
        )
        auto_filter: str | None = m.Field(
            default=None, description="Optional auto-filter range."
        )
        protection: FlextCliModelsXlsxSnapshot.XlsxSheetProtectionSnapshot = m.Field(
            description="Vendor-independent worksheet protection evidence."
        )
        formula_count: Annotated[int, m.Field(ge=0, description="Formula cell count.")]
        literal_count: Annotated[
            int, m.Field(ge=0, description="Nonblank literal cell count.")
        ]
        data_validation_count: Annotated[
            int, m.Field(ge=0, description="Data-validation rule count.")
        ]
        conditional_format_count: Annotated[
            int, m.Field(ge=0, description="Conditional-format rule count.")
        ]
        merge_count: Annotated[int, m.Field(ge=0, description="Merged-range count.")]

    class XlsxWorkbookSnapshot(m.FrozenModel):
        data_only: bool = m.Field(description="Whether cells expose cached values.")
        sheets: tuple[FlextCliModelsXlsxSnapshot.XlsxSheetSnapshot, ...] = m.Field(
            min_length=1, strict=False, description="Exact worksheet order."
        )
        defined_names: tuple[
            FlextCliModelsXlsxSnapshot.XlsxDefinedNameSnapshot, ...
        ] = m.Field(default=(), strict=False, description="Ordered defined names.")
        named_styles: tuple[str, ...] = m.Field(
            default=(), strict=False, description="Registered named styles."
        )
        formula_count: Annotated[
            int, m.Field(ge=0, description="Workbook formula cell count.")
        ]
        literal_count: Annotated[
            int, m.Field(ge=0, description="Workbook nonblank literal cell count.")
        ]


__all__: tuple[str, ...] = ("FlextCliModelsXlsxSnapshot",)

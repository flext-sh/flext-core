"""Typed conditional-format and protection declarations."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m

from .xlsx_cells import FlextCliModelsXlsxCells
from .xlsx_validation import FlextCliModelsXlsxValidation


class FlextCliModelsXlsxRules:
    """Immutable worksheet formatting and protection operations."""

    # NOTE (multi-agent, mro-j2yt.1): visual styles and protection remain
    # orthogonal so style assignment cannot silently unlock cells.
    class XlsxContainsTextFormatPlan(m.FrozenModel):
        kind: Literal["contains_text"] = m.Field(
            default="contains_text", description="Format kind."
        )
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Formatted range."
        )
        text: Annotated[str, m.Field(min_length=1, description="Searched text.")]
        style: Annotated[str, m.Field(min_length=1, description="Named style.")]
        stop_if_true: bool = m.Field(default=False, description="Stop later rules.")

    class XlsxCellFormatPlan(m.FrozenModel):
        kind: Literal["cell"] = m.Field(default="cell", description="Format kind.")
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Formatted range."
        )
        comparison: FlextCliModelsXlsxValidation.XlsxComparison = m.Field(
            description="Cell comparison."
        )
        style: Annotated[str, m.Field(min_length=1, description="Named style.")]
        stop_if_true: bool = m.Field(default=False, description="Stop later rules.")

    class XlsxFormulaFormatPlan(m.FrozenModel):
        kind: Literal["formula"] = m.Field(
            default="formula", description="Format kind."
        )
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Formatted range."
        )
        expressions: tuple[str, ...] = m.Field(
            min_length=1, strict=False, description="Conditional formulas."
        )
        style: Annotated[str, m.Field(min_length=1, description="Named style.")]
        stop_if_true: bool = m.Field(default=False, description="Stop later rules.")

    type XlsxConditionalFormatPlan = Annotated[
        XlsxContainsTextFormatPlan | XlsxCellFormatPlan | XlsxFormulaFormatPlan,
        m.Field(discriminator="kind"),
    ]

    class XlsxCellProtectionPlan(m.FrozenModel):
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Protected cell range."
        )
        locked: bool = m.Field(description="Locked state.")
        hidden: bool = m.Field(description="Formula-hidden state.")

    class XlsxPlainProtectionCredential(m.FrozenModel):
        kind: Literal["plain"] = m.Field(default="plain", description="Secret kind.")
        value: Annotated[str, m.Field(min_length=1, description="Resolved secret.")]

    class XlsxLegacyProtectionCredential(m.FrozenModel):
        kind: Literal["legacy_hash"] = m.Field(
            default="legacy_hash", description="Secret kind."
        )
        value: Annotated[
            str, m.Field(min_length=1, description="Legacy worksheet hash.")
        ]

    type XlsxProtectionCredential = Annotated[
        XlsxPlainProtectionCredential | XlsxLegacyProtectionCredential,
        m.Field(discriminator="kind"),
    ]

    class XlsxProtectionPermissions(m.FrozenModel):
        allow_select_locked: bool = m.Field(
            default=False, description="Allow selecting locked cells."
        )
        allow_select_unlocked: bool = m.Field(
            default=True, description="Allow selecting unlocked cells."
        )
        allow_format_cells: bool = m.Field(
            default=False, description="Allow formatting cells."
        )
        allow_format_columns: bool = m.Field(
            default=False, description="Allow formatting columns."
        )
        allow_format_rows: bool = m.Field(
            default=False, description="Allow formatting rows."
        )
        allow_insert_columns: bool = m.Field(
            default=False, description="Allow inserting columns."
        )
        allow_insert_rows: bool = m.Field(
            default=False, description="Allow inserting rows."
        )
        allow_insert_hyperlinks: bool = m.Field(
            default=False, description="Allow inserting hyperlinks."
        )
        allow_delete_columns: bool = m.Field(
            default=False, description="Allow deleting columns."
        )
        allow_delete_rows: bool = m.Field(
            default=False, description="Allow deleting rows."
        )
        allow_sort: bool = m.Field(default=False, description="Allow sorting.")
        allow_auto_filter: bool = m.Field(default=False, description="Allow filtering.")
        allow_pivot_tables: bool = m.Field(
            default=False, description="Allow pivot tables."
        )
        allow_edit_objects: bool = m.Field(
            default=False, description="Allow editing objects."
        )
        allow_edit_scenarios: bool = m.Field(
            default=False, description="Allow editing scenarios."
        )

    class XlsxSheetProtectionPlan(m.FrozenModel):
        credential: FlextCliModelsXlsxRules.XlsxProtectionCredential | None = m.Field(
            default=None, description="Optional resolved protection credential."
        )
        permissions: FlextCliModelsXlsxRules.XlsxProtectionPermissions = m.Field(
            description="Allowed worksheet operations."
        )
        cells: tuple[FlextCliModelsXlsxRules.XlsxCellProtectionPlan, ...] = m.Field(
            default=(), strict=False, description="Explicit cell protection."
        )

    class XlsxSheetRulesPlan(m.FrozenModel):
        validations: tuple[FlextCliModelsXlsxValidation.XlsxDataValidationPlan, ...] = (
            m.Field(default=(), strict=False, description="Data validations.")
        )
        conditional_formats: tuple[
            FlextCliModelsXlsxRules.XlsxConditionalFormatPlan, ...
        ] = m.Field(default=(), strict=False, description="Conditional formats.")
        protection: FlextCliModelsXlsxRules.XlsxSheetProtectionPlan | None = m.Field(
            default=None, description="Optional worksheet protection."
        )


__all__: tuple[str, ...] = ("FlextCliModelsXlsxRules",)

"""Apply typed data-validation plans through openpyxl."""

from __future__ import annotations

from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.worksheet import Worksheet

from flext_cli import c, m, p, r, t

from .xlsx_addresses import FlextCliUtilitiesXlsxAddresses
from .xlsx_formula_codec import FlextCliUtilitiesXlsxFormulaCodec


class FlextCliUtilitiesXlsxValidations(FlextCliUtilitiesXlsxAddresses):
    """Translate validation variants and positive UI behavior at runtime."""

    # NOTE (multi-agent, mro-j2yt.1): showDropDown is the only inverted
    # external flag; caller models consistently express show_dropdown.
    @staticmethod
    def _comparison_operator(mode: str) -> t.Cli.XlsxComparisonOperator:
        match mode:
            case "between" | "equal":
                return mode
            case "not_between":
                return "notBetween"
            case "not_equal":
                return "notEqual"
            case "less_than":
                return "lessThan"
            case "less_or_equal":
                return "lessThanOrEqual"
            case "greater_than":
                return "greaterThan"
            case "greater_or_equal":
                return "greaterThanOrEqual"
            case _:
                msg = f"Unsupported comparison operator: {mode}"
                raise ValueError(msg)

    @staticmethod
    def _validation_type(kind: str) -> t.Cli.XlsxValidationType:
        match kind:
            case "custom" | "date" | "decimal" | "list" | "time" | "whole":
                return kind
            case "text_length":
                return "textLength"
            case _:
                msg = f"Unsupported validation type: {kind}"
                raise ValueError(msg)

    @staticmethod
    def _inline_formula(values: tuple[str, ...]) -> str:
        if any("," in value or "\n" in value or "\r" in value for value in values):
            msg = "Inline validation values cannot contain commas or newlines"
            raise ValueError(msg)
        escaped = tuple(value.replace('"', '""') for value in values)
        formula = f'"{",".join(escaped)}"'
        if len(formula) > c.Cli.XLSX_INLINE_VALIDATION_FORMULA_LIMIT:
            msg = "Inline validation formula exceeds 255 characters"
            raise ValueError(msg)
        return formula

    @classmethod
    def _data_validation(cls, plan: m.Cli.XlsxDataValidationPlan) -> DataValidation:
        messages = plan.messages
        formula1: str | None = None
        formula2: str | None = None
        operator: t.Cli.XlsxComparisonOperator | None = None
        if plan.kind == "list":
            formula1 = (
                cls._inline_formula(plan.source.values)
                if plan.source.kind == "values"
                else FlextCliUtilitiesXlsxFormulaCodec.storage_formula(
                    plan.source.expression
                )
            )
        elif plan.kind == "custom":
            formula1 = FlextCliUtilitiesXlsxFormulaCodec.storage_formula(
                plan.expression
            )
        else:
            comparison = plan.comparison
            operator = cls._comparison_operator(comparison.mode)
            if isinstance(comparison, m.Cli.XlsxRangeComparison):
                formula1 = FlextCliUtilitiesXlsxFormulaCodec.storage_formula(
                    comparison.minimum
                )
                formula2 = FlextCliUtilitiesXlsxFormulaCodec.storage_formula(
                    comparison.maximum
                )
            else:
                formula1 = FlextCliUtilitiesXlsxFormulaCodec.storage_formula(
                    comparison.expression
                )
        return DataValidation(
            type=cls._validation_type(plan.kind),
            formula1=formula1,
            formula2=formula2,
            operator=operator,
            allowBlank=messages.allow_blank,
            showDropDown=not messages.show_dropdown,
            showInputMessage=messages.show_input,
            promptTitle=messages.input_title,
            prompt=messages.input_message,
            showErrorMessage=messages.show_error,
            errorStyle=messages.error_style,
            errorTitle=messages.error_title,
            error=messages.error_message,
        )

    @classmethod
    def _apply_validations(
        cls, worksheet: Worksheet, plans: tuple[m.Cli.XlsxDataValidationPlan, ...]
    ) -> p.Result[bool]:
        try:
            for plan in plans:
                validation = cls._data_validation(plan)
                worksheet.add_data_validation(validation)
                validation.add(cls._range_ref(plan.area))
        except (TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[bool].fail(f"{c.Cli.XlsxError.RENDER_FAILED}: {detail}")
        return r[bool].ok(True)


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxValidations",)

"""Compose worksheet validation, conditional-format, and protection rules."""

from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from flext_cli import m, p, r

from .xlsx_conditional import FlextCliUtilitiesXlsxConditional
from .xlsx_protection import FlextCliUtilitiesXlsxProtection
from .xlsx_validations import FlextCliUtilitiesXlsxValidations


class FlextCliUtilitiesXlsxRules(
    FlextCliUtilitiesXlsxConditional,
    FlextCliUtilitiesXlsxProtection,
    FlextCliUtilitiesXlsxValidations,
):
    """Apply every rule family through one fail-loud worksheet boundary."""

    # NOTE (multi-agent, mro-j2yt.1): each stage retains the same plan models;
    # no dump, revalidation, or rule-specific transport is introduced.
    @classmethod
    def _apply_rules(
        cls, worksheet: Worksheet, plan: m.Cli.XlsxSheetRulesPlan
    ) -> p.Result[bool]:
        validations = cls._apply_validations(worksheet, plan.validations)
        if validations.failure:
            return r[bool].fail(validations.error or "Data validation failed")
        conditional = cls._apply_conditional_formats(
            worksheet, plan.conditional_formats
        )
        if conditional.failure:
            return r[bool].fail(conditional.error or "Conditional formatting failed")
        protection = cls._apply_protection(worksheet, plan.protection)
        if protection.failure:
            return r[bool].fail(protection.error or "Worksheet protection failed")
        return r[bool].ok(True)


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxRules",)

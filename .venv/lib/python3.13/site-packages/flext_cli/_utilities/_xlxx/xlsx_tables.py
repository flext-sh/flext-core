"""Typed table and defined-name application for XLSX workbooks."""

from __future__ import annotations

from openpyxl import Workbook
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.worksheet.worksheet import Worksheet

from flext_cli import c, m, p, r

from .xlsx_addresses import FlextCliUtilitiesXlsxAddresses
from .xlsx_formula_codec import FlextCliUtilitiesXlsxFormulaCodec


class FlextCliUtilitiesXlsxTables(FlextCliUtilitiesXlsxAddresses):
    """Create tables and workbook names from immutable plans."""

    # NOTE (multi-agent, mro-j2yt.1): table/name uniqueness and header validity
    # are checked before external objects mutate the workbook.
    @classmethod
    def _apply_tables(
        cls,
        worksheet: Worksheet,
        plans: tuple[m.Cli.XlsxTablePlan, ...],
        used_names: frozenset[str],
    ) -> p.Result[frozenset[str]]:
        try:
            return cls._apply_tables_unchecked(worksheet, plans, used_names)
        except (KeyError, TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[frozenset[str]].fail(f"{c.Cli.XlsxError.RENDER_FAILED}: {detail}")

    @classmethod
    def _apply_tables_unchecked(
        cls,
        worksheet: Worksheet,
        plans: tuple[m.Cli.XlsxTablePlan, ...],
        used_names: frozenset[str],
    ) -> p.Result[frozenset[str]]:
        names = used_names
        for plan in plans:
            if plan.name in names:
                return r[frozenset[str]].fail(
                    f"{c.Cli.XlsxError.DUPLICATE_TABLE}: {plan.name}"
                )
            for column in range(plan.area.first.column, plan.area.last.column + 1):
                header = worksheet.cell(plan.area.first.row, column).value
                if not isinstance(header, str) or not header:
                    return r[frozenset[str]].fail(
                        f"Invalid table header: {plan.name} column {column}"
                    )
            table = Table(displayName=plan.name, ref=cls._range_ref(plan.area))
            table.tableStyleInfo = TableStyleInfo(
                name=plan.style,
                showFirstColumn=plan.show_first_column,
                showLastColumn=plan.show_last_column,
                showRowStripes=plan.show_row_stripes,
                showColumnStripes=plan.show_column_stripes,
            )
            worksheet.add_table(table)
            names = names.union((plan.name,))
        return r[frozenset[str]].ok(names)

    @classmethod
    def _apply_defined_names(
        cls, workbook: Workbook, plans: tuple[m.Cli.XlsxDefinedNamePlan, ...]
    ) -> p.Result[bool]:
        names: frozenset[str] = frozenset()
        for plan in plans:
            if plan.name in names or plan.name in workbook.defined_names:
                return r[bool].fail(
                    f"{c.Cli.XlsxError.DUPLICATE_DEFINED_NAME}: {plan.name}"
                )
            if plan.kind == "range":
                if plan.sheet not in workbook.sheetnames:
                    return r[bool].fail(
                        f"{c.Cli.XlsxError.SHEET_MISSING}: {plan.sheet}"
                    )
                expression = (
                    f"{cls._sheet_ref(plan.sheet)}!{cls._absolute_range_ref(plan.area)}"
                )
            else:
                expression = FlextCliUtilitiesXlsxFormulaCodec.storage_formula(
                    plan.expression
                )
            workbook.defined_names.add(
                DefinedName(name=plan.name, attr_text=expression)
            )
            names = names.union((plan.name,))
        return r[bool].ok(True)


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxTables",)

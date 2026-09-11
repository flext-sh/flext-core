"""Generic model-driven XLSX renderer."""

from __future__ import annotations

from openpyxl import Workbook

from flext_cli import c, m, p, r

from .xlsx_cells import FlextCliUtilitiesXlsxCells
from .xlsx_layout import FlextCliUtilitiesXlsxLayout
from .xlsx_rules import FlextCliUtilitiesXlsxRules
from .xlsx_tables import FlextCliUtilitiesXlsxTables
from .xlsx_workbook_plan import FlextCliUtilitiesXlsxWorkbookPlan


class FlextCliUtilitiesXlsxRenderer(
    FlextCliUtilitiesXlsxRules,
    FlextCliUtilitiesXlsxWorkbookPlan,
    FlextCliUtilitiesXlsxCells,
    FlextCliUtilitiesXlsxTables,
    FlextCliUtilitiesXlsxLayout,
):
    """Render one immutable workbook plan into XLSX bytes."""

    # NOTE (multi-agent, mro-j2yt.1): Rules precedes WorkbookPlan so the shared
    # style spine linearizes Builders -> Readers -> WorkbookIo consistently.
    # NOTE (multi-agent, mro-j2yt.1): stage order is canonical and fail-loud;
    # later stages never run after an earlier mutation reports failure.
    @classmethod
    def _render_sheet(
        cls, workbook: Workbook, plan: m.Cli.XlsxSheetPlan, table_names: frozenset[str]
    ) -> p.Result[frozenset[str]]:
        if plan.name not in workbook.sheetnames:
            return r[frozenset[str]].fail(
                f"{c.Cli.XlsxError.SHEET_MISSING}: {plan.name}"
            )
        # mro-j47u (codex): workbook planning creates only Worksheet instances.
        worksheet = workbook[plan.name]
        cells = cls._apply_cells(
            worksheet, plan.cells, frozenset(workbook.named_styles)
        )
        if cells.failure:
            return r[frozenset[str]].fail(cells.error or "Cell rendering failed")
        tables = cls._apply_tables(worksheet, plan.tables, table_names)
        if tables.failure:
            return r[frozenset[str]].fail(tables.error or "Table rendering failed")
        layout = cls._apply_layout(worksheet, plan.layout)
        if layout.failure:
            return r[frozenset[str]].fail(layout.error or "Layout rendering failed")
        rules = cls._apply_rules(worksheet, plan.rules)
        if rules.failure:
            return r[frozenset[str]].fail(rules.error or "Rule rendering failed")
        return r[frozenset[str]].ok(tables.value)

    @classmethod
    def xlsx_render(
        cls, request: m.Cli.XlsxRenderRequest
    ) -> p.Result[m.Cli.XlsxRenderResult]:
        """Render typed sheets, names, styles, and rules into workbook bytes."""
        workbook_result = cls._workbook_for_request(request)
        if workbook_result.failure:
            return r[m.Cli.XlsxRenderResult].fail(
                workbook_result.error or str(c.Cli.XlsxError.RENDER_FAILED)
            )
        workbook = workbook_result.value
        table_names: frozenset[str] = frozenset()
        for sheet in request.plan.sheets:
            rendered = cls._render_sheet(workbook, sheet, table_names)
            if rendered.failure:
                return r[m.Cli.XlsxRenderResult].fail(
                    rendered.error or str(c.Cli.XlsxError.RENDER_FAILED)
                )
            table_names = rendered.value
        names = cls._apply_defined_names(workbook, request.plan.defined_names)
        if names.failure:
            return r[m.Cli.XlsxRenderResult].fail(
                names.error or "Defined-name rendering failed"
            )
        content = cls._serialize_workbook(workbook)
        if content.failure:
            return r[m.Cli.XlsxRenderResult].fail(
                content.error or str(c.Cli.XlsxError.SERIALIZE_FAILED)
            )
        return r[m.Cli.XlsxRenderResult].ok(
            m.Cli.XlsxRenderResult(content=content.value, plan=request.plan)
        )


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxRenderer",)

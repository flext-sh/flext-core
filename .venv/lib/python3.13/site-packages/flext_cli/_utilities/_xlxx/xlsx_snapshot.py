"""Public typed semantic snapshot operation for XLSX bytes."""

from __future__ import annotations


from flext_cli import m, p, r

from .xlsx_snapshot_sheet import FlextCliUtilitiesXlsxSnapshotSheet
from .xlsx_workbook_io import FlextCliUtilitiesXlsxWorkbookIo


class FlextCliUtilitiesXlsxSnapshot(
    FlextCliUtilitiesXlsxSnapshotSheet, FlextCliUtilitiesXlsxWorkbookIo
):
    """Expose vendor-independent workbook parity evidence."""

    # NOTE (multi-agent, mro-j2yt.1): the formula view owns structure and
    # counts; a second data-only view supplies cached values only when asked.
    @classmethod
    def xlsx_snapshot(
        cls, request: m.Cli.XlsxSnapshotRequest
    ) -> p.Result[m.Cli.XlsxWorkbookSnapshot]:
        """Inspect workbook bytes into one immutable semantic snapshot."""
        try:
            snapshot = cls._snapshot_workbook(request)
        except (TypeError, m.ValidationError, ValueError) as exc:
            return r[m.Cli.XlsxWorkbookSnapshot].fail(
                f"Workbook snapshot failed ({exc.__class__.__name__}): {exc}"
            )
        return r[m.Cli.XlsxWorkbookSnapshot].ok(snapshot)

    @classmethod
    def _snapshot_workbook(
        cls, request: m.Cli.XlsxSnapshotRequest
    ) -> m.Cli.XlsxWorkbookSnapshot:
        formula_workbook = cls._require_success(
            cls._load_workbook(request.source, data_only=False)
        )
        value_workbook = (
            cls._require_success(cls._load_workbook(request.source, data_only=True))
            if request.data_only
            else formula_workbook
        )
        if len(formula_workbook.worksheets) != len(value_workbook.worksheets):
            msg = "Formula and value workbook views have different sheet counts"
            raise ValueError(msg)
        sheets: tuple[m.Cli.XlsxSheetSnapshot, ...] = ()
        for position, (formula_sheet, value_sheet) in enumerate(
            zip(formula_workbook.worksheets, value_workbook.worksheets, strict=True),
            start=1,
        ):
            sheet = cls._require_success(
                cls._snapshot_sheet(formula_sheet, value_sheet, position=position)
            )
            sheets = (*sheets, sheet)
        defined_names = cls._require_success(cls._snapshot_names(formula_workbook))
        return m.Cli.XlsxWorkbookSnapshot(
            data_only=request.data_only,
            sheets=sheets,
            defined_names=defined_names,
            named_styles=tuple(formula_workbook.named_styles),
            formula_count=sum(item.formula_count for item in sheets),
            literal_count=sum(item.literal_count for item in sheets),
        )


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxSnapshot",)

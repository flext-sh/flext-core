"""Public typed defined-name value resolution for XLSX bytes."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from openpyxl.cell.cell import Cell
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.worksheet import Worksheet

from flext_cli import c, m, p, r

from .xlsx_snapshot_values import FlextCliUtilitiesXlsxSnapshotValues
from .xlsx_workbook_io import FlextCliUtilitiesXlsxWorkbookIo


class FlextCliUtilitiesXlsxDefinedNameValues(
    FlextCliUtilitiesXlsxSnapshotValues, FlextCliUtilitiesXlsxWorkbookIo
):
    """Resolve one workbook defined name to typed cached cell values."""

    # NOTE (mro-uz8p): consumers ask for a defined name and receive the cached
    # values of its cell extent; workbook parsing stays inside this boundary so
    # no consumer re-implements openpyxl range resolution.
    @classmethod
    def xlsx_defined_name_values(
        cls, request: m.Cli.XlsxDefinedNameValuesRequest
    ) -> p.Result[m.Cli.XlsxDefinedNameValuesResult]:
        """Read cached values for a defined name from data-only workbook bytes."""
        try:
            return cls._defined_name_values_unchecked(request)
        except (TypeError, m.ValidationError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.XlsxDefinedNameValuesResult].fail(
                f"{c.Cli.XlsxError.DEFINED_NAME_INVALID}: {detail}"
            )

    @classmethod
    def _defined_name_values_unchecked(
        cls, request: m.Cli.XlsxDefinedNameValuesRequest
    ) -> r[m.Cli.XlsxDefinedNameValuesResult]:
        workbook = cls._require_success(
            cls._load_workbook(request.source, data_only=True)
        )
        defined_name = workbook.defined_names.get(request.name)
        if not isinstance(defined_name, DefinedName):
            return r[m.Cli.XlsxDefinedNameValuesResult].fail(
                f"{c.Cli.XlsxError.DEFINED_NAME_MISSING}: {request.name}"
            )
        cells: tuple[m.Cli.XlsxDefinedNameCell, ...] = ()
        for sheet_title, coordinate in defined_name.destinations:
            resolved = cls._destination_cells(workbook[sheet_title], coordinate)
            if resolved.failure:
                return r[m.Cli.XlsxDefinedNameValuesResult].fail(
                    resolved.error or str(c.Cli.XlsxError.DEFINED_NAME_INVALID)
                )
            cells = (*cells, *resolved.value)
        if not cells:
            return r[m.Cli.XlsxDefinedNameValuesResult].fail(
                f"{c.Cli.XlsxError.DEFINED_NAME_INVALID}: "
                f"{request.name} resolves to no worksheet cells"
            )
        return r[m.Cli.XlsxDefinedNameValuesResult].ok(
            m.Cli.XlsxDefinedNameValuesResult(name=request.name, cells=cells)
        )

    @classmethod
    def _destination_cells(
        cls, worksheet: Worksheet, coordinate: str
    ) -> r[tuple[m.Cli.XlsxDefinedNameCell, ...]]:
        cells: tuple[m.Cli.XlsxDefinedNameCell, ...] = ()
        selection = worksheet[coordinate]
        for cell in cls._flatten_cells(selection):
            if not isinstance(cell, Cell):
                return r[tuple[m.Cli.XlsxDefinedNameCell, ...]].fail(
                    f"{c.Cli.XlsxError.DEFINED_NAME_INVALID}: "
                    f"unsupported cell at {coordinate}"
                )
            cell_value = cell.value
            if cell_value is not None and not isinstance(
                cell_value, (str, int, float, bool, Decimal, date, datetime)
            ):
                return r[tuple[m.Cli.XlsxDefinedNameCell, ...]].fail(
                    f"{c.Cli.XlsxError.DEFINED_NAME_INVALID}: "
                    f"{cell_value.__class__.__name__} at {cell.coordinate}"
                )
            value = cls._require_success(
                cls._snapshot_value(cell_value, formula_view=False)
            )
            cells = (
                *cells,
                m.Cli.XlsxDefinedNameCell(
                    sheet=worksheet.title, coordinate=cell.coordinate, value=value
                ),
            )
        return r[tuple[m.Cli.XlsxDefinedNameCell, ...]].ok(cells)

    @staticmethod
    def _flatten_cells(selection: object) -> tuple[object, ...]:
        if isinstance(selection, Cell):
            return (selection,)
        if isinstance(selection, tuple):
            flattened: tuple[object, ...] = ()
            for item in selection:
                flattened = (
                    *flattened,
                    *FlextCliUtilitiesXlsxDefinedNameValues._flatten_cells(item),
                )
            return flattened
        return (selection,)


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxDefinedNameValues",)

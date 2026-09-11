"""Typed cell extraction for XLSX semantic snapshots."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import TypeVar

from openpyxl.cell.cell import Cell, MergedCell
from openpyxl.worksheet.worksheet import Worksheet

from flext_cli import c, m, r, t


T = TypeVar("T")


class FlextCliUtilitiesXlsxSnapshotValues:
    """Convert vendor cell values into canonical discriminated models."""

    # NOTE (multi-agent, mro-j2yt.1): external cell primitives are validated
    # once into XlsxCellValue and vendor instances never leave this utility.
    @staticmethod
    def _snapshot_style_name(cell: Cell) -> str | None:
        try:
            style_name: str | None = cell.style
        except IndexError:
            return None
        else:
            return style_name

    @staticmethod
    def _require_success(result: r[T]) -> T:
        if result.failure:
            error = result.error
            if error is None:
                msg = "Failed FlextResult omitted its error"
                raise ValueError(msg)
            raise ValueError(error)
        value: T = result.value
        return value

    @staticmethod
    def _snapshot_value(
        value: t.Cli.XlsxCellPrimitive, *, formula_view: bool
    ) -> r[m.Cli.XlsxCellValue]:
        try:
            return FlextCliUtilitiesXlsxSnapshotValues._snapshot_value_unchecked(
                value, formula_view=formula_view
            )
        except (InvalidOperation, m.ValidationError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.XlsxCellValue].fail(
                f"{c.Cli.XlsxError.CELL_VALUE_UNSUPPORTED}: {detail}"
            )

    @staticmethod
    def _snapshot_value_unchecked(
        value: t.Cli.XlsxCellPrimitive, *, formula_view: bool
    ) -> r[m.Cli.XlsxCellValue]:
        if formula_view:
            if isinstance(value, str) and value.startswith("="):
                converted: m.Cli.XlsxCellValue = m.Cli.XlsxFormulaValue(value=value)
                return r[m.Cli.XlsxCellValue].ok(converted)
            return r[m.Cli.XlsxCellValue].fail(
                f"{c.Cli.XlsxError.CELL_VALUE_UNSUPPORTED}: "
                "formula cell has no formula expression"
            )
        if value is None:
            converted = m.Cli.XlsxBlankValue()
        elif isinstance(value, bool):
            converted = m.Cli.XlsxBooleanValue(value=value)
        elif isinstance(value, int):
            converted = m.Cli.XlsxIntegerValue(value=value)
        elif isinstance(value, (float, Decimal)):
            converted = m.Cli.XlsxDecimalValue(value=Decimal(str(value)))
        elif isinstance(value, dt.datetime):
            converted = m.Cli.XlsxDateTimeValue(value=value)
        elif isinstance(value, dt.date):
            converted = m.Cli.XlsxDateValue(value=value)
        else:
            converted = m.Cli.XlsxTextValue(value=value)
        return r[m.Cli.XlsxCellValue].ok(converted)

    @staticmethod
    def _has_snapshot_content(cell: Cell) -> bool:
        return (
            cell.value is not None
            or cell.has_style
            or cell.comment is not None
            or cell.hyperlink is not None
        )

    @staticmethod
    def _formula(cell: Cell) -> str | None:
        if cell.data_type != "f":
            return None
        value = cell.value
        if not isinstance(value, str) or not value.startswith("="):
            msg = (
                f"{c.Cli.XlsxError.CELL_VALUE_UNSUPPORTED}: "
                f"invalid formula at {cell.coordinate}"
            )
            raise ValueError(msg)
        return value

    @classmethod
    def _snapshot_cell(
        cls, formula_cell: Cell, value_sheet: Worksheet, *, data_only: bool
    ) -> r[m.Cli.XlsxCellSnapshot]:
        try:
            return cls._snapshot_cell_unchecked(
                formula_cell, value_sheet, data_only=data_only
            )
        except (IndexError, TypeError, m.ValidationError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.XlsxCellSnapshot].fail(detail)

    @classmethod
    def _snapshot_cell_unchecked(
        cls, formula_cell: Cell, value_sheet: Worksheet, *, data_only: bool
    ) -> r[m.Cli.XlsxCellSnapshot]:
        formula = cls._formula(formula_cell)
        selected = (
            value_sheet.cell(formula_cell.row, formula_cell.column)
            if data_only
            else formula_cell
        )
        if not isinstance(selected, Cell):
            return r[m.Cli.XlsxCellSnapshot].fail(
                f"Unsupported selected cell: {formula_cell.coordinate}"
            )
        selected_value = selected.value
        if selected_value is not None and not isinstance(
            selected_value, (str, int, float, bool, Decimal, dt.date, dt.datetime)
        ):
            return r[m.Cli.XlsxCellSnapshot].fail(
                f"{c.Cli.XlsxError.CELL_VALUE_UNSUPPORTED}: "
                f"{selected_value.__class__.__name__} at {formula_cell.coordinate}"
            )
        value: m.Cli.XlsxCellValue = cls._require_success(
            cls._snapshot_value(
                selected_value, formula_view=formula is not None and not data_only
            )
        )
        return r[m.Cli.XlsxCellSnapshot].ok(
            m.Cli.XlsxCellSnapshot(
                coordinate=formula_cell.coordinate,
                position=m.Cli.XlsxCellAddress(
                    row=formula_cell.row, column=formula_cell.column
                ),
                value=value,
                formula=formula,
                style_name=cls._snapshot_style_name(formula_cell),
                style_id=formula_cell.style_id,
                number_format=formula_cell.number_format,
                locked=formula_cell.protection.locked,
                hidden=formula_cell.protection.hidden,
            )
        )

    @classmethod
    def _snapshot_cells(
        cls, formula_sheet: Worksheet, value_sheet: Worksheet, *, data_only: bool
    ) -> r[tuple[m.Cli.XlsxCellSnapshot, ...]]:
        try:
            cells = cls._snapshot_cells_unchecked(
                formula_sheet, value_sheet, data_only=data_only
            )
        except ValueError as exc:
            return r[tuple[m.Cli.XlsxCellSnapshot, ...]].fail(str(exc))
        return r[tuple[m.Cli.XlsxCellSnapshot, ...]].ok(cells)

    @classmethod
    def _snapshot_cells_unchecked(
        cls, formula_sheet: Worksheet, value_sheet: Worksheet, *, data_only: bool
    ) -> tuple[m.Cli.XlsxCellSnapshot, ...]:
        cells: tuple[m.Cli.XlsxCellSnapshot, ...] = ()
        for row in formula_sheet.iter_rows():
            for formula_cell in row:
                if isinstance(formula_cell, MergedCell):
                    continue
                if not cls._has_snapshot_content(formula_cell):
                    continue
                cell: m.Cli.XlsxCellSnapshot = cls._require_success(
                    cls._snapshot_cell(formula_cell, value_sheet, data_only=data_only)
                )
                cells = (*cells, cell)
        return cells


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxSnapshotValues",)

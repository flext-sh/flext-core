"""Typed structural extraction for XLSX semantic snapshots."""

from __future__ import annotations

from openpyxl import Workbook
from openpyxl.utils.cell import column_index_from_string
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.table import Table
from openpyxl.worksheet.worksheet import Worksheet
from pydantic import ValidationError

from flext_cli import m, r


class FlextCliUtilitiesXlsxSnapshotStructure:
    """Translate vendor tables, names, and dimensions into typed evidence."""

    # NOTE (multi-agent, mro-j2yt.1): vendor collections terminate here and
    # become ordered tuples of canonical models before entering the service.
    @staticmethod
    def _table_name(item: m.Cli.XlsxTableSnapshot) -> str:
        """Return the deterministic table sort key."""
        return item.name

    @staticmethod
    def _row_position(item: m.Cli.XlsxRowDimensionSnapshot) -> int:
        """Return the deterministic row-dimension sort key."""
        return item.position

    @staticmethod
    def _column_position(item: m.Cli.XlsxColumnDimensionSnapshot) -> int:
        """Return the deterministic column-dimension sort key."""
        return item.first

    @staticmethod
    def _defined_name(item: m.Cli.XlsxDefinedNameSnapshot) -> str:
        """Return the deterministic defined-name sort key."""
        return item.name

    @staticmethod
    def _snapshot_tables(
        worksheet: Worksheet,
    ) -> r[tuple[m.Cli.XlsxTableSnapshot, ...]]:
        tables: tuple[m.Cli.XlsxTableSnapshot, ...] = ()
        try:
            for item in worksheet.tables.values():
                if not isinstance(item, Table):
                    return r[tuple[m.Cli.XlsxTableSnapshot, ...]].fail(
                        f"Unsupported table value: {item.__class__.__name__}"
                    )
                if not isinstance(item.name, str) or not isinstance(item.ref, str):
                    return r[tuple[m.Cli.XlsxTableSnapshot, ...]].fail(
                        "Table requires a string name and reference"
                    )
                style_name = (
                    item.tableStyleInfo.name
                    if item.tableStyleInfo is not None
                    else None
                )
                tables = (
                    *tables,
                    m.Cli.XlsxTableSnapshot(
                        name=item.name, reference=item.ref, style_name=style_name
                    ),
                )
        except (AttributeError, TypeError, ValidationError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[tuple[m.Cli.XlsxTableSnapshot, ...]].fail(
                f"Table snapshot failed: {detail}"
            )
        return r[tuple[m.Cli.XlsxTableSnapshot, ...]].ok(
            tuple(
                sorted(tables, key=FlextCliUtilitiesXlsxSnapshotStructure._table_name)
            )
        )

    @staticmethod
    def _snapshot_rows(
        worksheet: Worksheet,
    ) -> r[tuple[m.Cli.XlsxRowDimensionSnapshot, ...]]:
        rows: tuple[m.Cli.XlsxRowDimensionSnapshot, ...] = ()
        try:
            for item in worksheet.row_dimensions.values():
                rows = (
                    *rows,
                    m.Cli.XlsxRowDimensionSnapshot(
                        position=item.index,
                        size=item.height,
                        hidden=item.hidden,
                        outline_level=item.outlineLevel,
                    ),
                )
        except (TypeError, ValidationError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[tuple[m.Cli.XlsxRowDimensionSnapshot, ...]].fail(
                f"Row-dimension snapshot failed: {detail}"
            )
        return r[tuple[m.Cli.XlsxRowDimensionSnapshot, ...]].ok(
            tuple(
                sorted(rows, key=FlextCliUtilitiesXlsxSnapshotStructure._row_position)
            )
        )

    @staticmethod
    def _snapshot_columns(
        worksheet: Worksheet,
    ) -> r[tuple[m.Cli.XlsxColumnDimensionSnapshot, ...]]:
        columns: tuple[m.Cli.XlsxColumnDimensionSnapshot, ...] = ()
        try:
            for item in worksheet.column_dimensions.values():
                anchor = column_index_from_string(item.index)
                columns = (
                    *columns,
                    m.Cli.XlsxColumnDimensionSnapshot(
                        name=item.index,
                        first=item.min if item.min is not None else anchor,
                        last=item.max if item.max is not None else anchor,
                        size=item.width,
                        hidden=item.hidden,
                        outline_level=item.outlineLevel,
                    ),
                )
        except (TypeError, ValidationError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[tuple[m.Cli.XlsxColumnDimensionSnapshot, ...]].fail(
                f"Column-dimension snapshot failed: {detail}"
            )
        return r[tuple[m.Cli.XlsxColumnDimensionSnapshot, ...]].ok(
            tuple(
                sorted(
                    columns, key=FlextCliUtilitiesXlsxSnapshotStructure._column_position
                )
            )
        )

    @staticmethod
    def _snapshot_names(
        workbook: Workbook,
    ) -> r[tuple[m.Cli.XlsxDefinedNameSnapshot, ...]]:
        names: tuple[m.Cli.XlsxDefinedNameSnapshot, ...] = ()
        try:
            for item in workbook.defined_names.values():
                if not isinstance(item, DefinedName):
                    return r[tuple[m.Cli.XlsxDefinedNameSnapshot, ...]].fail(
                        f"Unsupported defined name: {item.__class__.__name__}"
                    )
                if not isinstance(item.attr_text, str):
                    return r[tuple[m.Cli.XlsxDefinedNameSnapshot, ...]].fail(
                        f"Defined name has no expression or kind: {item.name}"
                    )
                names = (
                    *names,
                    m.Cli.XlsxDefinedNameSnapshot(
                        name=item.name,
                        expression=item.attr_text,
                        kind=item.type,
                        sheet_position=item.localSheetId,
                        hidden=item.hidden,
                    ),
                )
        except (TypeError, ValidationError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[tuple[m.Cli.XlsxDefinedNameSnapshot, ...]].fail(
                f"Defined-name snapshot failed: {detail}"
            )
        return r[tuple[m.Cli.XlsxDefinedNameSnapshot, ...]].ok(
            tuple(
                sorted(names, key=FlextCliUtilitiesXlsxSnapshotStructure._defined_name)
            )
        )


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxSnapshotStructure",)

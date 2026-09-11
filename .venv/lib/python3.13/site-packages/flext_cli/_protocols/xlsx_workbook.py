"""Structural contracts for external workbook objects."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from io import BytesIO
from typing import Protocol, runtime_checkable

from flext_cli import t

from .xlsx_rules import FlextCliProtocolsXlsxRules


class FlextCliProtocolsXlsxWorkbook:
    """Narrow implementation protocols kept inside the XLSX adapter."""

    # NOTE (multi-agent, mro-j2yt.1): these objects never enter consumer APIs;
    # they exist only to type the external adapter without concrete annotations.
    @runtime_checkable
    class XlsxCellProtection(Protocol):
        locked: bool
        hidden: bool

    @runtime_checkable
    class XlsxComment(Protocol):
        width: int
        height: int

    @runtime_checkable
    class XlsxCell(Protocol):
        value: t.Cli.XlsxCellPrimitive
        style: str
        protection: FlextCliProtocolsXlsxWorkbook.XlsxCellProtection
        comment: FlextCliProtocolsXlsxWorkbook.XlsxComment | None
        hyperlink: str | None

    @runtime_checkable
    class XlsxDimension(Protocol):
        width: float | None
        height: float | None
        hidden: bool
        outline_level: int

    @runtime_checkable
    class XlsxRowDimensions(Protocol):
        def __getitem__(
            self, index: int, /
        ) -> FlextCliProtocolsXlsxWorkbook.XlsxDimension: ...

        def group(
            self, start: int, end: int, *, outline_level: int, hidden: bool
        ) -> None: ...

    @runtime_checkable
    class XlsxColumnDimensions(Protocol):
        def __getitem__(
            self, index: str, /
        ) -> FlextCliProtocolsXlsxWorkbook.XlsxDimension: ...

        def group(
            self, start: str, end: str, *, outline_level: int, hidden: bool
        ) -> None: ...

    @runtime_checkable
    class XlsxAutoFilter(Protocol):
        ref: str | None

    @runtime_checkable
    class XlsxDefinedNames(Protocol):
        def add(
            self, value: FlextCliProtocolsXlsxWorkbook.XlsxDefinedName, /
        ) -> None: ...

        def values(self) -> Iterable[FlextCliProtocolsXlsxWorkbook.XlsxDefinedName]: ...

    @runtime_checkable
    class XlsxCalculation(Protocol):
        """Marker for external calculation properties."""

    @runtime_checkable
    class XlsxWorksheet(Protocol):
        title: str
        sheet_state: str
        freeze_panes: str | None
        auto_filter: FlextCliProtocolsXlsxWorkbook.XlsxAutoFilter
        row_dimensions: FlextCliProtocolsXlsxWorkbook.XlsxRowDimensions
        column_dimensions: FlextCliProtocolsXlsxWorkbook.XlsxColumnDimensions

        def cell(
            self, row: int, column: int
        ) -> FlextCliProtocolsXlsxWorkbook.XlsxCell: ...

        def merge_cells(self, range_string: str) -> None: ...

        def add_table(self, table: FlextCliProtocolsXlsxWorkbook.XlsxTable) -> None: ...

        def add_data_validation(
            self, validation: FlextCliProtocolsXlsxRules.XlsxDataValidation
        ) -> None: ...

        def iter_rows(
            self,
        ) -> Iterable[Sequence[FlextCliProtocolsXlsxWorkbook.XlsxCell]]: ...

    @runtime_checkable
    class XlsxWorkbook(Protocol):
        calculation: FlextCliProtocolsXlsxWorkbook.XlsxCalculation
        defined_names: FlextCliProtocolsXlsxWorkbook.XlsxDefinedNames
        named_styles: Sequence[str]
        sheetnames: Sequence[str]
        worksheets: Sequence[FlextCliProtocolsXlsxWorkbook.XlsxWorksheet]

        def __getitem__(
            self, name: str, /
        ) -> FlextCliProtocolsXlsxWorkbook.XlsxWorksheet: ...

        def create_sheet(
            self, title: str
        ) -> FlextCliProtocolsXlsxWorkbook.XlsxWorksheet: ...

        def remove(
            self, worksheet: FlextCliProtocolsXlsxWorkbook.XlsxWorksheet
        ) -> None: ...

        def add_named_style(
            self, style: FlextCliProtocolsXlsxRules.XlsxNamedStyle
        ) -> None: ...

        def save(self, filename: BytesIO) -> None: ...

    class XlsxDefinedName(Protocol): ...

    class XlsxTable(Protocol): ...


__all__: tuple[str, ...] = ("FlextCliProtocolsXlsxWorkbook",)

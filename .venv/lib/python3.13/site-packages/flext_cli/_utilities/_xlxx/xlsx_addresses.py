"""Canonical cell and range reference helpers for XLSX plans."""

from __future__ import annotations

from openpyxl.utils.cell import (
    absolute_coordinate,
    get_column_letter,
    quote_sheetname,
    range_boundaries,
)

from flext_cli import c, m, p, r


class FlextCliUtilitiesXlsxAddresses:
    """Translate validated one-based addresses into Excel references."""

    # NOTE (multi-agent, mro-j2yt.1): address rendering is implementation
    # policy, shared by cells, layout, tables, names, and worksheet rules.
    @staticmethod
    def _cell_ref(address: m.Cli.XlsxCellAddress) -> str:
        return f"{get_column_letter(address.column)}{address.row}"

    @classmethod
    def _range_ref(cls, area: m.Cli.XlsxCellRange) -> str:
        if area.first.row > area.last.row or area.first.column > area.last.column:
            msg = (
                "XLSX range starts after it ends: "
                f"{cls._cell_ref(area.first)}:{cls._cell_ref(area.last)}"
            )
            raise ValueError(msg)
        return f"{cls._cell_ref(area.first)}:{cls._cell_ref(area.last)}"

    @classmethod
    def _absolute_range_ref(cls, area: m.Cli.XlsxCellRange) -> str:
        reference: str = absolute_coordinate(cls._range_ref(area))
        return reference

    @staticmethod
    def _column_ref(index: int) -> str:
        reference: str = get_column_letter(index)
        return reference

    @staticmethod
    def _sheet_ref(name: str) -> str:
        reference: str = quote_sheetname(name)
        return reference

    @classmethod
    def _format_reference(cls, request: m.Cli.XlsxFormatReferenceRequest) -> str:
        if request.collapse_single_cell and request.area.first == request.area.last:
            reference = cls._cell_ref(request.area.first)
            if request.absolute:
                reference = cls._absolute_range_ref(request.area).partition(":")[0]
        elif request.absolute:
            reference = cls._absolute_range_ref(request.area)
        else:
            reference = cls._range_ref(request.area)
        if request.sheet is not None:
            reference = f"{cls._sheet_ref(request.sheet)}!{reference}"
        return reference

    @staticmethod
    def _range_failure(detail: str) -> p.Result[m.Cli.XlsxCellRange]:
        return r[m.Cli.XlsxCellRange].fail(f"{c.Cli.XlsxError.RANGE_INVALID}: {detail}")

    @classmethod
    def xlsx_parse_range(
        cls, request: m.Cli.XlsxParseRangeRequest
    ) -> p.Result[m.Cli.XlsxCellRange]:
        """Parse one concrete A1 cell/range through the XLSX adapter."""
        try:
            first_column, first_row, last_column, last_row = range_boundaries(
                request.reference
            )
        except (TypeError, ValueError) as exc:
            detail = str(exc).strip() or request.reference
            return cls._range_failure(detail)
        if (
            first_column is None
            or first_row is None
            or last_column is None
            or last_row is None
        ):
            return cls._range_failure(request.reference)
        if first_column > last_column or first_row > last_row:
            return cls._range_failure(request.reference)
        return r[m.Cli.XlsxCellRange].ok(
            m.Cli.XlsxCellRange(
                first=m.Cli.XlsxCellAddress(row=first_row, column=first_column),
                last=m.Cli.XlsxCellAddress(row=last_row, column=last_column),
            )
        )

    # mro-j2yt.1 (xlsx_reference_api): keep vendor formatting behind cli.
    @classmethod
    def xlsx_format_reference(
        cls, request: m.Cli.XlsxFormatReferenceRequest
    ) -> p.Result[m.Cli.XlsxReference]:
        """Format one validated range as a canonical Excel reference."""
        try:
            reference = cls._format_reference(request)
        except (TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.XlsxReference].fail(
                f"{c.Cli.XlsxError.RANGE_INVALID}: {detail}"
            )
        return r[m.Cli.XlsxReference].ok(m.Cli.XlsxReference(reference=reference))


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxAddresses",)

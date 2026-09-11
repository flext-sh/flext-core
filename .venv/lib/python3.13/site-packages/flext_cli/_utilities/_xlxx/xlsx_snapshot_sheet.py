"""Worksheet assembly for typed XLSX semantic snapshots."""

from __future__ import annotations

from typing import Literal

from openpyxl.worksheet.worksheet import Worksheet

from flext_cli import m, r

from .xlsx_snapshot_structure import FlextCliUtilitiesXlsxSnapshotStructure
from .xlsx_snapshot_values import FlextCliUtilitiesXlsxSnapshotValues


class FlextCliUtilitiesXlsxSnapshotSheet(
    FlextCliUtilitiesXlsxSnapshotValues, FlextCliUtilitiesXlsxSnapshotStructure
):
    """Build one immutable worksheet snapshot from formula and value views."""

    # NOTE (multi-agent, mro-j2yt.1): formula structure remains authoritative
    # while data_only selects only the typed value exposed for each cell.
    @staticmethod
    def _snapshot_state(
        worksheet: Worksheet,
    ) -> r[Literal["visible", "hidden", "veryHidden"]]:
        sheet_state: str = worksheet.sheet_state
        if sheet_state == "visible":
            return r[Literal["visible", "hidden", "veryHidden"]].ok("visible")
        if sheet_state == "hidden":
            return r[Literal["visible", "hidden", "veryHidden"]].ok("hidden")
        if sheet_state == "veryHidden":
            return r[Literal["visible", "hidden", "veryHidden"]].ok("veryHidden")
        return r[Literal["visible", "hidden", "veryHidden"]].fail(
            f"Unsupported worksheet state: {sheet_state}"
        )

    @classmethod
    def _snapshot_sheet(
        cls, formula_sheet: Worksheet, value_sheet: Worksheet, *, position: int
    ) -> r[m.Cli.XlsxSheetSnapshot]:
        try:
            snapshot = cls._snapshot_sheet_unchecked(
                formula_sheet, value_sheet, position=position
            )
        except (TypeError, m.ValidationError, ValueError) as exc:
            return r[m.Cli.XlsxSheetSnapshot].fail(
                f"Worksheet snapshot failed ({exc.__class__.__name__}): {exc}"
            )
        return r[m.Cli.XlsxSheetSnapshot].ok(snapshot)

    @classmethod
    def _snapshot_sheet_unchecked(
        cls, formula_sheet: Worksheet, value_sheet: Worksheet, *, position: int
    ) -> m.Cli.XlsxSheetSnapshot:
        if formula_sheet.title != value_sheet.title:
            msg = (
                f"Worksheet view mismatch: {formula_sheet.title} != {value_sheet.title}"
            )
            raise ValueError(msg)
        state = cls._require_success(cls._snapshot_state(formula_sheet))
        cells = cls._require_success(
            cls._snapshot_cells(
                formula_sheet, value_sheet, data_only=formula_sheet is not value_sheet
            )
        )
        tables = cls._require_success(cls._snapshot_tables(formula_sheet))
        rows = cls._require_success(cls._snapshot_rows(formula_sheet))
        columns = cls._require_success(cls._snapshot_columns(formula_sheet))
        merged_ranges = tuple(
            sorted(str(item) for item in formula_sheet.merged_cells.ranges)
        )
        legacy_password_hash = formula_sheet.protection.password
        if legacy_password_hash is not None and not isinstance(
            legacy_password_hash, str
        ):
            msg = "Worksheet legacy protection hash is not textual"
            raise TypeError(msg)
        return m.Cli.XlsxSheetSnapshot(
            name=formula_sheet.title,
            position=position,
            state=state,
            max_row=formula_sheet.max_row,
            max_column=formula_sheet.max_column,
            cells=cells,
            tables=tables,
            row_dimensions=rows,
            column_dimensions=columns,
            merged_ranges=merged_ranges,
            freeze_pane=formula_sheet.freeze_panes,
            auto_filter=formula_sheet.auto_filter.ref,
            protection=(
                m.Cli.XlsxSheetProtectionSnapshot(
                    enabled=formula_sheet.protection.sheet,
                    legacy_password_hash=legacy_password_hash,
                )
            ),
            formula_count=sum(item.formula is not None for item in cells),
            literal_count=sum(
                item.formula is None and item.value.kind != "blank" for item in cells
            ),
            data_validation_count=len(formula_sheet.data_validations.dataValidation),
            conditional_format_count=sum(
                len(formula_sheet.conditional_formatting[item])
                for item in formula_sheet.conditional_formatting
            ),
            merge_count=len(merged_ranges),
        )


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxSnapshotSheet",)

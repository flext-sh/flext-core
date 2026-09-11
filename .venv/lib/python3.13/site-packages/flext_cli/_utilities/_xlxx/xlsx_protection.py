"""Apply positive typed protection permissions at the openpyxl edge."""

from __future__ import annotations

from openpyxl.styles import Protection
from openpyxl.worksheet.worksheet import Worksheet

from flext_cli import c, m, p, r


class FlextCliUtilitiesXlsxProtection:
    """Translate explicit cell rules and positive worksheet permissions."""

    # NOTE (multi-agent, mro-j2yt.1): SheetProtection booleans express denied
    # actions, so positive allow_* model flags are inverted exactly once here.
    @classmethod
    def _apply_protection(
        cls, worksheet: Worksheet, plan: m.Cli.XlsxSheetProtectionPlan | None
    ) -> p.Result[bool]:
        if plan is None:
            return r[bool].ok(True)
        try:
            return cls._apply_protection_unchecked(worksheet, plan)
        except (TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[bool].fail(f"{c.Cli.XlsxError.RENDER_FAILED}: {detail}")

    @classmethod
    def _apply_protection_unchecked(
        cls, worksheet: Worksheet, plan: m.Cli.XlsxSheetProtectionPlan
    ) -> p.Result[bool]:
        for item in plan.cells:
            if (
                item.area.first.row > item.area.last.row
                or item.area.first.column > item.area.last.column
            ):
                return r[bool].fail("Cell protection range is inverted")
            for row in range(item.area.first.row, item.area.last.row + 1):
                for column in range(item.area.first.column, item.area.last.column + 1):
                    worksheet.cell(row, column).protection = Protection(
                        locked=item.locked, hidden=item.hidden
                    )
        permissions = plan.permissions
        protection = worksheet.protection
        protection.sheet = True
        protection.selectLockedCells = not permissions.allow_select_locked
        protection.selectUnlockedCells = not permissions.allow_select_unlocked
        protection.formatCells = not permissions.allow_format_cells
        protection.formatColumns = not permissions.allow_format_columns
        protection.formatRows = not permissions.allow_format_rows
        protection.insertColumns = not permissions.allow_insert_columns
        protection.insertRows = not permissions.allow_insert_rows
        protection.insertHyperlinks = not permissions.allow_insert_hyperlinks
        protection.deleteColumns = not permissions.allow_delete_columns
        protection.deleteRows = not permissions.allow_delete_rows
        protection.sort = not permissions.allow_sort
        protection.autoFilter = not permissions.allow_auto_filter
        protection.pivotTables = not permissions.allow_pivot_tables
        protection.objects = not permissions.allow_edit_objects
        protection.scenarios = not permissions.allow_edit_scenarios
        if plan.credential is not None:
            if plan.credential.kind == "legacy_hash":
                protection.set_password(plan.credential.value, already_hashed=True)
            else:
                protection.set_password(plan.credential.value)
        return r[bool].ok(True)


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxProtection",)

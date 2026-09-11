"""Safe byte IO for the private openpyxl adapter."""

from __future__ import annotations

from io import BytesIO
from zipfile import BadZipFile

from openpyxl import Workbook, load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from flext_cli import c, m, p, r

from .xlsx_archive import FlextCliUtilitiesXlsxArchive


class FlextCliUtilitiesXlsxWorkbookIo(FlextCliUtilitiesXlsxArchive):
    """Terminate workbook implementation objects at a bytes-only boundary."""

    # NOTE (multi-agent, mro-j2yt.1): source bytes pass the safe OOXML
    # inventory before openpyxl receives them; workbook objects stay private.
    @classmethod
    def _load_workbook(cls, source: bytes, *, data_only: bool = False) -> r[Workbook]:
        inspection_request = m.Cli.XlsxArchiveInspectionRequest(
            source=source, policy=m.Cli.XlsxArchivePolicy()
        )
        inspection_result = cls.xlsx_inspect(inspection_request)
        if inspection_result.failure:
            return r[Workbook].fail(
                inspection_result.error or str(c.Cli.XlsxError.ARCHIVE_INVALID)
            )
        inspection = inspection_result.value
        if not inspection.clean:
            detail = "; ".join(
                f"{item.kind}:{item.location}:{item.detail}"
                for item in inspection.violations
            )
            return r[Workbook].fail(
                f"{c.Cli.XlsxError.ARCHIVE_POLICY_VIOLATION}: {detail}"
            )
        try:
            workbook = load_workbook(
                BytesIO(source),
                read_only=False,
                data_only=data_only,
                keep_links=True,
                rich_text=False,
            )
        except (BadZipFile, InvalidFileException, KeyError, OSError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[Workbook].fail(f"{c.Cli.XlsxError.WORKBOOK_LOAD_FAILED}: {detail}")
        return r[Workbook].ok(workbook)

    @staticmethod
    def _new_workbook() -> Workbook:
        """Create one editable workbook with the implementation default sheet."""
        return Workbook(write_only=False, iso_dates=False)

    @staticmethod
    def _serialize_workbook(workbook: Workbook) -> p.Result[bytes]:
        """Serialize one private workbook to immutable bytes."""
        target = BytesIO()
        try:
            workbook.save(target)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[bytes].fail(f"{c.Cli.XlsxError.SERIALIZE_FAILED}: {detail}")
        content = target.getvalue()
        if not content:
            return r[bytes].fail(str(c.Cli.XlsxError.SERIALIZE_FAILED))
        return r[bytes].ok(content)


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxWorkbookIo",)

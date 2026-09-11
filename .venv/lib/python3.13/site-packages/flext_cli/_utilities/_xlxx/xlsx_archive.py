"""Generic safe OOXML archive inspector."""

from __future__ import annotations

from io import BytesIO
from zipfile import BadZipFile, LargeZipFile, ZipFile

from flext_cli import c, m, p, r

from .xlsx_archive_checks import FlextCliUtilitiesXlsxArchiveChecks


class FlextCliUtilitiesXlsxArchive(FlextCliUtilitiesXlsxArchiveChecks):
    """Inspect OOXML bytes against a caller-owned typed policy."""

    # NOTE (multi-agent, mro-j2yt.1): ZIP and XML implementations terminate at
    # this private adapter; consumers receive immutable inspection models.
    @classmethod
    def _inventory(
        cls, archive: p.Cli.XlsxArchiveReader, policy: m.Cli.XlsxArchivePolicy
    ) -> m.Cli.XlsxArchiveInventory:
        members: tuple[str, ...] = ()
        blocked: frozenset[str] = frozenset()
        seen: frozenset[str] = frozenset()
        total_size = 0
        violations: tuple[m.Cli.XlsxArchiveViolation, ...] = ()
        for info in archive.infolist():
            member = info.filename
            members = (*members, member)
            total_size += info.file_size
            if member in seen:
                violations = (
                    *violations,
                    cls._violation("duplicate_member", member, "duplicate ZIP member"),
                )
            seen = seen.union((member,))
            if info.file_size > policy.max_member_uncompressed_bytes:
                blocked = blocked.union((member,))
                violations = (
                    *violations,
                    cls._violation("member_size", member, str(info.file_size)),
                )
            if member in policy.forbidden_members:
                violations = (
                    *violations,
                    cls._violation("member", member, "forbidden exact member"),
                )
            for prefix in policy.forbidden_prefixes:
                if member.startswith(prefix):
                    violations = (
                        *violations,
                        cls._violation("member_prefix", member, prefix),
                    )
        if len(members) > policy.max_members:
            violations = (
                *violations,
                cls._violation("member_count", "archive", str(len(members))),
            )
        if total_size > policy.max_total_uncompressed_bytes:
            violations = (
                *violations,
                cls._violation("total_size", "archive", str(total_size)),
            )
        return m.Cli.XlsxArchiveInventory(
            members=members,
            blocked_members=blocked,
            total_uncompressed_bytes=total_size,
            violations=violations,
        )

    @classmethod
    def _inspect_archive(
        cls, archive: p.Cli.XlsxArchiveReader, policy: m.Cli.XlsxArchivePolicy
    ) -> p.Result[m.Cli.XlsxArchiveInspection]:
        inventory = cls._inventory(archive, policy)
        violations = inventory.violations
        workbook_member = c.Cli.XLSX_WORKBOOK_MEMBER
        styles_member = c.Cli.XLSX_STYLES_MEMBER
        if workbook_member not in inventory.members:
            violations = (
                *violations,
                cls._violation(
                    "required_member", workbook_member, "required OOXML member"
                ),
            )
        worksheets = tuple(
            member
            for member in inventory.members
            if member.startswith(c.Cli.XLSX_WORKSHEET_PREFIX)
            and member.endswith(c.Cli.XLSX_XML_SUFFIX)
        )
        if (
            policy.required_worksheet_count is not None
            and len(worksheets) != policy.required_worksheet_count
        ):
            violations = (
                *violations,
                cls._violation(
                    "worksheet_count",
                    "archive",
                    f"expected={policy.required_worksheet_count}, actual={len(worksheets)}",
                ),
            )
        xml_members = tuple(
            member
            for member in (*worksheets, workbook_member, styles_member)
            if member in inventory.members and member not in inventory.blocked_members
        )
        for member in xml_members:
            root_result = cls._xml_root(archive, member)
            if root_result.failure:
                return r[m.Cli.XlsxArchiveInspection].fail(
                    root_result.error or f"Invalid OOXML member: {member}"
                )
            root = root_result.value
            if member in worksheets:
                violations = (
                    *violations,
                    *cls._worksheet_violations(root, member, policy),
                )
            elif member == workbook_member:
                violations = (
                    *violations,
                    *cls._workbook_violations(root, member, policy),
                )
            else:
                violations = (*violations, *cls._style_violations(root, member, policy))
        inspection = m.Cli.XlsxArchiveInspection(
            member_count=len(inventory.members),
            worksheet_count=len(worksheets),
            total_uncompressed_bytes=inventory.total_uncompressed_bytes,
            violations=violations,
            clean=not violations,
        )
        return r[m.Cli.XlsxArchiveInspection].ok(inspection)

    @classmethod
    def xlsx_inspect(
        cls, request: m.Cli.XlsxArchiveInspectionRequest
    ) -> p.Result[m.Cli.XlsxArchiveInspection]:
        """Inspect workbook bytes without extracting or trusting package XML."""
        try:
            with ZipFile(BytesIO(request.source)) as archive:
                # mro-j47u (codex): ZipFile already satisfies the typed archive port.
                return cls._inspect_archive(archive, request.policy)
        except (BadZipFile, LargeZipFile, OSError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.XlsxArchiveInspection].fail(
                f"{c.Cli.XlsxError.ARCHIVE_INVALID}: {detail}"
            )


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxArchive",)

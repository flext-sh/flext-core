"""Raw cached-value evidence for recalculated XLSX workbooks."""

from __future__ import annotations

from io import BytesIO
from zipfile import BadZipFile, LargeZipFile, ZipFile

from flext_cli import c, p, r

from .xlsx_archive_checks import FlextCliUtilitiesXlsxArchiveChecks


class FlextCliUtilitiesXlsxRecalcEvidence(FlextCliUtilitiesXlsxArchiveChecks):
    """Read formula cache facts straight from safely parsed worksheet XML."""

    # NOTE (multi-agent, mro-j2yt.1): openpyxl cannot distinguish a missing
    # cached value from an empty-string result; the OOXML value element is the
    # only faithful evidence and stays behind this private adapter.
    @classmethod
    def _require_xml(
        cls, archive: p.Cli.XlsxArchiveReader, member: str
    ) -> p.Cli.XlsxXmlElement:
        result = cls._xml_root(archive, member)
        if result.failure:
            msg = result.error or f"Invalid OOXML member: {member}"
            raise ValueError(msg)
        return result.value

    @classmethod
    def _worksheet_targets(
        cls, workbook_root: p.Cli.XlsxXmlElement, rels_root: p.Cli.XlsxXmlElement
    ) -> tuple[tuple[str, str], ...]:
        relationships: tuple[tuple[str, str], ...] = ()
        for relationship in rels_root.iter():
            if cls._local_name(relationship.tag) != "Relationship":
                continue
            rel_id = relationship.get("Id")
            target = relationship.get("Target")
            if rel_id is None or target is None:
                continue
            relationships = (*relationships, (rel_id, target))
        targets: tuple[tuple[str, str], ...] = ()
        for sheet in workbook_root.iter():
            if cls._local_name(sheet.tag) != "sheet":
                continue
            name = sheet.get("name")
            rel_id = sheet.get(c.Cli.XLSX_RELATIONSHIPS_ID_ATTRIBUTE)
            if name is None or rel_id is None:
                continue
            target = next(
                (
                    candidate
                    for candidate_id, candidate in relationships
                    if candidate_id == rel_id
                ),
                None,
            )
            if target is None:
                continue
            member = target.lstrip("/")
            if not member.startswith(c.Cli.XLSX_PACKAGE_PREFIX):
                member = f"{c.Cli.XLSX_PACKAGE_PREFIX}{member}"
            targets = (*targets, (name, member))
        return targets

    @classmethod
    def _formula_cache_evidence(
        cls, source: bytes
    ) -> p.Result[tuple[tuple[str, ...], tuple[str, ...]]]:
        """Classify formula cells as uncached or empty-string cached."""
        try:
            evidence = cls._formula_cache_evidence_unchecked(source)
        except (BadZipFile, LargeZipFile, OSError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[tuple[tuple[str, ...], tuple[str, ...]]].fail(
                f"{c.Cli.XlsxError.PARITY_FAILED}: {detail}"
            )
        return r[tuple[tuple[str, ...], tuple[str, ...]]].ok(evidence)

    @classmethod
    def _formula_cache_evidence_unchecked(
        cls, source: bytes
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        with ZipFile(BytesIO(source)) as archive:
            workbook_root = cls._require_xml(archive, c.Cli.XLSX_WORKBOOK_MEMBER)
            rels_root = cls._require_xml(archive, c.Cli.XLSX_WORKBOOK_RELS_MEMBER)
            uncached: tuple[str, ...] = ()
            empty: tuple[str, ...] = ()
            for sheet_name, member in cls._worksheet_targets(workbook_root, rels_root):
                root = cls._require_xml(archive, member)
                for element in root.iter():
                    if cls._local_name(element.tag) != "c":
                        continue
                    has_formula = False
                    value_element: p.Cli.XlsxXmlElement | None = None
                    for child in element.iter():
                        local = cls._local_name(child.tag)
                        if local == "f":
                            has_formula = True
                        elif local == "v" and value_element is None:
                            value_element = child
                    if not has_formula:
                        continue
                    coordinate = element.get("r")
                    if coordinate is None:
                        msg = f"Formula cell without coordinate in {member}"
                        raise ValueError(msg)
                    if value_element is None:
                        uncached = (*uncached, f"{sheet_name}!{coordinate}")
                    elif not (value_element.text or "").strip():
                        empty = (*empty, f"{sheet_name}!{coordinate}")
        return uncached, empty


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxRecalcEvidence",)

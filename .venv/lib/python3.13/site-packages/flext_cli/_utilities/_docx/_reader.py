"""Generic model-driven DOCX reader."""

from __future__ import annotations

from io import BytesIO
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal
from zipfile import BadZipFile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_UNDERLINE
from docx.table import Table

from flext_cli import c, m, p, r, t

if TYPE_CHECKING:
    from docx.document import Document as DocumentType
    from docx.text.paragraph import Paragraph
    from docx.text.run import Run


_DOCX_UNDERLINE_VALUE = Literal[
    "single", "double", "singleAccounting", "doubleAccounting"
]


class FlextCliUtilitiesDocxReader:
    """Read DOCX bytes into an immutable document plan."""

    # NOTE (multi-agent, mro-j2yt.1): python-docx objects stay inside this
    # module; consumers receive only validated plans.

    @classmethod
    def docx_read(cls, source: bytes) -> p.Result[m.Cli.DocxDocumentPlan]:
        """Read document bytes into a validated plan."""
        try:
            document = Document(BytesIO(source))
        except (OSError, ValueError, KeyError, BadZipFile) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.DocxDocumentPlan].fail(
                f"{c.Cli.DocxError.DOCUMENT_LOAD_FAILED}: {detail}"
            )
        try:
            plan = cls._snapshot_document(document)
        except (TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.DocxDocumentPlan].fail(
                f"{c.Cli.DocxError.RENDER_FAILED}: {detail}"
            )
        return r[m.Cli.DocxDocumentPlan].ok(plan)

    @classmethod
    def _snapshot_document(cls, document: DocumentType) -> m.Cli.DocxDocumentPlan:
        paragraphs = tuple(cls._snapshot_paragraph(p) for p in document.paragraphs)
        tables = tuple(cls._snapshot_table(t) for t in document.tables)
        return m.Cli.DocxDocumentPlan(
            paragraphs=paragraphs,
            tables=tables,
            sections=(),
            core_properties=cls._snapshot_core_properties(document),
        )

    @classmethod
    def _snapshot_paragraph(cls, paragraph: Paragraph) -> m.Cli.DocxParagraphPlan:
        runs = tuple(cls._snapshot_run(run) for run in paragraph.runs)
        alignment = cls._snapshot_alignment(paragraph.paragraph_format.alignment)
        return m.Cli.DocxParagraphPlan(
            runs=runs,
            style=paragraph.style.name if paragraph.style else None,
            alignment=alignment,
        )

    @classmethod
    def _snapshot_run(cls, run: Run) -> m.Cli.DocxRunPlan:
        font = run.font
        color = None
        if font.color and font.color.rgb:
            color = m.Cli.DocxRgbColor(value=str(font.color.rgb))
        style = m.Cli.DocxRunStyleSpec(
            font=m.Cli.DocxFontSpec(
                name=font.name,
                size=font.size.pt if font.size else None,
                bold=font.bold,
                italic=font.italic,
                underline=cls._snapshot_underline(underline=font.underline),
                strike=font.strike,
                color=color,
            )
        )
        return m.Cli.DocxRunPlan(text=run.text, style=style)

    @classmethod
    def _snapshot_table(cls, table: Table) -> m.Cli.DocxTablePlan:
        rows: tuple[m.Cli.DocxTableRowPlan, ...] = ()
        for row in table.rows:
            cells = tuple(
                m.Cli.DocxTableCellPlan(
                    paragraphs=tuple(
                        cls._snapshot_paragraph(p) for p in cell.paragraphs
                    )
                )
                for cell in row.cells
            )
            rows = (*rows, m.Cli.DocxTableRowPlan(cells=cells))
        return m.Cli.DocxTablePlan(rows=rows)

    @classmethod
    def _snapshot_core_properties(cls, document: DocumentType) -> t.JsonMapping:
        core_props = document.core_properties
        properties: dict[str, str] = {}
        for key in ("author", "title", "subject", "keywords", "comments", "category"):
            value = getattr(core_props, key, None)
            if value:
                properties[key] = str(value)
        return MappingProxyType(properties)

    @staticmethod
    def _snapshot_alignment(alignment: WD_ALIGN_PARAGRAPH | None) -> str | None:
        if alignment is None:
            return None
        mapping: dict[WD_ALIGN_PARAGRAPH, str] = {
            WD_ALIGN_PARAGRAPH.LEFT: "left",
            WD_ALIGN_PARAGRAPH.CENTER: "center",
            WD_ALIGN_PARAGRAPH.RIGHT: "right",
            WD_ALIGN_PARAGRAPH.JUSTIFY: "justify",
            WD_ALIGN_PARAGRAPH.DISTRIBUTE: "distribute",
        }
        return mapping.get(alignment)

    @staticmethod
    def _snapshot_underline(
        *, underline: WD_UNDERLINE | int | bool | None
    ) -> _DOCX_UNDERLINE_VALUE | None:
        if underline is None or underline is False:
            return None
        mapping: dict[WD_UNDERLINE | int, _DOCX_UNDERLINE_VALUE] = {
            WD_UNDERLINE.SINGLE: "single",
            WD_UNDERLINE.DOUBLE: "double",
            9: "singleAccounting",
            11: "doubleAccounting",
        }
        return mapping.get(underline)


__all__: tuple[str, ...] = ("FlextCliUtilitiesDocxReader",)

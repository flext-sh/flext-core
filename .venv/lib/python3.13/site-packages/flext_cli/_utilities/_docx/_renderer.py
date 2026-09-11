"""Generic model-driven DOCX renderer."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING, ClassVar, Protocol
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from docx import Document
from docx.document import Document as DocumentType
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX, WD_UNDERLINE
from docx.shared import Inches, Pt, RGBColor

from flext_cli import c, m, p, r, t

if TYPE_CHECKING:
    from collections.abc import Sequence

    from docx.text.paragraph import Paragraph, ParagraphFormat
    from docx.text.run import Font

    class _DocxParagraphContainer(Protocol):
        def add_paragraph(self) -> Paragraph: ...

        @property
        def paragraphs(self) -> Sequence[Paragraph]: ...


_HEX_COLOR_WITH_ALPHA_LENGTH = 8
_ZIP_EPOCH_FLOOR_YEAR = 1980


class FlextCliUtilitiesDocxRenderer:
    """Render one immutable document plan into DOCX bytes."""

    # NOTE (multi-agent, mro-j2yt.1): python-docx objects stay inside this
    # module; consumers exchange only validated plans and bytes.

    _ALIGNMENT_MAP: ClassVar[dict[str, WD_ALIGN_PARAGRAPH]] = {
        "left": WD_ALIGN_PARAGRAPH.LEFT,
        "center": WD_ALIGN_PARAGRAPH.CENTER,
        "right": WD_ALIGN_PARAGRAPH.RIGHT,
        "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
        "distribute": WD_ALIGN_PARAGRAPH.DISTRIBUTE,
    }

    _UNDERLINE_MAP: ClassVar[dict[str, WD_UNDERLINE | None]] = {
        "single": WD_UNDERLINE.SINGLE,
        "double": WD_UNDERLINE.DOUBLE,
        "singleAccounting": WD_UNDERLINE(9),
        "doubleAccounting": WD_UNDERLINE(11),
    }

    _HIGHLIGHT_MAP: ClassVar[dict[str, WD_COLOR_INDEX | None]] = {
        name: getattr(WD_COLOR_INDEX, name.upper(), None)
        for name in (
            "yellow",
            "green",
            "cyan",
            "magenta",
            "blue",
            "red",
            "darkBlue",
            "darkCyan",
            "darkGreen",
            "darkMagenta",
            "darkRed",
            "darkYellow",
            "darkGray",
            "lightGray",
            "black",
        )
    }

    @classmethod
    def docx_render(
        cls, request: m.Cli.DocxRenderRequest
    ) -> p.Result[m.Cli.DocxRenderResult]:
        """Render typed paragraphs, tables, and sections into document bytes."""
        document_result = cls._document_for_request(request)
        if document_result.failure:
            return r[m.Cli.DocxRenderResult].fail(
                document_result.error or str(c.Cli.DocxError.RENDER_FAILED)
            )
        document = document_result.value
        try:
            cls._apply_document(document, request.plan)
            cls._apply_source_date(document, request.source_date_epoch)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.DocxRenderResult].fail(
                f"{c.Cli.DocxError.RENDER_FAILED}: {detail}"
            )
        content = cls._serialize_document(
            document, source_date_epoch=request.source_date_epoch
        )
        if content.failure:
            return r[m.Cli.DocxRenderResult].fail(
                content.error or str(c.Cli.DocxError.SERIALIZE_FAILED)
            )
        return r[m.Cli.DocxRenderResult].ok(
            m.Cli.DocxRenderResult(content=content.value, plan=request.plan)
        )

    @classmethod
    def _document_for_request(
        cls, request: m.Cli.DocxRenderRequest
    ) -> p.Result[DocumentType]:
        if request.template is None:
            return r[DocumentType].ok(Document())
        try:
            document = Document(BytesIO(request.template))
        except (OSError, ValueError, KeyError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[DocumentType].fail(
                f"{c.Cli.DocxError.DOCUMENT_LOAD_FAILED}: {detail}"
            )
        return r[DocumentType].ok(document)

    @classmethod
    def _apply_source_date(
        cls, document: DocumentType, source_date_epoch: int | None
    ) -> None:
        if source_date_epoch is None:
            return
        moment = datetime.fromtimestamp(source_date_epoch, tz=UTC)
        core_props = document.core_properties
        core_props.created = moment
        core_props.modified = moment

    @classmethod
    def _serialize_document(
        cls, document: DocumentType, *, source_date_epoch: int | None = None
    ) -> p.Result[bytes]:
        target = BytesIO()
        try:
            document.save(target)
        except (OSError, TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[bytes].fail(f"{c.Cli.DocxError.SERIALIZE_FAILED}: {detail}")
        content = target.getvalue()
        if not content:
            return r[bytes].fail(str(c.Cli.DocxError.SERIALIZE_FAILED))
        if source_date_epoch is None:
            return r[bytes].ok(content)
        return cls._normalize_archive(content, source_date_epoch)

    @classmethod
    def _normalize_archive(
        cls, content: bytes, source_date_epoch: int
    ) -> p.Result[bytes]:
        member_date = cls._stable_member_date(source_date_epoch)
        target = BytesIO()
        try:
            cls._rewrite_members(content, target, member_date)
        except (OSError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[bytes].fail(f"{c.Cli.DocxError.SERIALIZE_FAILED}: {detail}")
        normalized_content = target.getvalue()
        if not normalized_content:
            return r[bytes].fail(str(c.Cli.DocxError.SERIALIZE_FAILED))
        return r[bytes].ok(normalized_content)

    @staticmethod
    def _stable_member_date(
        source_date_epoch: int,
    ) -> tuple[int, int, int, int, int, int]:
        moment = datetime.fromtimestamp(source_date_epoch, tz=UTC)
        zip_floor = datetime(_ZIP_EPOCH_FLOOR_YEAR, 1, 1, tzinfo=UTC)
        stamped = max(moment, zip_floor)
        return (
            stamped.year,
            stamped.month,
            stamped.day,
            stamped.hour,
            stamped.minute,
            stamped.second,
        )

    @staticmethod
    def _rewrite_members(
        content: bytes,
        target: BytesIO,
        member_date: tuple[int, int, int, int, int, int],
    ) -> None:
        with (
            ZipFile(BytesIO(content)) as source,
            ZipFile(target, "w", ZIP_DEFLATED) as normalized,
        ):
            for info in source.infolist():
                stable = ZipInfo(filename=info.filename, date_time=member_date)
                stable.compress_type = info.compress_type
                stable.external_attr = info.external_attr
                stable.internal_attr = info.internal_attr
                stable.create_system = info.create_system
                normalized.writestr(stable, source.read(info.filename))

    @classmethod
    def _apply_document(
        cls, document: DocumentType, plan: m.Cli.DocxDocumentPlan
    ) -> None:
        cls._apply_core_properties(document, plan.core_properties)
        for paragraph in plan.paragraphs:
            cls._apply_paragraph(document, paragraph)
        for table in plan.tables:
            cls._apply_table(document, table)
        cls._apply_sections(document, plan.sections)

    @classmethod
    def _apply_core_properties(
        cls, document: DocumentType, properties: t.JsonMapping
    ) -> None:
        core_props = document.core_properties
        for key, value in properties.items():
            if hasattr(core_props, key):
                setattr(core_props, key, value)

    @classmethod
    def _apply_paragraph(
        cls,
        container: DocumentType | _DocxParagraphContainer,
        plan: m.Cli.DocxParagraphPlan,
        paragraph: Paragraph | None = None,
    ) -> Paragraph:
        if paragraph is None:
            paragraph = container.add_paragraph()
        if plan.style:
            paragraph.style = plan.style
        if plan.alignment:
            paragraph.alignment = cls._ALIGNMENT_MAP[plan.alignment]
        if plan.style_spec:
            cls._apply_paragraph_style(paragraph, plan.style_spec)
        for run in plan.runs:
            cls._apply_run(paragraph, run)
        return paragraph

    @classmethod
    def _apply_paragraph_style(
        cls, paragraph: Paragraph, spec: m.Cli.DocxParagraphStyleSpec
    ) -> None:
        if spec.font:
            if not paragraph.runs:
                paragraph.add_run()
            cls._apply_font(paragraph.runs[0].font, spec.font)
        if spec.paragraph_format:
            cls._apply_paragraph_format(
                paragraph.paragraph_format, spec.paragraph_format
            )

    @classmethod
    def _apply_run(cls, paragraph: Paragraph, plan: m.Cli.DocxRunPlan) -> None:
        run = paragraph.add_run(plan.text)
        if plan.style and plan.style.font:
            cls._apply_font(run.font, plan.style.font)

    @classmethod
    def _apply_font(cls, font: Font, spec: m.Cli.DocxFontSpec) -> None:
        if spec.name is not None:
            font.name = spec.name
        if spec.size is not None:
            font.size = Pt(spec.size)
        if spec.bold is not None:
            font.bold = spec.bold
        if spec.italic is not None:
            font.italic = spec.italic
        if spec.underline is not None:
            font.underline = cls._UNDERLINE_MAP.get(spec.underline, WD_UNDERLINE.SINGLE)
        if spec.strike is not None:
            font.strike = spec.strike
        if spec.color is not None:
            font.color.rgb = cls._color_value(spec.color)
        if spec.highlight is not None:
            font.highlight_color = cls._HIGHLIGHT_MAP.get(spec.highlight)
        if spec.superscript is not None:
            font.superscript = spec.superscript
        if spec.subscript is not None:
            font.subscript = spec.subscript
        if spec.all_caps is not None:
            font.all_caps = spec.all_caps
        if spec.small_caps is not None:
            font.small_caps = spec.small_caps

    @classmethod
    def _apply_paragraph_format(
        cls, fmt: ParagraphFormat, spec: m.Cli.DocxParagraphFormatSpec
    ) -> None:
        if spec.alignment is not None:
            fmt.alignment = cls._ALIGNMENT_MAP[spec.alignment]
        if spec.space_before is not None:
            fmt.space_before = Inches(spec.space_before / 72)  # approximate points
        if spec.space_after is not None:
            fmt.space_after = Inches(spec.space_after / 72)
        if spec.line_spacing is not None:
            fmt.line_spacing = spec.line_spacing
        if spec.first_line_indent is not None:
            fmt.first_line_indent = Inches(spec.first_line_indent / 72)
        if spec.left_indent is not None:
            fmt.left_indent = Inches(spec.left_indent / 72)
        if spec.right_indent is not None:
            fmt.right_indent = Inches(spec.right_indent / 72)
        if spec.keep_together is not None:
            fmt.keep_together = spec.keep_together
        if spec.keep_with_next is not None:
            fmt.keep_with_next = spec.keep_with_next
        if spec.page_break_before is not None:
            fmt.page_break_before = spec.page_break_before
        if spec.widow_control is not None:
            fmt.widow_control = spec.widow_control

    @classmethod
    def _apply_table(cls, document: DocumentType, plan: m.Cli.DocxTablePlan) -> None:
        table = document.add_table(rows=len(plan.rows), cols=cls._table_columns(plan))
        if plan.style:
            table.style = plan.style
        for row_idx, row_plan in enumerate(plan.rows):
            for cell_idx, cell_plan in enumerate(row_plan.cells):
                cell = table.rows[row_idx].cells[cell_idx]
                for idx, paragraph in enumerate(cell_plan.paragraphs):
                    if idx == 0:
                        cell.paragraphs[0].text = ""
                        cls._apply_paragraph(cell, paragraph, cell.paragraphs[0])
                    else:
                        cls._apply_paragraph(cell, paragraph)

    @classmethod
    def _table_columns(cls, plan: m.Cli.DocxTablePlan) -> int:
        if not plan.rows:
            return 0
        return max(len(row.cells) for row in plan.rows) if plan.rows else 0

    @classmethod
    def _apply_sections(
        cls, document: DocumentType, sections: tuple[m.Cli.DocxSectionPlan, ...]
    ) -> None:
        for section_plan in sections:
            section = document.add_section()
            if section_plan.width is not None:
                section.page_width = Inches(
                    section_plan.width / 914400
                )  # EMU to inches
            if section_plan.height is not None:
                section.page_height = Inches(section_plan.height / 914400)
            if section_plan.orientation == "landscape":
                section.orientation = WD_ORIENT.LANDSCAPE
            if section_plan.left_margin is not None:
                section.left_margin = Inches(section_plan.left_margin / 914400)
            if section_plan.right_margin is not None:
                section.right_margin = Inches(section_plan.right_margin / 914400)
            if section_plan.top_margin is not None:
                section.top_margin = Inches(section_plan.top_margin / 914400)
            if section_plan.bottom_margin is not None:
                section.bottom_margin = Inches(section_plan.bottom_margin / 914400)

    @staticmethod
    def _color_value(color: m.Cli.DocxColor) -> RGBColor:
        if color.kind == "rgb":
            value = color.value
            if len(value) == _HEX_COLOR_WITH_ALPHA_LENGTH:
                value = value[2:]
            return RGBColor.from_string(value)
        if color.kind == "theme":
            # python-docx does not expose theme colors directly; fall back to a
            # neutral gray so the document remains valid.
            return RGBColor(0x80, 0x80, 0x80)
        return RGBColor(0x00, 0x00, 0x00)


__all__: tuple[str, ...] = ("FlextCliUtilitiesDocxRenderer",)

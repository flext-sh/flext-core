"""Generic style catalog extraction and format-template emission."""

from __future__ import annotations

from hashlib import sha256

from flext_cli import m, p, r

from .xlsx_style_codec import FlextCliUtilitiesXlsxStyleCodec
from .xlsx_workbook_io import FlextCliUtilitiesXlsxWorkbookIo


class FlextCliUtilitiesXlsxStyleCatalog(
    FlextCliUtilitiesXlsxStyleCodec, FlextCliUtilitiesXlsxWorkbookIo
):
    """Own visual deduplication, deterministic names, and blank templates."""

    # NOTE (multi-agent, mro-j2yt.1): cells identify source style IDs through
    # public openpyxl properties; protection never enters the visual signature.
    @staticmethod
    def _style_name(prefix: str, visual: m.Cli.XlsxVisualStyleSpec) -> str:
        digest = sha256(repr(visual).encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

    @staticmethod
    def _source_style_id(item: m.Cli.XlsxSourceVisualStyle) -> int:
        """Return the deterministic source-style sort key."""
        return item.source_style_id

    @classmethod
    def _source_visuals(
        cls, source: bytes
    ) -> p.Result[tuple[m.Cli.XlsxSourceVisualStyle, ...]]:
        workbook_result = cls._load_workbook(source)
        if workbook_result.failure:
            return r[tuple[m.Cli.XlsxSourceVisualStyle, ...]].fail(
                workbook_result.error or "Workbook load failed"
            )
        seen: frozenset[int] = frozenset()
        source_styles: tuple[m.Cli.XlsxSourceVisualStyle, ...] = ()
        for worksheet in workbook_result.value.worksheets:
            for row in worksheet.iter_rows():
                for cell in row:
                    if (
                        cell.value is None
                        and not cell.has_style
                        and cell.comment is None
                        and cell.hyperlink is None
                    ):
                        continue
                    source_style_id = cell.style_id
                    if source_style_id in seen:
                        continue
                    visual_result = cls._visual_from_styleable(cell)
                    if visual_result.failure:
                        return r[tuple[m.Cli.XlsxSourceVisualStyle, ...]].fail(
                            visual_result.error
                            or f"Style extraction failed: {source_style_id}"
                        )
                    seen = seen.union((source_style_id,))
                    source_styles = (
                        *source_styles,
                        m.Cli.XlsxSourceVisualStyle(
                            source_style_id=source_style_id, visual=visual_result.value
                        ),
                    )
        ordered = tuple(sorted(source_styles, key=cls._source_style_id))
        return r[tuple[m.Cli.XlsxSourceVisualStyle, ...]].ok(ordered)

    @classmethod
    def xlsx_style_catalog(
        cls, request: m.Cli.XlsxStyleCatalogRequest
    ) -> p.Result[m.Cli.XlsxStyleCatalog]:
        """Extract all cell-used visual styles and deduplicate them."""
        source_result = cls._source_visuals(request.source)
        if source_result.failure:
            return r[m.Cli.XlsxStyleCatalog].fail(
                source_result.error or "Style catalog extraction failed"
            )
        styles: tuple[m.Cli.XlsxNamedStyleSpec, ...] = ()
        style_map: tuple[m.Cli.XlsxStyleMapEntry, ...] = ()
        for source in source_result.value:
            existing = next(
                (style for style in styles if style.visual == source.visual), None
            )
            if existing is None:
                existing = m.Cli.XlsxNamedStyleSpec(
                    name=cls._style_name(request.style_name_prefix, source.visual),
                    visual=source.visual,
                )
                if any(style.name == existing.name for style in styles):
                    return r[m.Cli.XlsxStyleCatalog].fail(
                        f"Deterministic style-name collision: {existing.name}"
                    )
                styles = (*styles, existing)
            style_map = (
                *style_map,
                m.Cli.XlsxStyleMapEntry(
                    source_style_id=source.source_style_id, style_name=existing.name
                ),
            )
        return r[m.Cli.XlsxStyleCatalog].ok(
            m.Cli.XlsxStyleCatalog(style_map=style_map, styles=styles)
        )

    @classmethod
    def xlsx_style_template(
        cls, request: m.Cli.XlsxStyleTemplateRequest
    ) -> p.Result[m.Cli.XlsxStyleTemplateResult]:
        """Emit a blank workbook containing only deduplicated visual styles."""
        catalog_result = cls.xlsx_style_catalog(
            m.Cli.XlsxStyleCatalogRequest(
                source=request.source, style_name_prefix=request.style_name_prefix
            )
        )
        if catalog_result.failure:
            return r[m.Cli.XlsxStyleTemplateResult].fail(
                catalog_result.error or "Style catalog extraction failed"
            )
        catalog = catalog_result.value
        workbook = cls._new_workbook()
        for spec in catalog.styles:
            workbook.add_named_style(cls._named_style(spec))
        content_result = cls._serialize_workbook(workbook)
        if content_result.failure:
            return r[m.Cli.XlsxStyleTemplateResult].fail(
                content_result.error or "Style template serialization failed"
            )
        return r[m.Cli.XlsxStyleTemplateResult].ok(
            m.Cli.XlsxStyleTemplateResult(
                content=content_result.value, style_map=catalog.style_map
            )
        )


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxStyleCatalog",)

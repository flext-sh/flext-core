"""Typed style catalog and format-template requests."""

from __future__ import annotations

from typing import Annotated

from flext_core import m

from .xlsx_styles import FlextCliModelsXlsxStyles


class FlextCliModelsXlsxStyleCatalog:
    """Immutable visual-style extraction and emission contracts."""

    # NOTE (multi-agent, mro-j2yt.1): source style identifiers are retained
    # while protection is intentionally excluded from visual signatures.
    class XlsxStyleMapEntry(m.FrozenModel):
        source_style_id: Annotated[
            int, m.Field(ge=0, description="Source workbook style identifier.")
        ]
        style_name: Annotated[
            str, m.Field(min_length=1, description="Generated named style key.")
        ]

    class XlsxSourceVisualStyle(m.FrozenModel):
        source_style_id: Annotated[
            int, m.Field(ge=0, description="Source workbook style identifier.")
        ]
        visual: FlextCliModelsXlsxStyles.XlsxVisualStyleSpec = m.Field(
            description="Protection-free source visual signature."
        )

    class XlsxStyleCatalog(m.FrozenModel):
        style_map: tuple[FlextCliModelsXlsxStyleCatalog.XlsxStyleMapEntry, ...] = (
            m.Field(default=(), strict=False, description="Source style assignments.")
        )
        styles: tuple[FlextCliModelsXlsxStyles.XlsxNamedStyleSpec, ...] = m.Field(
            default=(), strict=False, description="Unique visual styles."
        )

    class XlsxStyleCatalogRequest(m.FrozenModel):
        source: Annotated[
            bytes, m.Field(min_length=1, description="Source workbook bytes.")
        ]
        style_name_prefix: Annotated[
            str,
            m.Field(
                min_length=1,
                max_length=64,
                pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
                description="Generated style-name prefix.",
            ),
        ]

    class XlsxStyleTemplateRequest(m.FrozenModel):
        source: Annotated[
            bytes, m.Field(min_length=1, description="Source workbook bytes.")
        ]
        style_name_prefix: Annotated[
            str,
            m.Field(
                min_length=1,
                max_length=64,
                pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$",
                description="Generated style-name prefix.",
            ),
        ]

    class XlsxStyleTemplateResult(m.FrozenModel):
        content: Annotated[
            bytes, m.Field(min_length=1, description="Format-only workbook bytes.")
        ]
        style_map: tuple[FlextCliModelsXlsxStyleCatalog.XlsxStyleMapEntry, ...] = (
            m.Field(default=(), strict=False, description="Source style assignments.")
        )


__all__: tuple[str, ...] = ("FlextCliModelsXlsxStyleCatalog",)

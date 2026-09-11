"""Typed XLSX colors and font declarations."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m


class FlextCliModelsXlsxStylePrimitives:
    """Immutable style primitives independent of workbook implementations."""

    # NOTE (multi-agent, mro-j2yt.1): optional values preserve the source
    # OOXML distinction between an absent attribute and an explicit false.
    class XlsxRgbColor(m.FrozenModel):
        kind: Literal["rgb"] = m.Field(default="rgb", description="Color kind.")
        value: Annotated[
            str,
            m.Field(
                pattern=r"^(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$",
                description="RGB or ARGB value.",
            ),
        ]
        tint: Annotated[float, m.Field(ge=-1, le=1, description="Color tint.")] = 0

    class XlsxIndexedColor(m.FrozenModel):
        kind: Literal["indexed"] = m.Field(default="indexed", description="Color kind.")
        value: Annotated[int, m.Field(ge=0, description="Indexed color value.")]
        tint: Annotated[float, m.Field(ge=-1, le=1, description="Color tint.")] = 0

    class XlsxThemeColor(m.FrozenModel):
        kind: Literal["theme"] = m.Field(default="theme", description="Color kind.")
        value: Annotated[int, m.Field(ge=0, description="Theme color value.")]
        tint: Annotated[float, m.Field(ge=-1, le=1, description="Color tint.")] = 0

    class XlsxAutomaticColor(m.FrozenModel):
        kind: Literal["auto"] = m.Field(default="auto", description="Color kind.")
        tint: Annotated[float, m.Field(ge=-1, le=1, description="Color tint.")] = 0

    type XlsxColor = Annotated[
        XlsxRgbColor | XlsxIndexedColor | XlsxThemeColor | XlsxAutomaticColor,
        m.Field(discriminator="kind"),
    ]

    class XlsxFontSpec(m.FrozenModel):
        name: str | None = m.Field(default=None, description="Font family.")
        size: Annotated[float, m.Field(gt=0, description="Font size.")] | None = None
        bold: bool | None = m.Field(default=None, description="Bold font state.")
        italic: bool | None = m.Field(default=None, description="Italic font state.")
        strike: bool | None = m.Field(default=None, description="Strike font state.")
        outline: bool | None = m.Field(default=None, description="Outline font state.")
        shadow: bool | None = m.Field(default=None, description="Shadow font state.")
        condense: bool | None = m.Field(
            default=None, description="Condensed font state."
        )
        extend: bool | None = m.Field(default=None, description="Extended font state.")
        underline: (
            Literal["single", "double", "singleAccounting", "doubleAccounting"] | None
        ) = m.Field(default=None, description="Underline style.")
        vertical_align: Literal["superscript", "subscript", "baseline"] | None = (
            m.Field(default=None, description="Vertical font alignment.")
        )
        color: FlextCliModelsXlsxStylePrimitives.XlsxColor | None = m.Field(
            default=None, description="Optional font color."
        )
        charset: Annotated[int, m.Field(ge=0, description="Font charset.")] | None = (
            None
        )
        family: (
            Annotated[float, m.Field(ge=0, description="Font family class.")] | None
        ) = None
        scheme: Literal["major", "minor"] | None = m.Field(
            default=None, description="Theme font scheme."
        )


__all__: tuple[str, ...] = ("FlextCliModelsXlsxStylePrimitives",)

"""Typed DOCX style declarations for generic Word rendering."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m


class FlextCliModelsDocxStyles:
    """Immutable DOCX style primitives independent of python-docx."""

    # NOTE (multi-agent, mro-j2yt.1): style primitives are data-only and
    # carry no document-specific or customer policy.

    class DocxRgbColor(m.FrozenModel):
        kind: Literal["rgb"] = m.Field(default="rgb", description="Color kind.")
        value: Annotated[
            str,
            m.Field(
                pattern=r"^(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$",
                description="RGB or ARGB value.",
            ),
        ]
        tint: Annotated[float, m.Field(ge=-1, le=1, description="Color tint.")] = 0

    class DocxThemeColor(m.FrozenModel):
        kind: Literal["theme"] = m.Field(default="theme", description="Color kind.")
        value: Annotated[int, m.Field(ge=0, description="Theme color value.")]
        tint: Annotated[float, m.Field(ge=-1, le=1, description="Color tint.")] = 0

    class DocxAutomaticColor(m.FrozenModel):
        kind: Literal["auto"] = m.Field(default="auto", description="Color kind.")
        tint: Annotated[float, m.Field(ge=-1, le=1, description="Color tint.")] = 0

    type DocxColor = Annotated[
        DocxRgbColor | DocxThemeColor | DocxAutomaticColor,
        m.Field(discriminator="kind"),
    ]

    class DocxFontSpec(m.FrozenModel):
        name: str | None = m.Field(default=None, description="Font family.")
        size: Annotated[float, m.Field(gt=0, description="Font size.")] | None = None
        bold: bool | None = m.Field(default=None, description="Bold font state.")
        italic: bool | None = m.Field(default=None, description="Italic font state.")
        underline: (
            Literal["single", "double", "singleAccounting", "doubleAccounting"] | None
        ) = m.Field(default=None, description="Underline style.")
        strike: bool | None = m.Field(default=None, description="Strike font state.")
        color: FlextCliModelsDocxStyles.DocxColor | None = m.Field(
            default=None, description="Optional font color."
        )
        highlight: (
            Literal[
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
            ]
            | None
        ) = m.Field(default=None, description="Optional highlight color.")
        superscript: bool | None = m.Field(default=None, description="Superscript.")
        subscript: bool | None = m.Field(default=None, description="Subscript.")
        all_caps: bool | None = m.Field(default=None, description="All caps.")
        small_caps: bool | None = m.Field(default=None, description="Small caps.")

    class DocxParagraphFormatSpec(m.FrozenModel):
        alignment: (
            Literal["left", "center", "right", "justify", "distribute"] | None
        ) = m.Field(default=None, description="Paragraph alignment.")
        space_before: (
            Annotated[float, m.Field(ge=0, description="Space before.")] | None
        ) = None
        space_after: (
            Annotated[float, m.Field(ge=0, description="Space after.")] | None
        ) = None
        line_spacing: (
            Annotated[float, m.Field(gt=0, description="Line spacing multiplier.")]
            | None
        ) = None
        first_line_indent: (
            Annotated[float, m.Field(description="First line indent.")] | None
        ) = None
        left_indent: Annotated[float, m.Field(description="Left indent.")] | None = None
        right_indent: Annotated[float, m.Field(description="Right indent.")] | None = (
            None
        )
        keep_together: bool | None = m.Field(default=None, description="Keep together.")
        keep_with_next: bool | None = m.Field(
            default=None, description="Keep with next paragraph."
        )
        page_break_before: bool | None = m.Field(
            default=None, description="Page break before paragraph."
        )
        widow_control: bool | None = m.Field(default=None, description="Widow control.")

    class DocxRunStyleSpec(m.FrozenModel):
        font: FlextCliModelsDocxStyles.DocxFontSpec | None = m.Field(
            default=None, description="Optional run font specification."
        )

    class DocxParagraphStyleSpec(m.FrozenModel):
        font: FlextCliModelsDocxStyles.DocxFontSpec | None = m.Field(
            default=None, description="Optional paragraph font specification."
        )
        paragraph_format: FlextCliModelsDocxStyles.DocxParagraphFormatSpec | None = (
            m.Field(
                default=None, description="Optional paragraph format specification."
            )
        )


__all__: tuple[str, ...] = ("FlextCliModelsDocxStyles",)

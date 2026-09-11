"""Typed border, alignment, and named visual style declarations."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m

from .xlsx_style_fills import FlextCliModelsXlsxStyleFills
from .xlsx_style_primitives import FlextCliModelsXlsxStylePrimitives


class FlextCliModelsXlsxStyles(
    FlextCliModelsXlsxStylePrimitives, FlextCliModelsXlsxStyleFills
):
    """Immutable style specifications kept separate from protection rules."""

    # NOTE (multi-agent, mro-j2yt.1): named styles are visual-only; runtime
    # protection is represented by the dedicated protection models.
    class XlsxBorderSideSpec(m.FrozenModel):
        style: (
            Literal[
                "dashDot",
                "dashDotDot",
                "dashed",
                "dotted",
                "double",
                "hair",
                "medium",
                "mediumDashDot",
                "mediumDashDotDot",
                "mediumDashed",
                "slantDashDot",
                "thick",
                "thin",
            ]
            | None
        ) = m.Field(default=None, description="Border style.")
        color: FlextCliModelsXlsxStyles.XlsxColor | None = m.Field(
            default=None, description="Optional border color."
        )

    class XlsxBorderSpec(m.FrozenModel):
        left: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional left border."
        )
        right: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional right border."
        )
        top: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional top border."
        )
        bottom: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional bottom border."
        )
        start: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional logical start border."
        )
        end: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional logical end border."
        )
        diagonal: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional diagonal border."
        )
        vertical: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional internal vertical border."
        )
        horizontal: FlextCliModelsXlsxStyles.XlsxBorderSideSpec | None = m.Field(
            default=None, description="Optional internal horizontal border."
        )
        diagonal_up: bool = m.Field(default=False, description="Draw upward diagonal.")
        diagonal_down: bool = m.Field(
            default=False, description="Draw downward diagonal."
        )
        outline: bool = m.Field(default=True, description="Apply border as outline.")

    class XlsxAlignmentSpec(m.FrozenModel):
        horizontal: (
            Literal[
                "general",
                "left",
                "center",
                "right",
                "fill",
                "justify",
                "centerContinuous",
                "distributed",
            ]
            | None
        ) = m.Field(default=None, description="Horizontal alignment.")
        vertical: (
            Literal["top", "center", "bottom", "justify", "distributed"] | None
        ) = m.Field(default=None, description="Vertical alignment.")
        wrap_text: bool | None = m.Field(default=None, description="Wrap cell text.")
        shrink_to_fit: bool | None = m.Field(
            default=None, description="Shrink cell text."
        )
        text_rotation: Annotated[
            int, m.Field(ge=0, le=180, description="Text rotation.")
        ] = 0
        indent: Annotated[float, m.Field(ge=0, description="Text indent.")] = 0
        relative_indent: float = m.Field(default=0, description="Relative text indent.")
        justify_last_line: bool | None = m.Field(
            default=None, description="Justify the final line."
        )
        reading_order: Annotated[
            float, m.Field(ge=0, description="Text reading order.")
        ] = 0

    class XlsxVisualStyleSpec(m.FrozenModel):
        font: FlextCliModelsXlsxStylePrimitives.XlsxFontSpec = m.Field(
            description="Font specification."
        )
        fill: FlextCliModelsXlsxStyleFills.XlsxFillSpec = m.Field(
            description="Fill specification."
        )
        border: FlextCliModelsXlsxStyles.XlsxBorderSpec = m.Field(
            description="Border specification."
        )
        alignment: FlextCliModelsXlsxStyles.XlsxAlignmentSpec = m.Field(
            description="Alignment specification."
        )
        number_format: Annotated[
            str, m.Field(min_length=1, description="Excel number format.")
        ] = "General"

    class XlsxNamedStyleSpec(m.FrozenModel):
        name: Annotated[str, m.Field(min_length=1, description="Named style key.")]
        visual: FlextCliModelsXlsxStyles.XlsxVisualStyleSpec = m.Field(
            description="Protection-free visual signature."
        )


__all__: tuple[str, ...] = ("FlextCliModelsXlsxStyles",)

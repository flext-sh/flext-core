"""Typed XLSX pattern and gradient fill declarations."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m

from .xlsx_style_primitives import FlextCliModelsXlsxStylePrimitives


class FlextCliModelsXlsxStyleFills:
    """Immutable fill variants used by visual style specifications."""

    # NOTE (multi-agent, mro-j2yt.1): the discriminator mirrors public OOXML
    # fill variants and keeps gradient configuration outside implementation code.
    class XlsxPatternFillSpec(m.FrozenModel):
        kind: Literal["pattern"] = m.Field(default="pattern", description="Fill kind.")
        pattern: (
            Literal[
                "solid",
                "darkDown",
                "darkGray",
                "darkGrid",
                "darkHorizontal",
                "darkTrellis",
                "darkUp",
                "darkVertical",
                "gray0625",
                "gray125",
                "lightDown",
                "lightGray",
                "lightGrid",
                "lightHorizontal",
                "lightTrellis",
                "lightUp",
                "lightVertical",
                "mediumGray",
            ]
            | None
        ) = m.Field(default=None, description="Fill pattern.")
        foreground: FlextCliModelsXlsxStylePrimitives.XlsxColor | None = m.Field(
            default=None, description="Optional foreground color."
        )
        background: FlextCliModelsXlsxStylePrimitives.XlsxColor | None = m.Field(
            default=None, description="Optional background color."
        )

    class XlsxGradientStop(m.FrozenModel):
        color: FlextCliModelsXlsxStylePrimitives.XlsxColor = m.Field(
            description="Gradient stop color."
        )
        position: Annotated[
            float, m.Field(ge=0, le=1, description="Gradient stop position.")
        ]

    class XlsxGradientFillSpec(m.FrozenModel):
        kind: Literal["gradient"] = m.Field(
            default="gradient", description="Fill kind."
        )
        mode: Literal["linear", "path"] = m.Field(
            default="linear", description="Gradient mode."
        )
        degree: float = m.Field(default=0, description="Linear gradient angle.")
        left: float = m.Field(default=0, description="Path left extent.")
        right: float = m.Field(default=0, description="Path right extent.")
        top: float = m.Field(default=0, description="Path top extent.")
        bottom: float = m.Field(default=0, description="Path bottom extent.")
        stops: tuple[FlextCliModelsXlsxStyleFills.XlsxGradientStop, ...] = m.Field(
            min_length=1, strict=False, description="Ordered gradient stops."
        )

    type XlsxFillSpec = Annotated[
        XlsxPatternFillSpec | XlsxGradientFillSpec, m.Field(discriminator="kind")
    ]


__all__: tuple[str, ...] = ("FlextCliModelsXlsxStyleFills",)

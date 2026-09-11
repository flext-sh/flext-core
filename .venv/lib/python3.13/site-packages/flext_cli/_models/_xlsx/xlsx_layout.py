"""Typed worksheet layout declarations for generic XLSX rendering."""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m

from .xlsx_cells import FlextCliModelsXlsxCells


class FlextCliModelsXlsxLayout:
    """Immutable layout operations owned by one worksheet plan."""

    # NOTE (multi-agent, mro-j2yt.1): layout is data-only and contains no
    # document or customer policy.
    class XlsxMergePlan(m.FrozenModel):
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Merged cell range."
        )

    class XlsxCommentPlan(m.FrozenModel):
        at: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="Comment cell."
        )
        text: Annotated[str, m.Field(min_length=1, description="Comment text.")]
        author: Annotated[str, m.Field(min_length=1, description="Comment author.")]
        width: Annotated[int, m.Field(gt=0, description="Comment width.")] | None = None
        height: Annotated[int, m.Field(gt=0, description="Comment height.")] | None = (
            None
        )

    class XlsxExternalHyperlinkPlan(m.FrozenModel):
        kind: Literal["external"] = m.Field(
            default="external", description="Hyperlink kind."
        )
        at: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="Hyperlink cell."
        )
        target: Annotated[
            str, m.Field(min_length=1, description="External hyperlink target.")
        ]

    class XlsxNavigationHyperlinkPlan(m.FrozenModel):
        kind: Literal["navigation"] = m.Field(
            default="navigation", description="Hyperlink kind."
        )
        at: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="Hyperlink cell."
        )
        destination_sheet: Annotated[
            str, m.Field(min_length=1, description="Destination worksheet.")
        ]
        destination: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="Destination address."
        )

    type XlsxHyperlinkPlan = Annotated[
        XlsxExternalHyperlinkPlan | XlsxNavigationHyperlinkPlan,
        m.Field(discriminator="kind"),
    ]

    class XlsxDimensionPlan(m.FrozenModel):
        axis: Literal["row", "column"] = m.Field(description="Dimension axis.")
        first: Annotated[int, m.Field(ge=1, description="First index.")]
        last: Annotated[int, m.Field(ge=1, description="Last index.")]
        size: Annotated[float, m.Field(gt=0, description="Explicit size.")] | None = (
            None
        )
        hidden: bool = m.Field(default=False, description="Hidden state.")

    class XlsxGroupPlan(m.FrozenModel):
        axis: Literal["row", "column"] = m.Field(description="Grouped axis.")
        first: Annotated[int, m.Field(ge=1, description="First index.")]
        last: Annotated[int, m.Field(ge=1, description="Last index.")]
        outline_level: Annotated[int, m.Field(ge=1, le=8, description="Outline level.")]
        hidden: bool = m.Field(default=False, description="Hidden state.")

    class XlsxFreezePanePlan(m.FrozenModel):
        at: FlextCliModelsXlsxCells.XlsxCellAddress = m.Field(
            description="First scrolling cell."
        )

    class XlsxAutoFilterPlan(m.FrozenModel):
        area: FlextCliModelsXlsxCells.XlsxCellRange = m.Field(
            description="Auto-filter range."
        )

    class XlsxViewPlan(m.FrozenModel):
        visibility: Literal["visible", "hidden", "veryHidden"] = m.Field(
            default="visible", description="Worksheet visibility."
        )
        tab_color: (
            Annotated[
                str,
                m.Field(
                    pattern=r"^(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$",
                    description="RGB or ARGB tab color.",
                ),
            ]
            | None
        ) = None

    class XlsxSheetLayoutPlan(m.FrozenModel):
        merges: tuple[FlextCliModelsXlsxLayout.XlsxMergePlan, ...] = m.Field(
            default=(), strict=False, description="Merged ranges."
        )
        comments: tuple[FlextCliModelsXlsxLayout.XlsxCommentPlan, ...] = m.Field(
            default=(), strict=False, description="Cell comments."
        )
        hyperlinks: tuple[FlextCliModelsXlsxLayout.XlsxHyperlinkPlan, ...] = m.Field(
            default=(), strict=False, description="Cell hyperlinks."
        )
        dimensions: tuple[FlextCliModelsXlsxLayout.XlsxDimensionPlan, ...] = m.Field(
            default=(), strict=False, description="Row and column dimensions."
        )
        groups: tuple[FlextCliModelsXlsxLayout.XlsxGroupPlan, ...] = m.Field(
            default=(), strict=False, description="Row and column groups."
        )
        freeze_pane: FlextCliModelsXlsxLayout.XlsxFreezePanePlan | None = m.Field(
            default=None, description="Optional freeze pane."
        )
        auto_filter: FlextCliModelsXlsxLayout.XlsxAutoFilterPlan | None = m.Field(
            default=None, description="Optional auto-filter."
        )
        view: FlextCliModelsXlsxLayout.XlsxViewPlan | None = m.Field(
            default=None, description="Optional worksheet view."
        )


__all__: tuple[str, ...] = ("FlextCliModelsXlsxLayout",)

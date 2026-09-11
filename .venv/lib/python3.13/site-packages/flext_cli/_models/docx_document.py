"""Aggregate plans and byte-boundary results for generic DOCX rendering."""

from __future__ import annotations

from typing import Annotated

from flext_cli import t
from flext_cli._models._defaults import EMPTY_JSON_MAPPING
from flext_core import m

from .docx_styles import FlextCliModelsDocxStyles


class FlextCliModelsDocxDocument:
    """Immutable document, paragraph, and render request/result models."""

    # NOTE (multi-agent, mro-j2yt.1): consumers exchange only validated plans
    # and bytes; python-docx implementation objects never cross this boundary.

    class DocxRunPlan(m.FrozenModel):
        text: str = m.Field(default="", description="Run text.")
        style: FlextCliModelsDocxStyles.DocxRunStyleSpec = m.Field(
            default_factory=FlextCliModelsDocxStyles.DocxRunStyleSpec,
            description="Run style specification.",
        )

    class DocxParagraphPlan(m.FrozenModel):
        runs: tuple[FlextCliModelsDocxDocument.DocxRunPlan, ...] = m.Field(
            default=(), strict=False, description="Paragraph runs."
        )
        style: (
            Annotated[str, m.Field(min_length=1, description="Named paragraph style.")]
            | None
        ) = None
        style_spec: FlextCliModelsDocxStyles.DocxParagraphStyleSpec | None = m.Field(
            default=None, description="Inline paragraph style specification."
        )
        alignment: (
            Annotated[
                str,
                m.Field(
                    pattern=r"^(left|center|right|justify|distribute)$",
                    description="Paragraph alignment override.",
                ),
            ]
            | None
        ) = None

    class DocxTableCellPlan(m.FrozenModel):
        paragraphs: tuple[FlextCliModelsDocxDocument.DocxParagraphPlan, ...] = m.Field(
            default=(), strict=False, description="Cell paragraphs."
        )

    class DocxTableRowPlan(m.FrozenModel):
        cells: tuple[FlextCliModelsDocxDocument.DocxTableCellPlan, ...] = m.Field(
            default=(), strict=False, description="Row cells."
        )

    class DocxTablePlan(m.FrozenModel):
        rows: tuple[FlextCliModelsDocxDocument.DocxTableRowPlan, ...] = m.Field(
            default=(), strict=False, description="Table rows."
        )
        style: (
            Annotated[str, m.Field(min_length=1, description="Named table style.")]
            | None
        ) = None

    class DocxSectionPlan(m.FrozenModel):
        width: (
            Annotated[float, m.Field(gt=0, description="Page width in EMU.")] | None
        ) = None
        height: (
            Annotated[float, m.Field(gt=0, description="Page height in EMU.")] | None
        ) = None
        orientation: Annotated[
            str,
            m.Field(pattern=r"^(portrait|landscape)$", description="Page orientation."),
        ] = "portrait"
        left_margin: (
            Annotated[float, m.Field(ge=0, description="Left margin in EMU.")] | None
        ) = None
        right_margin: (
            Annotated[float, m.Field(ge=0, description="Right margin in EMU.")] | None
        ) = None
        top_margin: (
            Annotated[float, m.Field(ge=0, description="Top margin in EMU.")] | None
        ) = None
        bottom_margin: (
            Annotated[float, m.Field(ge=0, description="Bottom margin in EMU.")] | None
        ) = None

    class DocxDocumentPlan(m.FrozenModel):
        paragraphs: tuple[FlextCliModelsDocxDocument.DocxParagraphPlan, ...] = m.Field(
            default=(), strict=False, description="Document paragraphs."
        )
        tables: tuple[FlextCliModelsDocxDocument.DocxTablePlan, ...] = m.Field(
            default=(), strict=False, description="Document tables."
        )
        sections: tuple[FlextCliModelsDocxDocument.DocxSectionPlan, ...] = m.Field(
            default=(), strict=False, description="Document sections."
        )
        core_properties: t.JsonMapping = m.Field(
            default_factory=lambda: EMPTY_JSON_MAPPING,
            description="Core document properties.",
        )

    class DocxRenderRequest(m.FrozenModel):
        template: (
            Annotated[
                bytes, m.Field(min_length=1, description="Formatting template bytes.")
            ]
            | None
        ) = m.Field(default=None, description="Optional source document.")
        plan: FlextCliModelsDocxDocument.DocxDocumentPlan = m.Field(
            description="Validated document plan."
        )
        source_date_epoch: (
            Annotated[
                int,
                m.Field(
                    ge=0, description="Deterministic build epoch in seconds since 1970."
                ),
            ]
            | None
        ) = m.Field(
            default=None,
            description="Fix core-property and archive timestamps for reproducibility.",
        )

    class DocxRenderResult(m.FrozenModel):
        content: Annotated[
            bytes, m.Field(min_length=1, description="Rendered document bytes.")
        ]
        plan: FlextCliModelsDocxDocument.DocxDocumentPlan = m.Field(
            description="Exact source plan."
        )


__all__: tuple[str, ...] = ("FlextCliModelsDocxDocument",)

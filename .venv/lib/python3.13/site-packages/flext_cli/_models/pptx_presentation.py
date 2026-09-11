"""Aggregate plans and byte-boundary results for generic PPTX rendering."""

from __future__ import annotations

from typing import Annotated

from flext_cli import t
from flext_cli._models._defaults import EMPTY_JSON_MAPPING
from flext_core import m


class FlextCliModelsPptxPresentation:
    """Immutable presentation, slide, and render request/result models."""

    # NOTE (multi-agent, mro-j2yt.1): consumers exchange only validated plans
    # and bytes; python-pptx implementation objects never cross this boundary.

    class PptxSlidePlan(m.FrozenModel):
        title: str = m.Field(default="", description="Slide title.")

    class PptxPresentationPlan(m.FrozenModel):
        slides: tuple[FlextCliModelsPptxPresentation.PptxSlidePlan, ...] = m.Field(
            default=(), strict=False, description="Presentation slides."
        )
        core_properties: t.JsonMapping = m.Field(
            default_factory=lambda: EMPTY_JSON_MAPPING,
            description="Core document properties.",
        )

    class PptxRenderRequest(m.FrozenModel):
        template: (
            Annotated[
                bytes, m.Field(min_length=1, description="Formatting template bytes.")
            ]
            | None
        ) = m.Field(default=None, description="Optional source presentation.")
        plan: FlextCliModelsPptxPresentation.PptxPresentationPlan = m.Field(
            description="Validated presentation plan."
        )

    class PptxRenderResult(m.FrozenModel):
        content: Annotated[
            bytes, m.Field(min_length=1, description="Rendered presentation bytes.")
        ]
        plan: FlextCliModelsPptxPresentation.PptxPresentationPlan = m.Field(
            description="Exact source plan."
        )


__all__: tuple[str, ...] = ("FlextCliModelsPptxPresentation",)

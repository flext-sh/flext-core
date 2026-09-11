"""Generic model-driven PPTX renderer."""

from __future__ import annotations

from io import BytesIO

from pptx import Presentation
from pptx.presentation import Presentation as PresentationType

from flext_cli import c, m, p, r, t
from flext_cli._utilities._pptx._serializer import FlextCliUtilitiesPptxSerializer


class FlextCliUtilitiesPptxRenderer:
    """Render one immutable presentation plan into PPTX bytes."""

    # NOTE (multi-agent, mro-j2yt.1): python-pptx objects stay inside this
    # module; consumers exchange only validated plans and bytes.

    @classmethod
    def pptx_render(
        cls, request: m.Cli.PptxRenderRequest
    ) -> p.Result[m.Cli.PptxRenderResult]:
        """Render typed slides into presentation bytes."""
        presentation_result = cls._presentation_for_request(request)
        if presentation_result.failure:
            return r[m.Cli.PptxRenderResult].fail(
                presentation_result.error or str(c.Cli.PptxError.RENDER_FAILED)
            )
        presentation = presentation_result.value
        try:
            cls._apply_presentation(presentation, request.plan)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.PptxRenderResult].fail(
                f"{c.Cli.PptxError.RENDER_FAILED}: {detail}"
            )
        content = FlextCliUtilitiesPptxSerializer.pptx_save(presentation)
        if content.failure:
            return r[m.Cli.PptxRenderResult].fail(
                content.error or str(c.Cli.PptxError.SERIALIZE_FAILED)
            )
        return r[m.Cli.PptxRenderResult].ok(
            m.Cli.PptxRenderResult(content=content.value, plan=request.plan)
        )

    @classmethod
    def _presentation_for_request(
        cls, request: m.Cli.PptxRenderRequest
    ) -> p.Result[PresentationType]:
        if request.template is None:
            return r[PresentationType].ok(Presentation())
        try:
            presentation = Presentation(BytesIO(request.template))
        except (OSError, ValueError, KeyError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[PresentationType].fail(
                f"{c.Cli.PptxError.PRESENTATION_LOAD_FAILED}: {detail}"
            )
        return r[PresentationType].ok(presentation)

    @classmethod
    def _apply_presentation(
        cls, presentation: PresentationType, plan: m.Cli.PptxPresentationPlan
    ) -> None:
        cls._apply_pptx_core_properties(presentation, plan.core_properties)
        for slide in plan.slides:
            cls._apply_slide(presentation, slide)

    @classmethod
    def _apply_pptx_core_properties(
        cls, presentation: PresentationType, properties: t.JsonMapping
    ) -> None:
        core_props = presentation.core_properties
        for key, value in properties.items():
            if hasattr(core_props, key):
                setattr(core_props, key, value)

    @classmethod
    def _apply_slide(
        cls, presentation: PresentationType, plan: m.Cli.PptxSlidePlan
    ) -> None:
        layout = presentation.slide_layouts[0]
        new_slide = presentation.slides.add_slide(layout)
        if new_slide.shapes.title is not None:
            new_slide.shapes.title.text = plan.title


__all__: tuple[str, ...] = ("FlextCliUtilitiesPptxRenderer",)

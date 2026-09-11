"""Generic model-driven PPTX reader."""

from __future__ import annotations

from io import BytesIO
from types import MappingProxyType
from zipfile import BadZipFile

from pptx import Presentation
from pptx.exc import PackageNotFoundError
from pptx.presentation import Presentation as PresentationType

from flext_cli import c, m, p, r


class FlextCliUtilitiesPptxReader:
    """Read PPTX bytes into an immutable presentation plan."""

    # NOTE (multi-agent, mro-j2yt.1): python-pptx objects stay inside this
    # module; consumers receive only validated plans.

    @classmethod
    def pptx_read(cls, source: bytes) -> p.Result[m.Cli.PptxPresentationPlan]:
        """Read presentation bytes into a validated plan."""
        try:
            presentation = Presentation(BytesIO(source))
        except (OSError, ValueError, KeyError, BadZipFile, PackageNotFoundError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.PptxPresentationPlan].fail(
                f"{c.Cli.PptxError.PRESENTATION_LOAD_FAILED}: {detail}"
            )
        try:
            plan = cls._snapshot_presentation(presentation)
        except (TypeError, ValueError) as exc:
            detail = str(exc).strip() or exc.__class__.__name__
            return r[m.Cli.PptxPresentationPlan].fail(
                f"{c.Cli.PptxError.RENDER_FAILED}: {detail}"
            )
        return r[m.Cli.PptxPresentationPlan].ok(plan)

    @classmethod
    def _snapshot_presentation(
        cls, presentation: PresentationType
    ) -> m.Cli.PptxPresentationPlan:
        slides = tuple(
            m.Cli.PptxSlidePlan(
                title=slide.shapes.title.text if slide.shapes.title else ""
            )
            for slide in presentation.slides
        )
        properties: dict[str, str] = {}
        core_props = presentation.core_properties
        for key in ("author", "title", "subject", "keywords", "comments", "category"):
            value = getattr(core_props, key, None)
            if value:
                properties[key] = str(value)
        return m.Cli.PptxPresentationPlan(
            slides=slides, core_properties=MappingProxyType(properties)
        )


__all__: tuple[str, ...] = ("FlextCliUtilitiesPptxReader",)

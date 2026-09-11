"""Stable constants for the generic PPTX boundary."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextCliConstantsPptx:
    """Protocol constants shared by PPTX adapters and consumers."""

    # NOTE (multi-agent, mro-j2yt.1): keep PowerPoint protocol facts out of
    # consumer packages so every external PPTX dependency has one owner.

    PPTX_PRESENTATION_MEMBER: Final[str] = "ppt/presentation.xml"
    PPTX_SLIDES_MEMBER: Final[str] = "ppt/slides/slide1.xml"
    PPTX_CONTENT_TYPES_MEMBER: Final[str] = "[Content_Types].xml"
    PPTX_CORE_PROPERTIES_MEMBER: Final[str] = "docProps/core.xml"

    @unique
    class PptxError(StrEnum):
        """Stable failure codes returned by the PPTX service."""

        ARCHIVE_INVALID = "pptx_archive_invalid"
        PRESENTATION_LOAD_FAILED = "pptx_presentation_load_failed"
        PLAN_INVALID = "pptx_plan_invalid"
        RENDER_FAILED = "pptx_render_failed"
        SERIALIZE_FAILED = "pptx_serialize_failed"


__all__: tuple[str, ...] = ("FlextCliConstantsPptx",)

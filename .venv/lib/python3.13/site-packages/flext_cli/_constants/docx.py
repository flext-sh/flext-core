"""Stable constants for the generic DOCX boundary."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextCliConstantsDocx:
    """Protocol constants shared by DOCX adapters and consumers."""

    # NOTE (multi-agent, mro-j2yt.1): keep Word protocol facts out of
    # consumer packages so every external DOCX dependency has one owner.

    DOCX_DOCUMENT_MEMBER: Final[str] = "word/document.xml"
    DOCX_STYLES_MEMBER: Final[str] = "word/styles.xml"
    DOCX_RELATIONSHIPS_MEMBER: Final[str] = "word/_rels/document.xml.rels"
    DOCX_CORE_PROPERTIES_MEMBER: Final[str] = "docProps/core.xml"
    DOCX_CONTENT_TYPES_MEMBER: Final[str] = "[Content_Types].xml"
    DOCX_XMLNS_W: Final[str] = (
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    )

    @unique
    class DocxError(StrEnum):
        """Stable failure codes returned by the DOCX service."""

        ARCHIVE_INVALID = "docx_archive_invalid"
        DOCUMENT_LOAD_FAILED = "docx_document_load_failed"
        PLAN_INVALID = "docx_plan_invalid"
        RENDER_FAILED = "docx_render_failed"
        SERIALIZE_FAILED = "docx_serialize_failed"
        STYLE_MISSING = "docx_style_missing"


__all__: tuple[str, ...] = ("FlextCliConstantsDocx",)

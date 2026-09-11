"""Private MRO composition for generic PPTX models."""

from __future__ import annotations

from .pptx_presentation import FlextCliModelsPptxPresentation


class FlextCliModelsPptx(FlextCliModelsPptxPresentation):
    """Canonical private PPTX model namespace."""


__all__: tuple[str, ...] = ("FlextCliModelsPptx",)

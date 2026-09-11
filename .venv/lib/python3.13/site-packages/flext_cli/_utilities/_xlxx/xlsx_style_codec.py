"""Canonical MRO owner for typed XLSX visual-style translation."""

from __future__ import annotations

from .xlsx_style_builders import FlextCliUtilitiesXlsxStyleBuilders
from .xlsx_style_readers import FlextCliUtilitiesXlsxStyleReaders


class FlextCliUtilitiesXlsxStyleCodec(
    FlextCliUtilitiesXlsxStyleBuilders, FlextCliUtilitiesXlsxStyleReaders
):
    """Compose vendor-to-model and model-to-vendor style translation once."""


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsxStyleCodec",)

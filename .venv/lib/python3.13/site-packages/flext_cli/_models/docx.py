"""Private MRO composition for generic DOCX models."""

from __future__ import annotations

from .docx_document import FlextCliModelsDocxDocument
from .docx_styles import FlextCliModelsDocxStyles


class FlextCliModelsDocx(FlextCliModelsDocxDocument, FlextCliModelsDocxStyles):
    """Canonical private DOCX model namespace."""


__all__: tuple[str, ...] = ("FlextCliModelsDocx",)

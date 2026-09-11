"""Structural contracts for external XLSX rules and style components."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class FlextCliProtocolsXlsxRules:
    """Narrow protocols for objects created by the workbook adapter."""

    # NOTE (multi-agent, mro-j2yt.1): protocols expose only operations required
    # by the generic plan and never leak vendor classes to consumers.
    @runtime_checkable
    class XlsxDataValidation(Protocol):
        def add(self, cell_range: str) -> None: ...

    @runtime_checkable
    class XlsxConditionalFormatting(Protocol):
        def add(
            self, cell_range: str, rule: FlextCliProtocolsXlsxRules.XlsxRule
        ) -> None: ...

    @runtime_checkable
    class XlsxRule(Protocol):
        """Marker for conditional-format rules."""

    @runtime_checkable
    class XlsxNamedStyle(Protocol):
        name: str
        font: FlextCliProtocolsXlsxRules.XlsxFont
        fill: FlextCliProtocolsXlsxRules.XlsxFill
        border: FlextCliProtocolsXlsxRules.XlsxBorder
        alignment: FlextCliProtocolsXlsxRules.XlsxAlignment
        number_format: str

    class XlsxFont(Protocol): ...

    class XlsxFill(Protocol): ...

    class XlsxBorder(Protocol): ...

    class XlsxAlignment(Protocol): ...


__all__: tuple[str, ...] = ("FlextCliProtocolsXlsxRules",)

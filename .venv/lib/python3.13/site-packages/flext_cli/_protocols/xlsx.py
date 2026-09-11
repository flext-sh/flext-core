"""Public-shape protocols for the private XLSX byte boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core import p

from .xlsx_archive import FlextCliProtocolsXlsxArchive
from .xlsx_rules import FlextCliProtocolsXlsxRules
from .xlsx_snapshot import FlextCliProtocolsXlsxSnapshot
from .xlsx_workbook import FlextCliProtocolsXlsxWorkbook

if TYPE_CHECKING:
    # mro-j47u (codex): p -> m stays type-only through the canonical facade.
    from flext_cli import m


class FlextCliProtocolsXlsx(
    FlextCliProtocolsXlsxArchive,
    FlextCliProtocolsXlsxSnapshot,
    FlextCliProtocolsXlsxWorkbook,
    FlextCliProtocolsXlsxRules,
):
    """Consumer protocols for plans, requests, results, and services."""

    # NOTE (multi-agent, mro-j2yt.1): protocol properties retain the exact
    # source model objects; no dump, mapping, or adapter DTO is introduced.
    @runtime_checkable
    class XlsxSheetPlan(Protocol):
        @property
        def name(self) -> str: ...

    @runtime_checkable
    class XlsxWorkbookPlan(Protocol):
        @property
        def sheets(self) -> tuple[m.Cli.XlsxSheetPlan, ...]: ...

        @property
        def full_calculation_on_load(self) -> bool: ...

    @runtime_checkable
    class XlsxRenderRequest(Protocol):
        @property
        def template(self) -> bytes | None: ...

        @property
        def plan(self) -> m.Cli.XlsxWorkbookPlan: ...

    @runtime_checkable
    class XlsxRenderResult(Protocol):
        @property
        def content(self) -> bytes: ...

        @property
        def plan(self) -> m.Cli.XlsxWorkbookPlan: ...

    @runtime_checkable
    class XlsxParseRangeRequest(Protocol):
        @property
        def reference(self) -> str: ...

    # mro-j2yt.1 (xlsx_reference_api): public structural formatting boundary.
    @runtime_checkable
    class XlsxFormatReferenceRequest(Protocol):
        @property
        def area(self) -> m.Cli.XlsxCellRange: ...

        @property
        def sheet(self) -> str | None: ...

        @property
        def absolute(self) -> bool: ...

        @property
        def collapse_single_cell(self) -> bool: ...

    @runtime_checkable
    class XlsxReference(Protocol):
        @property
        def reference(self) -> str: ...

    @runtime_checkable
    class XlsxArchiveInspectionRequest(Protocol):
        @property
        def source(self) -> bytes: ...

        @property
        def policy(self) -> m.Cli.XlsxArchivePolicy: ...

    @runtime_checkable
    class XlsxStyleCatalogRequest(Protocol):
        @property
        def source(self) -> bytes: ...

        @property
        def style_name_prefix(self) -> str: ...

    @runtime_checkable
    class XlsxStyleTemplateRequest(XlsxStyleCatalogRequest, Protocol): ...

    @runtime_checkable
    class XlsxStyleMapEntry(Protocol):
        @property
        def source_style_id(self) -> int: ...

        @property
        def style_name(self) -> str: ...

    @runtime_checkable
    class XlsxStyleTemplateResult(Protocol):
        @property
        def content(self) -> bytes: ...

        @property
        def style_map(self) -> tuple[m.Cli.XlsxStyleMapEntry, ...]: ...

    @runtime_checkable
    class XlsxService(FlextCliProtocolsXlsxSnapshot.XlsxSnapshotService, Protocol):
        def xlsx_parse_range(
            self, request: FlextCliProtocolsXlsx.XlsxParseRangeRequest
        ) -> p.Result[m.Cli.XlsxCellRange]: ...

        def xlsx_format_reference(
            self, request: FlextCliProtocolsXlsx.XlsxFormatReferenceRequest
        ) -> p.Result[m.Cli.XlsxReference]: ...

        def xlsx_render(
            self, request: FlextCliProtocolsXlsx.XlsxRenderRequest
        ) -> p.Result[m.Cli.XlsxRenderResult]: ...

        def xlsx_inspect(
            self, request: FlextCliProtocolsXlsx.XlsxArchiveInspectionRequest
        ) -> p.Result[m.Cli.XlsxArchiveInspection]: ...

        def xlsx_style_catalog(
            self, request: FlextCliProtocolsXlsx.XlsxStyleCatalogRequest
        ) -> p.Result[m.Cli.XlsxStyleCatalog]: ...

        def xlsx_style_template(
            self, request: FlextCliProtocolsXlsx.XlsxStyleTemplateRequest
        ) -> p.Result[m.Cli.XlsxStyleTemplateResult]: ...


__all__: tuple[str, ...] = ("FlextCliProtocolsXlsx",)

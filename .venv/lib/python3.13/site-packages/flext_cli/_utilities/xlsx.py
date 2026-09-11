"""Private MRO composition for the generic XLSX byte boundary."""

from __future__ import annotations

from ._xlxx.xlsx_archive import FlextCliUtilitiesXlsxArchive
from ._xlxx.xlsx_defined_name_values import FlextCliUtilitiesXlsxDefinedNameValues
from ._xlxx.xlsx_recalc import FlextCliUtilitiesXlsxRecalc
from ._xlxx.xlsx_renderer import FlextCliUtilitiesXlsxRenderer
from ._xlxx.xlsx_snapshot import FlextCliUtilitiesXlsxSnapshot
from ._xlxx.xlsx_style_catalog import FlextCliUtilitiesXlsxStyleCatalog


class FlextCliUtilitiesXlsx(
    FlextCliUtilitiesXlsxRecalc,
    FlextCliUtilitiesXlsxDefinedNameValues,
    FlextCliUtilitiesXlsxSnapshot,
    FlextCliUtilitiesXlsxRenderer,
    FlextCliUtilitiesXlsxStyleCatalog,
    FlextCliUtilitiesXlsxArchive,
):
    """Compose rendering, snapshot, style-template, and inspection operations."""

    # NOTE (multi-agent, mro-j2yt.1): one MRO path exposes every generic XLSX
    # byte operation; snapshotting does not create a parallel service.


__all__: tuple[str, ...] = ("FlextCliUtilitiesXlsx",)

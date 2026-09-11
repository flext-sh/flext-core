"""Private MRO composition for generic XLSX models."""

from __future__ import annotations

from ._xlsx.xlsx_archive import FlextCliModelsXlsxArchive
from ._xlsx.xlsx_cells import FlextCliModelsXlsxCells
from ._xlsx.xlsx_layout import FlextCliModelsXlsxLayout
from ._xlsx.xlsx_recalc import FlextCliModelsXlsxRecalc
from ._xlsx.xlsx_rules import FlextCliModelsXlsxRules
from ._xlsx.xlsx_snapshot import FlextCliModelsXlsxSnapshot
from ._xlsx.xlsx_style_catalog import FlextCliModelsXlsxStyleCatalog
from ._xlsx.xlsx_styles import FlextCliModelsXlsxStyles
from ._xlsx.xlsx_tables import FlextCliModelsXlsxTables
from ._xlsx.xlsx_validation import FlextCliModelsXlsxValidation
from ._xlsx.xlsx_workbook import FlextCliModelsXlsxWorkbook


class FlextCliModelsXlsx(
    FlextCliModelsXlsxSnapshot,
    FlextCliModelsXlsxRecalc,
    FlextCliModelsXlsxWorkbook,
    FlextCliModelsXlsxArchive,
    FlextCliModelsXlsxStyleCatalog,
    FlextCliModelsXlsxRules,
    FlextCliModelsXlsxValidation,
    FlextCliModelsXlsxTables,
    FlextCliModelsXlsxLayout,
    FlextCliModelsXlsxStyles,
    FlextCliModelsXlsxCells,
):
    """Canonical private XLSX model namespace."""

    # NOTE (multi-agent, mro-j2yt.1): snapshot declarations join the existing
    # XLSX namespace without a parallel public model surface.


__all__: tuple[str, ...] = ("FlextCliModelsXlsx",)

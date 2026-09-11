"""Typed recalculation declarations for generic XLSX workbooks."""

from __future__ import annotations

from typing import Annotated

from flext_core import m


class FlextCliModelsXlsxRecalc:
    """Immutable recalculation requests, results, and parity evidence."""

    # NOTE (multi-agent, mro-j2yt.1): recalculation evidence is declarative;
    # producers store the verdict instead of recomputing it from properties.
    class XlsxRecalcRequest(m.FrozenModel):
        source: Annotated[
            bytes, m.Field(min_length=1, description="Source workbook bytes.")
        ]

    class XlsxRecalcResult(m.FrozenModel):
        content: Annotated[
            bytes, m.Field(min_length=1, description="Recalculated workbook bytes.")
        ]

    class XlsxRecalcParityRequest(m.FrozenModel):
        source: Annotated[
            bytes, m.Field(min_length=1, description="Source workbook bytes.")
        ]
        expected_formula_count: (
            Annotated[
                int, m.Field(ge=0, description="Expected source formula cell count.")
            ]
            | None
        ) = None

    class XlsxRecalcParityReport(m.FrozenModel):
        content: Annotated[
            bytes,
            m.Field(
                min_length=1,
                description="Exact recalculated workbook bytes validated by this report.",
            ),
        ]
        recalculated: bool = m.Field(
            description="Whether the engine produced recalculated bytes."
        )
        formula_count: Annotated[
            int, m.Field(ge=0, description="Source workbook formula cell count.")
        ]
        error_cells: tuple[str, ...] = m.Field(
            default=(),
            strict=False,
            description="Formula cells whose cached value is an error literal.",
        )
        uncached_cells: tuple[str, ...] = m.Field(
            default=(),
            strict=False,
            description="Formula cells without any cached value element.",
        )
        empty_result_cells: tuple[str, ...] = m.Field(
            default=(),
            strict=False,
            description="Formula cells cached as an empty string result.",
        )
        ok: bool = m.Field(
            description="Producer-stored verdict: caches complete, no errors, count matches."
        )


__all__: tuple[str, ...] = ("FlextCliModelsXlsxRecalc",)

"""CLI Pydantic domain models."""

from __future__ import annotations

from typing import Annotated

from flext_cli import c, t
from flext_core import m, u


class FlextCliModelsBase:
    """Implementation part for FlextCliModelsBase."""

    class TableConfig(m.Value):
        """Table display configuration for tabulate extending Value via inheritance.

        Fields map directly to tabulate() parameters.
        Inherits frozen=True and extra="forbid" from m.Value.
        """

        # Headers configuration
        headers: Annotated[
            t.Cli.TableHeaders,
            m.Field(
                description=(
                    "Table headers (string like 'keys', 'firstrow' "
                    "or sequence of header names)"
                )
            ),
        ] = "keys"
        title: Annotated[
            str | None,
            m.Field(description="Optional title printed before the rendered table"),
        ] = None
        show_header: Annotated[
            bool, m.Field(description="Whether to show table header")
        ] = True

        # Format configuration
        table_format: Annotated[
            c.Cli.TabularFormat,
            m.Field(description="Table format enum-derived literal authority"),
        ] = c.Cli.TabularFormat.SIMPLE

        @u.computed_field
        @property
        def table_backend_format(self) -> c.Cli.TabularFormat:
            """Canonical backend format used by tabulate rendering."""
            return (
                c.Cli.TabularFormat.SIMPLE
                if self.table_format == c.Cli.TabularFormat.TABLE
                else self.table_format
            )

        # Number formatting
        floatfmt: Annotated[str, m.Field(description="Float format string")] = ".4g"
        numalign: Annotated[
            str, m.Field(description="Number alignment (right, center, left, decimal)")
        ] = "decimal"

        # String formatting
        stralign: Annotated[
            str, m.Field(description="String alignment (left, center, right)")
        ] = "left"

        align: Annotated[
            str, m.Field(description="General alignment (left, center, right, decimal)")
        ] = "left"

        # Missing values
        missingval: Annotated[
            str, m.Field(description="String to use for missing values")
        ] = ""

        # Index display
        showindex: Annotated[
            t.Cli.TableShowIndex,
            m.Field(
                default=False,
                validate_default=True,
                description="Whether to show row indices",
            ),
        ] = False

        # Column alignment
        colalign: Annotated[
            t.Cli.TableColAlign,
            m.Field(description="Per-column alignment (left, center, right, decimal)"),
        ] = None

        # Number parsing
        disable_numparse: t.Cli.TableDisableNumparse = m.Field(
            False,
            validate_default=True,
            description="Disable number parsing (bool or list of column indices)",
        )


__all__: list[str] = ["FlextCliModelsBase"]

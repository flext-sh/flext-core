"""FLEXT CLI - Tabulate Integration Layer.

This module provides lightweight ASCII table formatting as an alternative
to Rich tables. Optimized for performance, plain text output, and large datasets.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_cli import c, m, p, s, t, u
from flext_cli.services.formatters import FlextCliFormatters


class FlextCliTables(s):
    """Tabulate integration for lightweight ASCII tables."""

    @staticmethod
    def format_table(
        data: t.Cli.TableDataSource,
        settings: m.Cli.TableConfig | None = None,
        **config_kwargs: t.Cli.TableConfigValue,
    ) -> p.Result[str]:
        """Format table data to a string using the public CLI API."""
        return u.Cli.tables_resolve_config(settings, **config_kwargs).flat_map(
            lambda config_final: u.Cli.tables_normalize_data(data).flat_map(
                lambda rows: u.Cli.tables_render(rows, config_final)
            )
        )

    @staticmethod
    def show_table(
        data: t.Cli.TableDataSource,
        settings: m.Cli.TableConfig | None = None,
        **config_kwargs: t.Cli.TableConfigValue,
    ) -> None:
        """Render and display a formatted table on the console."""

        def _render_with_title(rendered: str, title: str | None = None) -> str:
            if title:
                FlextCliFormatters.print(title, style=c.Cli.MessageStyles.BOLD)
            return rendered

        def _print_error(error: str | None) -> str:
            error_line, error_style = u.Cli.output_table_error(error)
            FlextCliFormatters.print(error_line, style=error_style)
            return ""

        outcome = u.Cli.tables_resolve_config(settings, **config_kwargs).flat_map(
            lambda config_final: u.Cli.tables_normalize_data(data).flat_map(
                lambda rows: u.Cli.tables_render(rows, config_final).map(
                    lambda rendered: _render_with_title(rendered, config_final.title)
                )
            )
        )
        FlextCliFormatters.print(
            outcome.fold(on_success=lambda value: value, on_failure=_print_error)
        )


__all__: t.MutableSequenceOf[str] = ["FlextCliTables"]

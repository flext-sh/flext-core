"""CLI formatter helpers shared through ``u.Cli``."""

from __future__ import annotations

from typing import ClassVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable

from flext_cli import m


class FlextCliUtilitiesFormatters:
    """Rich-backed formatter helpers for CLI services."""

    _console: ClassVar[Console] = Console()

    @classmethod
    def formatters_print(cls, message: str, style: str | None = None) -> None:
        """Print one message via Rich."""
        cls._console.print(message, style=style)

    @classmethod
    def formatters_render_rule(cls, text: str) -> None:
        """Render one horizontal rule via Rich."""
        cls._console.rule(text)

    @classmethod
    def formatters_render_panel(cls, content: str, *, title: str = "") -> None:
        """Render one panel via Rich."""
        cls._console.print(Panel(content, title=title or None))

    @classmethod
    def formatters_render_table(cls, request: m.Cli.TableRenderRequest) -> None:
        """Render one table via Rich."""
        table = RichTable(title=request.title or None)
        for col in request.columns:
            table.add_column(col)
        for row in request.rows:
            table.add_row(*row)
        cls._console.print(table)


__all__: list[str] = ["FlextCliUtilitiesFormatters"]

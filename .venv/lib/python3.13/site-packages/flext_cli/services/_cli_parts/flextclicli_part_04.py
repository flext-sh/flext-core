"""FLEXT CLI - Unified Typer abstraction service.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_cli import p, t, u
from flext_cli.services._cli_parts.flextclicli_part_03 import (
    FlextCliCli as FlextCliCliPart03,
)


class FlextCliCli(FlextCliCliPart03):
    """Implementation part for FlextCliCli."""

    @staticmethod
    def execute_app(
        app: p.Cli.Application, *, prog_name: str, args: t.StrSequence | None = None
    ) -> p.Result[bool]:
        """Execute an application through the private framework boundary."""
        return u.Cli.framework_execute(app, prog_name=prog_name, args=args)

    @staticmethod
    def execute_external_command(
        command: p.Cli.ExternalCommand,
        *,
        prog_name: str,
        args: t.StrSequence | None = None,
    ) -> p.Result[bool]:
        """Execute a foreign command through the private framework boundary."""
        return u.Cli.framework_execute_external(command, prog_name=prog_name, args=args)

    @staticmethod
    def external_command(app: p.Cli.Application) -> p.Cli.ExternalCommand:
        """Expose one adapter-owned application as an external command."""
        return u.Cli.framework_external_command(app)

    @staticmethod
    def register_callback(app: p.Cli.Application, *, command: t.Cli.CliCommand) -> None:
        """Register one model-backed root callback."""
        u.Cli.framework_register_callback(app, command)

    @staticmethod
    def exit(code: int = 0) -> None:
        """Terminate through the adapter-owned execution context."""
        u.Cli.framework_exit(code)

    @staticmethod
    def register_command(
        app: p.Cli.Application, *, name: str, help_text: str, command: t.Cli.CliCommand
    ) -> None:
        """Register a command through the private framework boundary."""
        u.Cli.framework_register_command(
            app, name=name, help_text=help_text, command=command
        )


__all__: list[str] = ["FlextCliCli"]

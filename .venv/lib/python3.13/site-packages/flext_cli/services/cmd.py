"""Command execution and settings bridge for flext-.

Encapsulates the bridge between registered commands, file utilities, and settings
helpers using `r` for predictable success/failure handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_cli import m, p, s, t, u


class FlextCliCmd(s):
    """Execute registered CLI commands and expose execution metadata.

    Extends s for consistent logging and container access.
    Delegates settings operations to direct ``u.Cli`` helpers.
    Railway-Oriented Programming via r for composable error handling.
    """

    @staticmethod
    def settings_snapshot() -> p.Result[m.Cli.SettingsSnapshot]:
        """Return the current settings snapshot using ``u.Cli``."""
        return u.Cli.cmd_settings_snapshot()

    def show_settings(self) -> p.Result[bool]:
        """Show current settings.

        Returns:
            r[bool]: True if displayed successfully, or error

        """
        return u.Cli.cmd_show_settings(self.logger)

    def validate_settings(self) -> p.Result[bool]:
        """Validate settings structure using u directly.

        Returns:
            r[bool]: True if validation passed, or error

        """
        return u.Cli.cmd_validate_settings(self.logger)


__all__: t.MutableSequenceOf[str] = ["FlextCliCmd"]

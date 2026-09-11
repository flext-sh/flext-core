"""FlextCli protocol definitions - Structural typing contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class FlextCliProtocolsBase:
    """Implementation part for FlextCliProtocolsBase."""

    @runtime_checkable
    class CliSettings(Protocol):
        """Flat CLI runtime settings (§2.6 — simple scalars only).

        NOTE (multi-agent): the nested ``Cli`` branch was removed; settings
        are flat ``cli_*`` scalars loadable from env/.env. Test-runtime
        detection moved to ``u.Cli.cli_test_env`` (behavior lives in the
        utilities layer, never on settings). Plain ``Protocol`` (not
        ``p.Model``): pyrefly cannot reconcile pydantic ``model_fields``
        metaclass descriptors on this hot path — structural field access is
        the whole contract.
        """

        @property
        def cli_app_name(self) -> str:
            """CLI application name."""
            ...

        @property
        def cli_log_level(self) -> str:
            """CLI log level."""
            ...

        @property
        def cli_log_verbosity(self) -> str:
            """Log verbosity mode."""
            ...

        @property
        def cli_no_color(self) -> bool:
            """Whether color output is disabled."""
            ...

        @property
        def cli_output_format(self) -> str:
            """Configured output format."""
            ...

        @property
        def cli_quiet(self) -> bool:
            """Whether quiet mode is enabled."""
            ...

        @property
        def cli_verbose(self) -> bool:
            """Whether verbose mode is enabled."""
            ...

        cli_config_file: str | None
        """Path to the configured settings file."""

        cli_token_file: str | None
        """Path to the configured authentication token file."""

        cli_ci: bool
        """Whether the current runtime is a CI environment."""

        cli_pytest_current_test: str | None
        """Current pytest test identifier, when present."""

        cli_shell_command: str | None
        """Current shell command propagated by the runtime environment."""


__all__: list[str] = ["FlextCliProtocolsBase"]

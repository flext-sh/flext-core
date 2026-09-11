"""FlextCli protocol definitions - Structural typing contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_cli._protocols._base_parts.flextcliprotocolsbase_part_02 import (
    FlextCliProtocolsBase as FlextCliProtocolsBasePart02,
)

if TYPE_CHECKING:
    from flext_cli import p, t


class FlextCliProtocolsBase(FlextCliProtocolsBasePart02):
    """Implementation part for FlextCliProtocolsBase."""

    @runtime_checkable
    class CommandRunner(Protocol):
        """Contract for generic command execution services."""

        def run(
            self,
            cmd: t.StrSequence,
            cwd: t.Cli.TextPath | None = None,
            timeout: int | None = None,
            env: t.StrMapping | None = None,
            remove_env_keys: t.StrSequence = (),
            input_data: str | bytes | None = None,
            *,
            capture: bool = True,
        ) -> p.Result[p.Cli.CommandOutput]:
            """Execute a command and require zero exit status."""
            ...

        def capture(
            self,
            cmd: t.StrSequence,
            cwd: t.Cli.TextPath | None = None,
            timeout: int | None = None,
            env: t.StrMapping | None = None,
            remove_env_keys: t.StrSequence = (),
            input_data: str | bytes | None = None,
        ) -> p.Result[str]:
            """Execute a command and return stripped stdout."""
            ...

        def run_raw(
            self,
            cmd: t.StrSequence,
            cwd: t.Cli.TextPath | None = None,
            timeout: int | None = None,
            env: t.StrMapping | None = None,
            remove_env_keys: t.StrSequence = (),
            input_data: str | bytes | None = None,
            *,
            capture: bool = True,
        ) -> p.Result[p.Cli.CommandOutput]:
            """Execute a command without enforcing zero exit status."""
            ...

        # mro-zf1s: binary execution shares the runner contract through p.Cli.
        def run_bytes(
            self,
            cmd: t.StrSequence,
            cwd: t.Cli.TextPath | None = None,
            timeout: int | None = None,
            env: t.StrMapping | None = None,
            remove_env_keys: t.StrSequence = (),
            input_data: str | bytes | None = None,
        ) -> p.Result[p.Cli.CommandBytesOutput]:
            """Execute a command and preserve byte-exact output."""
            ...

        def run_checked(
            self,
            cmd: t.StrSequence,
            cwd: t.Cli.TextPath | None = None,
            timeout: int | None = None,
            env: t.StrMapping | None = None,
            remove_env_keys: t.StrSequence = (),
            input_data: str | bytes | None = None,
            *,
            capture: bool = True,
        ) -> p.Result[bool]:
            """Execute a command and return a success flag."""
            ...

        def run_live(
            self,
            cmd: t.StrSequence,
            cwd: t.Cli.TextPath | None = None,
            timeout: int | None = None,
            env: t.StrMapping | None = None,
            remove_env_keys: t.StrSequence = (),
            input_data: str | bytes | None = None,
        ) -> p.Result[p.Cli.CommandOutput]:
            """Execute a checked command with inherited live output."""
            ...

        def run_to_file(
            self,
            cmd: t.StrSequence,
            output_file: t.Cli.TextPath,
            cwd: t.Cli.TextPath | None = None,
            timeout: int | None = None,
            env: t.StrMapping | None = None,
            remove_env_keys: t.StrSequence = (),
            input_data: str | bytes | None = None,
            *,
            live: bool = False,
            deadline: p.Cli.ProcessDeadline | None = None,
        ) -> p.Result[int]:
            """Execute once with byte-identical combined live and durable output."""
            ...

    @runtime_checkable
    class CliParamsConfig(Protocol):
        """Protocol for CLI parameters configuration."""

        @property
        def debug(self) -> bool | None:
            """Check if debug mode is enabled."""
            ...

        @property
        def log_format(self) -> str | None:
            """Configured log format."""
            ...

        @property
        def log_level(self) -> str | None:
            """Configured log level."""
            ...

        @property
        def no_color(self) -> bool | None:
            """Check if color is disabled."""
            ...

        @property
        def output_format(self) -> str | None:
            """Configured output format."""
            ...

        @property
        def params(self) -> t.JsonMapping:
            """Validated configuration parameters."""
            ...

        @property
        def quiet(self) -> bool | None:
            """Check if quiet mode is enabled."""
            ...

        @property
        def trace(self) -> bool | None:
            """Check if trace mode is enabled."""
            ...

        @property
        def verbose(self) -> bool | None:
            """Check if verbose mode is enabled."""
            ...


__all__: list[str] = ["FlextCliProtocolsBase"]

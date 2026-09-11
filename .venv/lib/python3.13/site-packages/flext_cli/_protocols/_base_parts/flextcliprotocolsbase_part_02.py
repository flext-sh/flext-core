"""FlextCli protocol definitions - Structural typing contracts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import IO, Protocol, runtime_checkable

from flext_cli._protocols._base_parts.flextcliprotocolsbase_part_01 import (
    FlextCliProtocolsBase as FlextCliProtocolsBasePart01,
)
from flext_core import p


class FlextCliProtocolsBase(FlextCliProtocolsBasePart01):
    """Implementation part for FlextCliProtocolsBase."""

    @runtime_checkable
    class Settings(p.Settings, FlextCliProtocolsBasePart01.CliSettings, Protocol):
        """Protocol for CLI runtime settings consumed by the public services.

        NOTE (multi-agent): settings are flat ``cli_*`` scalars (§2.6); the
        nested ``Cli`` branch and its pyrefly ``Final`` workaround were
        removed with it. The flat field contract comes from
        ``CliSettings`` (part_01) via protocol composition.
        """

        @property
        def debug(self) -> bool:
            """Check if debug mode is enabled."""
            ...

        @property
        def trace(self) -> bool:
            """Check if trace mode is enabled."""
            ...

        @classmethod
        def reset_for_testing(cls) -> None:
            """Reset the process-wide singleton (test isolation only)."""
            ...

    @runtime_checkable
    class CommandOutput(Protocol):
        """Minimal external command execution output contract."""

        @property
        def duration(self) -> float:
            """Command duration in seconds."""
            ...

        @property
        def exit_code(self) -> int:
            """Command exit code."""
            ...

        @property
        def stderr(self) -> str:
            """Command standard error."""
            ...

        @property
        def stdout(self) -> str:
            """Command standard output."""
            ...

    # mro-zf1s: binary command consumers type against p, never the model owner.
    @runtime_checkable
    class CommandBytesOutput(Protocol):
        """Byte-exact external command execution output contract."""

        @property
        def duration(self) -> float:
            """Command duration in seconds."""
            ...

        @property
        def exit_code(self) -> int:
            """Command exit code."""
            ...

        @property
        def stderr(self) -> bytes:
            """Command standard error as raw bytes."""
            ...

        @property
        def stdout(self) -> bytes:
            """Command standard output as raw bytes."""
            ...

    @runtime_checkable
    class ProcessDeadline(Protocol):
        """Absolute monotonic deadline for one complete process lifecycle."""

        @property
        def expires_at_monotonic(self) -> float:
            """Absolute time.monotonic expiry in seconds."""
            ...

        @property
        def termination_grace_seconds(self) -> float:
            """Reserved graceful termination and drain budget."""
            ...

        @property
        def timeout_exit_code(self) -> int:
            """Canonical exit code returned for deadline expiry."""
            ...

    @runtime_checkable
    class ProcessHandle(Protocol):
        """Restrictive process handle consumed by portable lifecycle utilities."""

        @property
        def pid(self) -> int:
            """Operating-system process identifier."""
            ...

        @property
        def stdout(self) -> IO[bytes] | None:
            """Binary standard-output pipe when requested."""
            ...

        @property
        def stderr(self) -> IO[bytes] | None:
            """Binary standard-error pipe when requested separately."""
            ...

        def kill(self) -> None:
            """Force termination of the root process."""
            ...

        def poll(self) -> int | None:
            """Return the root exit code when available."""
            ...

        def send_signal(self, sig: int) -> None:
            """Send one platform signal to the root process."""
            ...

        def wait(self) -> int:
            """Block until the root process is reaped and return its exit code."""
            ...


__all__: list[str] = ["FlextCliProtocolsBase"]

"""Real subprocess runner service for FLEXT infra tests.

Uses flext_tests base classes (c, r, t) for type-safe subprocess execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import r, t
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_tests import s


class RealSubprocessRunner(s[str]):
    """Real subprocess runner using flext_tests service base.

    Uses c.Infra.Cli constants for command validation and
    r (r) for railway-oriented error handling.
    """

    subprocess_utility: type[FlextInfraUtilitiesSubprocess] = (
        FlextInfraUtilitiesSubprocess
    )

    def __init__(self, **data: t.Scalar) -> None:
        super().__init__(**data)
        # Import c at runtime to avoid circular reference
        self.allowed_commands = frozenset({
            "echo",
            "pwd",
            "ls",
            "git",
        })

    def _validate_safe_command(self, cmd: list[str]) -> r[bool]:
        if not cmd:
            return r[bool].fail("command must not be empty")
        if cmd[0] not in self.allowed_commands:
            return r[bool].fail(f"command '{cmd[0]}' is not in the safe allowlist")
        return r[bool].ok(True)

    def _failure_message[T_Result](self, result: r[T_Result]) -> str:
        return str(result.error) if result.error else "subprocess execution failed"

    def run_safe(self, cmd: list[str]) -> r[str]:
        """Run safe command and return stdout."""
        validation = self._validate_safe_command(cmd)
        if validation.is_failure:
            return r[str].fail(validation.error or "unsafe command")

        result = self.subprocess_utility.run(cmd)
        if result.is_failure:
            return r[str].fail(self._failure_message(result))
        return r[str].ok(result.value.stdout.strip())

    def capture_output(self, cmd: list[str]) -> r[tuple[str, str]]:
        """Run command and capture both stdout and stderr."""
        validation = self._validate_safe_command(cmd)
        if validation.is_failure:
            return r[tuple[str, str]].fail(validation.error or "unsafe command")

        result = self.subprocess_utility.run(cmd)
        if result.is_failure:
            return r[tuple[str, str]].fail(self._failure_message(result))
        output = result.value
        return r[tuple[str, str]].ok((output.stdout.strip(), output.stderr.strip()))


__all__ = ["RealSubprocessRunner"]

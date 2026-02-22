"""Command execution service for infrastructure operations.

Wraps subprocess execution with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from flext_core.result import FlextResult, r

from flext_infra.models import im


class CommandRunner:
    """Infrastructure service for subprocess execution.

    Provides FlextResult-wrapped command execution, replacing the bare
    ``run_checked`` and ``run_capture`` functions from ``scripts/libs/subprocess.py``.

    Structurally satisfies ``InfraProtocols.CommandRunnerProtocol``.
    """

    def run(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
    ) -> FlextResult[im.CommandOutput]:
        """Run a command and return structured output.

        Args:
            cmd: Command line arguments as a sequence.
            cwd: Optional working directory for the command.

        Returns:
            FlextResult containing CommandOutput on success,
            or failure with error details.

        """
        try:
            result = subprocess.run(
                list(cmd),
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            output = im.CommandOutput(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.returncode,
            )
            if result.returncode != 0:
                cmd_str = shlex.join(list(cmd))
                detail = (result.stderr or result.stdout).strip()
                return r[im.CommandOutput].fail(
                    f"command failed ({result.returncode}): {cmd_str}: {detail}",
                )
            return r[im.CommandOutput].ok(output)
        except Exception as exc:
            return r[im.CommandOutput].fail(f"command execution error: {exc}")

    def run_checked(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
    ) -> FlextResult[bool]:
        """Run a command and return success/failure status.

        Args:
            cmd: Command line arguments as a sequence.
            cwd: Optional working directory for the command.

        Returns:
            FlextResult[bool] with True on success, or failure with error.

        """
        result = self.run(cmd, cwd=cwd)
        if result.is_success:
            return r[bool].ok(True)
        return r[bool].fail(result.error or "command failed")

    def capture(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
    ) -> FlextResult[str]:
        """Run a command and capture its stdout.

        Equivalent to the legacy ``run_capture`` function.

        Args:
            cmd: Command line arguments as a sequence.
            cwd: Optional working directory for the command.

        Returns:
            FlextResult[str] with stripped stdout on success.

        """
        result = self.run(cmd, cwd=cwd)
        if result.is_success:
            value = result.value
            return r[str].ok(value.stdout.strip())
        return r[str].fail(result.error or "capture failed")


__all__ = ["CommandRunner"]

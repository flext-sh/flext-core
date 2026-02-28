"""Command execution service for infrastructure operations.

Wraps subprocess execution with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from flext_core import FlextResult, r

from flext_infra import c, m


class CommandRunner:
    """Infrastructure service for subprocess execution.

    Provides FlextResult-wrapped command execution, replacing the bare
    ``run_checked`` and ``run_capture`` functions from ``scripts/libs/subprocess.py``.

    Structurally satisfies ``InfraProtocols.CommandRunnerProtocol``.
    """

    def run_raw(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> FlextResult[m.CommandOutput]:
        """Run a command without enforcing zero exit code.

        Args:
            cmd: Command line arguments as a sequence.
            cwd: Optional working directory for the command.
            timeout: Optional timeout in seconds.
            env: Optional environment override.

        Returns:
            FlextResult containing raw command output and exit code.

        """
        try:
            result = subprocess.run(
                list(cmd),
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
                env=env,
            )
            output = m.CommandOutput(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.returncode,
            )
            return r[m.CommandOutput].ok(output)
        except subprocess.TimeoutExpired as exc:
            cmd_str = shlex.join(list(cmd))
            timeout_text = str(exc.timeout)
            return r[m.CommandOutput].fail(
                f"command timeout after {timeout_text}s: {cmd_str}",
            )
        except (OSError, ValueError) as exc:
            return r[m.CommandOutput].fail(f"command execution error: {exc}")

    def run(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> FlextResult[m.CommandOutput]:
        """Run a command and return structured output.

        Args:
            cmd: Command line arguments as a sequence.
            cwd: Optional working directory for the command.

        Returns:
            FlextResult containing CommandOutput on success,
            or failure with error details.

        """
        raw_result = self.run_raw(cmd, cwd=cwd, timeout=timeout, env=env)
        if raw_result.is_failure:
            return r[m.CommandOutput].fail(
                raw_result.error or "command execution error",
            )

        output = raw_result.value
        if output.exit_code != 0:
            cmd_str = shlex.join(list(cmd))
            detail = (output.stderr or output.stdout).strip()
            return r[m.CommandOutput].fail(
                f"command failed ({output.exit_code}): {cmd_str}: {detail}",
            )
        return r[m.CommandOutput].ok(output)

    def run_checked(
        self,
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> FlextResult[bool]:
        """Run a command and return success/failure status.

        Args:
            cmd: Command line arguments as a sequence.
            cwd: Optional working directory for the command.

        Returns:
            FlextResult[bool] with True on success, or failure with error.

        """
        result = self.run(cmd, cwd=cwd, timeout=timeout, env=env)
        if result.is_success:
            return r[bool].ok(True)
        return r[bool].fail(result.error or "command failed")

    def run_to_file(
        self,
        cmd: Sequence[str],
        output_file: Path,
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> FlextResult[int]:
        """Run a command and stream combined output to a file.

        Args:
            cmd: Command line arguments as a sequence.
            output_file: Destination file path for command output.
            cwd: Optional working directory for the command.
            timeout: Optional timeout in seconds.
            env: Optional environment override.

        Returns:
            FlextResult containing the process exit code.

        """
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding=c.Encoding.DEFAULT) as handle:
                result = subprocess.run(
                    list(cmd),
                    cwd=cwd,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=timeout,
                    env=env,
                )
            return r[int].ok(result.returncode)
        except subprocess.TimeoutExpired as exc:
            cmd_str = shlex.join(list(cmd))
            timeout_text = str(exc.timeout)
            return r[int].fail(f"command timeout after {timeout_text}s: {cmd_str}")
        except OSError as exc:
            return r[int].fail(f"command file output error: {exc}")
        except ValueError as exc:
            return r[int].fail(f"command execution error: {exc}")

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

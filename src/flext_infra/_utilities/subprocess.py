"""Subprocess execution utilities for infrastructure operations.

Provides r-wrapped subprocess execution as static methods,
replacing bare subprocess calls with structured error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from flext_core import r
from flext_infra.constants import FlextInfraConstants as c
from flext_infra.models import FlextInfraModels as m


class FlextInfraUtilitiesSubprocess:
    """Subprocess execution helpers with r-wrapped error handling.

    All methods are static and do not require instantiation.

    Usage via namespace::

        from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess as S

        result = S.run(["ls", "-la"])
    """

    @staticmethod
    def run_raw(
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[m.Infra.Core.CommandOutput]:
        """Run a command without enforcing zero exit code."""
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
            output = m.Infra.Core.CommandOutput(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.returncode,
            )
            return r[m.Infra.Core.CommandOutput].ok(output)
        except subprocess.TimeoutExpired as exc:
            cmd_str = shlex.join(list(cmd))
            return r[m.Infra.Core.CommandOutput].fail(
                f"command timeout after {exc.timeout}s: {cmd_str}",
            )
        except (OSError, ValueError) as exc:
            return r[m.Infra.Core.CommandOutput].fail(
                f"command execution error: {exc}",
            )

    @staticmethod
    def run(
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[m.Infra.Core.CommandOutput]:
        """Run a command and return structured output with zero-exit enforcement."""
        raw_result = FlextInfraUtilitiesSubprocess.run_raw(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
        )
        if raw_result.is_failure:
            return r[m.Infra.Core.CommandOutput].fail(
                raw_result.error or "command execution error",
            )
        output = raw_result.value
        if output.exit_code != 0:
            cmd_str = shlex.join(list(cmd))
            detail = (output.stderr or output.stdout).strip()
            return r[m.Infra.Core.CommandOutput].fail(
                f"command failed ({output.exit_code}): {cmd_str}: {detail}",
            )
        return r[m.Infra.Core.CommandOutput].ok(output)

    @staticmethod
    def run_checked(
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[bool]:
        """Run a command and return success/failure boolean status."""
        result = FlextInfraUtilitiesSubprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
        )
        return result.fold(
            on_failure=lambda e: r[bool].fail(e or "command failed"),
            on_success=lambda _: r[bool].ok(True),
        )

    @staticmethod
    def capture(
        cmd: Sequence[str],
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[str]:
        """Run a command and capture its stripped stdout."""
        result = FlextInfraUtilitiesSubprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
        )
        return result.fold(
            on_failure=lambda e: r[str].fail(e or "capture failed"),
            on_success=lambda v: r[str].ok(v.stdout.strip()),
        )

    @staticmethod
    def run_to_file(
        cmd: Sequence[str],
        output_file: Path,
        cwd: Path | None = None,
        timeout: int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> r[int]:
        """Run a command and stream combined output to a file.

        Returns r containing the process exit code.
        """
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding=c.Infra.Encoding.DEFAULT) as handle:
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
            return r[int].fail(f"command timeout after {exc.timeout}s: {cmd_str}")
        except OSError as exc:
            return r[int].fail(f"command file output error: {exc}")
        except ValueError as exc:
            return r[int].fail(f"command execution error: {exc}")


__all__ = ["FlextInfraUtilitiesSubprocess"]

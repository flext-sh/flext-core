"""Subprocess execution utilities."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    """Run a command and raise RuntimeError if it fails.

    Args:
        command: The command line arguments as a list.
        cwd: Optional working directory for the command.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.

    """
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        cmd = shlex.join(command)
        msg = f"command failed ({result.returncode}): {cmd}"
        raise RuntimeError(msg)


def run_capture(command: list[str], cwd: Path | None = None) -> str:
    """Run a command, capture its output, and raise RuntimeError if it fails.

    Args:
        command: The command line arguments as a list.
        cwd: Optional working directory for the command.

    Returns:
        The stripped standard output of the command.

    Raises:
        RuntimeError: If the command returns a non-zero exit code.

    """
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        cmd = shlex.join(command)
        detail = (result.stderr or result.stdout).strip()
        msg = f"command failed ({result.returncode}): {cmd}: {detail}"
        raise RuntimeError(msg)
    return result.stdout.strip()

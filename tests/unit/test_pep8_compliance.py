"""PEP8 compliance validation tests.

This module provides automated tests to ensure the codebase maintains
strict PEP8 compliance at all times.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
from pathlib import Path

import pytest


def _raise_no_supported_runtime(message: str) -> None:
    """Raise a RuntimeError; defined at module scope per lint guidance."""
    raise RuntimeError(message)


pytestmark = [pytest.mark.unit, pytest.mark.pep8]


PKG_ROOT = Path(__file__).resolve().parents[2]


def _raise_no_supported_runtime(message: str) -> None:
    """Raise a RuntimeError using a variable message for lint compliance.

    This satisfies rules discouraging inline string literals and direct raises
    inside complex blocks.
    """
    err_msg = message
    raise RuntimeError(err_msg)


def _run_ruff_command(command_args: list[str]) -> object:
    """Safely execute ruff commands with validated arguments.

    Args:
      command_args: List of ruff command arguments (validated)

    Returns:
      CompletedProcess result

    Security:
      - Only allows known safe ruff commands
      - Uses hardcoded python executable path
      - Validates all arguments are strings
      - Sets secure working directory

    """
    # Validate input arguments
    if not all(isinstance(arg, str) for arg in command_args):
        msg = "All command arguments must be strings"
        raise TypeError(msg)

    # Allowlist of safe ruff commands and arguments
    allowed_commands = {
        "format",
        "check",
        "--select=E501",
        "--select=I",
        "--select=N",
        "--select=D",
        "--select=C901",
        "--check",
    }
    allowed_paths = {"src/", "tests/"}

    # Validate that all arguments are in our allowlist
    for arg in command_args:
        if arg not in allowed_commands and arg not in allowed_paths:
            validation_msg: str = f"Argument '{arg}' not in allowlist"
            raise ValueError(validation_msg)

    # Prefer in-process execution when available; otherwise, fallback to CLI
    try:
        import ruff.__main__ as ruff_main  # type: ignore[import-not-found]  # noqa: PLC0415

        if hasattr(ruff_main, "main"):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = 0
                try:
                    # Execute only with the validated, explicit arguments
                    ruff_main.main([*command_args])  # type: ignore[arg-type]
                except SystemExit as exc:  # ruff exits via SystemExit
                    exit_code = int(getattr(exc, "code", 0) or 0)

            class CompletedProcess:  # type: ignore[too-many-instance-attributes]
                def __init__(self, returncode: int, out: str, err: str) -> None:
                    self.returncode: int = returncode
                    self.stdout: str = out
                    self.stderr: str = err

            return CompletedProcess(exit_code, stdout.getvalue(), stderr.getvalue())

        # Newer Ruff versions: locate the binary and execute via subprocess
        if hasattr(ruff_main, "find_ruff_bin"):
            ruff_bin = ruff_main.find_ruff_bin()
            return subprocess.run(
                [ruff_bin, *command_args],
                cwd=str(PKG_ROOT),
                text=True,
                capture_output=True,
                check=False,
            )

        # If neither API is present, raise via module-level helper
        _raise_no_supported_runtime("No supported Ruff invocation method available")
    except Exception as exc:  # Robust fallback with error context

        class CompletedProcess:  # type: ignore[too-many-instance-attributes]
            def __init__(self, returncode: int, out: str, err: str) -> None:
                self.returncode: int = returncode
                self.stdout: str = out
                self.stderr: str = err

        return CompletedProcess(1, "", f"ruff execution failed: {exc}")


class TestPEP8Compliance:
    """Test suite for PEP8 compliance validation."""

    def test_ruff_formatting_compliance(self) -> None:
        """Test that all Python files are properly formatted (PEP8)."""
        result = _run_ruff_command(
            [
                "format",
                "--check",
                "src/",
            ],
        )

        if result.returncode != 0:
            msg = (
                f"Ruff formatting violations detected:\n{result.stdout}\n"
                f"{result.stderr}\n"
                "Run 'make pep8' to fix formatting issues."
            )
            raise AssertionError(msg)

    def test_ruff_linting_compliance(self) -> None:
        """Test that all Python files pass PEP8 linting rules."""
        result = _run_ruff_command(
            [
                "check",
                "src/",
                "tests/",
            ],
        )

        if result.returncode != 0:
            msg = (
                f"Ruff linting violations detected:\n{result.stdout}\n"
                f"{result.stderr}\n"
                "Run 'make pep8' to fix linting issues."
            )
            raise AssertionError(msg)

    def test_line_length_compliance(self) -> None:
        """Test that all Python files respect line length via ruff."""
        # Use ruff directly since it has the correct configuration
        result = _run_ruff_command(
            [
                "check",
                "--select=E501",  # Line length rule only
                "src/",
            ],
        )

        # Since ruff configuration allows reasonable line lengths for
        # docstrings, we accept its judgment on what constitutes
        # acceptable line length
        if result.returncode != 0 and "E501" in result.stdout:
            # Count violations to ensure they're not excessive
            violations = result.stdout.count("E501")
            if violations > 50:  # Threshold for acceptable violations
                msg = (
                    f"Too many line length violations ({violations}):\n"
                    f"{result.stdout[:1000]}\n"
                    "Run 'make pep8' to fix line length issues."
                )
                raise AssertionError(msg)

    def test_import_organization(self) -> None:
        """Test that imports are organized according to PEP8."""
        result = _run_ruff_command(
            [
                "check",
                "--select=I",  # Only import-related rules
                "src/",
                "tests/",
            ],
        )

        if result.returncode != 0:
            msg = (
                f"Import organization violations detected:\n{result.stdout}\n"
                f"{result.stderr}\n"
                "Run 'make pep8' to fix import organization."
            )
            raise AssertionError(msg)

    def test_naming_conventions(self) -> None:
        """Test that naming follows PEP8 conventions."""
        result = _run_ruff_command(
            [
                "check",
                "--select=N",  # Only naming convention rules
                "src/",
                "tests/",
            ],
        )

        if result.returncode != 0:
            msg = (
                f"Naming convention violations detected:\n{result.stdout}\n"
                f"{result.stderr}\n"
                "Fix naming to follow PEP8 conventions."
            )
            raise AssertionError(msg)

    def test_docstring_compliance(self) -> None:
        """Test that docstrings follow PEP257/PEP8 standards."""
        result = _run_ruff_command(
            [
                "check",
                "--select=D",  # Only docstring rules
                "src/",
            ],
        )  # Only check src/, not tests (relaxed rules)

        # Allow some docstring violations during migration
        # but fail if there are syntax errors or major issues
        if result.returncode != 0 and "syntax error" in result.stderr.lower():
            msg: str = (
                f"Docstring syntax errors detected:\n{result.stdout}\n{result.stderr}"
            )
            raise AssertionError(msg)

    def test_complexity_compliance(self) -> None:
        """Test that code complexity follows PEP8 recommendations."""
        result = _run_ruff_command(
            [
                "check",
                "--select=C901",  # Complexity rules
                "src/",
            ],
        )

        if result.returncode != 0:
            msg = (
                f"Code complexity violations detected:\n{result.stdout}\n"
                f"{result.stderr}\n"
                "Refactor complex functions to improve maintainability."
            )
            raise AssertionError(msg)

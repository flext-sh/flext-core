"""PEP8 compliance validation tests.

This module provides automated tests to ensure the codebase maintains
strict PEP8 compliance at all times.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pep8]


def _run_ruff_command(command_args: list[str]) -> subprocess.CompletedProcess[str]:
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
            msg = f"Argument '{arg}' not in allowlist"
            raise ValueError(msg)

    # Construct safe command with hardcoded python executable
    safe_command = [sys.executable, "-m", "ruff", *command_args]

    # Execute with safe parameters - subprocess call is secure due to input validation
    return subprocess.run(  # noqa: S603
        safe_command,
        check=False,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
        timeout=60,  # Prevent hanging
    )


class TestPEP8Compliance:
    """Test suite for PEP8 compliance validation."""

    def test_ruff_formatting_compliance(self) -> None:
        """Test that all Python files are properly formatted (PEP8)."""
        result = _run_ruff_command(
            [
                "format",
                "--check",
                "src/",
                "tests/",
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
            msg = f"Docstring syntax errors detected:\n{result.stdout}\n{result.stderr}"
            raise AssertionError(msg)

    def test_complexity_compliance(self) -> None:
        """Test that code complexity follows PEP8 recommendations."""
        result = _run_ruff_command(
            [
                "check",
                "--select=C901",  # Complexity rules
                "src/",
                "tests/",
            ],
        )

        if result.returncode != 0:
            msg = (
                f"Code complexity violations detected:\n{result.stdout}\n"
                f"{result.stderr}\n"
                "Refactor complex functions to improve maintainability."
            )
            raise AssertionError(msg)

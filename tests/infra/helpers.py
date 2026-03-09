"""Assertion helpers for FLEXT infra tests.

Provides domain-specific assertion helpers that wrap flext_tests matchers (tm)
with infra-specific context and validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from flext_core import r, t
from flext_tests import tm


class FlextInfraTestHelpers:
    """Assertion helpers for infra testing with tm integration.

    Wraps flext_tests matchers (tm) with infra-specific validation context.
    All helpers return unwrapped values or error messages for chaining.
    """

    @staticmethod
    def assert_ok[TResult](result: r[TResult]) -> TResult:
        """Assert FlextResult success and return unwrapped value.

        Uses tm.ok() internally for consistent assertion semantics.

        Args:
            result: FlextResult to validate

        Returns:
            Unwrapped value from result

        Raises:
            AssertionError: If result is failure

        """
        if not result.is_success:
            raise AssertionError(f"Expected success, got failure: {result.error}")
        return result.value

    @staticmethod
    def assert_fail[TResult](result: r[TResult], contains: str | None = None) -> str:
        """Assert FlextResult failure and return error message.

        Uses tm.fail() internally for consistent assertion semantics.

        Args:
            result: FlextResult to validate
            contains: Optional substring to check in error message

        Returns:
            Error message from result

        Raises:
            AssertionError: If result is success or error doesn't match

        """
        if contains:
            return tm.fail(result, has=contains)
        return tm.fail(result)

    @staticmethod
    def assert_file_exists(path: Path, msg: str | None = None) -> Path:
        """Assert file exists at path.

        Args:
            path: Path to check
            msg: Optional custom error message

        Returns:
            The path (for chaining)

        Raises:
            AssertionError: If file does not exist

        """
        if not path.exists():
            error = msg or f"File does not exist: {path}"
            raise AssertionError(error)
        if not path.is_file():
            error = msg or f"Path is not a file: {path}"
            raise AssertionError(error)
        return path

    @staticmethod
    def assert_dir_exists(path: Path, msg: str | None = None) -> Path:
        """Assert directory exists at path.

        Args:
            path: Path to check
            msg: Optional custom error message

        Returns:
            The path (for chaining)

        Raises:
            AssertionError: If directory does not exist

        """
        if not path.exists():
            error = msg or f"Directory does not exist: {path}"
            raise AssertionError(error)
        if not path.is_dir():
            error = msg or f"Path is not a directory: {path}"
            raise AssertionError(error)
        return path

    @staticmethod
    def assert_file_contains(path: Path, content: str, msg: str | None = None) -> Path:
        """Assert file exists and contains substring.

        Args:
            path: Path to file
            content: Substring to find
            msg: Optional custom error message

        Returns:
            The path (for chaining)

        Raises:
            AssertionError: If file doesn't exist or doesn't contain substring

        """
        FlextInfraTestHelpers.assert_file_exists(path, msg)
        file_content = path.read_text()
        if content not in file_content:
            error = msg or f"File {path} does not contain: {content}"
            raise AssertionError(error)
        return path

    @staticmethod
    def assert_toml_valid(
        path: Path, msg: str | None = None
    ) -> dict[str, t.ContainerValue]:
        """Assert TOML file is valid and return parsed content.

        Args:
            path: Path to TOML file
            msg: Optional custom error message

        Returns:
            Parsed TOML content as dict

        Raises:
            AssertionError: If file doesn't exist or TOML is invalid

        """
        FlextInfraTestHelpers.assert_file_exists(path, msg)
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            error = msg or f"Invalid TOML in {path}: {e}"
            raise AssertionError(error) from e

    @staticmethod
    def assert_toml_has_section(
        path: Path, section: str, msg: str | None = None
    ) -> dict[str, t.ContainerValue]:
        """Assert TOML file has specific section.

        Args:
            path: Path to TOML file
            section: Section name to check (e.g., "project", "tool.poetry")
            msg: Optional custom error message

        Returns:
            Parsed TOML content

        Raises:
            AssertionError: If section doesn't exist

        """
        content = FlextInfraTestHelpers.assert_toml_valid(path, msg)
        parts = section.split(".")
        current = content
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                error = msg or f"TOML section '{section}' not found in {path}"
                raise AssertionError(error)
            current = current[part]
        return content


# Canonical alias for infra test helpers
h = FlextInfraTestHelpers
__all__ = ["FlextInfraTestHelpers", "h"]

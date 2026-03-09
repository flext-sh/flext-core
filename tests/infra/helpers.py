"""Assertion helpers for FLEXT infra tests.

Provides domain-specific assertion helpers that wrap flext_tests matchers (tm)
and utilities (u) with infra-specific context and validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flext_core import r
from flext_tests import t, tm, u


class FlextInfraTestHelpers:
    """Assertion helpers for infra testing with tm integration.

    Wraps flext_tests matchers (tm) and utilities (u) with infra-specific
    validation context. All helpers return unwrapped values or error messages.
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
        return tm.ok(result)

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
        """Assert file exists at path using tm.that().

        Args:
            path: Path to check
            msg: Optional custom error message

        Returns:
            The path (for chaining)

        Raises:
            AssertionError: If file does not exist

        """
        tm.that(path.exists(), eq=True, msg=msg or f"File does not exist: {path}")
        tm.that(path.is_file(), eq=True, msg=msg or f"Path is not a file: {path}")
        return path

    @staticmethod
    def assert_dir_exists(path: Path, msg: str | None = None) -> Path:
        """Assert directory exists at path using tm.that().

        Args:
            path: Path to check
            msg: Optional custom error message

        Returns:
            The path (for chaining)

        Raises:
            AssertionError: If directory does not exist

        """
        tm.that(path.exists(), eq=True, msg=msg or f"Directory does not exist: {path}")
        tm.that(path.is_dir(), eq=True, msg=msg or f"Path is not a directory: {path}")
        return path

    @staticmethod
    def assert_file_contains(path: Path, content: str, msg: str | None = None) -> Path:
        """Assert file exists and contains substring using tm.that().

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
        file_content = path.read_text(encoding="utf-8")
        tm.that(
            file_content,
            contains=content,
            msg=msg or f"File {path} does not contain: {content}",
        )
        return path

    @staticmethod
    def assert_toml_valid(
        path: Path, msg: str | None = None
    ) -> dict[str, t.ContainerValue]:
        """Assert TOML file is valid and return parsed content using u.parse_toml().

        Args:
            path: Path to TOML file
            msg: Optional custom error message

        Returns:
            Parsed TOML content as dict

        Raises:
            AssertionError: If file doesn't exist or TOML is invalid

        """
        FlextInfraTestHelpers.assert_file_exists(path, msg)
        result = u.parse_toml(path)
        return tm.ok(result, msg=msg or f"Invalid TOML in {path}")

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
            tm.that(
                isinstance(current, dict) and part in current,
                eq=True,
                msg=msg or f"TOML section '{section}' not found in {path}",
            )
            current = current[part]
        return content

    @staticmethod
    def assert_valid_project_name(name: str, msg: str | None = None) -> str:
        """Assert project name is valid using c.Infra.Tests.Projects patterns.

        Args:
            name: Project name to validate
            msg: Optional custom error message

        Returns:
            The name (for chaining)

        Raises:
            AssertionError: If name is invalid

        """
        is_valid = name and name.replace("-", "").replace("_", "").isalnum()
        tm.that(is_valid, eq=True, msg=msg or f"Invalid project name: {name}")
        return name

    @staticmethod
    def assert_is_docker_available(msg: str | None = None) -> bool:
        """Assert Docker is available using c.Infra.Tests.Docker constants.

        Args:
            msg: Optional custom error message

        Returns:
            True if Docker is available

        Raises:
            AssertionError: If Docker is not available

        """
        is_available = shutil.which("docker") is not None
        tm.that(is_available, eq=True, msg=msg or "Docker is not available")
        return True


# Canonical alias for infra test helpers
h = FlextInfraTestHelpers
__all__ = ["FlextInfraTestHelpers", "h"]

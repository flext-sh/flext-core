"""Assertion helpers for FLEXT infra tests.

Provides domain-specific assertion helpers that wrap flext_tests matchers (tm)
and utilities (u) with infra-specific context and validation.

Also provides reusable factory methods for common test setup operations
(project creation, workspace setup, CLI namespace creation, subprocess stubs).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

from flext_core import r
from flext_tests import tm
from tests.infra.typings import t


class FlextInfraTestHelpers:
    """Assertion helpers and factories for infra testing with tm integration.

    Wraps flext_tests matchers (tm) with infra-specific validation context.

    Assertion methods:
        h.assert_ok(result)            — unwrap r success
        h.assert_fail(result)          — unwrap r failure
        h.assert_file_exists(path)     — assert file exists
        h.assert_dir_exists(path)      — assert directory exists
        h.assert_file_contains(p, s)   — assert file contains substring
        h.assert_toml_valid(path)      — parse & assert valid TOML
        h.assert_toml_has_section(p,s) — TOML has section
        h.assert_valid_project_name(n) — project name is valid
        h.assert_is_docker_available() — docker is on PATH

    Factory methods:
        h.mk_project(root, name)       — create minimal project dir
        h.write_project(root)          — full migration test project
        h.ns(**kw)                     — argparse.Namespace
        h.stub_run(stdout, stderr, rc) — SimpleNamespace for subprocess
        h.workspace(root)              — workspace root with markers
    """

    # ── Assertions ───────────────────────────────────────────────────

    @staticmethod
    def assert_ok[TResult](result: r[TResult]) -> TResult:
        """Assert r success and return unwrapped value.

        Args:
            result: r to validate

        Returns:
            Unwrapped value from result

        Raises:
            AssertionError: If result is failure

        """
        return tm.ok(result)

    @staticmethod
    def assert_fail[TResult](result: r[TResult], contains: str | None = None) -> str:
        """Assert r failure and return error message.

        Args:
            result: r to validate
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

        """
        tm.that(path.exists(), eq=True, msg=msg or f"File does not exist: {path}")
        tm.that(path.is_file(), eq=True, msg=msg or f"Path is not a file: {path}")
        return path

    @staticmethod
    def assert_dir_exists(path: Path, msg: str | None = None) -> Path:
        """Assert directory exists at path.

        Args:
            path: Path to check
            msg: Optional custom error message

        Returns:
            The path (for chaining)

        """
        tm.that(path.exists(), eq=True, msg=msg or f"Directory does not exist: {path}")
        tm.that(path.is_dir(), eq=True, msg=msg or f"Path is not a directory: {path}")
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
    ) -> dict[str, t.Infra.InfraValue]:
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
            content: dict[str, t.Infra.InfraValue] = tomllib.loads(
                path.read_text(encoding="utf-8")
            )
        except tomllib.TOMLDecodeError as exc:
            raise AssertionError(msg or f"Invalid TOML in {path}: {exc}") from exc
        return content

    @staticmethod
    def assert_toml_has_section(
        path: Path, section: str, msg: str | None = None
    ) -> dict[str, t.Infra.InfraValue]:
        """Assert TOML file has specific section.

        Args:
            path: Path to TOML file
            section: Section name (e.g., "project", "tool.poetry")
            msg: Optional custom error message

        Returns:
            Parsed TOML content

        """
        content = FlextInfraTestHelpers.assert_toml_valid(path, msg)
        parts = section.split(".")
        current: dict[str, t.Infra.InfraValue] | t.Infra.InfraValue = content
        for part in parts:
            tm.that(
                isinstance(current, dict),
                eq=True,
                msg=msg or f"TOML section '{section}' not found in {path}",
            )
            assert isinstance(current, dict)
            current_map: dict[str, t.Infra.InfraValue] = current
            tm.that(
                part in current_map,
                eq=True,
                msg=msg or f"TOML section '{section}' not found in {path}",
            )
            current = current_map[part]
        return content

    @staticmethod
    def assert_valid_project_name(name: str, msg: str | None = None) -> str:
        """Assert project name is valid.

        Args:
            name: Project name to validate
            msg: Optional custom error message

        Returns:
            The name (for chaining)

        """
        is_valid = name and name.replace("-", "").replace("_", "").isalnum()
        tm.that(is_valid, eq=True, msg=msg or f"Invalid project name: {name}")
        return name

    @staticmethod
    def assert_is_docker_available(msg: str | None = None) -> bool:
        """Assert Docker is available on PATH.

        Args:
            msg: Optional custom error message

        Returns:
            True if Docker is available

        """
        is_available = shutil.which("docker") is not None
        tm.that(is_available, eq=True, msg=msg or "Docker is not available")
        return True

    # ── Factory helpers ──────────────────────────────────────────────

    @staticmethod
    def mk_project(
        root: Path,
        name: str,
        *,
        pyproject: str = "[tool]\n",
        with_src: bool = False,
        with_git: bool = False,
    ) -> Path:
        """Create a minimal project directory for testing.

        Eliminates the ``_mk_proj(tmp_path, name)`` boilerplate duplicated
        across 20+ test files.

        Args:
            root: Parent directory (usually ``tmp_path``).
            name: Project directory name.
            pyproject: Content of ``pyproject.toml`` (default minimal).
            with_src: If True, create ``src/`` subdirectory.
            with_git: If True, create ``.git/`` directory.

        Returns:
            Path to the created project directory.

        """
        proj = root / name
        proj.mkdir(parents=True, exist_ok=True)
        (proj / "pyproject.toml").write_text(pyproject, encoding="utf-8")
        if with_src:
            (proj / "src").mkdir(exist_ok=True)
        if with_git:
            (proj / ".git").mkdir(exist_ok=True)
        return proj

    @staticmethod
    def write_project(
        project_root: Path,
        *,
        base_mk: str = "OLD_BASE\n",
        makefile: str = "",
        pyproject: str = "",
        gitignore: str = "",
    ) -> None:
        """Set up a full project directory for migration testing.

        Eliminates the ``_write_project(root)`` boilerplate from migrator tests.

        Args:
            project_root: Project root path.
            base_mk: Content for ``base.mk``.
            makefile: Content for ``Makefile`` (uses default if empty).
            pyproject: Content for ``pyproject.toml`` (uses default if empty).
            gitignore: Content for ``.gitignore`` (uses default if empty).

        """
        (project_root / ".git").mkdir(parents=True, exist_ok=True)
        (project_root / "base.mk").write_text(base_mk, encoding="utf-8")
        (project_root / "Makefile").write_text(
            makefile
            or (
                'python "$(WORKSPACE_ROOT)/scripts/check/fix_pyrefly_config.py"'
                ' "$(PROJECT_NAME)"\n'
                'python "$(WORKSPACE_ROOT)/scripts/check/workspace_check.py"'
                ' --gates lint "$(PROJECT_NAME)"\n'
            ),
            encoding="utf-8",
        )
        (project_root / "pyproject.toml").write_text(
            pyproject
            or (
                '[project]\nname = "project-a"\nversion = "0.1.0"\n'
                'dependencies = ["requests>=2"]\n'
            ),
            encoding="utf-8",
        )
        (project_root / ".gitignore").write_text(
            gitignore or "!scripts/\n!scripts/**\n*.pyc\n",
            encoding="utf-8",
        )

    @staticmethod
    def ns(**kwargs: t.Scalar | None) -> argparse.Namespace:
        """Create ``argparse.Namespace`` from keyword arguments.

        Eliminates the repeated ``_ns()`` / ``_build_args()`` helpers from
        CLI test files.

        Args:
            **kwargs: Attributes for the namespace.

        Returns:
            ``argparse.Namespace`` with the given attributes.

        """
        return argparse.Namespace(**kwargs)

    @staticmethod
    def stub_run(
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
    ) -> SimpleNamespace:
        """Create a ``SimpleNamespace`` mimicking a subprocess result.

        Replaces the pervasive ``SimpleNamespace(stdout=..., stderr=..., returncode=...)``
        pattern across checker/runner test stubs.

        Args:
            stdout: Standard output content.
            stderr: Standard error content.
            returncode: Process return code.

        Returns:
            ``SimpleNamespace`` with stdout, stderr, returncode attributes.

        """
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

    @staticmethod
    def workspace(
        root: Path,
        *,
        minor: int | None = None,
        with_makefile: bool = True,
    ) -> Path:
        """Create a workspace root directory with standard markers.

        Eliminates the ``_ws(root)`` helpers from python_version and discovery tests.

        Args:
            root: Root directory path.
            minor: Python minor version for requires-python (default: current).
            with_makefile: If True, create ``Makefile``.

        Returns:
            Path to the workspace root.

        """
        actual_minor = minor if minor is not None else sys.version_info.minor
        root.mkdir(exist_ok=True)
        (root / ".git").mkdir(exist_ok=True)
        if with_makefile:
            (root / "Makefile").touch()
        (root / "pyproject.toml").write_text(
            f'requires-python = ">=3.{actual_minor}"\n',
            encoding="utf-8",
        )
        return root


# Canonical alias for infra test helpers
h = FlextInfraTestHelpers
__all__ = ["FlextInfraTestHelpers", "h"]

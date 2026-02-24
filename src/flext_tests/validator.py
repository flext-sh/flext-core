"""FLEXT Architecture Validator.

Provides FlextTestsValidator (tv) for detecting architecture violations in pytest tests.

Usage:
    from flext_tests import tv

    # Validate imports
    result = tv.imports(Path("src"))
    assert result.is_success and result.value.passed

    # Validate types
    result = tv.types(Path("src"))

    # Validate pyproject.toml
    result = tv.validate_config(Path("pyproject.toml"))

    # Validate all
    result = tv.all(Path("src"), pyproject=Path("pyproject.toml"))

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from flext_core import r

from flext_tests._validator import (
    FlextValidatorBypass,
    FlextValidatorImports,
    FlextValidatorLayer,
    FlextValidatorSettings,
    FlextValidatorTests,
    FlextValidatorTypes,
)
from flext_tests.base import s
from flext_tests.constants import c
from flext_tests.models import m


class FlextTestsValidator(s[m.Tests.Validator.ScanResult]):
    """FLEXT Architecture Validator - detects code violations.

    Provides methods to validate:
    - imports: lazy imports, TYPE_CHECKING, ImportError handling
    - types: type:ignore, Any types, unapproved
    - tests: monkeypatch, mocks, @patch
    - config: pyproject.toml deviations
    - bypass: noqa, pragma, exception swallowing
    - layer: cross-layer import violations

    Uses c.Validator for all rules, messages, and defaults.

    Usage:
        from flext_tests import tv

        result = tv.imports(Path("src"))
        result = tv.types(Path("src"))
        result = tv.validate_config(Path("pyproject.toml"))
        result = tv.all(Path("src"))
    """

    # Re-export models for convenience via m.Tests.Validator namespace
    Violation: ClassVar[type[m.Tests.Validator.Violation]] = m.Tests.Validator.Violation
    ScanResult: ClassVar[type[m.Tests.Validator.ScanResult]] = (
        m.Tests.Validator.ScanResult
    )

    @classmethod
    def imports(
        cls,
        path: Path,
        exclude_patterns: list[str] | None = None,
        approved_exceptions: Mapping[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Validate imports in Python files.

        Detects:
        - IMPORT-001: Lazy imports (not at module top)
        - IMPORT-002: TYPE_CHECKING blocks
        - IMPORT-003: try/except ImportError
        - IMPORT-004: sys.path manipulation
        - IMPORT-005: Direct technology imports
        - IMPORT-006: Non-root flext-* imports

        Args:
            path: Directory or file to scan
            exclude_patterns: Glob patterns to exclude (defaults to common excludes)
            approved_exceptions: Dict mapping rule IDs to approved file patterns

        Returns:
            FlextResult[ScanResult] with violations found

        """
        files = cls._discover_files(path, exclude_patterns)
        return FlextValidatorImports.scan(files, approved_exceptions)

    @classmethod
    def types(
        cls,
        path: Path,
        exclude_patterns: list[str] | None = None,
        approved_exceptions: Mapping[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Validate type annotations in Python files.

        Detects:
        - TYPE-001: # type: ignore comments
        - TYPE-002: Any type annotations
        - TYPE-003: Unapproved  usage

        Args:
            path: Directory or file to scan
            exclude_patterns: Glob patterns to exclude
            approved_exceptions: Dict mapping rule IDs to approved file patterns

        Returns:
            FlextResult[ScanResult] with violations found

        """
        files = cls._discover_files(path, exclude_patterns)
        return FlextValidatorTypes.scan(files, approved_exceptions)

    @classmethod
    def tests(
        cls,
        path: Path,
        exclude_patterns: list[str] | None = None,
        approved_exceptions: Mapping[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Validate test patterns in Python files.

        Detects:
        - TEST-001: monkeypatch usage
        - TEST-002: Mock/MagicMock usage
        - TEST-003: @patch decorator usage

        Args:
            path: Directory or file to scan
            exclude_patterns: Glob patterns to exclude
            approved_exceptions: Dict mapping rule IDs to approved file patterns

        Returns:
            FlextResult[ScanResult] with violations found

        """
        files = cls._discover_files(path, exclude_patterns)
        return FlextValidatorTests.scan(files, approved_exceptions)

    @classmethod
    def validate_config(
        cls,
        pyproject_path: Path,
        approved_exceptions: Mapping[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Validate pyproject.toml configuration.

        Detects:
        - CONFIG-001: mypy ignore_errors = true
        - CONFIG-002: Custom ruff ignores beyond approved
        - CONFIG-003: disallow_incomplete_defs = false
        - CONFIG-004: warn_return_any = false
        - CONFIG-005: reportPrivateUsage = false

        Args:
            pyproject_path: Path to pyproject.toml
            approved_exceptions: Dict mapping rule IDs to approved file patterns

        Returns:
            FlextResult[ScanResult] with violations found

        """
        return FlextValidatorSettings.validate(pyproject_path, approved_exceptions)

    @classmethod
    def bypass(
        cls,
        path: Path,
        exclude_patterns: list[str] | None = None,
        approved_exceptions: Mapping[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Validate bypass patterns in Python files.

        Detects:
        - BYPASS-001: noqa comments
        - BYPASS-002: pragma: no cover (unapproved)
        - BYPASS-003: Exception swallowing

        Args:
            path: Directory or file to scan
            exclude_patterns: Glob patterns to exclude
            approved_exceptions: Dict mapping rule IDs to approved file patterns

        Returns:
            FlextResult[ScanResult] with violations found

        """
        files = cls._discover_files(path, exclude_patterns)
        return FlextValidatorBypass.scan(files, approved_exceptions)

    @classmethod
    def layer(
        cls,
        path: Path,
        exclude_patterns: list[str] | None = None,
        approved_exceptions: Mapping[str, list[str]] | None = None,
        layer_hierarchy: Mapping[str, int] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Validate layer dependencies in Python files.

        Detects:
        - LAYER-001: Lower layer importing upper layer

        Args:
            path: Directory or file to scan
            exclude_patterns: Glob patterns to exclude
            approved_exceptions: Dict mapping rule IDs to approved file patterns
            layer_hierarchy: Custom layer hierarchy (module_name -> layer_number)

        Returns:
            FlextResult[ScanResult] with violations found

        """
        files = cls._discover_files(path, exclude_patterns)
        return FlextValidatorLayer.scan(files, approved_exceptions, layer_hierarchy)

    @classmethod
    def all(
        cls,
        path: Path,
        pyproject_path: Path | None = None,
        exclude_patterns: list[str] | None = None,
        approved_exceptions: Mapping[str, list[str]] | None = None,
        *,
        include_tests_validation: bool = False,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Run all validations and combine results.

        Args:
            path: Directory or file to scan
            pyproject_path: Path to pyproject.toml (optional)
            exclude_patterns: Glob patterns to exclude
            approved_exceptions: Dict mapping rule IDs to approved file patterns
            include_tests_validation: Whether to include test pattern validation

        Returns:
            FlextResult[ScanResult] with combined violations from all validators

        """
        all_violations: list[m.Tests.Validator.Violation] = []
        total_files = 0

        # Run each validator
        validators: list[tuple[str, r[m.Tests.Validator.ScanResult]]] = [
            ("imports", cls.imports(path, exclude_patterns, approved_exceptions)),
            ("types", cls.types(path, exclude_patterns, approved_exceptions)),
            ("bypass", cls.bypass(path, exclude_patterns, approved_exceptions)),
            ("layer", cls.layer(path, exclude_patterns, approved_exceptions)),
        ]

        if include_tests_validation:
            validators.append((
                "tests",
                cls.tests(path, exclude_patterns, approved_exceptions),
            ))

        if pyproject_path and pyproject_path.exists():
            validators.append((
                "config",
                cls.validate_config(pyproject_path, approved_exceptions),
            ))

        for name, result in validators:
            if result.is_failure:
                return r[m.Tests.Validator.ScanResult].fail(
                    f"Validator '{name}' failed: {result.error}",
                )
            scan_result = result.value
            all_violations.extend(scan_result.violations)
            total_files = max(total_files, scan_result.files_scanned)

        return r[m.Tests.Validator.ScanResult].ok(
            m.Tests.Validator.ScanResult.create(
                validator_name="all",
                files_scanned=total_files,
                violations=all_violations,
            ),
        )

    @classmethod
    def _discover_files(
        cls,
        path: Path,
        exclude_patterns: list[str] | None = None,
    ) -> list[Path]:
        """Discover Python files to scan.

        Args:
            path: Directory or file to scan
            exclude_patterns: Glob patterns to exclude

        Returns:
            List of Python file paths

        """
        excludes = exclude_patterns or list(c.Tests.Validator.Defaults.EXCLUDE_PATTERNS)

        if path.is_file():
            return [path] if path.suffix == ".py" else []

        files: list[Path] = []
        for py_file in path.rglob("*.py"):
            # Check if file matches any exclude pattern
            file_str = str(py_file)
            excluded = False
            for pattern in excludes:
                if fnmatch.fnmatch(file_str, pattern):
                    excluded = True
                    break
            if not excluded:
                files.append(py_file)

        return files


# Short alias for convenient usage
tv = FlextTestsValidator

__all__ = ["FlextTestsValidator", "tv"]

"""Test validation for FLEXT architecture.

Detects test violations: monkeypatch, mocks, @patch decorator usage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

from flext_core.result import r
from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.utilities import u


class FlextValidatorTests:
    """Test validation methods for FlextTestsValidator.

    Uses c.Tests.Validator, m.Tests.Validator, u.Tests.Validator.
    """

    @classmethod
    def scan(
        cls,
        files: list[Path],
        approved_exceptions: Mapping[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Scan files for test violations.

        Args:
            files: List of Python files to scan
            approved_exceptions: Dict mapping rule IDs to list of approved file patterns

        Returns:
            FlextResult with ScanResult containing all violations found

        """
        violations: list[m.Tests.Validator.Violation] = []
        approved = approved_exceptions or {}

        for file_path in files:
            file_violations = cls._scan_file(file_path, approved)
            violations.extend(file_violations)

        return r[m.Tests.Validator.ScanResult].ok(
            m.Tests.Validator.ScanResult.create(
                validator_name=c.Tests.Validator.Defaults.VALIDATOR_TESTS,
                files_scanned=len(files),
                violations=violations,
            ),
        )

    @classmethod
    def _scan_file(
        cls,
        file_path: Path,
        approved: Mapping[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Scan a single file for test violations."""
        violations: list[m.Tests.Validator.Violation] = []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return violations

        lines = content.splitlines()

        # Check for monkeypatch usage
        violations.extend(cls._check_monkeypatch(file_path, tree, lines, approved))

        # Check for Mock/MagicMock usage
        violations.extend(cls._check_mock_usage(file_path, tree, lines, approved))

        # Check for @patch decorator
        violations.extend(cls._check_patch_decorator(file_path, tree, lines, approved))

        return violations

    @classmethod
    def _check_monkeypatch(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: Mapping[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect monkeypatch usage in function parameters and calls."""
        if u.Tests.Validator.is_approved("TEST-001", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        for node in ast.walk(tree):
            # Check function parameters for monkeypatch
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if arg.arg == "monkeypatch":
                        violation = u.Tests.Validator.create_violation(
                            file_path,
                            node.lineno,
                            "TEST-001",
                            lines,
                            c.Tests.Validator.Messages.TEST_MONKEYPATCH.format(
                                func=node.name,
                            ),
                        )
                        violations.append(violation)

            # Check for monkeypatch.setattr, monkeypatch.delattr, etc.
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "monkeypatch"
            ):
                violation = u.Tests.Validator.create_violation(
                    file_path,
                    node.lineno,
                    "TEST-001",
                    lines,
                    f"monkeypatch.{node.attr}",
                )
                violations.append(violation)

        return violations

    @classmethod
    def _check_mock_usage(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: Mapping[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect Mock and MagicMock usage."""
        if u.Tests.Validator.is_approved("TEST-002", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []
        mock_names = c.Tests.Validator.Approved.MOCK_NAMES

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.ImportFrom):
                if node.module and "mock" in node.module.lower():
                    for alias in node.names:
                        if alias.name in mock_names:
                            violation = u.Tests.Validator.create_violation(
                                file_path,
                                node.lineno,
                                "TEST-002",
                                lines,
                                f"import {alias.name}",
                            )
                            violations.append(violation)

            # Check calls to Mock(), MagicMock(), etc.
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in mock_names
            ):
                violation = u.Tests.Validator.create_violation(
                    file_path,
                    node.lineno,
                    "TEST-002",
                    lines,
                    f"{node.func.id}()",
                )
                violations.append(violation)

        return violations

    @classmethod
    def _check_patch_decorator(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: Mapping[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect @patch decorator usage."""
        if u.Tests.Validator.is_approved("TEST-003", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        for node in ast.walk(tree):
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                continue
            for decorator in node.decorator_list:
                if cls._is_patch_decorator(decorator):
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        decorator.lineno,
                        "TEST-003",
                        lines,
                    )
                    violations.append(violation)

        return violations

    @classmethod
    def _is_patch_decorator(cls, decorator: ast.expr) -> bool:
        """Check if decorator is @patch or @patch.object, etc."""
        # @patch
        if type(decorator) is ast.Name and decorator.id == "patch":
            return True

        # @patch(...)
        if type(decorator) is ast.Call:
            if type(decorator.func) is ast.Name and decorator.func.id == "patch":
                return True
            # @patch.object(...)
            if type(decorator.func) is ast.Attribute:
                if (
                    type(decorator.func.value) is ast.Name
                    and decorator.func.value.id == "patch"
                ):
                    return True
                # @mock.patch(...)
                if decorator.func.attr == "patch":
                    return True

        # @patch.object
        return (
            type(decorator) is ast.Attribute
            and type(decorator.value) is ast.Name
            and decorator.value.id == "patch"
        )


__all__ = ["FlextValidatorTests"]

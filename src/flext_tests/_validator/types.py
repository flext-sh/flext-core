"""Type validation for FLEXT architecture.

Detects type violations: type:ignore comments, Any types, unapproved  usage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path

from flext_core import r

from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.utilities import u


class FlextValidatorTypes:
    """Type validation methods for FlextTestsValidator.

    Uses c.Tests.Validator, m.Tests.Validator, u.Tests.Validator.
    """

    @classmethod
    def scan(
        cls,
        files: list[Path],
        approved_exceptions: Mapping[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Scan files for type violations.

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
                validator_name=c.Tests.Validator.Defaults.VALIDATOR_TYPES,
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
        """Scan a single file for type violations."""
        violations: list[m.Tests.Validator.Violation] = []

        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return violations

        lines = content.splitlines()

        # Check for type: ignore comments (regex-based)
        violations.extend(cls._check_type_ignore(file_path, lines, approved))

        # AST-based checks
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return violations

        # Check for Any type annotations
        violations.extend(cls._check_any_types(file_path, tree, lines, approved))

        # Check for unapproved  usage
        violations.extend(cls._check_cast_usage(file_path, tree, lines, approved))

        return violations

    @classmethod
    def _check_type_ignore(
        cls,
        file_path: Path,
        lines: list[str],
        approved: Mapping[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect type: ignore comments in code (not in strings/docstrings)."""
        if u.Tests.Validator.is_approved("TYPE-001", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []
        pattern = re.compile(r"#\s*type:\s*ignore")

        for i, line in enumerate(lines, start=1):
            # Match pattern and verify it's in a real comment (not inside strings)
            is_real = u.Tests.Validator.is_real_comment(line, pattern)
            if pattern.search(line) and is_real:
                violation = u.Tests.Validator.create_violation(
                    file_path,
                    i,
                    "TYPE-001",
                    lines,
                )
                violations.append(violation)

        return violations

    @classmethod
    def _check_any_types(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: Mapping[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect Any type annotations."""
        if u.Tests.Validator.is_approved("TYPE-002", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        for node in ast.walk(tree):
            # Check function annotations
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check return annotation
                if node.returns and u.Tests.Validator.is_any_type(node.returns):
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        node.lineno,
                        "TYPE-002",
                        lines,
                        c.Tests.Validator.Messages.TYPE_ANY_RETURN,
                    )
                    violations.append(violation)

                # Check argument annotations
                for arg in node.args.args + node.args.kwonlyargs:
                    if arg.annotation and u.Tests.Validator.is_any_type(arg.annotation):
                        violation = u.Tests.Validator.create_violation(
                            file_path,
                            arg.lineno if hasattr(arg, "lineno") else node.lineno,
                            "TYPE-002",
                            lines,
                            c.Tests.Validator.Messages.TYPE_ANY_ARG.format(arg=arg.arg),
                        )
                        violations.append(violation)

            # Check variable annotations
            elif isinstance(node, ast.AnnAssign):
                if node.annotation and u.Tests.Validator.is_any_type(node.annotation):
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        node.lineno,
                        "TYPE-002",
                        lines,
                        "in variable annotation",
                    )
                    violations.append(violation)

        return violations

    @classmethod
    def _check_cast_usage(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: Mapping[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect unapproved  usage."""
        # Check both custom approved patterns and defaults
        patterns = list(approved.get("TYPE-003", [])) + list(
            c.Tests.Validator.Approved.CAST_PATTERNS,
        )
        file_str = str(file_path)
        if any(re.search(pattern, file_str) for pattern in patterns):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Check for cast call
            is_cast_name = isinstance(node.func, ast.Name) and node.func.id == "cast"
            is_cast_typing = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "cast"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "typing"
            )
            if is_cast_name or is_cast_typing:
                violation = u.Tests.Validator.create_violation(
                    file_path,
                    node.lineno,
                    "TYPE-003",
                    lines,
                )
                violations.append(violation)

        return violations


__all__ = ["FlextValidatorTypes"]

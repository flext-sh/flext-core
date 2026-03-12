"""Bypass validation for FLEXT architecture.

Detects bypass patterns: noqa comments, pragma: no cover, exception swallowing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path

from flext_core import r
from flext_tests import c, m, u


class FlextValidatorBypass:
    """Bypass validation methods for FlextTestsValidator.

    Uses c.Tests.Validator for constants and m.Tests.Validator for models.
    """

    @classmethod
    def _check_exception_swallowing(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: Mapping[str, list[str]],
    ) -> list[m.Tests.Violation]:
        """Detect exception swallowing patterns (bare except or except with pass)."""
        if u.Tests.Validator.is_approved("BYPASS-003", file_path, approved):
            return []
        violations: list[m.Tests.Violation] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        node.lineno,
                        "BYPASS-003",
                        lines,
                        c.Tests.Validator.Messages.BYPASS_BARE_EXCEPT,
                    )
                    violations.append(violation)
                elif u.Tests.Validator.is_only_pass(node.body):
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        node.lineno,
                        "BYPASS-003",
                        lines,
                        c.Tests.Validator.Messages.BYPASS_ONLY_PASS,
                    )
                    violations.append(violation)
        return violations

    @classmethod
    def _check_noqa(
        cls, file_path: Path, lines: list[str], approved: Mapping[str, list[str]]
    ) -> list[m.Tests.Violation]:
        """Detect # noqa comments."""
        if u.Tests.Validator.is_approved("BYPASS-001", file_path, approved):
            return []
        violations: list[m.Tests.Violation] = []
        pattern = re.compile(r"#\s*noqa", re.IGNORECASE)
        for i, line in enumerate(lines, start=1):
            is_real = u.Tests.Validator.is_real_comment(line, pattern)
            if pattern.search(line) and is_real:
                violation = u.Tests.Validator.create_violation(
                    file_path, i, "BYPASS-001", lines
                )
                violations.append(violation)
        return violations

    @classmethod
    def _check_pragma_no_cover(
        cls, file_path: Path, lines: list[str], approved: Mapping[str, list[str]]
    ) -> list[m.Tests.Violation]:
        """Detect # pragma: no cover comments."""
        patterns = list(approved.get("BYPASS-002", [])) + list(
            c.Tests.Validator.Approved.PRAGMA_PATTERNS
        )
        file_str = str(file_path)
        if any(re.search(pattern, file_str) for pattern in patterns):
            return []
        violations: list[m.Tests.Violation] = []
        pattern = re.compile(r"#\s*pragma:\s*no\s*cover", re.IGNORECASE)
        for i, line in enumerate(lines, start=1):
            is_real = u.Tests.Validator.is_real_comment(line, pattern)
            if pattern.search(line) and is_real:
                violation = u.Tests.Validator.create_violation(
                    file_path, i, "BYPASS-002", lines
                )
                violations.append(violation)
        return violations

    @classmethod
    def _scan_file(
        cls, file_path: Path, approved: Mapping[str, list[str]]
    ) -> list[m.Tests.Violation]:
        """Scan a single file for bypass violations."""
        violations: list[m.Tests.Violation] = []
        try:
            content = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return violations
        lines = content.splitlines()
        violations.extend(cls._check_noqa(file_path, lines, approved))
        violations.extend(cls._check_pragma_no_cover(file_path, lines, approved))
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return violations
        violations.extend(
            cls._check_exception_swallowing(file_path, tree, lines, approved)
        )
        return violations

    @classmethod
    def scan(
        cls,
        files: list[Path],
        approved_exceptions: Mapping[str, list[str]] | None = None,
    ) -> r[m.Tests.ScanResult]:
        """Scan files for bypass violations.

        Args:
            files: List of Python files to scan
            approved_exceptions: Dict mapping rule IDs to list of approved file patterns

        Returns:
            r with ScanResult containing all violations found

        """
        violations: list[m.Tests.Violation] = []
        approved = approved_exceptions or {}
        for file_path in files:
            file_violations = cls._scan_file(file_path, approved)
            violations.extend(file_violations)
        return r[m.Tests.ScanResult].ok(
            m.Tests.ScanResult.create(
                validator_name=c.Tests.Validator.Defaults.VALIDATOR_BYPASS,
                files_scanned=len(files),
                violations=violations,
            )
        )


__all__ = ["FlextValidatorBypass"]

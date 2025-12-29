"""Import validation for FLEXT architecture.

Detects import violations: lazy imports, TYPE_CHECKING, ImportError handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

from flext_core.result import r
from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.utilities import u

if TYPE_CHECKING:
    from pathlib import Path


class FlextValidatorImports:
    """Import validation methods for FlextTestsValidator.

    Uses c.Tests.Validator for constants and m.Tests.Validator for models.
    """

    @classmethod
    def scan(
        cls,
        files: list[Path],
        approved_exceptions: dict[str, list[str]] | None = None,
    ) -> r[m.Tests.Validator.ScanResult]:
        """Scan files for import violations.

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
                validator_name=c.Tests.Validator.Defaults.VALIDATOR_IMPORTS,
                files_scanned=len(files),
                violations=violations,
            ),
        )

    @classmethod
    def _scan_file(
        cls,
        file_path: Path,
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Scan a single file for import violations."""
        violations: list[m.Tests.Validator.Violation] = []

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError):
            return violations

        lines = content.splitlines()

        # Check for lazy imports (imports not at top)
        violations.extend(cls._check_lazy_imports(file_path, tree, lines, approved))

        # Check for TYPE_CHECKING blocks
        violations.extend(cls._check_type_checking(file_path, tree, lines, approved))

        # Check for try/except ImportError
        violations.extend(
            cls._check_import_error_handling(file_path, tree, lines, approved),
        )

        # Check for sys.path manipulation
        violations.extend(cls._check_sys_path(file_path, tree, lines, approved))

        # Check for direct technology imports
        violations.extend(
            cls._check_direct_tech_imports(file_path, tree, lines, approved),
        )

        # Check for non-root flext imports
        violations.extend(
            cls._check_non_root_flext_imports(file_path, tree, lines, approved),
        )

        return violations

    @classmethod
    def _check_lazy_imports(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect imports not at module top level."""
        if u.Tests.Validator.is_approved("IMPORT-001", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Check if import is inside a function or class
                parent = u.Tests.Validator.get_parent(tree, node)
                if isinstance(
                    parent,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        node.lineno,
                        "IMPORT-001",
                        lines,
                    )
                    violations.append(violation)

        return violations

    @classmethod
    def _check_type_checking(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect TYPE_CHECKING blocks."""
        if u.Tests.Validator.is_approved("IMPORT-002", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        for node in ast.walk(tree):
            # Check if condition is TYPE_CHECKING
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                violation = u.Tests.Validator.create_violation(
                    file_path,
                    node.lineno,
                    "IMPORT-002",
                    lines,
                )
                violations.append(violation)

        return violations

    @classmethod
    def _check_import_error_handling(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect try/except ImportError patterns."""
        if u.Tests.Validator.is_approved("IMPORT-003", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            # Check if any handler catches ImportError or ModuleNotFoundError
            for handler in node.handlers:
                if handler.type is None:
                    continue
                handler_names = u.Tests.Validator.get_exception_names(handler.type)
                if (
                    "ImportError" in handler_names
                    or "ModuleNotFoundError" in handler_names
                ):
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        node.lineno,
                        "IMPORT-003",
                        lines,
                    )
                    violations.append(violation)

        return violations

    @classmethod
    def _check_sys_path(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect sys.path manipulation."""
        if u.Tests.Validator.is_approved("IMPORT-004", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        for node in ast.walk(tree):
            # Check for sys.path
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and node.attr == "path"
            ):
                # Check if it's being modified (append, insert, extend)
                parent = u.Tests.Validator.get_parent(tree, node)
                if isinstance(parent, ast.Call):
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        node.lineno,
                        "IMPORT-004",
                        lines,
                    )
                    violations.append(violation)

        return violations

    @classmethod
    def _check_direct_tech_imports(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect direct technology imports."""
        if u.Tests.Validator.is_approved("IMPORT-005", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []
        tech_imports = c.Tests.Validator.Approved.TECH_IMPORTS

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in tech_imports:
                        violation = u.Tests.Validator.create_violation(
                            file_path,
                            node.lineno,
                            "IMPORT-005",
                            lines,
                            alias.name,
                        )
                        violations.append(violation)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] in tech_imports
            ):
                violation = u.Tests.Validator.create_violation(
                    file_path,
                    node.lineno,
                    "IMPORT-005",
                    lines,
                    node.module,
                )
                violations.append(violation)

        return violations

    @classmethod
    def _check_non_root_flext_imports(
        cls,
        file_path: Path,
        tree: ast.AST,
        lines: list[str],
        approved: dict[str, list[str]],
    ) -> list[m.Tests.Validator.Violation]:
        """Detect non-root imports from flext-* packages internal modules.

        Detects imports from internal modules (prefixed with _) like:
        - from flext_core._models import domain  (violation)
        - from flext_tests._validator import imports  (violation)

        Allows public module imports:
        - from flext_core.result import r  (OK)
        - from flext_tests.models import m  (OK)

        Allows __init__.py inside internal packages to import sibling modules:
        - _validator/__init__.py can import from flext_tests._validator.* (OK)
        """
        if u.Tests.Validator.is_approved("IMPORT-006", file_path, approved):
            return []

        # Check if file is an internal package __init__.py (allowed to import siblings)
        file_str = str(file_path)
        internal_init_patterns = c.Tests.Validator.Approved.INTERNAL_INIT_PATTERNS
        if any(re.search(pattern, file_str) for pattern in internal_init_patterns):
            return []

        violations: list[m.Tests.Validator.Violation] = []
        flext_packages = c.Tests.Validator.Approved.FLEXT_PACKAGES

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if len(parts) > 1 and parts[0] in flext_packages:
                    # Check if any part is internal (starts with _)
                    internal_parts = [p for p in parts[1:] if p.startswith("_")]
                    if internal_parts:
                        internal = internal_parts[0]
                        violation = u.Tests.Validator.create_violation(
                            file_path,
                            node.lineno,
                            "IMPORT-006",
                            lines,
                            f"from {node.module} (internal: {internal})",
                        )
                        violations.append(violation)

        return violations


__all__ = ["FlextValidatorImports"]

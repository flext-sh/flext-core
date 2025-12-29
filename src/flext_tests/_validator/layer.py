"""Layer validation for FLEXT architecture.

Detects layer violations: lower layer importing upper layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path

from flext_core.result import r
from flext_tests.constants import c
from flext_tests.models import m
from flext_tests.utilities import u


class FlextValidatorLayer:
    """Layer validation methods for FlextTestsValidator.

    Uses c.Tests.Validator, m.Tests.Validator, u.Tests.Validator.
    """

    @classmethod
    def scan(
        cls,
        files: list[Path],
        approved_exceptions: dict[str, list[str]] = Field(default_factory=dict[str, list[str]]),
        layer_hierarchy: dict[str, int] = Field(default_factory=dict[str, int]),
    ) -> r[m.Tests.Validator.ScanResult]:
        """Scan files for layer violations.

        Args:
            files: List of Python files to scan
            approved_exceptions: Dict mapping rule IDs to list of approved file patterns
            layer_hierarchy: Custom layer hierarchy (module_name -> layer_number)

        Returns:
            FlextResult with ScanResult containing all violations found

        """
        violations: list[m.Tests.Validator.Violation] = []
        approved = approved_exceptions or {}
        hierarchy = layer_hierarchy or c.Tests.Validator.LayerHierarchy.as_dict()

        for file_path in files:
            file_violations = cls._scan_file(file_path, approved, hierarchy)
            violations.extend(file_violations)

        return r[m.Tests.Validator.ScanResult].ok(
            m.Tests.Validator.ScanResult.create(
                validator_name=c.Tests.Validator.Defaults.VALIDATOR_LAYER,
                files_scanned=len(files),
                violations=violations,
            ),
        )

    @classmethod
    def _scan_file(
        cls,
        file_path: Path,
        approved: dict[str, list[str]],
        hierarchy: dict[str, int],
    ) -> list[m.Tests.Validator.Violation]:
        """Scan a single file for layer violations."""
        if u.Tests.Validator.is_approved("LAYER-001", file_path, approved):
            return []

        violations: list[m.Tests.Validator.Violation] = []

        # Get the layer of the current file
        current_module = file_path.stem
        current_layer = hierarchy.get(current_module)

        if current_layer is None:
            # Not a tracked module, skip
            return violations

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return violations

        lines = content.splitlines()

        # Check all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_module = cls._extract_module_name(node.module)
                imported_layer = hierarchy.get(imported_module)

                if imported_layer is not None and imported_layer > current_layer:
                    # Lower layer importing higher layer - violation!
                    violation = u.Tests.Validator.create_violation(
                        file_path,
                        node.lineno,
                        "LAYER-001",
                        lines,
                        c.Tests.Validator.Messages.LAYER_VIOLATION.format(
                            current=current_module,
                            current_level=current_layer,
                            imported=imported_module,
                            imported_level=imported_layer,
                        ),
                    )
                    violations.append(violation)

        return violations

    @classmethod
    def _extract_module_name(cls, module_path: str) -> str:
        """Extract the final module name from an import path.

        Examples:
            'flext_core.result' -> 'result'
            'flext_core._models.domain' -> 'domain'
            'result' -> 'result'

        """
        parts = module_path.split(".")
        # Return the last part
        return parts[-1]


__all__ = ["FlextValidatorLayer"]

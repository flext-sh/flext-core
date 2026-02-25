"""Models for FLEXT architecture validation.

Provides shared models for all validator extensions using FlextTestsModels patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import Field

from flext_core._models.entity import FlextModelsEntity
from flext_tests.constants import FlextTestsConstants, c
from flext_tests.models import m

# Type aliases for readability
type _SeverityLiteral = FlextTestsConstants.Tests.Validator.SeverityLiteral


class FlextValidatorModels(m):
    """Models for FLEXT architecture validation - extends FlextTestsModels.

    Uses c.Tests.Validator for constants (Severity, Rules, Defaults, Approved patterns).
    """

    # Re-export Severity from constants for convenience
    Severity = c.Tests.Validator.Severity

    class Violation(FlextModelsEntity.Value):
        """A detected architecture violation."""

        file_path: Path
        line_number: int
        rule_id: str
        severity: _SeverityLiteral
        description: str
        code_snippet: str = ""

        def format(self) -> str:
            """Format violation as string using c.Tests.Validator.Messages."""
            return c.Tests.Validator.Messages.VIOLATION_WITH_SNIPPET.format(
                rule_id=self.rule_id,
                description=self.description,
                snippet=self.code_snippet or "(no snippet)",
            )

        def format_short(self) -> str:
            """Format violation as short string."""
            return c.Tests.Validator.Messages.VIOLATION.format(
                rule_id=self.rule_id,
                file=self.file_path.name,
                line=self.line_number,
            )

    class ScanResult(FlextModelsEntity.Value):
        """Result of a validation scan."""

        validator_name: str
        files_scanned: int
        violations: list[FlextValidatorModels.Violation]
        passed: bool

        @classmethod
        def create(
            cls,
            validator_name: str,
            files_scanned: int,
            violations: list[FlextValidatorModels.Violation],
        ) -> FlextValidatorModels.ScanResult:
            """Create a ScanResult from violations."""
            return cls(
                validator_name=validator_name,
                files_scanned=files_scanned,
                violations=violations,
                passed=len(violations) == 0,
            )

        def format(self) -> str:
            """Format scan result as string using c.Tests.Validator.Messages."""
            if self.passed:
                return c.Tests.Validator.Messages.SCAN_PASSED.format(
                    count=self.files_scanned,
                )
            return c.Tests.Validator.Messages.SCAN_FAILED.format(
                violations=len(self.violations),
                count=self.files_scanned,
            )

    class ScanConfig(FlextModelsEntity.Value):
        """Configuration for validation scan."""

        target_path: Path
        include_patterns: list[str] = Field(
            default_factory=lambda: list(c.Tests.Validator.Defaults.INCLUDE_PATTERNS),
        )
        exclude_patterns: list[str] = Field(
            default_factory=lambda: list(c.Tests.Validator.Defaults.EXCLUDE_PATTERNS),
        )
        approved_exceptions: Mapping[str, list[str]] = Field(default_factory=dict)


# Short alias
vm = FlextValidatorModels

__all__ = ["FlextValidatorModels", "vm"]

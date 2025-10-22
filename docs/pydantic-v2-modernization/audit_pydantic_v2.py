#!/usr/bin/env python3
"""Automated Pydantic v2 compliance audit script for FLEXT projects.

Usage:
    # Audit current project
    python audit_pydantic_v2.py

    # Audit specific project
    python audit_pydantic_v2.py --project ../flext-cli

    # Audit all 33 projects (from workspace root)
    for dir in flext-*; do python "$dir/docs/pydantic-v2-modernization/audit_pydantic_v2.py"; done

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class AuditViolation:
    """Represents a single audit violation."""

    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    pattern: str  # What pattern was violated
    file: str  # File path
    line: int  # Line number
    code: str  # Code snippet
    detail: str  # Detailed explanation


@dataclass
class AuditResult:
    """Results of auditing a project."""

    project: str
    status: str  # PASS, FAIL, WARNING
    critical: list[AuditViolation] = field(default_factory=list)
    high: list[AuditViolation] = field(default_factory=list)
    medium: list[AuditViolation] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_violations(self) -> int:
        """Total number of violations."""
        return len(self.critical) + len(self.high) + len(self.medium)

    def __str__(self) -> str:
        """Format audit result for output."""
        lines = [
            "=" * 70,
            f"PROJECT AUDIT REPORT: {self.project}",
            "=" * 70,
            f"Status: {self.status}",
            f"Violations: {self.total_violations}",
            "",
        ]

        if self.critical:
            lines.extend([
                "🔴 CRITICAL VIOLATIONS (MUST FIX):",
                "-" * 70,
            ])
            for v in self.critical:
                lines.extend([
                    f"  {v.pattern}",
                    f"    File: {v.file}:{v.line}",
                    f"    Detail: {v.detail}",
                    "",
                ])

        if self.high:
            lines.extend([
                "🟠 HIGH PRIORITY VIOLATIONS (SHOULD FIX):",
                "-" * 70,
            ])
            for v in self.high:
                lines.extend([
                    f"  {v.pattern}",
                    f"    File: {v.file}:{v.line}",
                    f"    Detail: {v.detail}",
                    "",
                ])

        if self.recommendations:
            lines.extend([
                "💡 RECOMMENDATIONS:",
                "-" * 70,
            ])
            lines.extend([f"  • {rec}" for rec in self.recommendations])
            lines.append("")

        lines.extend([
            "📊 STATISTICS:",
            "-" * 70,
        ])
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class PydanticV2Auditor:
    """Audits project for Pydantic v2 compliance."""

    # CRITICAL: Pydantic v1 patterns (MUST NOT EXIST)
    CRITICAL_PATTERNS: ClassVar[dict[str, str]] = {
        r"class\s+\w+.*:\s*\n\s*class\s+Config": "Pydantic v1 `class Config` pattern",
        r"\.dict\(": "Pydantic v1 `.dict()` method (use `model_dump()`)",
        # NOTE: .json() pattern excluded due to HTTP library false positives (requests.json(), httpx.json())
        # Pydantic v1 .json() is less common in modern codebases with model_dump_json()
        # r"\.json\(": "Pydantic v1 `.json()` method (use `model_dump_json()`)",
        r"parse_obj\(": "Pydantic v1 `parse_obj()` (use `model_validate()`)",
        r"@validator\(": "Pydantic v1 `@validator` (use `@field_validator`)",
        r"@root_validator": "Pydantic v1 `@root_validator` (use `@model_validator`)",
    }

    # HIGH: Missing Pydantic v2 patterns (SHOULD EXIST)
    HIGH_PATTERNS: ClassVar[dict[str, str]] = {
        r"model_dump\(": "Uses `model_dump()` for serialization",
        r"model_validate\(": "Uses `model_validate()` for parsing",
        r"@field_validator": "Uses `@field_validator` decorator",
        r"ConfigDict": "Uses `ConfigDict` for model configuration",
    }

    # Anti-patterns: Custom validation methods that should NOT exist
    # (Removed in Phase 3: validate_port, validate_email, validate_url, validate_positive_integer,
    # validate_non_negative_integer, validate_string_length, validate_string_pattern,
    # validate_file_path, validate_directory_path, validate_timeout_seconds, validate_retry_count,
    # validate_log_level, validate_string_not_none, validate_string_not_empty, validate_string,
    # validate_host, validate_pipeline were consolidated into Pydantic v2 native types)
    REMOVED_VALIDATORS: ClassVar[dict[str, str]] = {}

    def __init__(self, project_path: str | None = None) -> None:
        """Initialize auditor."""
        self.project_path = Path(project_path or ".").resolve()
        self.src_path = self.project_path / "src"
        self.result = AuditResult(project=str(self.project_path.name), status="PENDING")

    def audit(self) -> AuditResult:
        """Run complete audit."""
        if not self.src_path.exists():
            self.result.status = "FAIL"
            self.result.critical.append(
                AuditViolation(
                    severity="CRITICAL",
                    pattern="Missing src directory",
                    file=str(self.src_path),
                    line=0,
                    code="",
                    detail="No src/ directory found in project",
                )
            )
            return self.result

        # Collect all Python files
        py_files = list(self.src_path.rglob("*.py"))
        if not py_files:
            self.result.status = "SKIP"
            self.result.recommendations.append("No Python files found in src/")
            return self.result

        # Scan files
        for py_file in py_files:
            self._audit_file(py_file)

        # Determine status
        if self.result.critical:
            self.result.status = "FAIL"
        elif self.result.high:
            self.result.status = "WARNING"
        else:
            self.result.status = "PASS"

        # Add statistics
        self._add_statistics(py_files)

        return self.result

    def _audit_file(self, py_file: Path) -> None:
        """Audit a single Python file."""
        try:
            content = py_file.read_text(encoding="utf-8")
            lines = content.split("\n")
        except Exception as e:
            self.result.recommendations.append(f"Could not read {py_file}: {e}")
            return

        # Check critical patterns
        for pattern, description in self.CRITICAL_PATTERNS.items():
            matches = self._find_pattern(pattern, lines)
            for line_num in matches:
                self.result.critical.append(
                    AuditViolation(
                        severity="CRITICAL",
                        pattern=description,
                        file=str(py_file.relative_to(self.project_path)),
                        line=line_num + 1,
                        code=lines[line_num].strip() if line_num < len(lines) else "",
                        detail="Pydantic v1 pattern detected",
                    )
                )

        # Check for removed validators (should not exist in codebase)
        # All 17 duplicate validators were removed in Phase 3
        # This check ensures they don't accidentally reappear
        # NOTE: validate_pipeline is EXCLUDED - it's legitimate business logic for composing validators
        removed_validator_patterns = {
            "validate_port",
            "validate_email",
            "validate_url",
            "validate_positive_integer",
            "validate_non_negative_integer",
            "validate_string_length",
            "validate_string_pattern",
            "validate_file_path",
            "validate_directory_path",
            "validate_timeout_seconds",
            "validate_retry_count",
            "validate_log_level",
            "validate_string_not_none",
            "validate_string_not_empty",
            "validate_string",
            "validate_host",
        }
        for validator_name in removed_validator_patterns:
            pattern = rf"def {validator_name}\("
            matches = self._find_pattern(pattern, lines)
            for line_num in matches:
                self.result.critical.append(
                    AuditViolation(
                        severity="CRITICAL",
                        pattern=f"REGRESSION: Removed validator reappeared: {validator_name}",
                        file=str(py_file.relative_to(self.project_path)),
                        line=line_num + 1,
                        code=lines[line_num].strip(),
                        detail="This validator was removed in Phase 3 and should not be re-added. Use Pydantic v2 native types instead.",
                    )
                )

    def _find_pattern(
        self,
        pattern: str,
        lines: list[str],
    ) -> list[int]:
        """Find all lines matching a pattern."""
        matches = []
        for idx, line in enumerate(lines):
            if re.search(pattern, line):
                matches.append(idx)
        return matches

    def _add_statistics(self, py_files: list[Path]) -> None:
        """Add audit statistics."""
        self.result.stats["Total Python files"] = len(py_files)
        self.result.stats["Total violations"] = self.result.total_violations
        self.result.stats["Critical violations"] = len(self.result.critical)
        self.result.stats["High violations"] = len(self.result.high)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audit Pydantic v2 compliance")
    parser.add_argument(
        "--project",
        default=".",
        help="Project path to audit (default: current directory)",
    )
    args = parser.parse_args()

    auditor = PydanticV2Auditor(args.project)
    result = auditor.audit()

    print(result)

    # Return exit code based on status
    return 1 if result.status == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for FlextInfraCodegenCensus service.

Validates violation parsing, fixability classification, and project exclusion
logic without hitting the real workspace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_infra.codegen.census import FlextInfraCodegenCensus
from flext_infra.constants import c
from flext_infra.models import FlextInfraModels
from flext_tests import tm


@pytest.fixture
def census(tmp_path: Path) -> FlextInfraCodegenCensus:
    """Census service pointed at an empty temporary workspace."""
    return FlextInfraCodegenCensus(workspace_root=tmp_path)


class TestParseViolationValid:
    """_parse_violation correctly parses well-formed violation strings."""

    @pytest.mark.parametrize(
        (
            "violation_str",
            "expected_rule",
            "expected_module",
            "expected_line",
            "expected_msg",
        ),
        [
            (
                "[NS-000-001] src/file.py:42 — Multiple outer classes found (expected 1, got 2)",
                "NS-000",
                "src/file.py",
                42,
                "Multiple outer classes found (expected 1, got 2)",
            ),
            (
                "[NS-001-001] src/file.py:10 — Loose Final constant 'X' belongs in constants.py",
                "NS-001",
                "src/file.py",
                10,
                "Loose Final constant 'X' belongs in constants.py",
            ),
            (
                "[NS-002-001] src/file.py:5 — TypeVar 'T' belongs in typings.py",
                "NS-002",
                "src/file.py",
                5,
                "TypeVar 'T' belongs in typings.py",
            ),
            (
                "[NS-001-099] src/deep/nested/module.py:999 — Some long message with special chars: !@#",
                "NS-001",
                "src/deep/nested/module.py",
                999,
                "Some long message with special chars: !@#",
            ),
        ],
        ids=["ns000", "ns001", "ns002", "deep-path"],
    )
    def test_parses_fields(
        self,
        violation_str: str,
        expected_rule: str,
        expected_module: str,
        expected_line: int,
        expected_msg: str,
    ) -> None:
        result = FlextInfraCodegenCensus._parse_violation(violation_str)
        tm.that(result is not None, eq=True)
        tm.that(
            isinstance(result, FlextInfraModels.Infra.Codegen.CensusViolation), eq=True
        )
        tm.that(result.rule, eq=expected_rule)
        tm.that(result.module, eq=expected_module)
        tm.that(result.line, eq=expected_line)
        tm.that(result.message, eq=expected_msg)


class TestParseViolationInvalid:
    """_parse_violation returns None for malformed strings."""

    @pytest.mark.parametrize(
        "violation_str",
        [
            "",
            "random text without brackets",
            "[WRONG-FORMAT] missing fields",
            "[NS-001] src/file.py:10 - wrong dash instead of em-dash",
            "src/file.py:10 — Missing rule prefix",
            "[NS-001-001] no-colon-line — message",
            "[NS-001-001] src/file.py:notanumber — message",
        ],
        ids=[
            "empty",
            "no-brackets",
            "wrong-format",
            "wrong-dash",
            "missing-rule",
            "no-line-number",
            "non-numeric-line",
        ],
    )
    def test_returns_none(self, violation_str: str) -> None:
        tm.that(
            FlextInfraCodegenCensus._parse_violation(violation_str) is None, eq=True
        )


class TestFixabilityClassification:
    """NS-000 is not fixable; NS-001 and NS-002 are fixable."""

    def test_ns000_not_fixable(self) -> None:
        result = FlextInfraCodegenCensus._parse_violation(
            "[NS-000-001] src/file.py:1 — Structure violation",
        )
        tm.that(result is not None, eq=True)
        tm.that(result.fixable, eq=False)

    def test_ns001_fixable(self) -> None:
        result = FlextInfraCodegenCensus._parse_violation(
            "[NS-001-001] src/file.py:1 — Constant violation",
        )
        tm.that(result is not None, eq=True)
        tm.that(result.fixable, eq=True)

    def test_ns002_fixable(self) -> None:
        result = FlextInfraCodegenCensus._parse_violation(
            "[NS-002-001] src/file.py:1 — TypeVar violation",
        )
        tm.that(result is not None, eq=True)
        tm.that(result.fixable, eq=True)

    def test_ns000_multiple_sub_rules_not_fixable(self) -> None:
        """Different sub-rule numbers under NS-000 are still not fixable."""
        for sub in ("001", "002", "099"):
            result = FlextInfraCodegenCensus._parse_violation(
                f"[NS-000-{sub}] src/x.py:1 — msg"
            )
            tm.that(result is not None, eq=True)
            tm.that(result.fixable, eq=False)


class TestExcludedProjects:
    """Census skips the 'flexcore' project."""

    def test_flexcore_in_excluded_set(self) -> None:
        tm.that("flexcore" in c.Infra.Codegen.EXCLUDED_PROJECTS, eq=True)

    def test_excluded_set_is_frozenset(self) -> None:
        tm.that(isinstance(c.Infra.Codegen.EXCLUDED_PROJECTS, frozenset), eq=True)


class TestViolationPattern:
    """Compiled regex captures named groups correctly."""

    def test_named_groups_present(self) -> None:
        match = c.Infra.Codegen.VIOLATION_PATTERN.match(
            "[NS-001-001] src/file.py:10 — msg"
        )
        tm.that(match is not None, eq=True)
        tm.that(set(match.groupdict().keys()), eq={"rule", "module", "line", "message"})


_CV = FlextInfraModels.Infra.Codegen.CensusViolation
_CR = FlextInfraModels.Infra.Codegen.CensusReport


class TestCensusViolationModel:
    """CensusViolation model validates parsed data correctly."""

    def test_model_fields(self) -> None:
        v = _CV(
            module="src/file.py",
            rule="NS-001",
            line=10,
            message="Test message",
            fixable=True,
        )
        tm.that(v.module, eq="src/file.py")
        tm.that(v.rule, eq="NS-001")
        tm.that(v.line, eq=10)
        tm.that(v.message, eq="Test message")
        tm.that(v.fixable, eq=True)


class TestCensusReportModel:
    """CensusReport aggregates violations correctly."""

    def test_empty_report(self) -> None:
        report = _CR(project="test-project", violations=[], total=0, fixable=0)
        tm.that(report.project, eq="test-project")
        tm.that(report.total, eq=0)
        tm.that(report.fixable, eq=0)
        tm.that(report.violations, eq=[])

    def test_report_with_mixed_violations(self) -> None:
        violations = [
            _CV(module="src/a.py", rule="NS-000", line=1, message="m1", fixable=False),
            _CV(module="src/b.py", rule="NS-001", line=2, message="m2", fixable=True),
            _CV(module="src/c.py", rule="NS-002", line=3, message="m3", fixable=True),
        ]
        report = _CR(
            project="test-project",
            violations=violations,
            total=len(violations),
            fixable=sum(1 for v in violations if v.fixable),
        )
        tm.that(report.total, eq=3)
        tm.that(report.fixable, eq=2)


__all__: list[str] = []

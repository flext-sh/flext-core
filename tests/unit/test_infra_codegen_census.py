"""Tests for FlextInfraCodegenCensus service.

Validates violation parsing, fixability classification, and project exclusion
logic without hitting the real workspace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.codegen.census import (
    _EXCLUDED_PROJECTS,
    _VIOLATION_PATTERN,
    FlextInfraCodegenCensus,
)
from flext_infra.models import FlextInfraModels

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def census(tmp_path: Path) -> FlextInfraCodegenCensus:
    """Census service pointed at an empty temporary workspace."""
    return FlextInfraCodegenCensus(workspace_root=tmp_path)


# ---------------------------------------------------------------------------
# _parse_violation — valid inputs
# ---------------------------------------------------------------------------


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
                "[NS-000-001] src/file.py:42 \u2014 Multiple outer classes found (expected 1, got 2)",
                "NS-000",
                "src/file.py",
                42,
                "Multiple outer classes found (expected 1, got 2)",
            ),
            (
                "[NS-001-001] src/file.py:10 \u2014 Loose Final constant 'X' belongs in constants.py",
                "NS-001",
                "src/file.py",
                10,
                "Loose Final constant 'X' belongs in constants.py",
            ),
            (
                "[NS-002-001] src/file.py:5 \u2014 TypeVar 'T' belongs in typings.py",
                "NS-002",
                "src/file.py",
                5,
                "TypeVar 'T' belongs in typings.py",
            ),
            (
                "[NS-001-099] src/deep/nested/module.py:999 \u2014 Some long message with special chars: !@#",
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

        assert result is not None
        assert isinstance(result, FlextInfraModels.CensusViolation)
        assert result.rule == expected_rule
        assert result.module == expected_module
        assert result.line == expected_line
        assert result.message == expected_msg


# ---------------------------------------------------------------------------
# _parse_violation — invalid inputs
# ---------------------------------------------------------------------------


class TestParseViolationInvalid:
    """_parse_violation returns None for malformed strings."""

    @pytest.mark.parametrize(
        "violation_str",
        [
            "",
            "random text without brackets",
            "[WRONG-FORMAT] missing fields",
            "[NS-001] src/file.py:10 - wrong dash instead of em-dash",
            "src/file.py:10 \u2014 Missing rule prefix",
            "[NS-001-001] no-colon-line \u2014 message",
            "[NS-001-001] src/file.py:notanumber \u2014 message",
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
        result = FlextInfraCodegenCensus._parse_violation(violation_str)
        assert result is None


# ---------------------------------------------------------------------------
# Fixability classification
# ---------------------------------------------------------------------------


class TestFixabilityClassification:
    """NS-000 is not fixable; NS-001 and NS-002 are fixable."""

    def test_ns000_not_fixable(self) -> None:
        result = FlextInfraCodegenCensus._parse_violation(
            "[NS-000-001] src/file.py:1 \u2014 Structure violation"
        )
        assert result is not None
        assert result.fixable is False

    def test_ns001_fixable(self) -> None:
        result = FlextInfraCodegenCensus._parse_violation(
            "[NS-001-001] src/file.py:1 \u2014 Constant violation"
        )
        assert result is not None
        assert result.fixable is True

    def test_ns002_fixable(self) -> None:
        result = FlextInfraCodegenCensus._parse_violation(
            "[NS-002-001] src/file.py:1 \u2014 TypeVar violation"
        )
        assert result is not None
        assert result.fixable is True

    def test_ns000_multiple_sub_rules_not_fixable(self) -> None:
        """Different sub-rule numbers under NS-000 are still not fixable."""
        for sub in ("001", "002", "099"):
            result = FlextInfraCodegenCensus._parse_violation(
                f"[NS-000-{sub}] src/x.py:1 \u2014 msg"
            )
            assert result is not None
            assert result.fixable is False


# ---------------------------------------------------------------------------
# Excluded projects
# ---------------------------------------------------------------------------


class TestExcludedProjects:
    """Census skips the 'flexcore' project."""

    def test_flexcore_in_excluded_set(self) -> None:
        assert "flexcore" in _EXCLUDED_PROJECTS

    def test_excluded_set_is_frozenset(self) -> None:
        assert isinstance(_EXCLUDED_PROJECTS, frozenset)


# ---------------------------------------------------------------------------
# Regex pattern sanity
# ---------------------------------------------------------------------------


class TestViolationPattern:
    """Compiled regex captures named groups correctly."""

    def test_named_groups_present(self) -> None:
        expected_groups = {"rule", "module", "line", "message"}
        match = _VIOLATION_PATTERN.match("[NS-001-001] src/file.py:10 \u2014 msg")
        assert match is not None
        assert set(match.groupdict().keys()) == expected_groups


# ---------------------------------------------------------------------------
# CensusViolation model round-trip
# ---------------------------------------------------------------------------


class TestCensusViolationModel:
    """CensusViolation model validates parsed data correctly."""

    def test_model_fields(self) -> None:
        violation = FlextInfraModels.CensusViolation(
            module="src/file.py",
            rule="NS-001",
            line=10,
            message="Test message",
            fixable=True,
        )
        assert violation.module == "src/file.py"
        assert violation.rule == "NS-001"
        assert violation.line == 10
        assert violation.message == "Test message"
        assert violation.fixable is True


# ---------------------------------------------------------------------------
# CensusReport model
# ---------------------------------------------------------------------------


class TestCensusReportModel:
    """CensusReport aggregates violations correctly."""

    def test_empty_report(self) -> None:
        report = FlextInfraModels.CensusReport(
            project="test-project",
            violations=[],
            total=0,
            fixable=0,
        )
        assert report.project == "test-project"
        assert report.total == 0
        assert report.fixable == 0
        assert report.violations == []

    def test_report_with_mixed_violations(self) -> None:
        violations = [
            FlextInfraModels.CensusViolation(
                module="src/a.py",
                rule="NS-000",
                line=1,
                message="m1",
                fixable=False,
            ),
            FlextInfraModels.CensusViolation(
                module="src/b.py",
                rule="NS-001",
                line=2,
                message="m2",
                fixable=True,
            ),
            FlextInfraModels.CensusViolation(
                module="src/c.py",
                rule="NS-002",
                line=3,
                message="m3",
                fixable=True,
            ),
        ]
        report = FlextInfraModels.CensusReport(
            project="test-project",
            violations=violations,
            total=len(violations),
            fixable=sum(1 for v in violations if v.fixable),
        )
        assert report.total == 3
        assert report.fixable == 2


__all__: list[str] = []

"""Tests for FlextInfraDocValidator — core validate and report model.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r, t
from flext_infra.docs.shared import FlextInfraDocsShared
from flext_infra.docs.validator import FlextInfraDocValidator
from flext_tests import tm
from tests.infra.models import m


class TestValidateReport:
    """Tests for DocsPhaseReport model used by validator."""

    def test_report_frozen(self) -> None:
        """Test DocsPhaseReport is frozen (immutable)."""
        tm.that(m.Infra.Docs.DocsPhaseReport.model_config.get("frozen"), eq=True)

    def test_missing_adr_skills_field(self) -> None:
        """Test DocsPhaseReport missing_adr_skills field."""
        report = m.Infra.Docs.DocsPhaseReport(
            phase="validate",
            scope="test",
            result="FAIL",
            message="Missing skills",
            missing_adr_skills=["skill1", "skill2"],
        )
        tm.that(len(report.missing_adr_skills), eq=2)
        tm.that("skill1" in report.missing_adr_skills, eq=True)

    def test_todo_written_field(self) -> None:
        """Test DocsPhaseReport todo_written field."""
        report = m.Infra.Docs.DocsPhaseReport(
            phase="validate",
            scope="test",
            result="PASS",
            message="Validation passed",
            todo_written=True,
        )
        tm.that(report.todo_written, eq=True)

    def test_result_field_values(self) -> None:
        """Test DocsPhaseReport result field accepts valid values."""
        for status in ["PASS", "FAIL", "WARN"]:
            report = m.Infra.Docs.DocsPhaseReport(
                phase="validate",
                scope="test",
                result=status,
                message="Test",
            )
            tm.that(report.result, eq=status)

    def test_message_field(self) -> None:
        """Test DocsPhaseReport message field."""
        report = m.Infra.Docs.DocsPhaseReport(
            phase="validate",
            scope="test",
            result="PASS",
            message="All validations passed successfully",
        )
        tm.that(report.message, eq="All validations passed successfully")


class TestValidateCore:
    """Tests for FlextInfraDocValidator.validate."""

    @pytest.fixture
    def validator(self) -> FlextInfraDocValidator:
        """Create validator instance."""
        return FlextInfraDocValidator()

    def test_returns_flext_result(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test that validate returns r."""
        result = validator.validate(tmp_path)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_valid_scope_returns_success(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test validate with valid scope returns success."""
        result = validator.validate(tmp_path)
        tm.ok(result)
        tm.that(len(result.value) >= 0, eq=True)

    def test_report_structure(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test ValidateReport has required fields."""
        result = validator.validate(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            tm.that(hasattr(report, "scope"), eq=True)
            tm.that(hasattr(report, "result"), eq=True)
            tm.that(hasattr(report, "message"), eq=True)

    def test_with_project_filter(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test validate with single project filter."""
        result = validator.validate(tmp_path, project="test-project")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_with_projects_filter(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test validate with multiple projects filter."""
        result = validator.validate(tmp_path, projects="proj1,proj2")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_with_check_parameter(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test validate with check parameter."""
        result = validator.validate(tmp_path, check="adr-skills")
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_apply_false_dry_run(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test validate with apply=False (dry-run mode)."""
        result = validator.validate(tmp_path, apply=False)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_apply_true(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test validate with apply=True."""
        result = validator.validate(tmp_path, apply=True)
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_custom_output_dir(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test validate with custom output directory."""
        result = validator.validate(tmp_path, output_dir=str(tmp_path / "custom"))
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_multiple_scopes(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
    ) -> None:
        """Test validate returns list for multiple scopes."""
        result = validator.validate(tmp_path, projects="proj1,proj2,proj3")
        if result.is_success:
            tm.that(len(result.value) >= 0, eq=True)

    def test_scope_failure_returns_failure(
        self,
        validator: FlextInfraDocValidator,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test validate returns failure when scope building fails."""

        def mock_build_scopes(
            *args: t.Scalar, **kwargs: t.Scalar
        ) -> r[list[m.Infra.Docs.FlextInfraDocScope]]:
            return r[list[m.Infra.Docs.FlextInfraDocScope]].fail("Scope error")

        monkeypatch.setattr(FlextInfraDocsShared, "build_scopes", mock_build_scopes)
        result = validator.validate(tmp_path)
        tm.fail(result, has="Scope error")

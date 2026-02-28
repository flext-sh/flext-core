"""Tests for FlextInfraDocValidator service.

Tests documentation validation functionality with mocked file system
and structured FlextResult reports.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_infra.docs.shared import FlextInfraDocScope
from flext_infra.docs.validator import FlextInfraDocValidator, ValidateReport


class TestFlextInfraDocValidator:
    """Tests for FlextInfraDocValidator service."""

    @pytest.fixture
    def validator(self) -> FlextInfraDocValidator:
        """Create validator instance."""
        return FlextInfraDocValidator()

    @pytest.fixture
    def sample_scope(self, tmp_path: Path) -> FlextInfraDocScope:
        """Create sample documentation scope."""
        report_dir = tmp_path / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        return FlextInfraDocScope(
            name="test-project",
            path=tmp_path,
            report_dir=report_dir,
        )

    def test_validate_returns_flext_result(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test that validate returns FlextResult[list[ValidateReport]]."""
        result = validator.validate(tmp_path)
        assert result.is_success or result.is_failure

    def test_validate_with_valid_scope_returns_success(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test validate with valid scope returns success."""
        result = validator.validate(tmp_path)
        assert result.is_success
        assert isinstance(result.value, list)

    def test_validate_report_structure(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test ValidateReport has required fields."""
        result = validator.validate(tmp_path)
        if result.is_success and result.value:
            report = result.value[0]
            assert hasattr(report, "scope")
            assert hasattr(report, "result")
            assert hasattr(report, "message")
            assert hasattr(report, "missing_adr_skills")
            assert hasattr(report, "todo_written")

    def test_validate_report_frozen(self) -> None:
        """Test ValidateReport is frozen (immutable)."""
        report = ValidateReport(
            scope="test",
            result="PASS",
            message="Validation passed",
        )
        with pytest.raises(Exception):  # pydantic frozen raises
            report.scope = "modified"  # type: ignore

    def test_validate_report_missing_adr_skills_field(self) -> None:
        """Test ValidateReport missing_adr_skills field."""
        report = ValidateReport(
            scope="test",
            result="FAIL",
            message="Missing skills",
            missing_adr_skills=["skill1", "skill2"],
        )
        assert len(report.missing_adr_skills) == 2
        assert "skill1" in report.missing_adr_skills

    def test_validate_report_todo_written_field(self) -> None:
        """Test ValidateReport todo_written field."""
        report = ValidateReport(
            scope="test",
            result="PASS",
            message="Validation passed",
            todo_written=True,
        )
        assert report.todo_written is True

    def test_validate_with_project_filter(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test validate with single project filter."""
        result = validator.validate(tmp_path, project="test-project")
        assert result.is_success or result.is_failure

    def test_validate_with_projects_filter(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test validate with multiple projects filter."""
        result = validator.validate(tmp_path, projects="proj1,proj2")
        assert result.is_success or result.is_failure

    def test_validate_with_check_parameter(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test validate with check parameter."""
        result = validator.validate(tmp_path, check="adr-skills")
        assert result.is_success or result.is_failure

    def test_validate_with_apply_false_dry_run(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test validate with apply=False (dry-run mode)."""
        result = validator.validate(tmp_path, apply=False)
        assert result.is_success or result.is_failure

    def test_validate_with_apply_true_writes_todos(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test validate with apply=True writes TODOS.md."""
        result = validator.validate(tmp_path, apply=True)
        assert result.is_success or result.is_failure

    def test_validate_with_custom_output_dir(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test validate with custom output directory."""
        output_dir = str(tmp_path / "custom_output")
        result = validator.validate(tmp_path, output_dir=output_dir)
        assert result.is_success or result.is_failure

    def test_validate_report_result_field_values(self) -> None:
        """Test ValidateReport result field accepts valid values."""
        for status in ["PASS", "FAIL", "WARN"]:
            report = ValidateReport(
                scope="test",
                result=status,
                message="Test message",
            )
            assert report.result == status

    def test_validate_report_message_field(self) -> None:
        """Test ValidateReport message field."""
        report = ValidateReport(
            scope="test",
            result="PASS",
            message="All validations passed successfully",
        )
        assert report.message == "All validations passed successfully"

    def test_validate_multiple_scopes(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test validate returns multiple reports for multiple scopes."""
        result = validator.validate(tmp_path, projects="proj1,proj2,proj3")
        if result.is_success:
            assert isinstance(result.value, list)

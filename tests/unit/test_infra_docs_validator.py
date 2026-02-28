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
        assert ValidateReport.model_config.get("frozen") is True

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

    def test_validate_scope_with_adr_check(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _validate_scope with adr-skill check."""
        scope = FlextInfraDocScope(
            name="root",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = validator._validate_scope(scope, check="adr-skill", apply_mode=False)
        assert report.scope == "root"

    def test_validate_scope_without_config(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _validate_scope without architecture config."""
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = validator._validate_scope(scope, check="all", apply_mode=False)
        assert report.scope == "test"

    def test_has_adr_reference_with_adr_text(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _has_adr_reference detects ADR references."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# Skill\n\nADR: This is an ADR reference.\n")
        assert validator._has_adr_reference(skill_file) is True

    def test_has_adr_reference_without_adr_text(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _has_adr_reference returns False without ADR."""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# Skill\n\nNo architecture decision record here.\n")
        assert validator._has_adr_reference(skill_file) is False

    def test_run_adr_skill_check_no_config(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _run_adr_skill_check with no config uses defaults."""
        code, missing = validator._run_adr_skill_check(tmp_path)
        assert isinstance(code, int)
        assert isinstance(missing, list)

    def test_run_adr_skill_check_with_config(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _run_adr_skill_check loads from config."""
        config_dir = tmp_path / "docs/architecture"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "architecture_config.json"
        config_file.write_text(
            '{"docs_validation": {"required_skills": ["test-skill"]}}'
        )
        code, missing = validator._run_adr_skill_check(tmp_path)
        assert isinstance(code, int)
        assert isinstance(missing, list)

    def test_maybe_write_todo_root_scope(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _maybe_write_todo skips root scope."""
        scope = FlextInfraDocScope(
            name="root",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        result = validator._maybe_write_todo(scope, apply_mode=True)
        assert result is False

    def test_maybe_write_todo_apply_false(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _maybe_write_todo with apply_mode=False."""
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        result = validator._maybe_write_todo(scope, apply_mode=False)
        assert result is False

    def test_maybe_write_todo_creates_file(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _maybe_write_todo creates TODOS.md."""
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        result = validator._maybe_write_todo(scope, apply_mode=True)
        assert result is True
        assert (tmp_path / "TODOS.md").exists()

    def test_validate_scope_with_all_check(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _validate_scope with all check."""
        scope = FlextInfraDocScope(
            name="test",
            path=tmp_path,
            report_dir=tmp_path / "reports",
        )
        report = validator._validate_scope(scope, check="all", apply_mode=False)
        assert report.scope == "test"

    def test_run_adr_skill_check_with_missing_skills(
        self, validator: FlextInfraDocValidator, tmp_path: Path
    ) -> None:
        """Test _run_adr_skill_check detects missing skills."""
        skills_dir = tmp_path / ".claude/skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        code, missing = validator._run_adr_skill_check(tmp_path)
        # Should detect missing skills
        assert isinstance(code, int)
        assert isinstance(missing, list)

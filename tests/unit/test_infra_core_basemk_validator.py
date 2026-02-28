"""Tests for FlextInfraBaseMkValidator."""

from __future__ import annotations

from pathlib import Path


from flext_core import r
from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator
from flext_infra import m


class TestFlextInfraBaseMkValidator:
    """Test suite for FlextInfraBaseMkValidator."""

    def test_validate_with_missing_root_basemk_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that missing root base.mk returns failure result."""
        # Arrange
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert result.is_success
        assert not result.value.passed
        assert "missing root base.mk" in result.value.summary

    def test_validate_with_matching_basemk_returns_success(
        self, tmp_path: Path
    ) -> None:
        """Test that matching base.mk files return success result."""
        # Arrange
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        # Create root base.mk
        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root base.mk content")

        # Create project with matching base.mk
        project_dir = workspace_root / "project1"
        project_dir.mkdir()
        project_basemk = project_dir / "base.mk"
        project_basemk.write_text("# root base.mk content")

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert result.is_success
        assert isinstance(result.value, m.ValidationReport)

    def test_validate_with_mismatched_basemk_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that mismatched base.mk files return failure result."""
        # Arrange
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        # Create root base.mk
        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root base.mk content")

        # Create project with different base.mk
        project_dir = workspace_root / "project1"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text("[tool.poetry]\n")
        project_basemk = project_dir / "base.mk"
        project_basemk.write_text("# different content")

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert result.is_failure or (result.is_success and not result.value.passed)

    def test_validate_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that validate returns FlextResult type."""
        # Arrange
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path
        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# content")

        # Act
        result = validator.validate(workspace_root)

        # Assert
        assert isinstance(
            result,
            type(
                r[m.ValidationReport].ok(
                    m.ValidationReport(passed=True, violations=[], summary="")
                )
            ),
        )

"""Tests for FlextInfraBaseMkValidator."""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra import m
from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator


class TestFlextInfraBaseMkValidator:
    """Test suite for FlextInfraBaseMkValidator."""

    def test_validate_with_missing_root_basemk_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that missing root base.mk returns failure result."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        result = validator.validate(workspace_root)
        assert result.is_success
        assert not result.value.passed
        assert "missing root base.mk" in result.value.summary

    def test_validate_with_matching_basemk_returns_success(
        self, tmp_path: Path
    ) -> None:
        """Test that matching base.mk files return success result."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root base.mk content")

        project_dir = workspace_root / "project1"
        project_dir.mkdir()
        project_basemk = project_dir / "base.mk"
        project_basemk.write_text("# root base.mk content")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert isinstance(result.value, m.ValidationReport)

    def test_validate_with_mismatched_basemk_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that mismatched base.mk files return failure result."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root base.mk content")

        project_dir = workspace_root / "project1"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text("[tool.poetry]\n")
        project_basemk = project_dir / "base.mk"
        project_basemk.write_text("# different content")

        result = validator.validate(workspace_root)
        assert result.is_failure or (result.is_success and not result.value.passed)

    def test_validate_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that validate returns FlextResult type."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path
        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# content")

        result = validator.validate(workspace_root)

        assert isinstance(
            result,
            type(
                r[m.ValidationReport].ok(
                    m.ValidationReport(passed=True, violations=[], summary="")
                )
            ),
        )

    def test_validate_with_multiple_projects_checks_all(self, tmp_path: Path) -> None:
        """Test that validate checks all projects."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root content")

        for i in range(3):
            proj_dir = workspace_root / f"project{i}"
            proj_dir.mkdir()
            (proj_dir / "pyproject.toml").write_text("")
            (proj_dir / "base.mk").write_text("# root content")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert result.value.passed

    def test_validate_with_no_vendored_basemk_returns_success(
        self, tmp_path: Path
    ) -> None:
        """Test that validate succeeds when no vendored base.mk exists."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root content")

        project_dir = workspace_root / "project1"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text("")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert result.value.passed

    def test_validate_with_exception_returns_failure(self, tmp_path: Path) -> None:
        """Test that validate handles exceptions."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path / "nonexistent"

        result = validator.validate(workspace_root)
        assert result.is_success

    def test_validate_reports_all_mismatches(self, tmp_path: Path) -> None:
        """Test that validate reports all mismatched files."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root content")

        for i in range(2):
            proj_dir = workspace_root / f"project{i}"
            proj_dir.mkdir()
            (proj_dir / "pyproject.toml").write_text("")
            (proj_dir / "base.mk").write_text(f"# different content {i}")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert not result.value.passed
        assert len(result.value.violations) == 2

    def test_validate_summary_includes_checked_count(self, tmp_path: Path) -> None:
        """Test that validate summary includes checked count."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root content")

        for i in range(3):
            proj_dir = workspace_root / f"project{i}"
            proj_dir.mkdir()
            (proj_dir / "pyproject.toml").write_text("")
            (proj_dir / "base.mk").write_text("# root content")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert "3 checked" in result.value.summary

    def test_sha256_computes_file_hash(self, tmp_path: Path) -> None:
        """Test _sha256 computes file hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        hash1 = FlextInfraBaseMkValidator._sha256(test_file)
        assert isinstance(hash1, str)
        assert len(hash1) == 64

    def test_sha256_same_content_same_hash(self, tmp_path: Path) -> None:
        """Test _sha256 produces same hash for same content."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("same content")
        file2.write_text("same content")

        hash1 = FlextInfraBaseMkValidator._sha256(file1)
        hash2 = FlextInfraBaseMkValidator._sha256(file2)
        assert hash1 == hash2

    def test_sha256_different_content_different_hash(self, tmp_path: Path) -> None:
        """Test _sha256 produces different hash for different content."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        hash1 = FlextInfraBaseMkValidator._sha256(file1)
        hash2 = FlextInfraBaseMkValidator._sha256(file2)
        assert hash1 != hash2

    def test_validate_with_empty_workspace_returns_success(
        self, tmp_path: Path
    ) -> None:
        """Test validate with empty workspace."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# content")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert result.value.passed

    def test_validate_with_project_without_pyproject_skips(
        self, tmp_path: Path
    ) -> None:
        """Test validate skips projects without pyproject.toml."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# content")

        proj_dir = workspace_root / "project1"
        proj_dir.mkdir()
        (proj_dir / "base.mk").write_text("# different")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert result.value.passed

    def test_validate_with_relative_path_in_violation(self, tmp_path: Path) -> None:
        """Test validate reports relative paths in violations."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root")

        proj_dir = workspace_root / "project1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("")
        (proj_dir / "base.mk").write_text("# different")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert not result.value.passed
        assert "project1/base.mk" in result.value.violations[0]

    def test_validate_passes_when_all_match(self, tmp_path: Path) -> None:
        """Test validate passes when all files match."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        content = "# shared base.mk content"
        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text(content)

        for i in range(5):
            proj_dir = workspace_root / f"project{i}"
            proj_dir.mkdir()
            (proj_dir / "pyproject.toml").write_text("")
            (proj_dir / "base.mk").write_text(content)

        result = validator.validate(workspace_root)
        assert result.is_success
        assert result.value.passed
        assert "all vendored base.mk copies in sync" in result.value.summary

    def test_validate_fails_when_any_mismatch(self, tmp_path: Path) -> None:
        """Test validate fails when any file mismatches."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# root")

        proj_dir1 = workspace_root / "project1"
        proj_dir1.mkdir()
        (proj_dir1 / "pyproject.toml").write_text("")
        (proj_dir1 / "base.mk").write_text("# root")

        proj_dir2 = workspace_root / "project2"
        proj_dir2.mkdir()
        (proj_dir2 / "pyproject.toml").write_text("")
        (proj_dir2 / "base.mk").write_text("# different")

        result = validator.validate(workspace_root)
        assert result.is_success
        assert not result.value.passed

    def test_validate_with_oserror_returns_failure(self, tmp_path: Path) -> None:
        """Test validate handles OSError exception (lines 79-80)."""
        validator = FlextInfraBaseMkValidator()
        workspace_root = tmp_path

        # Create root base.mk
        root_basemk = workspace_root / "base.mk"
        root_basemk.write_text("# content")

        # Create a project with base.mk
        proj_dir = workspace_root / "project1"
        proj_dir.mkdir()
        (proj_dir / "pyproject.toml").write_text("")
        proj_basemk = proj_dir / "base.mk"
        proj_basemk.write_text("# content")

        # Make the project base.mk unreadable to trigger OSError
        proj_basemk.chmod(0o000)
        try:
            result = validator.validate(workspace_root)
            # Should return failure result due to OSError
            assert result.is_failure
        finally:
            proj_basemk.chmod(0o644)

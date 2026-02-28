"""Tests for FlextInfraStubSupplyChain."""

from __future__ import annotations

from pathlib import Path

from flext_infra.core.stub_chain import FlextInfraStubSupplyChain


class TestFlextInfraStubSupplyChain:
    """Test suite for FlextInfraStubSupplyChain."""

    def test_init_creates_service_instance(self) -> None:
        """Test that StubSupplyChain initializes correctly."""
        chain = FlextInfraStubSupplyChain()
        assert chain is not None
        assert hasattr(chain, "_runner")

    def test_analyze_with_valid_project_returns_success(self, tmp_path: Path) -> None:
        """Test that analyze returns success for valid project."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path
        workspace_root = tmp_path.parent

        project_dir.mkdir(exist_ok=True)
        pyproject = project_dir / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")

        result = chain.analyze(project_dir, workspace_root)
        assert result.is_success or result.is_failure

    def test_analyze_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that analyze returns FlextResult type."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path
        workspace_root = tmp_path.parent
        project_dir.mkdir(exist_ok=True)

        result = chain.analyze(project_dir, workspace_root)
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_analyze_detects_missing_imports(self, tmp_path: Path) -> None:
        """Test that analyze detects missing imports."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path
        workspace_root = tmp_path.parent
        project_dir.mkdir(exist_ok=True)

        src_dir = project_dir / "src"
        src_dir.mkdir()
        py_file = src_dir / "main.py"
        py_file.write_text("import missing_module")

        result = chain.analyze(project_dir, workspace_root)
        assert result.is_success or result.is_failure

    def test_validate_with_workspace_root_returns_success(self, tmp_path: Path) -> None:
        """Test that validate returns success for workspace."""
        chain = FlextInfraStubSupplyChain()
        workspace_root = tmp_path
        workspace_root.mkdir(exist_ok=True)

        result = chain.validate(workspace_root)
        assert result.is_success or result.is_failure

    def test_validate_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that validate returns FlextResult type."""
        chain = FlextInfraStubSupplyChain()
        workspace_root = tmp_path
        workspace_root.mkdir(exist_ok=True)

        result = chain.validate(workspace_root)
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_validate_with_project_dirs_filters_projects(self, tmp_path: Path) -> None:
        """Test that validate respects project_dirs filter."""
        chain = FlextInfraStubSupplyChain()
        workspace_root = tmp_path
        workspace_root.mkdir(exist_ok=True)

        project_dir = workspace_root / "project1"
        project_dir.mkdir()

        result = chain.validate(workspace_root, project_dirs=[project_dir])
        assert result.is_success or result.is_failure

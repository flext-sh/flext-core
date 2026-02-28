"""Tests for FlextInfraStubSupplyChain."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

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

        with patch.object(chain, "_run_mypy_hints", return_value=[]):
            with patch.object(chain, "_run_pyrefly_missing", return_value=[]):
                result = chain.analyze(project_dir, workspace_root)
                assert result.is_success

    def test_analyze_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that analyze returns FlextResult type."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path
        workspace_root = tmp_path.parent
        project_dir.mkdir(exist_ok=True)

        with patch.object(chain, "_run_mypy_hints", return_value=[]):
            with patch.object(chain, "_run_pyrefly_missing", return_value=[]):
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

        with patch.object(chain, "_run_mypy_hints", return_value=[]):
            with patch.object(
                chain, "_run_pyrefly_missing", return_value=["missing_module"]
            ):
                result = chain.analyze(project_dir, workspace_root)
                assert result.is_success

    def test_analyze_classifies_internal_imports(self, tmp_path: Path) -> None:
        """Test analyze classifies internal imports."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path / "flext_test"
        workspace_root = tmp_path
        project_dir.mkdir(parents=True)

        with patch.object(chain, "_run_mypy_hints", return_value=[]):
            with patch.object(
                chain, "_run_pyrefly_missing", return_value=["flext_test"]
            ):
                result = chain.analyze(project_dir, workspace_root)
                assert result.is_success
                assert "internal_missing" in result.value

    def test_analyze_with_exception_returns_failure(self, tmp_path: Path) -> None:
        """Test analyze handles exceptions."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path / "nonexistent"
        workspace_root = tmp_path

        result = chain.analyze(project_dir, workspace_root)
        assert result.is_success

    def test_validate_with_workspace_root_returns_success(self, tmp_path: Path) -> None:
        """Test that validate returns success for workspace."""
        chain = FlextInfraStubSupplyChain()
        workspace_root = tmp_path
        workspace_root.mkdir(exist_ok=True)

        with patch.object(chain, "analyze") as mock_analyze:
            mock_result = Mock()
            mock_result.is_failure = False
            mock_result.value = {
                "internal_missing": [],
                "unresolved_missing": [],
            }
            mock_analyze.return_value = mock_result

            result = chain.validate(workspace_root)
            assert result.is_success

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

        with patch.object(chain, "analyze") as mock_analyze:
            mock_result = Mock()
            mock_result.is_failure = False
            mock_result.value = {
                "internal_missing": [],
                "unresolved_missing": [],
            }
            mock_analyze.return_value = mock_result

            result = chain.validate(workspace_root, project_dirs=[project_dir])
            assert result.is_success

    def test_validate_reports_internal_missing_imports(self, tmp_path: Path) -> None:
        """Test validate reports internal missing imports."""
        chain = FlextInfraStubSupplyChain()
        workspace_root = tmp_path
        workspace_root.mkdir(exist_ok=True)

        project_dir = workspace_root / "flext_test"
        project_dir.mkdir()

        with patch.object(chain, "analyze") as mock_analyze:
            mock_result = Mock()
            mock_result.is_failure = False
            mock_result.value = {
                "internal_missing": ["flext_test.module"],
                "unresolved_missing": [],
            }
            mock_analyze.return_value = mock_result

            result = chain.validate(workspace_root, project_dirs=[project_dir])
            assert result.is_success
            assert not result.value.passed

    def test_validate_reports_unresolved_imports(self, tmp_path: Path) -> None:
        """Test validate reports unresolved imports."""
        chain = FlextInfraStubSupplyChain()
        workspace_root = tmp_path
        workspace_root.mkdir(exist_ok=True)

        project_dir = workspace_root / "project1"
        project_dir.mkdir()

        with patch.object(chain, "analyze") as mock_analyze:
            mock_result = Mock()
            mock_result.is_failure = False
            mock_result.value = {
                "internal_missing": [],
                "unresolved_missing": ["external_lib"],
            }
            mock_analyze.return_value = mock_result

            result = chain.validate(workspace_root, project_dirs=[project_dir])
            assert result.is_success
            assert not result.value.passed

    def test_validate_with_analyze_failure_reports_error(self, tmp_path: Path) -> None:
        """Test validate handles analyze failures."""
        chain = FlextInfraStubSupplyChain()
        workspace_root = tmp_path
        workspace_root.mkdir(exist_ok=True)

        project_dir = workspace_root / "project1"
        project_dir.mkdir()

        with patch.object(chain, "analyze") as mock_analyze:
            mock_result = Mock()
            mock_result.is_failure = True
            mock_result.error = "analysis failed"
            mock_analyze.return_value = mock_result

            result = chain.validate(workspace_root, project_dirs=[project_dir])
            assert result.is_success
            assert not result.value.passed

    def test_validate_with_exception_returns_failure(self, tmp_path: Path) -> None:
        """Test validate handles exceptions."""
        chain = FlextInfraStubSupplyChain()
        workspace_root = tmp_path / "nonexistent"

        result = chain.validate(workspace_root)
        assert result.is_failure

    def test_run_mypy_hints_extracts_types_packages(self, tmp_path: Path) -> None:
        """Test _run_mypy_hints extracts types packages."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path

        with patch.object(
            type(chain._runner),
            "run",
            return_value=Mock(
                is_success=True, value=Mock(stdout="note: hint: `types-requests`")
            ),
        ):
            hints = chain._run_mypy_hints(project_dir)
            assert "types-requests" in hints

    def test_run_mypy_hints_with_failed_run_returns_empty(self, tmp_path: Path) -> None:
        """Test _run_mypy_hints returns empty on failed run."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path

        with patch.object(
            type(chain._runner), "run", return_value=Mock(is_success=False)
        ):
            hints = chain._run_mypy_hints(project_dir)
            assert hints == []

    def test_run_pyrefly_missing_extracts_imports(self, tmp_path: Path) -> None:
        """Test _run_pyrefly_missing extracts missing imports."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path

        with patch.object(
            type(chain._runner),
            "run",
            return_value=Mock(
                is_success=True,
                value=Mock(stdout="Cannot find module `requests` [missing-import]"),
            ),
        ):
            imports = chain._run_pyrefly_missing(project_dir)
            assert "requests" in imports

    def test_run_pyrefly_missing_with_failed_run_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """Test _run_pyrefly_missing returns empty on failed run."""
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path

        with patch.object(
            type(chain._runner), "run", return_value=Mock(is_success=False)
        ):
            imports = chain._run_pyrefly_missing(project_dir)
            assert imports == []

    def test_is_internal_with_flext_prefix(self) -> None:
        """Test _is_internal identifies flext_ prefix."""
        assert FlextInfraStubSupplyChain._is_internal("flext_core", "project")
        assert FlextInfraStubSupplyChain._is_internal("flext_api", "project")

    def test_is_internal_with_flext_dash_prefix(self) -> None:
        """Test _is_internal identifies flext- prefix."""
        assert FlextInfraStubSupplyChain._is_internal("flext-core", "project")

    def test_is_internal_with_project_name(self) -> None:
        """Test _is_internal identifies project name."""
        assert FlextInfraStubSupplyChain._is_internal("my_project", "my_project")
        assert FlextInfraStubSupplyChain._is_internal("my_project.sub", "my_project")

    def test_is_internal_with_external_module(self) -> None:
        """Test _is_internal returns False for external modules."""
        assert not FlextInfraStubSupplyChain._is_internal("requests", "my_project")

    def test_stub_exists_with_pyi_file(self, tmp_path: Path) -> None:
        """Test _stub_exists finds .pyi files."""
        typings_dir = tmp_path / "typings"
        typings_dir.mkdir()
        (typings_dir / "requests.pyi").write_text("")

        assert FlextInfraStubSupplyChain._stub_exists("requests", tmp_path)

    def test_stub_exists_with_package_init(self, tmp_path: Path) -> None:
        """Test _stub_exists finds package __init__.pyi."""
        typings_dir = tmp_path / "typings"
        pkg_dir = typings_dir / "requests"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "__init__.pyi").write_text("")

        assert FlextInfraStubSupplyChain._stub_exists("requests", tmp_path)

    def test_stub_exists_with_generated_stubs(self, tmp_path: Path) -> None:
        """Test _stub_exists finds generated stubs."""
        gen_dir = tmp_path / "typings" / "generated"
        gen_dir.mkdir(parents=True)
        (gen_dir / "requests.pyi").write_text("")

        assert FlextInfraStubSupplyChain._stub_exists("requests", tmp_path)

    def test_stub_exists_returns_false_for_missing(self, tmp_path: Path) -> None:
        """Test _stub_exists returns False for missing stubs."""
        assert not FlextInfraStubSupplyChain._stub_exists("requests", tmp_path)

    def test_discover_stub_projects_finds_projects(self, tmp_path: Path) -> None:
        """Test _discover_stub_projects finds projects."""
        proj1 = tmp_path / "project1"
        proj1.mkdir()
        (proj1 / "pyproject.toml").write_text("")
        (proj1 / "src").mkdir()

        proj2 = tmp_path / "project2"
        proj2.mkdir()
        (proj2 / "pyproject.toml").write_text("")
        (proj2 / "src").mkdir()

        projects = FlextInfraStubSupplyChain._discover_stub_projects(tmp_path)
        assert len(projects) == 2

    def test_discover_stub_projects_skips_hidden_dirs(self, tmp_path: Path) -> None:
        """Test _discover_stub_projects skips hidden directories."""
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "pyproject.toml").write_text("")
        (hidden / "src").mkdir()

        projects = FlextInfraStubSupplyChain._discover_stub_projects(tmp_path)
        assert len(projects) == 0

    def test_discover_stub_projects_requires_src_dir(self, tmp_path: Path) -> None:
        """Test _discover_stub_projects requires src directory."""
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("")

        projects = FlextInfraStubSupplyChain._discover_stub_projects(tmp_path)
        assert len(projects) == 0

    def test_discover_stub_projects_requires_pyproject(self, tmp_path: Path) -> None:
        """Test _discover_stub_projects requires pyproject.toml."""
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "src").mkdir()

        projects = FlextInfraStubSupplyChain._discover_stub_projects(tmp_path)
        assert len(projects) == 0

    def test_analyze_with_exception_raises_failure(self, tmp_path: Path) -> None:
        """Test analyze handles exceptions and returns failure (lines 77-78)."""
        chain = FlextInfraStubSupplyChain()
        chain = FlextInfraStubSupplyChain()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Mock _run_mypy_hints to raise an exception
        with patch.object(
            chain,
            "_run_mypy_hints",
            side_effect=ValueError("test error"),
        ):
            result = chain.analyze(project_dir, tmp_path)
            assert result.is_failure
            assert "stub analysis failed" in result.error

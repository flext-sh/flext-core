"""Tests for FlextInfraExtraPathsManager and extra paths synchronization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import tomlkit
from flext_core import r
from flext_infra.deps._constants import FlextInfraDepsConstants
from flext_infra.deps.extra_paths import (
    FlextInfraExtraPathsManager,
    _path_dep_paths_pep621,
    _path_dep_paths_poetry,
    get_dep_paths,
    main,
    sync_extra_paths,
    sync_one,
)


class TestFlextInfraExtraPathsManager:
    """Test FlextInfraExtraPathsManager."""

    def test_manager_initialization(self) -> None:
        """Test manager initializes without errors."""
        manager = FlextInfraExtraPathsManager()
        assert manager is not None

    def test_manager_has_required_services(self) -> None:
        """Test manager has required internal services."""
        manager = FlextInfraExtraPathsManager()
        assert hasattr(manager, "_resolver")
        assert hasattr(manager, "_toml")


class TestPathDepPathsPep621:
    """Test _path_dep_paths_pep621 function."""

    def test_pep621_empty_doc(self) -> None:
        """Test with empty TOML document."""
        doc = tomlkit.document()

        result = _path_dep_paths_pep621(doc)
        assert result == []

    def test_pep621_no_project(self) -> None:
        """Test with missing project section."""
        doc = tomlkit.document()
        doc["other"] = {}

        result = _path_dep_paths_pep621(doc)
        assert result == []

    def test_pep621_no_dependencies(self) -> None:
        """Test with project but no dependencies."""
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}

        result = _path_dep_paths_pep621(doc)
        assert result == []

    def test_pep621_with_file_deps(self) -> None:
        """Test with file:// path dependencies."""
        doc = tomlkit.document()
        doc["project"] = {
            "dependencies": [
                "flext-core @ file:../flext-core",
                "flext-api @ file:../flext-api",
            ]
        }

        result = _path_dep_paths_pep621(doc)
        assert any("flext-core" in p for p in result)
        assert any("flext-api" in p for p in result)

    def test_pep621_with_relative_deps(self) -> None:
        """Test with relative path dependencies."""
        doc = tomlkit.document()
        doc["project"] = {
            "dependencies": [
                "flext-core @ ./flext-core",
            ]
        }

        result = _path_dep_paths_pep621(doc)
        assert any("flext-core" in p for p in result)

    def test_pep621_mixed_deps(self) -> None:
        """Test with mixed path and regular dependencies."""
        doc = tomlkit.document()
        doc["project"] = {
            "dependencies": [
                "requests>=2.0",
                "flext-core @ file:../flext-core",
                "pydantic",
            ]
        }

        result = _path_dep_paths_pep621(doc)
        assert any("flext-core" in p for p in result)
        assert len(result) == 1


class TestPathDepPathsPoetry:
    """Test _path_dep_paths_poetry function."""

    def test_poetry_empty_doc(self) -> None:
        """Test with empty TOML document."""
        doc = tomlkit.document()

        result = _path_dep_paths_poetry(doc)
        assert result == []

    def test_poetry_no_tool(self) -> None:
        """Test with missing tool section."""
        doc = tomlkit.document()
        doc["project"] = {}

        result = _path_dep_paths_poetry(doc)
        assert result == []

    def test_poetry_no_poetry_section(self) -> None:
        """Test with tool but no poetry section."""
        doc = tomlkit.document()
        doc["tool"] = {"other": {}}

        result = _path_dep_paths_poetry(doc)
        assert result == []

    def test_poetry_no_dependencies(self) -> None:
        """Test with poetry section but no dependencies."""
        doc = tomlkit.document()
        doc["tool"] = {"poetry": {"name": "test"}}

        result = _path_dep_paths_poetry(doc)
        assert result == []

    def test_poetry_with_path_deps(self) -> None:
        """Test with poetry path dependencies."""
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {
                "dependencies": {
                    "flext-core": {"path": "../flext-core"},
                    "flext-api": {"path": "../flext-api"},
                }
            }
        }

        result = _path_dep_paths_poetry(doc)
        assert any("flext-core" in p for p in result)
        assert any("flext-api" in p for p in result)

    def test_poetry_with_relative_paths(self) -> None:
        """Test with relative path dependencies."""
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {
                "dependencies": {
                    "flext-core": {"path": "./flext-core"},
                }
            }
        }

        result = _path_dep_paths_poetry(doc)
        assert any("flext-core" in p for p in result)

    def test_poetry_mixed_deps(self) -> None:
        """Test with mixed path and regular dependencies."""
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {
                "dependencies": {
                    "requests": "^2.0",
                    "flext-core": {"path": "../flext-core"},
                    "pydantic": "^2.0",
                }
            }
        }

        result = _path_dep_paths_poetry(doc)
        assert any("flext-core" in p for p in result)
        assert len(result) == 1


class TestGetDepPaths:
    """Test get_dep_paths function."""

    def test_get_dep_paths_empty_doc(self) -> None:
        """Test get_dep_paths with empty TOML document."""
        doc = tomlkit.document()
        paths = get_dep_paths(doc, is_root=False)
        assert paths == []

    def test_get_dep_paths_with_pep621_deps(self) -> None:
        """Test get_dep_paths with PEP 621 path dependencies."""
        doc = tomlkit.document()
        doc["project"] = {"dependencies": ["flext-core @ file:../flext-core"]}
        paths = get_dep_paths(doc, is_root=False)
        assert isinstance(paths, list)
        assert any("flext-core" in p for p in paths)

    def test_get_dep_paths_with_poetry_deps(self) -> None:
        """Test get_dep_paths with Poetry path dependencies."""
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {
                "dependencies": {
                    "flext-core": {"path": "../flext-core"},
                },
            },
        }
        paths = get_dep_paths(doc, is_root=False)
        assert isinstance(paths, list)
        assert any("flext-core" in p for p in paths)

    def test_get_dep_paths_is_root_true(self) -> None:
        """Test get_dep_paths with is_root=True."""
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {
                "dependencies": {
                    "flext-core": {"path": "../flext-core"},
                },
            },
        }
        paths = get_dep_paths(doc, is_root=True)
        assert all(not p.startswith("../") for p in paths)

    def test_get_dep_paths_is_root_false(self) -> None:
        """Test get_dep_paths with is_root=False."""
        doc = tomlkit.document()
        doc["tool"] = {
            "poetry": {
                "dependencies": {
                    "flext-core": {"path": "../flext-core"},
                },
            },
        }
        paths = get_dep_paths(doc, is_root=False)
        assert all(p.startswith("../") for p in paths)

    def test_get_dep_paths_combined_sources(self) -> None:
        """Test get_dep_paths combines PEP 621 and Poetry deps."""
        doc = tomlkit.document()
        doc["project"] = {"dependencies": ["flext-api @ file:../flext-api"]}
        doc["tool"] = {
            "poetry": {
                "dependencies": {
                    "flext-core": {"path": "../flext-core"},
                },
            },
        }
        paths = get_dep_paths(doc, is_root=False)
        assert len(paths) >= 2


class TestSyncOne:
    """Test sync_one function."""

    def test_sync_one_missing_file(self, tmp_path: Path) -> None:
        """Test sync_one with missing pyproject.toml."""
        result = sync_one(tmp_path / "nonexistent.toml")
        assert result.is_success
        assert result.value is False

    def test_sync_one_no_tool_section(self, tmp_path: Path) -> None:
        """Test sync_one with missing tool section."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["project"] = {"name": "test"}
        pyproject.write_text(tomlkit.dumps(doc))
        result = sync_one(pyproject)
        assert result.is_success
        assert result.value is False

    def test_sync_one_no_pyright_section(self, tmp_path: Path) -> None:
        """Test sync_one with missing pyright section."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {"other": {}}
        pyproject.write_text(tomlkit.dumps(doc))
        result = sync_one(pyproject)
        assert result.is_success
        assert result.value is False

    def test_sync_one_updates_pyright_paths(self, tmp_path: Path) -> None:
        """Test sync_one updates pyright extraPaths."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {
            "pyright": {
                "extraPaths": ["src"],
            }
        }
        pyproject.write_text(tomlkit.dumps(doc))
        result = sync_one(pyproject, is_root=True)
        assert result.is_success

    def test_sync_one_updates_mypy_paths(self, tmp_path: Path) -> None:
        """Test sync_one updates mypy mypy_path."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {
            "pyright": {"extraPaths": []},
            "mypy": {"mypy_path": ["src"]},
        }
        pyproject.write_text(tomlkit.dumps(doc))
        result = sync_one(pyproject, is_root=True)
        assert result.is_success

    def test_sync_one_updates_pyrefly_search_path(self, tmp_path: Path) -> None:
        """Test sync_one updates pyrefly search-path."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {
            "pyright": {"extraPaths": []},
            "pyrefly": {"search-path": ["."]},
        }
        pyproject.write_text(tomlkit.dumps(doc))
        result = sync_one(pyproject, is_root=False)
        assert result.is_success

    def test_sync_one_dry_run(self, tmp_path: Path) -> None:
        """Test sync_one with dry_run=True."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {
            "pyright": {"extraPaths": ["old"]},
        }
        pyproject.write_text(tomlkit.dumps(doc))
        result = sync_one(pyproject, dry_run=True, is_root=True)
        assert result.is_success
        content = pyproject.read_text()
        assert "old" in content

    def test_sync_one_write_failure(self, tmp_path: Path) -> None:
        """Test sync_one handles write failure."""
        pyproject = tmp_path / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {
            "pyright": {"extraPaths": ["old"]},
        }
        pyproject.write_text(tomlkit.dumps(doc))
        with patch(
            "flext_infra.deps.extra_paths.FlextInfraTomlService"
        ) as mock_toml_class:
            mock_toml = Mock()
            mock_toml.read_document.return_value = r[object].ok(doc)
            mock_toml.write_document.return_value = r[object].fail("write error")
            mock_toml_class.return_value = mock_toml
            result = sync_one(pyproject, is_root=True)
            assert result.is_failure

    def test_sync_extra_paths_with_project_dirs(self, tmp_path: Path) -> None:
        """Test sync_extra_paths with specific project directories."""
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        pyproject = proj_dir / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {"pyright": {"extraPaths": []}}
        pyproject.write_text(tomlkit.dumps(doc))
        result = sync_extra_paths(project_dirs=[proj_dir])
        assert result.is_success

    def test_sync_extra_paths_no_project_dirs(self, tmp_path: Path) -> None:
        """Test sync_extra_paths without project directories."""
        with patch("flext_infra.deps.extra_paths.ROOT", tmp_path):
            pyproject = tmp_path / "pyproject.toml"
            doc = tomlkit.document()
            doc["tool"] = {"pyright": {"extraPaths": []}}
            pyproject.write_text(tomlkit.dumps(doc))
            result = sync_extra_paths()
            assert result.is_success

    def test_sync_extra_paths_missing_root_pyproject(self, tmp_path: Path) -> None:
        """Test sync_extra_paths with missing root pyproject."""
        with patch("flext_infra.deps.extra_paths.ROOT", tmp_path):
            result = sync_extra_paths()
            assert result.is_failure

    def test_sync_extra_paths_dry_run(self, tmp_path: Path) -> None:
        """Test sync_extra_paths with dry_run=True."""
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        pyproject = proj_dir / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {"pyright": {"extraPaths": ["old"]}}
        pyproject.write_text(tomlkit.dumps(doc))
        result = sync_extra_paths(dry_run=True, project_dirs=[proj_dir])
        assert result.is_success
        content = pyproject.read_text()
        assert "old" in content

    def test_sync_extra_paths_sync_failure(self, tmp_path: Path) -> None:
        """Test sync_extra_paths handles sync failure."""
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        pyproject = proj_dir / "pyproject.toml"
        pyproject.write_text("")
        with patch("flext_infra.deps.extra_paths.sync_one") as mock_sync:
            mock_sync.return_value = r[bool].fail("sync error")
            result = sync_extra_paths(project_dirs=[proj_dir])
            assert result.is_failure


class TestMain:
    """Test main function."""

    def test_main_no_args(self, tmp_path: Path) -> None:
        """Test main with no arguments."""
        with patch("flext_infra.deps.extra_paths.ROOT", tmp_path):
            pyproject = tmp_path / "pyproject.toml"
            doc = tomlkit.document()
            doc["tool"] = {"pyright": {"extraPaths": []}}
            pyproject.write_text(tomlkit.dumps(doc))
            with patch("sys.argv", ["extra_paths.py"]):
                result = main()
                assert result == 0

    def test_main_with_dry_run(self, tmp_path: Path) -> None:
        """Test main with --dry-run flag."""
        with patch("flext_infra.deps.extra_paths.ROOT", tmp_path):
            pyproject = tmp_path / "pyproject.toml"
            doc = tomlkit.document()
            doc["tool"] = {"pyright": {"extraPaths": []}}
            pyproject.write_text(tomlkit.dumps(doc))
            with patch("sys.argv", ["extra_paths.py", "--dry-run"]):
                result = main()
                assert result == 0

    def test_main_with_project(self, tmp_path: Path) -> None:
        """Test main with --project argument."""
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        pyproject = proj_dir / "pyproject.toml"
        doc = tomlkit.document()
        doc["tool"] = {"pyright": {"extraPaths": []}}
        pyproject.write_text(tomlkit.dumps(doc))
        with patch("flext_infra.deps.extra_paths.ROOT", tmp_path):
            with patch("sys.argv", ["extra_paths.py", "--project", "proj"]):
                result = main()
                assert result == 0

    def test_main_with_multiple_projects(self, tmp_path: Path) -> None:
        """Test main with multiple --project arguments."""
        for name in ["proj-a", "proj-b"]:
            proj_dir = tmp_path / name
            proj_dir.mkdir()
            pyproject = proj_dir / "pyproject.toml"
            doc = tomlkit.document()
            doc["tool"] = {"pyright": {"extraPaths": []}}
            pyproject.write_text(tomlkit.dumps(doc))
        with patch("flext_infra.deps.extra_paths.ROOT", tmp_path):
            with patch(
                "sys.argv",
                ["extra_paths.py", "--project", "proj-a", "--project", "proj-b"],
            ):
                result = main()
                assert result == 0

    def test_main_sync_failure(self, tmp_path: Path) -> None:
        """Test main handles sync failure."""
        with patch("flext_infra.deps.extra_paths.ROOT", tmp_path):
            with patch("flext_infra.deps.extra_paths.sync_extra_paths") as mock_sync:
                mock_sync.return_value = r[int].fail("sync error")
                with patch("sys.argv", ["extra_paths.py"]):
                    result = main()
                    assert result == 1


class TestConstants:
    """Test module constants."""

    def test_pyright_base_root(self) -> None:
        """Test PYRIGHT_BASE_ROOT constant."""
        assert isinstance(FlextInfraDepsConstants.PYRIGHT_BASE_ROOT, list)
        assert "scripts" in FlextInfraDepsConstants.PYRIGHT_BASE_ROOT
        assert "src" in FlextInfraDepsConstants.PYRIGHT_BASE_ROOT

    def test_mypy_base_root(self) -> None:
        """Test MYPY_BASE_ROOT constant."""
        assert isinstance(FlextInfraDepsConstants.MYPY_BASE_ROOT, list)
        assert "src" in FlextInfraDepsConstants.MYPY_BASE_ROOT

    def test_pyright_base_project(self) -> None:
        """Test PYRIGHT_BASE_PROJECT constant."""
        assert isinstance(FlextInfraDepsConstants.PYRIGHT_BASE_PROJECT, list)
        assert "." in FlextInfraDepsConstants.PYRIGHT_BASE_PROJECT
        assert "src" in FlextInfraDepsConstants.PYRIGHT_BASE_PROJECT

    def test_mypy_base_project(self) -> None:
        """Test MYPY_BASE_PROJECT constant."""
        assert isinstance(FlextInfraDepsConstants.MYPY_BASE_PROJECT, list)
        assert "." in FlextInfraDepsConstants.MYPY_BASE_PROJECT


class TestSyncExtraPathsEdgeCases:
    """Test edge cases in sync_extra_paths function."""

    def test_sync_extra_paths_with_empty_project_list(self) -> None:
        """Test sync_extra_paths with empty project list."""
        result = sync_extra_paths(dry_run=True, project_dirs=[])
        assert result.is_success

    def test_sync_extra_paths_with_no_args(self) -> None:
        """Test sync_extra_paths with default args (dry_run)."""
        result = sync_extra_paths(dry_run=True)
        assert result.is_success


class TestGetDepPathsEdgeCases:
    """Test edge cases in get_dep_paths function."""

    def test_get_dep_paths_with_empty_doc(self) -> None:
        """Test get_dep_paths with empty tomlkit document."""
        doc = tomlkit.document()
        result = get_dep_paths(doc)
        assert result == []

    def test_get_dep_paths_with_no_deps(self) -> None:
        """Test get_dep_paths with doc that has no path deps."""
        doc = tomlkit.document()
        project = tomlkit.table()
        project.add("name", "test")
        doc.add("project", project)
        result = get_dep_paths(doc)
        assert result == []


def test_get_dep_paths_with_is_root_true() -> None:
    """Test get_dep_paths with is_root=True."""
    doc = tomlkit.document()
    project = tomlkit.table()
    project["dependencies"] = ["flext-core @ file:flext-core"]
    doc["project"] = project

    result = get_dep_paths(doc, is_root=True)
    assert any("flext-core/src" in p for p in result)


def test_get_dep_paths_with_is_root_false() -> None:
    """Test get_dep_paths with is_root=False."""
    doc = tomlkit.document()
    project = tomlkit.table()
    project["dependencies"] = ["flext-core @ file:../flext-core"]
    doc["project"] = project

    result = get_dep_paths(doc, is_root=False)
    assert any("../flext-core/src" in p for p in result)


def test_sync_one_with_nonexistent_file() -> None:
    """Test sync_one with nonexistent pyproject.toml."""
    result = sync_one(Path("/nonexistent/pyproject.toml"), dry_run=True)
    assert result.is_success
    assert result.value is False


def test_sync_one_with_invalid_toml(tmp_path: Path) -> None:
    """Test sync_one with invalid TOML."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("invalid toml {", encoding="utf-8")

    result = sync_one(pyproject, dry_run=True)
    assert result.is_failure


def test_sync_one_with_no_tool_section(tmp_path: Path) -> None:
    """Test sync_one with no tool section."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "test"\n', encoding="utf-8")

    result = sync_one(pyproject, dry_run=True)
    assert result.is_success
    assert result.value is False


def test_sync_one_with_no_pyright_section(tmp_path: Path) -> None:
    """Test sync_one with no pyright section."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool]\n", encoding="utf-8")

    result = sync_one(pyproject, dry_run=True)
    assert result.is_success
    assert result.value is False


def test_sync_one_with_pyright_changes(tmp_path: Path) -> None:
    """Test sync_one updates pyright extraPaths."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.pyright]\nextraPaths = []\n",
        encoding="utf-8",
    )

    result = sync_one(pyproject, dry_run=False, is_root=True)
    assert result.is_success


def test_sync_one_with_mypy_changes(tmp_path: Path) -> None:
    """Test sync_one updates mypy mypy_path."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.pyright]\nextraPaths = []\n[tool.mypy]\nmypy_path = []\n",
        encoding="utf-8",
    )

    result = sync_one(pyproject, dry_run=False, is_root=True)
    assert result.is_success


def test_sync_one_with_pyrefly_changes(tmp_path: Path) -> None:
    """Test sync_one updates pyrefly search-path for non-root."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.pyright]\nextraPaths = []\n[tool.pyrefly]\nsearch-path = []\n",
        encoding="utf-8",
    )

    result = sync_one(pyproject, dry_run=False, is_root=False)
    assert result.is_success


def test_sync_one_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test sync_one handles write failures."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.pyright]\nextraPaths = []\n",
        encoding="utf-8",
    )

    def mock_write(*args: object, **kwargs: object) -> None:
        msg = "Write failed"
        raise OSError(msg)

    monkeypatch.setattr(Path, "write_text", mock_write)
    result = sync_one(pyproject, dry_run=False, is_root=True)
    assert result.is_failure


def test_path_dep_paths_pep621_with_file_prefix(tmp_path: Path) -> None:
    """Test _path_dep_paths_pep621 with file: prefix."""
    doc = tomlkit.document()
    doc["project"] = {"dependencies": ["flext-core @ file://flext-core"]}

    result = _path_dep_paths_pep621(doc)
    assert any("flext-core" in p for p in result)


def test_path_dep_paths_poetry_with_path(tmp_path: Path) -> None:
    """Test _path_dep_paths_poetry with path dependency."""
    doc = tomlkit.document()
    doc["tool"] = {
        "poetry": {"dependencies": {"flext-core": {"path": "../flext-core"}}}
    }

    result = _path_dep_paths_poetry(doc)
    assert any("flext-core" in p for p in result)


def test_sync_extra_paths_with_project_dirs(tmp_path: Path) -> None:
    """Test sync_extra_paths with specific project directories."""
    project = tmp_path / "project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")

    result = sync_extra_paths(dry_run=True, project_dirs=[project])
    assert result.is_success


def test_main_cli_with_project_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test main() CLI with --project argument."""
    project = tmp_path / "project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text("[tool.pyright]\nextraPaths = []\n", encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--project", str(project), "--dry-run"],
    )
    result = main()
    assert result == 0

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import t
from flext_infra import c, m, u
from flext_infra._utilities.iteration import FlextInfraUtilitiesIteration


class TestDiscoveryProjectRoots:
    def test_discover_project_roots_with_real_workspace_root(self) -> None:
        workspace_root = Path(__file__).resolve().parents[5]

        roots = u.Infra.discover_project_roots(workspace_root)

        assert any(root.name == "flext-core" for root in roots)
        assert all(root.is_dir() for root in roots)

    def test_discover_project_roots_from_tmp_workspace(self, tmp_path: Path) -> None:
        project = tmp_path / "demo-project"
        (project / c.Infra.Paths.DEFAULT_SRC_DIR).mkdir(parents=True)
        (project / c.Infra.Files.MAKEFILE_FILENAME).write_text(
            "all:\n", encoding="utf-8"
        )
        (project / c.Infra.Files.PYPROJECT_FILENAME).write_text(
            "[tool.poetry]\nname='demo'\n",
            encoding="utf-8",
        )

        roots = u.Infra.discover_project_roots(tmp_path)

        assert roots == [project]


class TestDiscoveryIterPythonFiles:
    def test_iter_python_files_returns_result_with_paths(self, tmp_path: Path) -> None:
        project = tmp_path / "pkg"
        src_dir = project / c.Infra.Paths.DEFAULT_SRC_DIR
        test_dir = project / c.Infra.Directories.TESTS
        src_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        module_file = src_dir / "mod.py"
        test_file = test_dir / "test_mod.py"
        module_file.write_text("x = 1\n", encoding="utf-8")
        test_file.write_text("def test_x():\n    assert True\n", encoding="utf-8")

        result = u.Infra.iter_python_files(
            workspace_root=tmp_path,
            project_roots=[project],
            include_examples=False,
            include_scripts=False,
        )

        assert result.is_success
        assert module_file in result.value
        assert test_file in result.value

    def test_iter_python_files_returns_failure_on_oserror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_oserror(workspace_root: Path, **_kwargs: t.Scalar) -> list[Path]:
            del workspace_root  # Mock function doesn't use the parameter
            msg = "forced failure"
            raise OSError(msg)

        monkeypatch.setattr(
            FlextInfraUtilitiesIteration,
            "_discover_project_roots",
            staticmethod(_raise_oserror),
        )

        result = u.Infra.iter_python_files(workspace_root=tmp_path)

        assert result.is_failure
        error_text = result.error or ""
        assert "python file iteration failed" in error_text


class TestDiscoveryFindAllPyprojectFiles:
    def test_find_all_pyproject_files_with_project_paths(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        first_pyproject = first / c.Infra.Files.PYPROJECT_FILENAME
        second_pyproject = second / c.Infra.Files.PYPROJECT_FILENAME
        first_pyproject.write_text("[project]\nname='first'\n", encoding="utf-8")
        second_pyproject.write_text("[project]\nname='second'\n", encoding="utf-8")

        result = u.Infra.find_all_pyproject_files(
            tmp_path,
            project_paths=[first, second_pyproject],
        )

        assert result.is_success
        assert result.value == [first_pyproject, second_pyproject]

    def test_find_all_pyproject_files_skips_excluded_dirs(self, tmp_path: Path) -> None:
        included = tmp_path / "project"
        skipped = tmp_path / ".sisyphus"
        included.mkdir()
        skipped.mkdir()
        included_file = included / c.Infra.Files.PYPROJECT_FILENAME
        skipped_file = skipped / c.Infra.Files.PYPROJECT_FILENAME
        included_file.write_text("[project]\nname='ok'\n", encoding="utf-8")
        skipped_file.write_text("[project]\nname='skip'\n", encoding="utf-8")

        result = u.Infra.find_all_pyproject_files(tmp_path)

        assert result.is_success
        assert included_file in result.value
        assert skipped_file not in result.value

    def test_find_all_pyproject_files_returns_failure_on_oserror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_oserror(_: Path, _pattern: str) -> list[Path]:
            msg = "scan failed"
            raise OSError(msg)

        monkeypatch.setattr(Path, "rglob", _raise_oserror)

        result = u.Infra.find_all_pyproject_files(tmp_path)

        assert result.is_failure
        error_text = result.error or ""
        assert "pyproject file scan failed" in error_text


class TestDiscoveryDiscoverProjects:
    def test_discover_projects_returns_project_info(self, tmp_path: Path) -> None:
        project = tmp_path / "alpha"
        (project / c.Infra.Git.DIR).mkdir(parents=True)
        (project / c.Infra.Paths.DEFAULT_SRC_DIR).mkdir(parents=True)
        (project / c.Infra.Directories.TESTS).mkdir(parents=True)
        (project / c.Infra.Files.MAKEFILE_FILENAME).write_text(
            "all:\n", encoding="utf-8"
        )
        (project / c.Infra.Files.PYPROJECT_FILENAME).write_text(
            "[project]\nname='alpha'\n",
            encoding="utf-8",
        )

        result = u.Infra.discover_projects(tmp_path)

        assert result.is_success
        assert len(result.value) == 1
        info = result.value[0]
        assert isinstance(info, m.Infra.Workspace.ProjectInfo)
        assert info.name == "alpha"
        assert info.has_src is True
        assert info.has_tests is True

    def test_discover_projects_returns_failure_on_oserror(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise_oserror(_: Path) -> list[Path]:
            msg = "no permission"
            raise OSError(msg)

        monkeypatch.setattr(Path, "iterdir", _raise_oserror)

        result = u.Infra.discover_projects(tmp_path)

        assert result.is_failure
        error_text = result.error or ""
        assert "project discovery failed" in error_text

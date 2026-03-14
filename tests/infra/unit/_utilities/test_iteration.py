from __future__ import annotations

from pathlib import Path

from flext_infra import u


class TestIterWorkspacePythonModules:
    @staticmethod
    def _create_project(workspace: Path, name: str) -> Path:
        project_root = workspace / name
        (project_root / "src" / name).mkdir(parents=True)
        (project_root / "tests").mkdir()
        (project_root / "Makefile").touch()
        (project_root / "pyproject.toml").touch()
        (project_root / "src" / name / "core.py").write_text(
            "from __future__ import annotations\n",
            encoding="utf-8",
        )
        (project_root / "tests" / "test_core.py").write_text(
            "from __future__ import annotations\n",
            encoding="utf-8",
        )
        return project_root

    def test_returns_success_with_project_and_file_tuples(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        alpha_root = self._create_project(tmp_path, "alpha")
        beta_root = self._create_project(tmp_path, "beta")

        result = u.Infra.iter_workspace_python_modules(workspace_root=tmp_path)

        assert result.is_success
        expected = {
            (alpha_root, alpha_root / "src" / "alpha" / "core.py"),
            (alpha_root, alpha_root / "tests" / "test_core.py"),
            (beta_root, beta_root / "src" / "beta" / "core.py"),
            (beta_root, beta_root / "tests" / "test_core.py"),
        }
        assert set(result.value) == expected
        assert all(
            project_root in file_path.parents
            for project_root, file_path in result.value
        )

    def test_excludes_files_in_skip_directories(self, tmp_path: Path) -> None:
        project_root = self._create_project(tmp_path, "gamma")
        (project_root / ".venv" / "lib").mkdir(parents=True)
        (project_root / ".venv" / "lib" / "ignored.py").write_text(
            "from __future__ import annotations\n",
            encoding="utf-8",
        )

        result = u.Infra.iter_workspace_python_modules(workspace_root=tmp_path)

        assert result.is_success
        assert (project_root, project_root / ".venv" / "lib" / "ignored.py") not in set(
            result.value,
        )

    def test_exclude_packages_parameter(self, tmp_path: Path) -> None:
        alpha_root = self._create_project(tmp_path, "alpha")
        self._create_project(tmp_path, "beta")

        result = u.Infra.iter_workspace_python_modules(
            workspace_root=tmp_path,
            exclude_packages=frozenset({"beta"}),
        )

        assert result.is_success
        assert result.value
        assert all(project_root == alpha_root for project_root, _ in result.value)

    def test_include_tests_parameter(self, tmp_path: Path) -> None:
        project_root = self._create_project(tmp_path, "delta")

        result = u.Infra.iter_workspace_python_modules(
            workspace_root=tmp_path,
            include_tests=False,
        )

        assert result.is_success
        expected = {(project_root, project_root / "src" / "delta" / "core.py")}
        assert set(result.value) == expected

    def test_empty_workspace_returns_empty_list(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()

        result = u.Infra.iter_workspace_python_modules(workspace_root=tmp_path)

        assert result.is_success
        assert result.value == []

    def test_non_existent_workspace_root_raises(self) -> None:
        missing_root = Path("/tmp/flext-missing-workspace-root")

        try:
            result = u.Infra.iter_workspace_python_modules(workspace_root=missing_root)
            assert result.is_failure
            assert result.error is not None
            assert "failed" in result.error
        except FileNotFoundError:
            assert True


__all__: list[str] = []
